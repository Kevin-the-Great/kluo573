#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Dict, Any


def run_cmd(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)


def detect_gpus() -> List[int]:
    out = run_cmd(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"])
    return [int(x.strip()) for x in out.strip().splitlines() if x.strip()]


def sample_metrics() -> List[Dict[str, Any]]:
    query = (
        "index,name,power.draw,power.limit,"
        "utilization.gpu,utilization.memory,"
        "temperature.gpu,clocks.sm,clocks.mem,"
        "memory.used,memory.total,pstate"
    )
    out = run_cmd(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    ts = datetime.now().isoformat(timespec="seconds")

    rows: List[Dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 12:
            continue
        idx, name, pwr, plim, util_g, util_m, temp, smclk, memclk, mem_used, mem_total, pstate = parts
        rows.append({
            "timestamp": ts,
            "idx": int(idx),
            "name": name,
            "power_w": float(pwr),
            "power_limit_w": float(plim),
            "util_gpu": float(util_g),
            "util_mem": float(util_m),
            "temp_c": float(temp),
            "sm_clock_mhz": float(smclk),
            "mem_clock_mhz": float(memclk),
            "mem_used_mib": float(mem_used),
            "mem_total_mib": float(mem_total),
            "pstate": pstate,
        })
    return rows


def stress_one_gpu(gpu_id: int, dtype: str, size: int, mem_fill: float, chunk_mib: int, gpu_duty: float):
    """
    每张 GPU 一个进程：
      1) 可选：按比例填充可用显存（常驻，不释放）
      2) 用 GEMM 压测 GPU
      3) 用 gpu_duty 做占空比控制：让平均功耗/平均利用率更可控、更稳定
    """
    import torch
    device = torch.device(f"cuda:{gpu_id}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if dtype == "bf16":
        dt = torch.bfloat16
    elif dtype == "fp16":
        dt = torch.float16
    else:
        dt = torch.float32

    # -------- (1) VRAM fill：吃掉“当前空闲显存”的一定比例 --------
    fillers = []
    if mem_fill and mem_fill > 0:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)

        # 预留 buffer 给 CUDA context / allocator / GEMM workspace，避免 OOM
        reserve_bytes = int(1.5 * 1024**3)  # 1.5 GiB
        usable = max(0, free_bytes - reserve_bytes)
        target_bytes = int(usable * min(mem_fill, 0.95))

        chunk_bytes = int(chunk_mib * 1024**2)
        chunk_bytes = max(chunk_bytes, 256 * 1024**2)

        allocated = 0
        while allocated + chunk_bytes <= target_bytes:
            try:
                # uint8：1字节=1字节显存，最稳定
                fillers.append(torch.empty((chunk_bytes,), device=device, dtype=torch.uint8))
                allocated += chunk_bytes
            except RuntimeError:
                break

        if fillers:
            fillers[0].fill_(1)
        torch.cuda.synchronize(device)

    # -------- (2) GEMM 压测：矩阵乘 --------
    a = torch.randn((size, size), device=device, dtype=dt)
    b = torch.randn((size, size), device=device, dtype=dt)

    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize(device)

    # -------- (3) GPU duty：占空比控制平均功耗/平均util --------
    gpu_duty = max(0.0, min(1.0, gpu_duty))
    period = 0.2                     # 200ms 一个周期；太小会抖，太大响应慢
    busy_time = gpu_duty * period
    idle_time = period - busy_time

    while True:
        t0 = time.time()
        while (time.time() - t0) < busy_time:
            c = a @ b
            c = c + 1

        # 确保 busy 段的 kernel 真做完再进入 idle 段
        torch.cuda.synchronize(device)

        if idle_time > 0:
            time.sleep(idle_time)


def stress_one_cpu_worker(duty: float, mem_mib: int):
    """
    CPU 压测 worker（一个进程 ≈ 尝试占一个核）
    - duty: 0~1 占空比；1=满载，0.5≈50%
    - mem_mib: 每个 worker 占用并触碰多少 MiB 内存（总 RAM ≈ cpu_stress * mem_mib）
    """
    import math

    buf = None
    if mem_mib and mem_mib > 0:
        buf = bytearray(mem_mib * 1024 * 1024)

    duty = max(0.0, min(1.0, duty))
    period = 0.1
    busy_time = duty * period
    sleep_time = period - busy_time

    x = 0.0
    while True:
        t0 = time.time()
        while (time.time() - t0) < busy_time:
            x = math.sin(x + 1e-6) + math.cos(x + 2e-6)
            if buf is not None:
                buf[0] = (buf[0] + 1) & 0xFF
                buf[len(buf)//2] = (buf[len(buf)//2] + 1) & 0xFF
                buf[-1] = (buf[-1] + 1) & 0xFF
        if sleep_time > 0:
            time.sleep(sleep_time)


def main():
    ap = argparse.ArgumentParser(
        description="Live GPU monitor (rich) + optional GPU stress + VRAM fill + GPU duty + CPU stress/RAM fill."
    )

    ap.add_argument("--gpus", type=str, default="", help="GPU indices, e.g. '0,1' (default: auto-detect)")
    ap.add_argument("--interval", type=float, default=1.0, help="Refresh interval seconds (default: 1.0)")
    ap.add_argument("--duration", type=int, default=0, help="Run for N seconds then exit. 0 = forever")

    # GPU
    ap.add_argument("--stress", action="store_true", help="Enable GPU stress workload")
    ap.add_argument("--size", type=int, default=20480, help="Stress GEMM matrix size (default: 20480)")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Stress dtype")
    ap.add_argument("--gpu-duty", type=float, default=1.0,
                    help="GPU duty cycle 0~1 (default: 1.0). Lower -> lower average power/util.")
    ap.add_argument("--mem-fill", type=float, default=0.0,
                    help="Fill VRAM by ratio of current free memory (0~0.95). e.g. 0.9")
    ap.add_argument("--chunk-mib", type=int, default=1024,
                    help="Allocation chunk size in MiB when filling VRAM (default: 1024)")

    # CPU
    ap.add_argument("--cpu-stress", type=int, default=0,
                    help="Number of CPU worker processes (0=off). e.g. 64")
    ap.add_argument("--cpu-duty", type=float, default=1.0,
                    help="CPU duty cycle 0~1 (default: 1.0)")
    ap.add_argument("--cpu-mem-mib", type=int, default=0,
                    help="Per CPU worker memory in MiB (total ~ cpu_stress * cpu_mem_mib)")

    args = ap.parse_args()

    # check nvidia-smi
    try:
        _ = run_cmd(["nvidia-smi", "-L"])
    except Exception as e:
        print("ERROR: nvidia-smi not available or no NVIDIA GPU visible.", file=sys.stderr)
        print(str(e), file=sys.stderr)
        sys.exit(1)

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()] if args.gpus.strip() else detect_gpus()
    if not gpus:
        print("ERROR: No GPUs detected.", file=sys.stderr)
        sys.exit(1)

    # start stress processes
    import multiprocessing as mp
    gpu_procs = []
    cpu_procs = []

    if args.stress:
        for gid in gpus:
            p = mp.Process(
                target=stress_one_gpu,
                args=(gid, args.dtype, args.size, args.mem_fill, args.chunk_mib, args.gpu_duty),
                daemon=True
            )
            p.start()
            gpu_procs.append(p)

    if args.cpu_stress and args.cpu_stress > 0:
        for _ in range(args.cpu_stress):
            p = mp.Process(
                target=stress_one_cpu_worker,
                args=(args.cpu_duty, args.cpu_mem_mib),
                daemon=True
            )
            p.start()
            cpu_procs.append(p)

    # rich UI
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.live import Live
    except ImportError:
        print("ERROR: rich not installed. Run: pip install rich", file=sys.stderr)
        sys.exit(1)

    console = Console()
    start = time.time()

    def make_table(rows: List[Dict[str, Any]]) -> Table:
        title = (
            f"GPU Live Monitor (refresh {args.interval}s) | "
            f"gpu_stress={'ON' if args.stress else 'OFF'} duty={args.gpu_duty} mem_fill={args.mem_fill} | "
            f"cpu_stress={args.cpu_stress} duty={args.cpu_duty} mem/worker={args.cpu_mem_mib}MiB | Ctrl+C"
        )
        t = Table(title=title, show_lines=False)
        t.add_column("GPU", justify="right")
        t.add_column("Name")
        t.add_column("Power", justify="right")
        t.add_column("P-Limit", justify="right")
        t.add_column("Util", justify="right")
        t.add_column("Mem", justify="right")
        t.add_column("Temp", justify="right")
        t.add_column("SM clk", justify="right")
        t.add_column("Mem clk", justify="right")
        t.add_column("Pstate", justify="right")

        for r in sorted(rows, key=lambda x: x["idx"]):
            mem_pct = 100.0 * r["mem_used_mib"] / max(1.0, r["mem_total_mib"])
            t.add_row(
                str(r["idx"]),
                r["name"],
                f"{r['power_w']:.1f} W",
                f"{r['power_limit_w']:.0f} W",
                f"{r['util_gpu']:.0f}%",
                f"{r['mem_used_mib']:.0f}/{r['mem_total_mib']:.0f} MiB ({mem_pct:.0f}%)",
                f"{r['temp_c']:.0f} C",
                f"{r['sm_clock_mhz']:.0f} MHz",
                f"{r['mem_clock_mhz']:.0f} MHz",
                r["pstate"],
            )
        return t

    try:
        with Live(
            console=console,
            refresh_per_second=max(1, int(1 / max(0.05, args.interval))),
            transient=False
        ) as live:
            while True:
                if args.duration and (time.time() - start) >= args.duration:
                    break
                rows = sample_metrics()
                rows = [r for r in rows if r["idx"] in gpus]
                live.update(make_table(rows))
                time.sleep(max(0.05, args.interval))
    except KeyboardInterrupt:
        console.print("\n[bold]Stopped by user (Ctrl+C).[/bold]")
    finally:
        for p in gpu_procs + cpu_procs:
            if p.is_alive():
                p.terminate()
        for p in gpu_procs + cpu_procs:
            try:
                p.join(timeout=2)
            except Exception:
                pass


if __name__ == "__main__":
    main()
