"""
GPU Load Generator v6 - 最终版
用法: 
  python gpu_load_v6.py              # 默认 512，先试这个
  python gpu_load_v6.py --size 256   # 功耗更低
  python gpu_load_v6.py --size 1024  # 功耗更高

原理:
  默认 stream + 无 sleep + 无 sync = GPU 队列永远满着 = UTL 稳定
  通过矩阵大小控制功耗：
    256  → 功耗低，SM 占用极少
    512  → 功耗中等（推荐先试这个）
    1024 → 功耗较高
    2048 → 功耗拉满（别用）
"""

import torch
import time
import argparse
import subprocess
import signal


def get_gpu_stats(device_id: int) -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={device_id}",
             "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = [x.strip() for x in result.stdout.strip().split(",")]
        return {
            "util": int(parts[0]),
            "power": float(parts[1]),
            "mem_used": int(parts[2]),
            "mem_total": int(parts[3]),
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="GPU Load Generator v6")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--size", type=int, default=1024,
                        help="矩阵大小: 256(低功耗) / 512(推荐) / 1024(高)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    N = args.size

    a = torch.randn(N, N, device=device, dtype=torch.float32)
    b = torch.randn(N, N, device=device, dtype=torch.float32)
    mem_mb = N * N * 4 * 3 / 1024 / 1024

    running = True
    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\n正在停止...")
    signal.signal(signal.SIGINT, signal_handler)

    print(f"GPU Load Generator v6 启动")
    print(f"  设备: cuda:{args.device}")
    print(f"  矩阵: {N}x{N}，额外显存 ~{mem_mb:.0f}MB")
    print(f"  模式: 默认stream，持续提交，无sync无sleep")
    print(f"  如果功耗太高用 --size 256，太低用 --size 1024")
    print(f"  Ctrl+C 停止\n")
    print(f"{'时间':>6s} | {'UTL':>5s} | {'功耗':>8s} | {'显存':>14s}")
    print("-" * 50)

    start = time.time()
    last_report = start

    while running:
        _ = torch.mm(a, b)

        now = time.time()
        if now - last_report >= 30:
            torch.cuda.synchronize(device)
            last_report = now
            stats = get_gpu_stats(args.device)
            if stats:
                elapsed = now - start
                print(f"{elapsed:6.0f}s | {stats['util']:4d}% | {stats['power']:6.1f} W "
                      f"| {stats['mem_used']:5d}/{stats['mem_total']:5d} MB")

    del a, b
    torch.cuda.empty_cache()
    print("已停止，GPU 资源已释放。")


if __name__ == "__main__":
    main()