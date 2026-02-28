#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

# ==========================================================================================
# 你只需要改 CONFIG，然后在 VSCode 里运行本文件（run_monitor.py）即可。
# 它会把 CONFIG 里的参数拼成命令行，去调用同目录的 gpu_live_monitor.py。
#
# 你想控制的东西对应关系：
#   - GPU 平均功耗/平均利用率：主要看 gpu_duty（最稳）+ size/dtype（负载强度）
#   - GPU 显存占用率：主要看 mem_fill（最直接）+ chunk_mib（分配粒度）
#   - CPU 利用率：主要看 cpu_stress（多少核）+ cpu_duty（占空比）
#   - 主机内存(RAM)占用：主要看 cpu_mem_mib（每个CPU进程占多少MiB，总≈cpu_stress*cpu_mem_mib）
# ==========================================================================================

CONFIG = {
    # =====================
    # 监控刷新与运行时长
    # =====================
    "gpus": "0,1",         # 用哪些GPU： "0,1" / "0" / "1" / ""(自动检测所有可见GPU)
    "interval": 1,         # 监控窗口刷新间隔(秒)。0.5更流畅但更频繁调用nvidia-smi；2~5更省资源
    "duration": 0,         # 运行多久自动退出(秒)。0=一直跑，直到你Ctrl+C

    # =====================
    # GPU 计算负载（影响GPU功耗/利用率）
    # =====================
    "stress": True,        # True=开启GPU压测（矩阵乘）；False=只监控（GPU功耗可能接近idle）
    "dtype": "bf16",       # 压测精度："bf16"/"fp16"/"fp32"。bf16/fp16通常更容易跑满、更快
    "size": 10240,         # 矩阵规模（越大负载越重 -> GPU util/功耗更高；太大可能OOM或速度反而下降）
                           # 常用建议：8192/10240/12288/16384/20480（显存越满时不要设太大）

    # 【关键】GPU占空比（最推荐用它控“平均功耗=上限的多少%”）
    "gpu_duty": 0.2,      # 0~1：1=持续满载；0.25≈工作25%时间、休息75%时间 -> 平均功耗大幅下降
                           # 目标功耗=PowerLimit*25% (例如300W*0.25=75W) 就从0.25开始，然后看功耗微调：
                           #   - 功耗偏高：gpu_duty从0.25降到0.20
                           #   - 功耗偏低：gpu_duty从0.25升到0.30
                           # 注意：功耗不是严格线性，但gpu_duty是最容易稳定控制的旋钮

    # =====================
    # GPU 显存占用（影响显存占用率）
    # =====================
    "mem_fill": 0.2,      # 0~0.95：额外占用“当前空闲显存”的比例（脚本内部会预留约1.5GiB避免OOM）
                           # 这不是“占总显存的比例”，所以想达到“总显存25%”通常要试一次再微调：
                           #   - 显存偏低：mem_fill稍增（例如0.18->0.22）
                           #   - 显存偏高：mem_fill稍减（例如0.18->0.15）
                           # 建议范围：0.05~0.30 用来做低/中占用；0.85~0.92 用来接近满显存（高风险OOM）

    "chunk_mib": 512,      # 填显存的块大小(MiB)。1024=块大分配次数少；512=更容易在碎片化时“吃到更准”
                           # 如果你发现mem_fill不太能到目标（卡在某个比例），试试把1024改成512

    # =====================
    # CPU 利用率（影响CPU%）
    # =====================
    "cpu_stress": 0,       # 开多少个CPU压测进程（一个进程≈占一个核）。
                           # 如果你想把CPU打到接近满载：
                           #   - 你实际可用64核 -> 设64
                           #   - 你实际只分到16核 -> 设16
                           # 不确定就先32，看CPU%再调。设太大只会抢占更严重，不会更“满”

    "cpu_duty": 1.0,       # 0~1 CPU占空比：1=满载；0.5≈50%；0.2≈20%

    # =====================
    # 主机内存(RAM)占用（影响系统内存占用）
    # =====================
    "cpu_mem_mib": 0,      # 每个CPU进程占用并持续触碰多少MiB内存
                           # 总内存大约=cpu_stress*cpu_mem_mib（+少量进程开销）
                           # 例子：cpu_stress=64, cpu_mem_mib=128 -> 约8GB
                           #      cpu_stress=64, cpu_mem_mib=512 -> 约32GB
}

MONITOR_SCRIPT = Path(__file__).with_name("gpu_live_monitor.py")


def build_cmd(cfg: dict) -> list:
    cmd = [sys.executable, str(MONITOR_SCRIPT)]

    cmd += ["--gpus", str(cfg.get("gpus", ""))]
    cmd += ["--interval", str(cfg.get("interval", 1))]
    cmd += ["--duration", str(cfg.get("duration", 0))]

    if cfg.get("stress", False):
        cmd += ["--stress"]
        cmd += ["--dtype", str(cfg.get("dtype", "bf16"))]
        cmd += ["--size", str(cfg.get("size", 20480))]
        cmd += ["--gpu-duty", str(cfg.get("gpu_duty", 1.0))]
        cmd += ["--mem-fill", str(cfg.get("mem_fill", 0.0))]
        cmd += ["--chunk-mib", str(cfg.get("chunk_mib", 1024))]
    else:
        # 不压测也允许填显存（如果你想只控显存占用）
        cmd += ["--gpu-duty", str(cfg.get("gpu_duty", 1.0))]
        cmd += ["--mem-fill", str(cfg.get("mem_fill", 0.0))]
        cmd += ["--chunk-mib", str(cfg.get("chunk_mib", 1024))]

    if int(cfg.get("cpu_stress", 0)) > 0:
        cmd += ["--cpu-stress", str(cfg.get("cpu_stress", 0))]
        cmd += ["--cpu-duty", str(cfg.get("cpu_duty", 1.0))]
        cmd += ["--cpu-mem-mib", str(cfg.get("cpu_mem_mib", 0))]

    return cmd


def main():
    if not MONITOR_SCRIPT.exists():
        print(
            f"ERROR: {MONITOR_SCRIPT} not found. Put run_monitor.py and gpu_live_monitor.py in the same folder.",
            file=sys.stderr
        )
        sys.exit(1)

    cmd = build_cmd(CONFIG)
    print("Launching:\n  " + " ".join(cmd) + "\n")

    # 继承当前终端，rich 实时界面会正常显示
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
