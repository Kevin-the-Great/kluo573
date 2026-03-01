"""
GPU Load Generator - 目标 ~25% 利用率，低显存低功耗
用法: python gpu_load_25pct.py [--device 0] [--target 25]

原理: 通过 duty cycling 控制 GPU 利用率
- 执行一小段矩阵运算（拉高瞬时利用率）
- 然后 sleep 一段时间（降低平均利用率）
- 动态调节 compute/sleep 比例来逼近目标 UTL
"""

import torch
import time
import argparse
import subprocess
import re
import signal
import sys


def get_gpu_utilization(device_id: int) -> int:
    """通过 nvidia-smi 查询当前 GPU 利用率"""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={device_id}", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip())
    except Exception:
        return -1


def get_gpu_stats(device_id: int) -> dict:
    """查询 GPU 利用率、功耗、显存"""
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
    parser = argparse.ArgumentParser(description="GPU Load Generator")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--target", type=int, default=25, help="目标利用率 %%")
    parser.add_argument("--matrix-size", type=int, default=512,
                        help="矩阵大小 (越小显存越低，默认512)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    target_util = args.target

    # 用较小的矩阵来最小化显存占用
    # 512x512 float32 矩阵 ≈ 1MB，两个矩阵 + 结果 ≈ 3MB
    N = args.matrix_size
    a = torch.randn(N, N, device=device, dtype=torch.float32)
    b = torch.randn(N, N, device=device, dtype=torch.float32)

    # Duty cycle 参数（秒）
    compute_time = 0.005   # 每次计算持续时间
    sleep_time = 0.015     # 每次休息时间（初始 25% duty cycle）

    # PID 控制器参数
    Kp = 0.0003
    Ki = 0.00001
    integral = 0.0

    running = True
    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\n正在停止...")
    signal.signal(signal.SIGINT, signal_handler)

    print(f"GPU Load Generator 启动")
    print(f"  设备: cuda:{args.device}")
    print(f"  目标利用率: {target_util}%")
    print(f"  矩阵大小: {N}x{N} (显存 ≈ {N*N*4*3/1024/1024:.1f} MB)")
    print(f"  Ctrl+C 停止\n")
    print(f"{'时间':>6s} | {'UTL':>5s} | {'功耗':>8s} | {'显存':>12s} | {'compute':>8s} | {'sleep':>8s}")
    print("-" * 70)

    start = time.time()
    iter_count = 0

    while running:
        # === Compute phase ===
        t0 = time.time()
        while time.time() - t0 < compute_time:
            _ = torch.mm(a, b)
        torch.cuda.synchronize(device)

        # === Sleep phase ===
        if sleep_time > 0:
            time.sleep(sleep_time)

        iter_count += 1

        # 每 2 秒打印状态并调节参数
        if iter_count % 100 == 0:
            stats = get_gpu_stats(args.device)
            if stats:
                error = target_util - stats["util"]
                integral += error
                integral = max(-5000, min(5000, integral))  # anti-windup

                # 调节 sleep_time
                adjustment = Kp * error + Ki * integral
                sleep_time -= adjustment
                sleep_time = max(0.001, min(0.1, sleep_time))

                elapsed = time.time() - start
                print(f"{elapsed:6.0f}s | {stats['util']:4d}% | {stats['power']:6.1f} W "
                      f"| {stats['mem_used']:5d}/{stats['mem_total']:5d} MB "
                      f"| {compute_time*1000:5.1f} ms | {sleep_time*1000:5.1f} ms")

    # 清理
    del a, b
    torch.cuda.empty_cache()
    print("已停止，GPU 资源已释放。")


if __name__ == "__main__":
    main()