import subprocess
import time
import sys
import os
from datetime import datetime

# 配置参数
SCRIPT_PATH = "/hpc2hdd/home/kluo573/qwen3_0.6b_learning/test.py"
PYTHON_PATH = "/hpc2hdd/home/kluo573/qwen3_0.6b_learning/qwen_env/bin/python"
NUM_RUNS = -1  # 运行次数，设为 -1 表示无限循环
DELAY_BETWEEN_RUNS = 2  # 每次运行之间的延迟（秒）
LOG_DIR = "/hpc2hdd/home/kluo573/qwen3_0.6b_learning/logs"  # 日志目录

# 创建日志目录
os.makedirs(LOG_DIR, exist_ok=True)

def run_script(run_number):
    """运行一次脚本"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOG_DIR, f"run_{run_number}_{timestamp}.log")
    
    print(f"\n{'='*60}")
    print(f"第 {run_number} 次运行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志文件: {log_file}")
    print('='*60)
    
    try:
        # 调用脚本
        with open(log_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                [PYTHON_PATH, SCRIPT_PATH],
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
        
        # 读取并显示输出
        with open(log_file, 'r', encoding='utf-8') as f:
            output = f.read()
            print(output)
        
        if result.stderr:
            print("警告信息:")
            print(result.stderr)
            # 将警告也追加到日志
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n\n--- 警告信息 ---\n{result.stderr}")
        
        print(f"✓ 第 {run_number} 次运行完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 第 {run_number} 次运行失败!")
        print(f"错误码: {e.returncode}")
        print(f"错误输出:\n{e.stderr}")
        
        # 保存错误信息
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n--- 运行失败 ---\n")
            f.write(f"错误码: {e.returncode}\n")
            f.write(f"错误输出:\n{e.stderr}")
        
        return False
    
    except KeyboardInterrupt:
        print("\n用户中断运行")
        raise
    
    except Exception as e:
        print(f"✗ 发生异常: {e}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n--- 异常 ---\n{str(e)}")
        return False


def main():
    """主循环"""
    print(f"开始循环运行脚本: {SCRIPT_PATH}")
    print(f"Python 路径: {PYTHON_PATH}")
    print(f"日志目录: {LOG_DIR}")
    
    if NUM_RUNS == -1:
        print(f"模式: 无限循环 (按 Ctrl+C 停止)")
    else:
        print(f"模式: 运行 {NUM_RUNS} 次")
    
    print(f"间隔时间: {DELAY_BETWEEN_RUNS} 秒")
    
    run_count = 0
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    try:
        while True:
            run_count += 1
            
            # 如果设置了运行次数限制，检查是否达到
            if NUM_RUNS != -1 and run_count > NUM_RUNS:
                break
            
            # 运行脚本
            run_start = time.time()
            success = run_script(run_count)
            run_duration = time.time() - run_start
            
            if success:
                success_count += 1
            else:
                fail_count += 1
            
            print(f"本次运行耗时: {run_duration:.2f} 秒")
            
            # 如果还有下一次，等待一段时间
            if NUM_RUNS == -1 or run_count < NUM_RUNS:
                if DELAY_BETWEEN_RUNS > 0:
                    print(f"\n等待 {DELAY_BETWEEN_RUNS} 秒后继续...")
                    time.sleep(DELAY_BETWEEN_RUNS)
    
    except KeyboardInterrupt:
        print("\n\n收到中断信号，停止运行")
    
    finally:
        total_time = time.time() - start_time
        
        # 打印统计信息
        print(f"\n{'='*60}")
        print("运行统计:")
        print(f"  总运行次数: {run_count}")
        print(f"  成功: {success_count}")
        print(f"  失败: {fail_count}")
        print(f"  总耗时: {total_time:.2f} 秒")
        if run_count > 0:
            print(f"  平均每次: {total_time/run_count:.2f} 秒")
        print(f"  日志保存在: {LOG_DIR}")
        print('='*60)


if __name__ == "__main__":
    main()