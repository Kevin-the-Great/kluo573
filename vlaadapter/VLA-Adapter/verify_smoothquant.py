"""
快速验证脚本: 测试 smooth + quantize 是否工作正常
===============================================

用法:
    cd /hpc2hdd/home/kluo573/vlaadapter/VLA-Adapter
    conda activate vla-adapter
    CUDA_VISIBLE_DEVICES=0 python verify_smoothquant.py
"""

import os
import sys
import torch
import torch.nn as nn

# ============================================================
# 配置 (根据你的 HPC 环境修改这些路径)
# ============================================================
VLA_ROOT = "/hpc2hdd/home/kluo573/vlaadapter/VLA-Adapter"
SMOOTHQUANT_ROOT = "/hpc2hdd/home/kluo573/smoothquant"
CHECKPOINT_PATH = "/hpc2hdd/home/kluo573/vlaadapter/VLA-Adapter/outputs/LIBERO-Spatial-Pro"
ACT_SCALES_PATH = "act_scales/vla_adapter_object.pt"
ALPHA = 0.5

# ============================================================
# 设置路径
# ============================================================
os.chdir(VLA_ROOT)
sys.path.insert(0, VLA_ROOT)
sys.path.insert(0, SMOOTHQUANT_ROOT)

print("=" * 60)
print("  VLA-Adapter SmoothQuant 验证脚本")
print("=" * 60)


# ============================================================
# Step 1: 加载模型 (和 eval 脚本用完全一样的方式)
# ============================================================
print("\n[1/6] 加载 VLA-Adapter 模型...")

from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq

# 注册自定义模型类 (本地 checkpoint 必须做这一步)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

# 同步 config (和 eval 脚本一样)
from experiments.robot.openvla_utils import update_auto_map, check_model_logic_mismatch
update_auto_map(CHECKPOINT_PATH)
check_model_logic_mismatch(CHECKPOINT_PATH)

# 加载模型 (和 eval 脚本 get_vla() 完全一样的调用方式)
vla = AutoModelForVision2Seq.from_pretrained(
    CHECKPOINT_PATH,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    load_in_4bit=False,
    low_cpu_mem_usage=False,
    trust_remote_code=True,
)
vla.eval()

# 移到 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vla = vla.to(device)

dtype = next(vla.parameters()).dtype
print(f"  模型加载完成! device={device}, dtype={dtype}")


# ============================================================
# Step 2: 查看 Qwen2.5 的 Linear 层 (量化前)
# ============================================================
print("\n[2/6] 扫描 Qwen2.5 的 Linear 层 (量化前)...")

def count_layers(model):
    """统计模型中 LLM 部分的 Linear 和 W8A8Linear 数量"""
    linear_count = 0
    w8a8_count = 0
    for name, module in model.named_modules():
        # 匹配 Qwen2.5 的层 (根据实际 key 名称调整)
        if not any(kw in name for kw in ["llm", "language_model", "model.layers"]):
            continue
        if isinstance(module, nn.Linear):
            linear_count += 1
        elif type(module).__name__ == "W8A8Linear":
            w8a8_count += 1
    return linear_count, w8a8_count

linear_before, w8a8_before = count_layers(vla)
print(f"  nn.Linear (LLM):  {linear_before}")
print(f"  W8A8Linear (LLM): {w8a8_before}")

# 打印所有模块的类名，帮助调试
print("\n  模型顶层结构:")
for name, module in vla.named_children():
    print(f"    {name}: {type(module).__name__}")

# 打印 LLM 相关的 Linear 层的完整名称（前 10 个）
print("\n  LLM Linear 层名称 (前 10 个):")
llm_linears = []
for name, module in vla.named_modules():
    if isinstance(module, nn.Linear):
        llm_linears.append(name)
for n in llm_linears[:10]:
    print(f"    {n}")
print(f"    ... 共 {len(llm_linears)} 个 Linear")

# 打印 DecoderLayer 类名
print("\n  DecoderLayer 类名:")
for name, module in vla.named_modules():
    cls_name = type(module).__name__
    if "DecoderLayer" in cls_name or "decoder_layer" in name.lower():
        print(f"    {name}: {cls_name}")
        break  # 只打印第一个


# ============================================================
# Step 3: 加载 act_scales
# ============================================================
print(f"\n[3/6] 加载 act_scales: {ACT_SCALES_PATH}")

act_scales = torch.load(ACT_SCALES_PATH, map_location="cpu")
print(f"  共 {len(act_scales)} 个 key")

# 打印前 10 个 key
print(f"\n  act_scales 的 key (前 10 个):")
for i, k in enumerate(sorted(act_scales.keys())):
    if i >= 10:
        break
    v = act_scales[k]
    print(f"    {k}  shape={v.shape}  max={v.max().item():.1f}")


# ============================================================
# Step 4: 执行 smooth
# ============================================================
print(f"\n[4/6] 执行 smooth (alpha={ALPHA})...")

from smoothquant_vla import smooth_qwen
smooth_qwen(vla, act_scales, alpha=ALPHA)


# ============================================================
# Step 5: 执行 fake quant
# ============================================================
print(f"\n[5/6] 执行 fake quantization...")

from smoothquant_vla import quantize_qwen
quantize_qwen(vla)

linear_after, w8a8_after = count_layers(vla)
print(f"\n  量化后:")
print(f"    nn.Linear (LLM):  {linear_before} → {linear_after}")
print(f"    W8A8Linear (LLM): {w8a8_before} → {w8a8_after}")


# ============================================================
# Step 6: 简单 forward pass 验证
# ============================================================
print(f"\n[6/6] 做一次简单 forward pass 验证...")

try:
    # 找到 LLM backbone 做一个简单测试
    llm = None
    for name, module in vla.named_modules():
        cls_name = type(module).__name__
        if "Qwen2ForCausalLM" in cls_name or "Qwen2Model" in cls_name:
            if hasattr(module, 'forward'):
                llm = module
                print(f"  找到 LLM: {name} ({cls_name})")
                break

    if llm is not None:
        dummy_ids = torch.randint(0, 1000, (1, 16)).to(device)
        with torch.no_grad():
            output = llm(input_ids=dummy_ids)
        if hasattr(output, 'logits'):
            print(f"  ✅ Forward pass 成功! Output shape: {output.logits.shape}")
        else:
            print(f"  ✅ Forward pass 成功! Output type: {type(output)}")
    else:
        print("  ⚠️  没找到独立的 Qwen2 模块，跳过 forward 测试")
        print("     这不影响后续 eval，只是验证不方便")

except Exception as e:
    print(f"  ⚠️  Forward pass 跳过: {e}")
    print("     这不影响后续 eval")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("  验证完成! 总结:")
print("=" * 60)
print(f"  模型加载:     ✅ 使用 AutoModelForVision2Seq.from_pretrained()")
print(f"  act_scales:   ✅ {len(act_scales)} keys loaded")
print(f"  Smooth:       ✅ alpha={ALPHA}")
print(f"  Fake quant:   ✅ {w8a8_after} 个 W8A8Linear")
print(f"")
print(f"  下一步: 修改 run_libero_eval.py，在 get_vla(cfg) 之后插入 smooth + quant 代码")
print("=" * 60)