"""
轻量验证脚本：只加载模型 + 量化，检查 W8A8Linear 是否真的替换成功。
不跑 LIBERO 仿真，几分钟就能跑完。
"""
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/hpc2hdd/home/kluo573/smoothquant")
sys.path.insert(0, "/hpc2hdd/home/kluo573/vlaadapter/VLA-Adapter")

from smoothquant.fake_quant import W8A8Linear


def count_modules(model):
    """统计模型中各类 Linear 模块的数量"""
    w8a8_count = 0
    linear_count = 0
    w8a8_by_prefix = {}
    linear_by_prefix = {}

    for name, m in model.named_modules():
        prefix = name.split(".")[0] if "." in name else name
        if isinstance(m, W8A8Linear):
            w8a8_count += 1
            w8a8_by_prefix[prefix] = w8a8_by_prefix.get(prefix, 0) + 1
        elif isinstance(m, nn.Linear):
            linear_count += 1
            linear_by_prefix[prefix] = linear_by_prefix.get(prefix, 0) + 1

    return w8a8_count, linear_count, w8a8_by_prefix, linear_by_prefix


def check_weight_quantized(module, name):
    """检查权重是否真的经过了量化（round 后的离散值）"""
    w = module.weight.float()
    # 取前 100 个元素
    sample = w.flatten()[:100]
    unique = sample.unique().numel()
    
    # INT8 per-channel: 每行最多 256 个值，100 个样本中 unique 应该很少
    print(f"\n  [{name}]")
    print(f"    type:  {type(module).__name__}")
    print(f"    dtype: {module.weight.dtype}, shape: {module.weight.shape}")
    print(f"    前10个值:    {w[0, :10].tolist()}")
    print(f"    unique/100:  {unique} (量化后应 << 100, 未量化接近 100)")
    
    # 检查是否有量化阶梯特征: 值是 scale 的整数倍
    row0 = w[0]
    scale = row0.abs().max() / 127.0  # INT8 的 scale
    rounded = (row0 / scale).round()
    reconstruction_error = ((row0 - rounded * scale).abs().max()).item()
    print(f"    scale (row 0): {scale.item():.6f}")
    print(f"    max reconstruction error: {reconstruction_error:.8f}")
    print(f"    是否量化: {'✅ YES' if reconstruction_error < 1e-4 else '❌ NO (误差太大)'}")


def main():
    QUANT_MODE = os.environ.get("QUANT_MODE", "smoothquant")
    QUANT_VISION = os.environ.get("QUANT_VISION", "0") == "1"
    ACT_SCALES_PATH = os.environ.get("ACT_SCALES_PATH", "act_scales/vla_adapter_object.pt")
    SMOOTH_ALPHA = float(os.environ.get("SMOOTH_ALPHA", "0.5"))

    print("=" * 60)
    print("SmoothQuant 量化验证脚本")
    print("=" * 60)
    print(f"  QUANT_MODE:      {QUANT_MODE}")
    print(f"  QUANT_VISION:    {QUANT_VISION}")
    print(f"  ACT_SCALES_PATH: {ACT_SCALES_PATH}")
    print(f"  SMOOTH_ALPHA:    {SMOOTH_ALPHA}")
    print()

    # 1. 加载模型
    print("[Step 1] Loading model...")
    from experiments.robot.robot_utils import get_model
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Union, Optional

    @dataclass
    class MinimalConfig:
        model_family: str = "openvla"
        pretrained_checkpoint: Union[str, Path] = os.environ.get(
            "CHECKPOINT", "outputs/LIBERO-Spatial-Pro")
        load_in_8bit: bool = False
        load_in_4bit: bool = False
        use_l1_regression: bool = True
        use_minivlm: bool = True
        num_diffusion_steps: int = 50
        use_film: bool = False
        num_images_in_input: int = 2
        use_proprio: bool = True
        center_crop: bool = True
        use_pro_version: bool = True

    cfg = MinimalConfig()
    model = get_model(cfg)

    # 2. 量化前状态
    print("\n[Step 2] 量化前模型状态:")
    w8a8_before, linear_before, _, _ = count_modules(model)
    print(f"  W8A8Linear: {w8a8_before}, nn.Linear: {linear_before}")

    # 3. 应用量化
    print("\n[Step 3] 应用量化...")
    from smoothquant_vla_full import smooth_qwen, quantize_qwen, quantize_vision_projector

    if QUANT_MODE in ("smoothquant", "naive_w8a8"):
        if QUANT_MODE == "smoothquant":
            print(f"  Loading act scales from {ACT_SCALES_PATH}...")
            act_scales = torch.load(ACT_SCALES_PATH, map_location="cpu")
            smooth_qwen(model, act_scales, alpha=SMOOTH_ALPHA)

        quantize_qwen(model)

    if QUANT_VISION:
        quantize_vision_projector(model)

    # 4. 量化后状态
    print("\n[Step 4] 量化后模型状态:")
    w8a8_after, linear_after, w8a8_by_prefix, linear_by_prefix = count_modules(model)
    print(f"  W8A8Linear: {w8a8_after}, nn.Linear: {linear_after}")
    print(f"  新增 W8A8Linear: {w8a8_after - w8a8_before}")
    print(f"\n  按组件统计 W8A8Linear:")
    for prefix, count in sorted(w8a8_by_prefix.items()):
        print(f"    {prefix}: {count}")
    print(f"\n  剩余 nn.Linear (未量化):")
    for prefix, count in sorted(linear_by_prefix.items()):
        print(f"    {prefix}: {count}")

    # 5. 抽样检查权重
    print("\n[Step 5] 抽样检查权重量化状态:")

    checked = {"qwen_attn": False, "qwen_mlp": False, "vit": False, "projector": False}
    for name, m in model.named_modules():
        if isinstance(m, W8A8Linear):
            if "self_attn.q_proj" in name and not checked["qwen_attn"]:
                check_weight_quantized(m, name)
                checked["qwen_attn"] = True
            elif "mlp.gate_proj" in name and not checked["qwen_mlp"]:
                check_weight_quantized(m, name)
                checked["qwen_mlp"] = True
            elif "vision_backbone" in name and not checked["vit"]:
                check_weight_quantized(m, name)
                checked["vit"] = True
            elif "projector" in name and not checked["projector"]:
                check_weight_quantized(m, name)
                checked["projector"] = True

        if all(checked.values()):
            break

    # 6. 做一次 dummy forward 确认可以正常推理
    print("\n[Step 6] Dummy forward test...")
    try:
        device = next(model.parameters()).device
        # 找一个 W8A8Linear 做单层 forward
        for name, m in model.named_modules():
            if isinstance(m, W8A8Linear):
                dummy_input = torch.randn(1, m.in_features, device=device, dtype=torch.float16)
                output = m(dummy_input)
                print(f"  Layer: {name}")
                print(f"  Input:  shape={dummy_input.shape}, dtype={dummy_input.dtype}")
                print(f"  Output: shape={output.shape}, dtype={output.dtype}")
                print(f"  ✅ Forward pass successful!")
                break
    except Exception as e:
        print(f"  ❌ Forward failed: {e}")

    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
