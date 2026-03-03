"""
SmoothQuant for VLA-Adapter (Qwen2.5-0.5B)
============================================

这个文件做两件事：
1. smooth_qwen()  — 对 Qwen2.5 的 RMSNorm 和 Linear 做 smooth（转移 outlier）
2. quantize_qwen() — 把 Qwen2.5 的 Linear 替换成 W8A8Linear（fake quant）

用法（在 HPC 上）：
    from smoothquant_vla import smooth_qwen, quantize_qwen

    # 加载模型后...
    act_scales = torch.load("act_scales/vla_adapter_object.pt")
    smooth_qwen(model, act_scales, alpha=0.5)
    quantize_qwen(model)
    # 然后跑 eval...

注意：这个文件依赖 smoothquant 仓库里的 W8A8Linear。
      确保 smoothquant/ 在 Python path 里。
"""

import torch
import torch.nn as nn

# ============================================================
# 从 smoothquant 仓库导入 W8A8Linear
# 如果你把这个文件放在 smoothquant 仓库外面，
# 需要先: sys.path.insert(0, "/path/to/smoothquant")
# ============================================================
from smoothquant.fake_quant import W8A8Linear


# ============================================================
# Step 1: smooth 函数
# ============================================================

@torch.no_grad()
def smooth_ln_fcs_qwen(ln, fcs, act_scales, alpha=0.5):
    """
    对一组 RMSNorm + Linear 做 smooth。

    这个函数和 SmoothQuant 源码里的 smooth_ln_fcs_llama_like() 逻辑完全一样，
    只是去掉了 isinstance 检查（因为 Qwen2RMSNorm 不在源码的 assert 里）。

    参数:
        ln:         Qwen2RMSNorm 层（只有 gamma，没有 beta）
        fcs:        [nn.Linear, ...]  下游的 Linear 层列表
        act_scales: tensor, shape=(hidden_size,)  这组 Linear 的激活最大值
        alpha:      float, smooth 强度 (0~1), 默认 0.5

    数学原理:
        scales = act_scales^alpha / weight_scales^(1-alpha)
        ln.weight /= scales     ← RMSNorm 输出变小
        fc.weight *= scales     ← Linear 权重变大，补偿回来
    """
    if not isinstance(fcs, list):
        fcs = [fcs]

    # 基本检查
    for fc in fcs:
        assert isinstance(fc, nn.Linear), f"Expected nn.Linear, got {type(fc)}"
        assert ln.weight.numel() == fc.in_features == act_scales.numel(), \
            f"Size mismatch: ln.weight={ln.weight.numel()}, " \
            f"fc.in_features={fc.in_features}, act_scales={act_scales.numel()}"

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    # weight_scales: 取所有下游 Linear 权重的 per-channel 最大值
    # 每个 fc.weight 的 shape 是 (out_features, in_features)
    # .abs().max(dim=0) 沿 out_features 方向取最大值 → shape=(in_features,)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    # 计算 smooth 缩放因子
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    # 修改 RMSNorm 的 gamma（没有 beta）
    ln.weight.div_(scales)

    # 修改 Linear 的 weight
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_qwen(model, act_scales, alpha=0.5):
    """
    对 VLA-Adapter 模型中的 Qwen2.5 部分做 smooth。

    遍历 Qwen2.5 的 24 个 decoder layer，每层做两组 smooth：
      配对1: input_layernorm     → q_proj, k_proj, v_proj
      配对2: post_attention_layernorm → gate_proj, up_proj

    参数:
        model:      VLA-Adapter 模型（PrismaticVLM / OpenVLA 对象）
        act_scales: dict, 从 act_scales/vla_adapter_object.pt 加载的校准数据
        alpha:      float, smooth 强度

    act_scales 的 key 格式举例:
        "llm_backbone.llm.model.layers.0.self_attn.q_proj"
        "llm_backbone.llm.model.layers.0.mlp.gate_proj"
    """
    count = 0  # 统计处理了多少层

    for name, module in model.named_modules():
        # ---- 判断是否是 Qwen2 的 DecoderLayer ----
        # 方法: 检查 module 是否同时有 input_layernorm + self_attn + mlp
        # 这比 isinstance 检查更通用，不依赖特定 transformers 版本
        module_type = type(module).__name__
        if "DecoderLayer" not in module_type:
            continue

        # 再确认这个 DecoderLayer 属于 Qwen2.5（不是 Policy 里的层）
        if not hasattr(module, 'input_layernorm'):
            continue
        if not hasattr(module, 'self_attn'):
            continue
        if not hasattr(module.self_attn, 'q_proj'):
            continue

        # ---- 配对1: input_layernorm → q/k/v_proj ----
        attn_ln = module.input_layernorm
        qkv = [
            module.self_attn.q_proj,
            module.self_attn.k_proj,
            module.self_attn.v_proj,
        ]
        # act_scales 的 key: 用 q_proj 的 key（q/k/v 共享同一个 layernorm 输出）
        scale_key = name + ".self_attn.q_proj"
        if scale_key in act_scales:
            smooth_ln_fcs_qwen(attn_ln, qkv, act_scales[scale_key], alpha)
        else:
            print(f"  [WARNING] Scale key not found: {scale_key}")

        # ---- 配对2: post_attention_layernorm → gate_proj, up_proj ----
        ffn_ln = module.post_attention_layernorm
        fcs = [module.mlp.gate_proj, module.mlp.up_proj]
        scale_key = name + ".mlp.gate_proj"
        if scale_key in act_scales:
            smooth_ln_fcs_qwen(ffn_ln, fcs, act_scales[scale_key], alpha)
        else:
            print(f"  [WARNING] Scale key not found: {scale_key}")

        count += 1

    print(f"\n[smooth_qwen] Done! Smoothed {count} decoder layers")
    print(f"  每层修改了: 2 个 RMSNorm (gamma÷s) + 5 个 Linear (weight×s)")
    print(f"  共修改: {count * 2} 个 RMSNorm, {count * 5} 个 Linear")


# ============================================================
# Step 2: fake quant 函数
# ============================================================

@torch.no_grad()
def quantize_qwen(model, weight_quant="per_channel", act_quant="per_token",
                  quantize_bmm_input=False):
    """
    对 VLA-Adapter 模型中的 Qwen2.5 部分做 fake quantization。

    遍历 Qwen2.5 的所有层，把 nn.Linear 替换成 W8A8Linear：
      - Attention: q_proj, k_proj, v_proj, o_proj  (4个)
      - MLP:       gate_proj, up_proj, down_proj     (3个)
      - 每层共 7 个 Linear → W8A8Linear

    参数:
        model:               VLA-Adapter 模型
        weight_quant:        "per_channel" (推荐) 或 "per_tensor"
        act_quant:           "per_token" (推荐) 或 "per_tensor"
        quantize_bmm_input:  False (推荐, GQA 不适合量化 BMM)

    W8A8Linear.from_float() 做的事情:
        1. 把原始 FP16 权重量化到 INT8（round → 再反量化回 FP16）
        2. 推理时: 输入激活也做同样的 量化→反量化
        3. 然后做 FP16 矩阵乘法
        → 这就是 "fake quant": 模拟了量化误差，但实际计算还是 FP16
    """
    replaced_count = 0

    for name, module in model.named_modules():
        module_type = type(module).__name__

        # ---- 处理 Attention 模块 ----
        # 匹配 Qwen2Attention / Qwen2SdpaAttention / Qwen2FlashAttention2
        if "Attention" in module_type and hasattr(module, 'q_proj'):
            # 检查是否属于 LLM（排除 Policy 的 attention）
            if not isinstance(module.q_proj, nn.Linear):
                continue

            module.q_proj = W8A8Linear.from_float(
                module.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,  # False: BMM 保持 FP16
            )
            module.k_proj = W8A8Linear.from_float(
                module.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            module.v_proj = W8A8Linear.from_float(
                module.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            module.o_proj = W8A8Linear.from_float(
                module.o_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
            )
            replaced_count += 4

        # ---- 处理 MLP 模块 ----
        # 匹配 Qwen2MLP
        elif "MLP" in module_type and hasattr(module, 'gate_proj'):
            if not isinstance(module.gate_proj, nn.Linear):
                continue

            module.gate_proj = W8A8Linear.from_float(
                module.gate_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
            )
            module.up_proj = W8A8Linear.from_float(
                module.up_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
            )
            module.down_proj = W8A8Linear.from_float(
                module.down_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
            )
            replaced_count += 3

    print(f"\n[quantize_qwen] Done! Replaced {replaced_count} Linear → W8A8Linear")
    print(f"  weight_quant={weight_quant}, act_quant={act_quant}")
    print(f"  quantize_bmm_input={quantize_bmm_input}")
    print(f"  预期: 24层 × 7个 = 168 个 (实际: {replaced_count})")

    return model