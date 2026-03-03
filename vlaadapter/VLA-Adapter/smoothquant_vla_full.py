"""
SmoothQuant for VLA-Adapter
============================

四个函数：
1. smooth_qwen()               — Qwen2.5: smooth（转移 outlier 到权重）
2. quantize_qwen()             — Qwen2.5: nn.Linear → W8A8Linear (fake quant)
3. quantize_vision_projector() — DINOv2 + SigLIP + Projector: nn.Linear → W8A8Linear (naive)
4. quantize_all()              — 一键全量化（1+2+3）

文件位置:
    /hpc2hdd/home/kluo573/vlaadapter/VLA-Adapter/smoothquant_vla.py

依赖:
    /hpc2hdd/home/kluo573/smoothquant/smoothquant/fake_quant.py  (W8A8Linear)
"""

import torch
import torch.nn as nn

from smoothquant.fake_quant import W8A8Linear


# ============================================================
# smooth 底层函数
# ============================================================

@torch.no_grad()
def smooth_ln_fcs_qwen(ln, fcs, act_scales, alpha=0.5):
    """
    对一组 RMSNorm + Linear 做 smooth。

    数学原理:
        scales = act_scales^alpha / weight_scales^(1-alpha)
        ln.weight /= scales     ← RMSNorm 输出变小
        fc.weight *= scales     ← Linear 权重变大，补偿回来
    """
    if not isinstance(fcs, list):
        fcs = [fcs]

    for fc in fcs:
        assert isinstance(fc, nn.Linear), f"Expected nn.Linear, got {type(fc)}"
        assert ln.weight.numel() == fc.in_features == act_scales.numel(), \
            f"Size mismatch: ln.weight={ln.weight.numel()}, " \
            f"fc.in_features={fc.in_features}, act_scales={act_scales.numel()}"

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


# ============================================================
# 1. smooth_qwen — 仅 Qwen2.5
# ============================================================

@torch.no_grad()
def smooth_qwen(model, act_scales, alpha=0.5):
    """
    遍历 Qwen2.5 的 24 个 DecoderLayer，每层做两组 smooth：
      配对1: input_layernorm          → q_proj, k_proj, v_proj
      配对2: post_attention_layernorm → gate_proj, up_proj
    """
    count = 0

    for name, module in model.named_modules():
        module_type = type(module).__name__
        if "DecoderLayer" not in module_type:
            continue
        if not hasattr(module, 'input_layernorm'):
            continue
        if not hasattr(module, 'self_attn'):
            continue
        if not hasattr(module.self_attn, 'q_proj'):
            continue

        # 配对1: input_layernorm → q/k/v_proj
        attn_ln = module.input_layernorm
        qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
        scale_key = name + ".self_attn.q_proj"
        if scale_key in act_scales:
            smooth_ln_fcs_qwen(attn_ln, qkv, act_scales[scale_key], alpha)
        else:
            print(f"  [WARNING] Scale key not found: {scale_key}")

        # 配对2: post_attention_layernorm → gate_proj, up_proj
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
# 2. quantize_qwen — Qwen2.5 fake quant
# ============================================================

@torch.no_grad()
def quantize_qwen(model, weight_quant="per_channel", act_quant="per_token",
                  quantize_bmm_input=False):
    """
    对 Qwen2.5 的 Attention + MLP 做 W8A8 fake quant。

    匹配逻辑:
      - 类名含 "Attention" 且有 q_proj → 替换 q/k/v/o_proj
      - 类名含 "MLP" 且有 gate_proj   → 替换 gate/up/down_proj

    不会误伤 ViT 层（TIMM 的 Attention 用 fused qkv 没有 q_proj，
    Mlp 类名大小写不匹配 "MLP"）。

    预期: 24层 × 7 = 168 个替换
    """
    replaced_count = 0

    for name, module in model.named_modules():
        module_type = type(module).__name__

        if "Attention" in module_type and hasattr(module, 'q_proj'):
            if not isinstance(module.q_proj, nn.Linear):
                continue
            module.q_proj = W8A8Linear.from_float(
                module.q_proj, weight_quant=weight_quant,
                act_quant=act_quant, quantize_output=quantize_bmm_input)
            module.k_proj = W8A8Linear.from_float(
                module.k_proj, weight_quant=weight_quant,
                act_quant=act_quant, quantize_output=quantize_bmm_input)
            module.v_proj = W8A8Linear.from_float(
                module.v_proj, weight_quant=weight_quant,
                act_quant=act_quant, quantize_output=quantize_bmm_input)
            module.o_proj = W8A8Linear.from_float(
                module.o_proj, weight_quant=weight_quant, act_quant=act_quant)
            replaced_count += 4

        elif "MLP" in module_type and hasattr(module, 'gate_proj'):
            if not isinstance(module.gate_proj, nn.Linear):
                continue
            module.gate_proj = W8A8Linear.from_float(
                module.gate_proj, weight_quant=weight_quant, act_quant=act_quant)
            module.up_proj = W8A8Linear.from_float(
                module.up_proj, weight_quant=weight_quant, act_quant=act_quant)
            module.down_proj = W8A8Linear.from_float(
                module.down_proj, weight_quant=weight_quant, act_quant=act_quant)
            replaced_count += 3

    print(f"\n[quantize_qwen] Done! Replaced {replaced_count} Linear → W8A8Linear")
    print(f"  weight_quant={weight_quant}, act_quant={act_quant}")
    print(f"  quantize_bmm_input={quantize_bmm_input}")
    print(f"  预期: 24层 × 7个 = 168 个 (实际: {replaced_count})")

    return model


# ============================================================
# 3. quantize_vision_projector — ViT + Projector naive W8A8
# ============================================================

@torch.no_grad()
def quantize_vision_projector(model, weight_quant="per_channel", act_quant="per_token",
                               include_lm_head=False):
    """
    对 DINOv2 + SigLIP + Projector 做 Naive W8A8 fake quant。

    替换范围：
      vision_backbone.featurizer       (DINOv2)   24 blocks × 4 =  96 层
      vision_backbone.fused_featurizer (SigLIP)    27 blocks × 4 = 108 层
      projector                        (MLP)        3 层
      language_model.lm_head           (可选)       1 层
                                                   合计 = 212 (或 208)
    """
    target_prefixes = ["vision_backbone.", "projector."]
    if include_lm_head:
        target_prefixes.append("language_model.lm_head")
    


    # 第一步: 收集（不能边遍历边改）
    all_modules = dict(model.named_modules())
    replacements = []

    for name, module in all_modules.items():
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.startswith(p) for p in target_prefixes):
            continue

        if "." in name:
            parent_path, attr_name = name.rsplit(".", 1)
        else:
            parent_path, attr_name = "", name

        parent = all_modules[parent_path] if parent_path else model
        replacements.append((name, parent, attr_name, module))

    # 第二步: 替换
    replaced_count = 0
    component_counts = {"DINOv2": 0, "SigLIP": 0, "Projector": 0, "lm_head": 0}

    for name, parent, attr_name, old_linear in replacements:
        new_linear = W8A8Linear.from_float(
            old_linear, weight_quant=weight_quant, act_quant=act_quant)
        setattr(parent, attr_name, new_linear)
        replaced_count += 1

        if "featurizer" in name and "fused" not in name:
            component_counts["DINOv2"] += 1
        elif "fused_featurizer" in name:
            component_counts["SigLIP"] += 1
        elif name.startswith("projector"):
            component_counts["Projector"] += 1
        elif "lm_head" in name:
            component_counts["lm_head"] += 1

    print(f"\n[quantize_vision_projector] Done! Replaced {replaced_count} Linear → W8A8Linear")
    print(f"  weight_quant={weight_quant}, act_quant={act_quant}")
    print(f"  各组件替换数量:")
    print(f"    DINOv2:    {component_counts['DINOv2']:3d} 层  (预期 96)")
    print(f"    SigLIP:    {component_counts['SigLIP']:3d} 层  (预期 113)")
    print(f"    Projector: {component_counts['Projector']:3d} 层  (预期 3)")
    if include_lm_head:
        print(f"    lm_head:   {component_counts['lm_head']:3d} 层  (预期 1)")
    total_expected = 212 + (1 if include_lm_head else 0)
    print(f"  合计: {replaced_count} 层 (预期 {total_expected})")

    if replaced_count != total_expected:
        print(f"  ⚠️  数量不匹配! 请检查模型结构是否有变化")
    else:
        print(f"  ✅ 数量匹配!")

    return model


# ============================================================
# 4. quantize_all — 一键全量化
# ============================================================

@torch.no_grad()
def quantize_all(model, act_scales, alpha=0.5,
                 weight_quant="per_channel", act_quant="per_token",
                 include_lm_head=False):
    """
    一键量化全模型（除 Policy Head）：
      Qwen2.5:       SmoothQuant → 168 层
      ViT+Projector:  Naive W8A8  → 212 层
      Policy Head:    不动
    """
    print("=" * 60)
    print("[quantize_all] 开始全模型量化")
    print("=" * 60)

    print("\n--- Step 1/3: Smooth Qwen2.5 ---")
    smooth_qwen(model, act_scales, alpha=alpha)

    print("\n--- Step 2/3: Quantize Qwen2.5 (SmoothQuant) ---")
    quantize_qwen(model, weight_quant=weight_quant, act_quant=act_quant)

    print("\n--- Step 3/3: Quantize ViT + Projector (Naive W8A8) ---")
    quantize_vision_projector(model, weight_quant=weight_quant, act_quant=act_quant,
                               include_lm_head=include_lm_head)

    total = 168 + 212 + (1 if include_lm_head else 0)
    print("\n" + "=" * 60)
    print(f"[quantize_all] 全模型量化完成! 共 {total} 个 Linear → W8A8Linear")
    print(f"  Qwen2.5:       SmoothQuant (smooth α={alpha:.1f} + W8A8)  168 层")
    print(f"  DINOv2/SigLIP: Naive W8A8 (无 smooth)                     204 层")
    print(f"  Projector:     Naive W8A8 (无 smooth)                       3 层")
    lm_str = "Naive W8A8  1 层" if include_lm_head else "保持 FP16"
    print(f"  lm_head:       {lm_str}")
    print(f"  Policy Head:   保持 FP16 (不量化)")
    print("=" * 60)

    return model