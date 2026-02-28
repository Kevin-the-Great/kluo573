import argparse
import os
import shutil
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForVision2Seq, BitsAndBytesConfig


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize a VLA checkpoint to 8-bit")
    parser.add_argument(
        "--src",
        default="outputs/LIBERO-Long-Pro",
        help="Source checkpoint directory (full fp16/fp32 weights)",
    )
    parser.add_argument(
        "--dst",
        default="outputs/LIBERO-Long-Pro-8bit",
        help="Output directory for 8-bit weights",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing dst directory",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=DTYPE_MAP.keys(),
        help="Compute dtype to use while loading the 8-bit model",
    )
    parser.add_argument(
        "--trust-remote-code",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable trust_remote_code for custom model classes",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map for accelerate (e.g. 'auto', 'cuda:0').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=6.0,
        help="Outlier threshold. Set to 0.0 for pure INT8 (no outlier decomposition).",
    )
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    dtype: Optional[torch.dtype] = DTYPE_MAP.get(name)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {name}")
    return dtype


def _is_tensor_like(value: object) -> bool:
    return torch.is_tensor(value) or isinstance(value, torch.nn.Parameter)


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.src):
        raise FileNotFoundError(f"Source checkpoint not found: {args.src}")

    if os.path.exists(args.dst):
        if args.overwrite:
            shutil.rmtree(args.dst)
        else:
            raise RuntimeError(f"{args.dst} already exists; use --overwrite to replace it")

    print(f"Copying {args.src} -> {args.dst} ...", flush=True)
    shutil.copytree(args.src, args.dst)

    print("Loading config with trust_remote_code=True ...", flush=True)
    config = AutoConfig.from_pretrained(args.dst, trust_remote_code=args.trust_remote_code)

    # 唯一的改动：用 BitsAndBytesConfig 控制 threshold
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=args.threshold,
    )
    
    print(f"Loading model in 8bit (threshold={args.threshold}) ...", flush=True)
    low_cpu_mem_usage = True
    model = AutoModelForVision2Seq.from_pretrained(
        args.dst,
        config=config,
        quantization_config=quantization_config,  # 改这里
        torch_dtype=_resolve_dtype(args.dtype),
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
    )

    print("Saving 8bit weights ...", flush=True)
    state_dict = model.state_dict()
    tensor_state_dict = {k: v for k, v in state_dict.items() if _is_tensor_like(v)}
    if len(tensor_state_dict) != len(state_dict):
        print(
            "Warning: some non-tensor entries were found in state_dict; "
            "saving only tensor weights.",
            flush=True,
        )
    
    # 用 HF 官方方法，但先临时替换 state_dict
    original_state_dict = model.state_dict
    try:
        model.state_dict = lambda: tensor_state_dict  # 临时让 save_pretrained 只看到张量
        model.save_pretrained(args.dst)
    except Exception as exc:
        print(
            f"Warning: save_pretrained failed ({exc}); saving raw state_dict instead.",
            flush=True,
        )
        torch.save(tensor_state_dict, os.path.join(args.dst, "pytorch_model.bin"))
        model.config.save_pretrained(args.dst)
    finally:
        model.state_dict = original_state_dict  # 恢复原方法
    print(f"✅ 8bit model saved to: {args.dst}")


if __name__ == "__main__":
    main()