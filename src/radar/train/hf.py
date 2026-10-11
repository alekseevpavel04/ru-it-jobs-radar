"""Model loading for training and inference (WSL training venv: torch, transformers, peft, unsloth).

Training loads the base in 4 bits (bitsandbytes NF4, double quantization, bf16 compute) with LoRA on
every linear projection: through Unsloth by default, through PEFT as the fallback (DECISIONS 9,
results/env/gpu_smoke.json). The dry run builds a two-layer random Qwen3 with the real tokenizer and
trains it on the CPU in float32, without quantization: it checks data, masking, callbacks, tracking,
checkpoints and the adapter round trip, not the GPU path.
"""

import json
from pathlib import Path
from typing import Any

from radar.train.config import RunConfig

TINY = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
}


def tiny_base(model: str, out_dir: Path, seed: int) -> Path:
    """A randomly initialised two-layer copy of `model`'s architecture, saved with its tokenizer."""
    import torch  # type: ignore[import-not-found]
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    if (out_dir / "config.json").exists():
        return out_dir
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(model)
    for key, value in TINY.items():
        setattr(config, key, value)
    if getattr(config, "layer_types", None):
        config.layer_types = config.layer_types[: TINY["num_hidden_layers"]]
    AutoModelForCausalLM.from_config(config).save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model).save_pretrained(out_dir)
    return out_dir


def _bnb() -> Any:
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_for_training(cfg: RunConfig, model_path: str, cpu: bool) -> tuple[Any, Any]:
    if cfg.backend == "unsloth" and not cpu:
        import torch
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]

        model, tok = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=cfg.train.max_seq_len,
            load_in_4bit=True,
            dtype=torch.bfloat16,
            use_exact_model_name=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.targets),
            use_gradient_checkpointing="unsloth",
            random_state=cfg.train.seed,
        )
        return model, tok

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cpu:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=_bnb(), dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.targets),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora), AutoTokenizer.from_pretrained(model_path)


def adapter_base(adapter: Path) -> str:
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    return str(config["base_model_name_or_path"])


def load_for_inference(adapter: Path | None, model: str | None, backend: str, max_seq_len: int, cpu: bool) -> Any:
    """Model and tokenizer for generation: an adapter on its base, or a base model alone."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if backend == "unsloth" and not cpu:
        from unsloth import FastLanguageModel

        llm, tok = FastLanguageModel.from_pretrained(
            model_name=str(adapter or model),
            max_seq_length=max_seq_len,
            load_in_4bit=True,
            dtype=torch.bfloat16,
        )
        FastLanguageModel.for_inference(llm)
    else:
        base = adapter_base(adapter) if adapter else str(model)
        if cpu:
            llm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        else:
            llm = AutoModelForCausalLM.from_pretrained(
                base, quantization_config=_bnb(), dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
            )
        if adapter:
            from peft import PeftModel

            llm = PeftModel.from_pretrained(llm, str(adapter))
        tok = AutoTokenizer.from_pretrained(str(adapter or model))
        llm.eval()
    tok.padding_side = "left"
    return llm, tok
