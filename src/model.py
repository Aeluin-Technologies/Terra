"""Model and processor initialization and LoRA wrapping configurations."""

import importlib.util
from typing import Any, Tuple
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, PeftModel


def load_processor(model_name: str) -> Any:
    """Loads the Hugging Face AutoProcessor for the specified model name.

    Args:
        model_name: Pretrained model name or path.

    Returns:
        The loaded processor.
    """
    return AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )


def load_model(model_name: str) -> Any:
    """Loads the main VLM model with optimal precision and attention settings.

    Args:
        model_name: Pretrained model name or path.

    Returns:
        The loaded model instance.
    """
    use_flash_attn = importlib.util.find_spec("flash_attn") is not None
    print(f"FlashAttention2 available: {use_flash_attn}")

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        **model_kwargs,
    )
    return model


def apply_lora_wrapper(
    model: Any,
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
) -> PeftModel:
    """Wraps the target model using LoRA config and enables gradient checkpointing.

    Args:
        model: The raw base model.
        r: Rank parameter for the LoRA adapters.
        lora_alpha: Alpha scaling parameter for LoRA.
        lora_dropout: Dropout probability for LoRA layers.

    Returns:
        The wrapped PeftModel ready for training.
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model.gradient_checkpointing_enable()
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def get_parameter_counts(model: Any) -> Tuple[int, int]:
    """Calculates the count of trainable and total parameters in the model.

    Args:
        model: The model to analyze.

    Returns:
        A tuple of (trainable_parameters_count, total_parameters_count).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
