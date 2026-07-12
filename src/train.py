"""Main orchestration module for fine-tuning Qwen3-VL using LoRA."""

import time
import torch
from tqdm.auto import tqdm
import mlflow

from src import config
from src.dataset import build_dataloader
from src.model import (
    load_processor,
    load_model,
    apply_lora_wrapper,
    get_parameter_counts,
)


def run_training() -> None:
    """Orchestrates the entire training pipeline for the Qwen3-VL VLM."""
    config.initialize_directories()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Qwen3VL_RSRCC")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"CUDA available: {torch.cuda.is_available()}")

    processor = load_processor(config.MODEL_NAME)
    base_model = load_model(config.MODEL_NAME)
    model = apply_lora_wrapper(
        base_model,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
    )

    if torch.cuda.is_available():
        model.cuda()
    model.train()

    train_loader = build_dataloader(
        dataset_name=config.DATASET_NAME,
        split="train",
        processor=processor,
        streaming=config.STREAMING,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        buffer_size=config.BUFFER_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=config.WEIGHT_DECAY,
    )

    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.03 * config.MAX_STEPS),
        num_training_steps=config.MAX_STEPS,
    )

    trainable_params, total_params = get_parameter_counts(model)

    with mlflow.start_run(run_name="qwen3vl-lora"):
        mlflow.log_params(
            {
                "model": config.MODEL_NAME,
                "dataset": config.DATASET_NAME,
                "learning_rate": config.LEARNING_RATE,
                "epochs": config.NUM_EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "gradient_accumulation": config.GRADIENT_ACCUMULATION_STEPS,
                "max_length": config.MAX_LENGTH,
                "image_size": config.IMAGE_SIZE,
                "weight_decay": config.WEIGHT_DECAY,
                "optimizer": "AdamW",
                "scheduler": "CosineWarmup",
                "streaming": config.STREAMING,
                "lora_rank": 64,
                "lora_alpha": 128,
                "seed": config.SEED,
            }
        )

        mlflow.log_metrics(
            {
                "trainable_parameters": trainable_params,
                "total_parameters": total_params,
                "trainable_ratio": trainable_params / total_params,
            }
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        global_step = 0
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(
            total=config.MAX_STEPS,
            desc="Training",
        )

        should_stop = False

        for epoch in range(config.NUM_EPOCHS):
            if should_stop:
                break

            mlflow.log_metric(
                "epoch",
                epoch,
                step=global_step,
            )

            for batch_idx, batch in enumerate(train_loader):
                if torch.cuda.is_available():
                    batch = {
                        k: v.cuda(non_blocking=True) if hasattr(v, "cuda") else v
                        for k, v in batch.items()
                    }

                with torch.autocast(
                    device_type=config.DEVICE,
                    dtype=torch.bfloat16,
                ):
                    outputs = model(**batch)
                    raw_loss = outputs.loss
                    loss = raw_loss / config.GRADIENT_ACCUMULATION_STEPS

                loss.backward()

                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    progress_bar.update(1)

                    loss_value = raw_loss.item()
                    current_lr = scheduler.get_last_lr()[0]

                    mlflow.log_metric("train_loss", loss_value, step=global_step)
                    mlflow.log_metric("learning_rate", current_lr, step=global_step)

                    if torch.cuda.is_available():
                        mlflow.log_metric(
                            "gpu_memory_allocated_GB",
                            torch.cuda.memory_allocated() / (1024**3),
                            step=global_step,
                        )
                        mlflow.log_metric(
                            "gpu_memory_reserved_GB",
                            torch.cuda.memory_reserved() / (1024**3),
                            step=global_step,
                        )

                    if global_step % config.LOG_EVERY == 0:
                        print(
                            f"step={global_step:6d} | "
                            f"loss={loss_value:.4f} | "
                            f"lr={current_lr:.2e}"
                        )

                    if global_step % config.SAVE_EVERY == 0:
                        checkpoint = f"checkpoint-{global_step}"
                        model.save_pretrained(checkpoint)
                        processor.save_pretrained(checkpoint)

                        mlflow.log_artifacts(
                            checkpoint,
                            artifact_path=f"checkpoints/{checkpoint}",
                        )

                    if global_step >= config.MAX_STEPS:
                        should_stop = True
                        break

        progress_bar.close()
        print(f"Training completed after {time.time() - start_time:.2f} seconds.")

        model.save_pretrained("qwen3vl-lora-final")
        processor.save_pretrained("qwen3vl-lora-final")


if __name__ == "__main__":
    run_training()
