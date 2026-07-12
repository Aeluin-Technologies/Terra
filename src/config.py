"""Configuration parameters for the Qwen3-VL LoRA training pipeline."""

import os
from pathlib import Path
import torch

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
DATASET_NAME = "google/RSRCC"
OUTPUT_DIR = Path("outputs")
DATASET_PATH = "./dataset"

STREAMING = True
BUFFER_SIZE = 10_000
SEED = 42
MAX_LENGTH = 2048
IMAGE_SIZE = 512

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MAX_STEPS = 15000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 if STREAMING else min(2, os.cpu_count() or 1)
PIN_MEMORY = True
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = None

LOG_EVERY = 20
SAVE_EVERY = 1000


def initialize_directories() -> None:
    """Creates directory structures required for training outputs.

    Raises:
        OSError: If directories cannot be created.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
