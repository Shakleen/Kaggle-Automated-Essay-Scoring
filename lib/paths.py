import os

from .config import config

class Paths:
    # Rata paths
    ROOT_DATA_PATH: str = "data"

    # Path to data
    TRAIN_CSV_PATH: str = "data/processed/train.csv"
    TEST_CSV_PATH: str = "data/processed/test.csv"

    # Dataloader path
    DATA_LOADER_PATH: str = f"data/dataloader_v{config.data_version}"

    # Output paths
    MODEL_OUTPUT_PATH: str = f"output/{config.model}"
    TOKENIZER_PATH: str = os.path.join(MODEL_OUTPUT_PATH, f"tokenizer_v{config.data_version}")

    # Best Model Path
    BEST_MODEL_PATH: str = "output/microsoft/deberta-v3-base/best_model"
