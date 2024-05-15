import os

from .config import config

class Paths:
    # Rata paths
    ROOT_DATA_PATH: str = "data"

    # Path to data
    TRAIN_CSV_PATH: str = "data/processed/train.csv"
    TEST_CSV_PATH: str = "data/processed/test.csv"
    
    # Competition data
    COMPETITION_TRAIN_CSV_PATH: str = "data/competition-data/train.csv"
    COMPETITION_TEST_CSE_PATH: str = "data/competition-data/test.csv"

    # Pursuade data
    PERSUADE_CSV_PATH: str = "data/external/persuade_2.0_human_scores_demo_id_github.csv"

    # Dataloader path
    DATA_LOADER_PATH: str = f"data/dataloader_v{config.data_version}"

    # Refit dataloader path
    REFIT_DATA_LOADER_PATH: str = f"data/refit_dataloader"

    ENG_WORDS_HX: str = "data/words.txt"

    # Output paths
    MODEL_OUTPUT_PATH: str = os.path.join("output", config.model)
    CONFIG_PATH: str = os.path.join(MODEL_OUTPUT_PATH, "config.pth")
    TOKENIZER_PATH: str = os.path.join(MODEL_OUTPUT_PATH, f"tokenizer_v{config.tokenizer_version}")
