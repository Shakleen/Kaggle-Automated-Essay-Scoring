from .config import Config

class Paths:
    # Rata paths
    ROOT_DATA_PATH: str = "data"

    # Path to data
    TRAIN_CSV_PATH: str = "data/processed/train.csv"
    TEST_CSV_PATH: str = "data/processed/test.csv"

    # Dataloader path
    DATA_LOADER_PATH: str = f"data/dataloader_v{Config.DATA_VERSION}"

    # Output paths
    TOKENIZER_PATH: str = f"output/tokenizer_v{Config.DATA_VERSION}"
