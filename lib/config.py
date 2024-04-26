class Config:
    DEBUG: bool = True

    # Common seed value for seeding everything.
    RANDOM_SEED: int = 29

    # Number of folds for Cross-validation
    N_FOLDS: int = 5

    # Model Name
    MODEL: str = "microsoft/deberta-v3-base"

    # Max number of tokens per sequence
    MAX_LENGTH: int = 512

    # Batch Size for train and test
    BATCH_SIZE_TRAIN: int = 1
    BATCH_SIZE_VALID: int = 1

    # For parallel processing in dataset creation
    NUM_WORKERS: int = 6
