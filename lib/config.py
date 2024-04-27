class Config:
    APEX: bool = True # Automatic Precision Enabled
    BATCH_SCHEDULER: bool = True
    BATCH_SIZE_TRAIN: int = 1
    BATCH_SIZE_VALID: int = 1
    DATA_VERSION: int = 1 # Dataset version
    DEBUG: bool = True
    GRADIENT_CHECKPOINTING: bool = True
    GRADIENT_ACCUMULATION_STEPS: int = 1
    MAX_LENGTH: int = 512 # Max number of tokens per sequence
    MAX_GRAD_NORM: int = 1000
    MODEL: str = "microsoft/deberta-v3-base" # Model Name
    N_FOLDS: int = 5 # Number of folds for Cross-validation
    NUM_CLASSES: int = 6 # Number of classes for classification models
    NUM_WORKERS: int = 6 # For parallel processing in dataset creation
    PRINT_FREQ: int = 20
    RANDOM_SEED: int = 29 # Common seed value for seeding everything.
