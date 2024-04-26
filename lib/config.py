class Config:
    # Common seed value for seeding everything.
    RANDOM_SEED: int = 29

    # Number of folds for Cross-validation
    N_FOLDS: int = 5

    # Model Name
    MODEL: str = "microsoft/deberta-v3-base"