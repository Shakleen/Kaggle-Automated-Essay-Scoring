from typing import Tuple, List

from types import SimpleNamespace

config = SimpleNamespace(
    apex=True,
    batch_scheduler=True,
    batch_size_train=16,
    batch_size_valid=16,
    betas=[0.9, 0.999],
    data_version=1,
    debug=False,
    decoder_lr=2e-5,
    encoder_lr=2e-5,
    epochs=2,
    eps=1e-6,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    max_grad_norm=1000,
    max_length=512,
    min_lr=1e-6,
    model="microsoft/deberta-v3-base",
    n_folds=5,
    num_classes=6,
    num_cycles=0.5,
    num_warmup_steps=0,
    num_workers=6,
    print_freq=6,
    random_seed=20,
    scheduler="cosine",  # ['linear', 'cosine']
    train=True,
    train_folds=[0, 1, 2, 3, 4],
    weight_decay=0.01,
)


# class Config:
#     APEX: bool = True  # Automatic Precision Enabled
#     BATCH_SCHEDULER: bool = True
#     BATCH_SIZE_TRAIN: int = 16
#     BATCH_SIZE_VALID: int = 16
#     BETAS: Tuple[float] = (0.9, 0.999)
#     DATA_VERSION: int = 1  # Dataset version
#     DEBUG: bool = False
#     DECODER_LR: float = 2e-5
#     ENCODER_LR: float = 2e-5
#     EPOCHS: int = 2
#     EPS: float = 1e-6
#     GRADIENT_ACCUMULATION_STEPS: int = 1
#     GRADIENT_CHECKPOINTING: bool = True
#     MAX_GRAD_NORM: int = 1000
#     MAX_LENGTH: int = 512  # Max number of tokens per sequence
#     MIN_LR: float = 1e-6
#     MODEL: str = "microsoft/deberta-v3-base"  # Model Name
#     N_FOLDS: int = 5  # Number of folds for Cross-validation
#     NUM_CLASSES: int = 6  # Number of classes for classification models
#     NUM_CYCLES = 0.5
#     NUM_WARMUP_STEPS: int = 0
#     NUM_WORKERS: int = 6  # For parallel processing in dataset creation
#     PRINT_FREQ: int = 20
#     RANDOM_SEED: int = 29  # Common seed value for seeding everything.
#     SCHEDULER: str = "cosine"  # ['linear', 'cosine']
#     TRAIN: bool = True
#     TRAIN_FOLDS: List[int] = [0, 1, 2, 3, 4]
#     WEIGHT_DECAY: float = 0.01


# if Config.DEBUG:
#     Config.EPOCHS = 2
#     Config.TRAIN_FOLDS = [0]
