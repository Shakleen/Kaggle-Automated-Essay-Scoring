from types import SimpleNamespace

config = SimpleNamespace(
    apex=True,
    batch_scheduler=True,
    batch_size_train=32,
    batch_size_valid=32,
    betas=[0.9, 0.999],
    data_version=3,
    debug=False,
    decoder_lr=2e-5,
    encoder_lr=2e-5,
    epochs=2,
    eps=1e-6,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    lgbm_a=2.998,
    lgbm_b=1.092,
    lgbm_n_folds=7,
    max_grad_norm=1000,
    max_length=1024,
    min_lr=1e-6,
    model="microsoft/deberta-v3-xsmall",
    n_folds=7,
    negative_sample=False,
    negative_sample_partitions=3,
    num_classes=6,
    num_cycles=0.5,
    num_warmup_steps=0,
    num_workers=6,
    positive_classes=[0, 5],
    negative_classes=[1, 2, 3, 4],
    print_freq=6,
    random_seed=20,
    regression=False,
    scheduler="cosine",  # ['linear', 'cosine']
    stride=384,
    tokenizer_version=2,
    train=True,
    train_folds=[0, 1, 2, 3, 4, 5, 6, 7],
    weight_decay=0.01,
)


if config.debug:
    config.epochs = 1
    config.train_folds = [0]
