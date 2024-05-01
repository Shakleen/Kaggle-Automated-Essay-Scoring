# [Kaggle Competition: Learning Agency Lab - Automated Essay Scoring 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2)

The goal of this competition is to train a model to score student essays.

Submissions are scored based on the quadratic weighted kappa, which measures the agreement between two outcomes. This metric typically varies from 0 (random agreement) to 1 (complete agreement). In the event that there is less agreement than expected by chance, the metric may go below 0.

The competition dataset comprises about 24000 student-written argumentative essays. Each essay was scored on a scale of 1 to 6.

## Attempts Thus Far

1. Classification Approach
    * DeBERTA-v3-base 
        * 512 max length
            * Baseline [CV: 0.77, PL: 0.743]
            * Added extra token to tokenizer ("\n") [CV: 0.7768, PL: 0.781]
        * 1024 max length
            * Added extra token to tokenizer ("\n") [CV: 0.8009, PL: ???]
            * Added extra tokens to tokenizer ("\n", "  ") [CV: 0.8009, PL: ???]