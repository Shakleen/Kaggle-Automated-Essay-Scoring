# [Kaggle Competition: Learning Agency Lab - Automated Essay Scoring 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2)

This repository contains code for the **Automated Essay Scoring 2.0** competition on Kaggle. In this competition, the task was to evaluate essays written by students on various topics and score them on a scale of 1 to 6, where 1 is the lowest and 6 is the highest score. The goal was to build a model that could automatically assign scores to the essays based on their content.

## Challenge Overview

The main challenges in this competition were:

- **Class Imbalance**: The dataset exhibited significant class imbalance, similar to a bell curve. Most of the essays had scores of 3, 4, or 5, while very few essays were scored 1, 2, or 6. This imbalance made it difficult to effectively train a model for rare classes.
  
- **Topic Imbalance**: In addition to class imbalance, there was also a discrepancy in the number of examples per topic. Some topics had over 1000 examples, while others had as few as 100. This made it challenging to generalize across topics.

## Approach

### 1. **DeBERTa-based Model for Text Scoring**

The core of the solution was based on a pre-trained **DeBERTa** (Decoding-enhanced BERT with disentangled attention) model for natural language understanding. Here’s how the model was designed and implemented:

- **Max Sequence Length**: Essays often exceeded the token limit of the DeBERTa model. To ensure the model processed the entire essay, the maximum sequence length was set to 1024 tokens. Essays longer than 1024 tokens were split into overlapping segments. Each segment was scored individually, and the final score for the essay was calculated as the **median score** across all segments.

- **Regression vs. Classification**: I experimented with both regression and classification approaches for scoring the essays. Regression yielded better results, especially when paired with post-processing techniques to round up or down scores based on certain thresholds.

- **Model Ensemble**: Instead of using a single DeBERTa model, I trained an ensemble of **7 DeBERTa models**. The final score was computed as the average of the predictions from these models, which improved robustness and overall performance.

### 2. **Manual Feature Engineering**

In addition to using DeBERTa for text scoring, I incorporated several manually engineered features to enrich the input data:

- **Text Statistics**: Counting statistics such as the minimum, median, and maximum values across paragraphs, sentences, and words.
- **Spelling Mistakes**: Counting the number of spelling mistakes in each essay.
- **Grammatical Inconsistencies**: Detecting grammatical inconsistencies using a pre-trained **T5 model**.
- **Sentence Cohesion**: Calculating the cohesion between sentences using **sentencepiece transformer** embeddings.

### 3. **LightGBM Ensemble for Final Scoring**

The manually engineered features, along with the predictions from the DeBERTa model, were then fed into an ensemble of **25 LightGBM models**. These models produced the final predictions, leveraging both the deep learning model's text understanding and the crafted features.

### 4. **Experiment Tracking with Weights & Biases**

To efficiently manage and track the experiments, datasets, models, and metrics, I used **Weights & Biases**. This allowed for rapid iteration and systematic improvement of the model pipeline, ensuring the best performance throughout the development process.

## Results

This approach, which combined a powerful pre-trained model with manual feature engineering and an ensemble of models, resulted in an effective solution for the Automated Essay Scoring 2.0 competition. The use of ensemble techniques and post-processing further improved the model’s ability to handle class and topic imbalance, ultimately leading to improved accuracy in scoring.

## Requirements

To reproduce the results, ensure you have the following libraries and tools installed:

- Python 3.x
- Transformers (for DeBERTa and T5 models)
- LightGBM
- Sentencepiece
- Weights & Biases (for experiment tracking)

You can install the necessary Python packages by running:

```bash
pip install -r requirements.txt
```

## Conclusion

This project demonstrated how combining advanced NLP models like DeBERTa with feature engineering and model ensembling can lead to robust performance in challenging tasks like automated essay scoring. By addressing class and topic imbalances and leveraging the strengths of different models, this approach achieved competitive results in the Kaggle competition.

Feel free to fork, experiment, and improve upon this approach!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Resources
1. [Holistic Rating](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf)
