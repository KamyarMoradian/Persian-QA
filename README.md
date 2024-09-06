# Persian Question Answering Model

## Project Overview

This project focuses on developing and fine-tuning question answering models for the Persian language. evaluated and improved the performance of two models - large_roberta_xlm_persian and PQuAD_answering_question_parsbert - using the PersianQA dataset. The project aims to enhance the capability of these models in understanding and answering questions in Persian.

## Dataset

Utilized the PersianQA dataset for training and evaluation. This dataset is specifically designed for question answering tasks in Persian and contains a diverse range of questions and contexts.

Dataset link: https://github.com/sajjjadayobi/PersianQA

## Models

1. **large_roberta_xlm_persian**: A multilingual RoBERTa model fine-tuned for Persian.
2. **PQuAD_answering_question_parsbert**: A BERT-based model specifically trained on Persian question answering tasks.

## Methodology

Designed approach consists of several key steps:

1. **Data Preparation**: loaded and preprocessed the PersianQA dataset, ensuring it's in the correct format for the models.

2. **Performance Evaluation**: assessed the initial performance of both models on the PersianQA dataset using custom evaluation metrics.

3. **Fine-tuning**: fine-tuned both models using the PersianQA dataset to improve their performance on Persian question answering tasks. Fine-tuning is done by experimenting with different hyper-parameters.

4. **Post-fine-tuning Evaluation**: re-evaluated the models after fine-tuning to measure the improvement in performance.

## Evaluation Metrics

Used two primary methods to evaluate the performance of utilized models:

1. **SQuAD v2 Metric**: employed the squad_v2_metric to assess the models' performance. This metric provides a comprehensive evaluation of the models' accuracy, including exact match and F1 scores.

2. **Judge Similarity**: introduced an additional evaluation method called judge_similarity. This method utilizes the all-MiniLM-L6-v2 model to generate encodings of the models' outputs and compare them to encodings of the actual answers using cosine similarity. This approach allows us to evaluate the semantic similarity between the predicted and ground truth answers.

### Judge Similarity Process

1. Generate model outputs for the test set
2. Use all-MiniLM-L6-v2 to create encodings for both model outputs and ground truth answers
3. Calculate cosine similarity between the encodings
4. Aggregate similarity scores to assess overall performance

This dual evaluation approach provides a comprehensive view of utilized models' capabilities, considering both exact match metrics and semantic similarity.

## Implementation Details

The project is implemented using Python and leverages several key libraries:

- **Transformers**: For loading and fine-tuning the pre-trained models
- **Datasets**: For efficient data handling and preprocessing
- **Evaluate**: For implementing evaluation metrics
- **Sentence-Transformers**: For generating sentence embeddings in the judge_similarity method

The implementation includes custom functions for data loading, model evaluation, and fine-tuning. Additionally, I implemented a custom trainer class to handle the training process efficiently.

## Results

The notebook includes detailed results of the evaluation process, comparing the performance of both models before and after fine-tuning. The results are presented in terms of exact match scores, F1 scores, and similarity scores.

Here's a berief result acheived after fine-tuning:
1. **large_roberta_xlm_persian**: 
    - EM: 57.84
    - F1: 72.67
    - Judge-Similarity: 80.06
2. **PQuAD_answering_question_parsbert**:
    - EM: 70.96
    - F1: 85.15
    - Judge-Similarity: 90.01

## Usage

To replicate this project:

1. Clone the repository
2. Install the required dependencies (listed in the notebook)
3. Download the PersianQA dataset
4. Run the Jupyter notebook to execute the evaluation and fine-tuning processes

## License

This project is under MIT LICENSE.
