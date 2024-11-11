# Project Structure

- **0. Import Libraries**
  
- **1. Data Preprocessing**
  - 1.1 Importing and Understanding the Data
  - 1.2 Handling Class Imbalance

- **2. Benchmark: VaderSentiment Dictionary**
  - 2.1 Clean the Tweets for VaderSentiment
  - 2.2 Apply VaderSentiment on Test Data
  - 2.3 Generate Report for VaderSentiment

- **3. Recurrent Neural Network (RNN)**
  - 3.1 Clean the Tweets for Training Embeddings
  - 3.2 Train FastText Embeddings
  - 3.3 Build and Train the RNN Model
  - 3.4 Classification Report for RNN
  - 3.5 Hyperparameter Tuning

- **4. DistilBERT**
  - 4.1 Clean the Tweets for DistilBERT
  - 4.2 Predictions with Pre-trained and Fine-tuned DistilBERT
  - 4.3 Classification Report for Pre-trained DistilBERT
  - 4.4 Classification Report for Fine-tuned DistilBERT

- **5. Final Comparison**
  - 5.1 Comparison Table and Conclusion



# Project Description: Bitcoin Tweet Sentiment Analysis

This project involves performing sentiment analysis on 2000 tweets about Bitcoin, with each tweet labeled as either positive (1) or negative (0).  
Utilize multiple approaches to analyze the sentiment, ranging from a sentiment dictionary to RNN and Transformer models.  
The data is divided into a training set of 1500 tweets and a test set of 500 tweets.

The following elements are included in this project:

### 1. Text Data Preprocessing

The tweets are web-scraped, meaning they contain HTML elements, hyperlinks, and unicode characters.  
Each model requires specific preprocessing, and informed decisions are made on what to keep or remove.  
Also the dataset is imbalanced handling which is the real challenge. Many measures like using different measures, SMOTE, data augmentation, using class weights have been taken to help model perform better.  
Detailed justifications are provided in the notebook.

### 2. Sentiment Dictionary Benchmark

As a baseline for sentiment analysis, I apply the vaderSentiment dictionary on the test dataset.  
This provides a benchmark to compare the performance of next deep learning models.

### 3. RNN Model with Custom Embeddings

An RNN-based classifier is implemented using Keras:
- Custom embeddings are trained on the train dataset.
- The model is evaluated using a validation split to analyze the training process.
- The trained RNN model is then applied to the test set for performance evaluation.
- Functionalities like LSTM, GRU, bidirectional, regularization are implemented to explore all the possibilities.
- A grid search is done at the end to find the best hyperparameter combination.

### 4. Pre-trained DistilBERT Model

Using Hugging Face's `transformers` library, I apply a pre-trained DistilBERT model fine-tuned for sentiment analysis:
- Model source: [DistilBERT Sentiment Analysis](https://huggingface.co/DT12the/distilbert-sentiment-analysis)
- Predictions are generated on the test dataset.

### 5. Fine-tuned DistilBERT Model

The DistilBERT model is also fine-tuned on the training dataset for one epoch:
- Due to computational demands, only one epoch is used, and training process analysis is omitted.
- This fine-tuned model is applied to the test dataset for sentiment predictions.

### 6. Evaluation of Models

Each approach is evaluated on the test dataset using the following binary classification metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC AUC**
- **PR Curve**

## Model Performance Comparison

| Model                 | Precision (Class 0) | Recall (Class 0) | F1 Score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1 Score (Class 1) | Accuracy |
|-----------------------|---------------------|-------------------|---------------------|----------------------|-------------------|---------------------|----------|
| **VaderSentiment**    | 0.58               | 0.45             | 0.51               | 0.87                | 0.92             | 0.89               | 0.83     |
| **RNN (LSTM 64)**     | 0.48               | 0.24             | 0.32               | 0.83                | 0.94             | 0.88               | 0.80     |
| **Pre-trained DistilBERT** | 0.73        | 0.60             | 0.66               | 0.91                | 0.95             | 0.93               | 0.88     |
| **Fine-tuned DistilBERT** | 0.79        | 0.62             | 0.70               | 0.91                | 0.96             | 0.93               | 0.89     |

- The table compares the classification performane of all 4 models tracking their precision, recall and F1 score for both the classes.
- From the numbers, **Fine-tuned DistilBERT** is the best overall, consistently giving high scores in precision, recall, and F1 score for both categories. It outperforms other models, particularly the minority class, which is crucial in this imbalanced dataset.
- Having said that, it is worth to note the performance of our benchmark model i.e. VaderSentiment. Although comparatively simpler, it outperforms RNN in every other metric (optimistically thinking that I have fitted the best possible RNN).
- I have noticed improved performance every model as I refined my data preprocessing, after learning from every iteration. Hence I will attribute a large share of improved performance to data preprocessing and I believe that there still sope for improvement.