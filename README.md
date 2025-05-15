# Fake News Headline Classification: BERT vs LSTM

This project demonstrates the classification of fake news headlines using two different deep learning approaches:  
- **BERT (Bidirectional Encoder Representations from Transformers)** – a pre-trained transformer-based large language model (LLM)  
- **LSTM (Long Short-Term Memory)** – a recurrent neural network model using word embeddings

---

## Project Overview

Fake news detection is critical to combat misinformation on digital platforms. This repository provides a comparative implementation and evaluation of:

- **BERT**, leveraging contextual embeddings and transformer architecture for state-of-the-art performance in text classification.
- **LSTM**, utilizing sequential modeling with embeddings for understanding temporal dependencies in text.

Both models are trained and evaluated on a labeled dataset of news headlines categorized as *Fake* or *True*.

---

## Model Details

### 1. BERT for Fake News Detection

- Uses `bert-base-uncased` pre-trained model from Hugging Face Transformers
- Fine-tuned for sequence classification with 2 output labels (Fake News, True News)
- Utilizes Adam optimizer and cross-entropy loss for training

**Performance on test set:**

| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Fake News   | 0.97      | 0.98   | 0.98     | 3578    |
| True News   | 0.98      | 0.98   | 0.98     | 4168    |
| **Accuracy**|           |        | **0.98** | 7746    |

---

### 2. LSTM with Embeddings

- Uses pre-trained or custom embeddings for input representation
- LSTM layers capture sequential dependencies in headline text
- Trained with binary classification output (Fake/True)

**Performance on test set:**

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Fake News (0) | 0.92      | 0.92   | 0.92     | 2639    |
| True News (1) | 0.93      | 0.94   | 0.93     | 3171    |
| **Accuracy**  |           |        | **0.93** | 5810    |

---

### Steps

1. **Prepare the dataset:** Ensure the dataset contains news headlines and their labels (`Fake` or `True`).

2. **Train the models:**  
   - Fine-tune BERT model for classification  
   - Train the LSTM model with embeddings

3. **Evaluate the models:**  
   - Use classification reports, accuracy, confusion matrices, and other metrics to compare performance.

---

## Results Summary

| Model | Accuracy | Remarks                             |
|-------|----------|-----------------------------------|
| BERT  | 98%      | Strong performance, better F1-scores across classes |
| LSTM  | 93%      | Good baseline model with embeddings, but lower than BERT |

---

## Conclusion

The BERT model demonstrates superior performance on fake news headline classification, achieving higher precision, recall, and F1-scores compared to the LSTM model with embeddings. This highlights the advantage of transformer-based language models in capturing contextual information and subtle linguistic nuances critical for detecting misinformation.

---
