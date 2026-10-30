# Spam vs Legitimate Message Classifier

Laboratory assignment: Designing an AI-Based System to Solve a Real-World Problem
Course: Software Design in Intelligent Systems
Author: Maftuna Sharabbayeva (ID 380261200262)
University: Muhammad al-Khwarizmi Tashkent University of Information Technologies

## Description
A prototype that classifies text messages as spam or legitimate using
TF-IDF features and a Multinomial Naive Bayes classifier.

## Files
- `spam_classifier.py` - dataset generation, preprocessing, training, evaluation, test cases
- `messages_dataset.csv` - synthetic labelled dataset (100 spam, 100 legitimate)
- `requirements.txt` - Python dependencies

## How to run
pip install -r requirements.txt
python spam_classifier.py

## Results
Hold-out test set (50 messages): accuracy, precision, recall and F1 all 1.00.
The dataset is synthetic and template-based, so these scores are optimistic.
8 hand-written test messages were all classified correctly.

## Limitations
Small synthetic dataset, English only, bag-of-words model.
Output should support, not replace, human judgment.
