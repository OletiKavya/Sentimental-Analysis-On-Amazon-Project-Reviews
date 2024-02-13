import numpy as np
import pandas as pd
import bz2
import re
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import joblib

def visualize_label_distribution(labels):
    plt.figure(figsize=(6, 6))
    sns.countplot(labels)
    plt.title("Label Distribution in Testing Data")
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.show()

def visualize_text_length_distribution(texts):
    text_lengths = [len(text.split()) for text in texts]
    plt.figure(figsize=(8, 6))
    plt.hist(text_lengths, bins=20)
    plt.title("Text Length Distribution in Testing Data")
    plt.xlabel("Text Length")
    plt.ylabel("Count")
    plt.show()



def load_data(file_path, fraction=1.0):
    file = bz2.BZ2File(file_path, 'r')
    texts, labels = [], []
    count = 0  # Initialize a count
    for line in file:
        x = line.decode('utf-8')  # decode binary to string
        labels.append(int(x[9]) - 1)  # extract labels
        texts.append(x[10:].strip())  # extract texts
        count += 1
        if count >= fraction * 1e6:  # Stop after processing the desired fraction of data
            break
    return np.array(labels), texts

def clean_text(texts):
    stwords = stopwords.words('english')
    l = len(texts) / 10
    temp_texts = []
    for i, text in enumerate(texts):
        text = re.sub('\d', '0', text)
        if 'www.' in text or 'http:' in text or 'https:' in text or '.com' in text:
            text = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", " ", text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = text.split()
        text = [word for word in text if not word in stwords]
        text = ' '.join(text)
        temp_texts.append(text)
        if i % l == 0:
            print('--' + str(int(i / l) * 10) + '%', end='')
    print('--100%--Done !')
    return temp_texts

def train_models(train_texts, test_texts, train_labels, test_labels):
    # Text preprocessing
    train_texts = clean_text(train_texts)
    test_texts = clean_text(test_texts)
    
    # Count Vectorization
    print('Fitting data...')
    count_vect = CountVectorizer()
    count_vect.fit(train_texts)
    
    print('Transforming training set...')
    train_texts_vec = count_vect.transform(train_texts)
    print('Transforming test set...')
    test_texts_vec = count_vect.transform(test_texts)

    # Multinomial Naive Bayes
    print("==================== MULTINOMIAL NAIVE BAYES ===========================")
    nb = MultinomialNB()
    nb.fit(train_texts_vec, train_labels)
    y_pred_nb = nb.predict(test_texts_vec)
    accuracy_nb = accuracy_score(test_labels, y_pred_nb)
    print('Accuracy for Multinomial Naive Bayes:', accuracy_nb)

    # Logistic Regression
    print("==================== LOGISTIC REGRESSION =======================")
    lr_model = LogisticRegression(n_jobs=-1, max_iter=1500)
    lr_model.fit(train_texts_vec, train_labels)
    pred_lr = lr_model.predict(test_texts_vec)
    accuracy_lr = accuracy_score(test_labels, pred_lr)
    print('Accuracy for Logistic Regression:', accuracy_lr)

    # Random Forest
    print("==================== RANDOM FOREST =======================")
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # You can adjust the number of trees (n_estimators)
    rf_model.fit(train_texts_vec, train_labels)
    pred_rf = rf_model.predict(test_texts_vec)
    accuracy_rf = accuracy_score(test_labels, pred_rf)
    print('Accuracy for Random Forest:', accuracy_rf)

    return count_vect, lr_model, rf_model  # Return count_vect from this function



def demonstrate_model_predictions(model, count_vect, text_samples, labels):
    """
    Demonstrates model predictions on a list of sample texts.

    Parameters:
    - model: Trained machine learning model (e.g., Logistic Regression or Random Forest)
    - count_vect: Trained CountVectorizer
    - text_samples: List of sample texts from the testing data
    - labels: True labels corresponding to the sample texts

    Returns:
    - None
    """
    # Preprocess the sample texts
    preprocessed_samples = clean_text(text_samples)

    # Vectorize the sample texts using the trained CountVectorizer
    sample_vec = count_vect.transform(preprocessed_samples)

    # Predict labels and probabilities for each sample
    predictions = model.predict(sample_vec)
    probabilities = model.predict_proba(sample_vec)

    # Display predictions and probabilities
    for i, sample in enumerate(text_samples):
        print("Sample Text:")
        print(sample)
        print("True Label:", labels[i])
        print("Predicted Label:", predictions[i])
        print("Probability of Being Label 1:", probabilities[i][1])
        print("Probability of Being Label 2:", probabilities[i][0])
        print("=" * 50)
        
def main():
    # Using a raw string to avoid escaping backslashes
    directory_path = r'C:\Users\Kavya Sree\Downloads\archive (1)'
    file_list = os.listdir(directory_path)
    
    for file in file_list:
        print(file)

    # Specify the full file paths
    train_file_path = r'C:\Users\Kavya Sree\Downloads\archive (1)\train.ft.txt.bz2'
    test_file_path = r'C:\Users\Kavya Sree\Downloads\archive (1)\test.ft.txt.bz2'
    
    # Load data
    train_labels, train_texts = load_data(train_file_path, fraction=1/3)
    test_labels, test_texts = load_data(test_file_path, fraction=1/3)
# Train models
    count_vect, lr_model, rf_model = train_models(train_texts, test_texts, train_labels, test_labels)
    
    # Example prediction
    sample = test_texts[100]
    sample_vec = count_vect.transform([sample])
    pred = lr_model.predict(sample_vec)
    print('Predicted label (Logistic Regression):', pred[0])
    print('Actual label:', test_labels[100])

    pred_rf = rf_model.predict(sample_vec)
    print('Predicted label (Random Forest):', pred_rf[0])
    
    # Save models
    pickle.dump(lr_model, open('model.pkl', 'wb'))
    pickle.dump(count_vect, open('countvect.pkl', 'wb'))
    
    joblib.dump(lr_model, 'model_joblib.pkl')
    joblib.dump(count_vect, 'countvect_joblib.pkl')
    joblib.dump(rf_model, 'rf_model_joblib.pkl')

    # Additional data visualization
    visualize_label_distribution(test_labels)
    visualize_text_length_distribution(test_texts)
    sample_texts = test_texts[:5]  # Example: Get predictions for the first 5 samples
    true_labels = test_labels[:5]  # True labels corresponding to the sample texts

    print("Demonstrating Model Predictions (Logistic Regression):")
    demonstrate_model_predictions(lr_model, sample_texts, true_labels)

    print("Demonstrating Model Predictions (Random Forest):")
    demonstrate_model_predictions(rf_model, sample_texts, true_labels)

if __name__ == "__main__":
    main()

