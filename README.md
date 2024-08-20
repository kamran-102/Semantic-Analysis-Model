
# Semantic Analysis Model 

This repository contains a script for training a semantic analysis model using custom text data. The goal of the script is to classify text data (such as reviews) as either positive or negative. The model uses Natural Language Processing (NLP) techniques for preprocessing the data and XGBoost as the classifier.

## Installation

1. Clone the repository or download the script.
2. Install the required Python packages:

```bash
pip install xgboost pandas nltk scikit-learn
```

## Data Preprocessing

### 1. Import Libraries
The script begins by importing the necessary libraries, including `pandas`, `nltk`, `xgboost`, and others required for data manipulation, natural language processing, and model training.

### 2. Text Preprocessing
The `preprocess` function performs several cleaning tasks on the input text:
- Converts text to lowercase.
- Removes HTML tags.
- Removes punctuation, numbers, and extra spaces.
- Strips any leading or trailing whitespace.

### 3. Stopword Removal
The `stopword` function removes common English stopwords from the text, which are words that carry less meaning (like 'the', 'and', 'is', etc.).

### 4. Lemmatization
The `lemmatizer` function uses the WordNet Lemmatizer to reduce words to their base or root form, which helps in normalizing the text data.

### 5. Tokenization
The `tokenize` function splits the text into individual tokens (words).

### 6. Bag of Words Conversion
The `bag_of_words` function converts tokenized sentences into a bag-of-words representation, which is a vector of numbers indicating the presence (or absence) of specific words in the sentence.

### 7. Final Preprocessing
The `finalpreprocess` function combines all the preprocessing steps (preprocessing, stopword removal, lemmatization, and tokenization) into a single function for easy use on text data.

## Data Loading and Cleaning

The script loads a CSV file containing text reviews and their associated sentiments (positive or negative). The reviews are then processed using the `finalpreprocess` function, and converted into a bag-of-words representation.



## Vocabulary Box

A vocabulary box is created, which contains all the unique words in the dataset after preprocessing. This vocabulary is saved to a file for later use.



## Model Training

The data is split into training and test sets using `train_test_split`. An XGBoost classifier (`xgb.XGBClassifier()`) is then trained on the bag-of-words representation of the reviews.



## Saving the Model

The trained XGBoost model is saved using `pickle` for later use in making predictions.



## Inferences

To make predictions on new data, the vocabulary box and trained model are loaded from their respective files. The input sentence is processed using the same steps as the training data and then fed into the model to predict the sentiment.


## Conclusion

This script provides a basic yet effective approach to building a semantic analysis model that can classify text data as positive or negative. It covers the essential steps of data preprocessing, model training, and inference, making it a useful tool for various text classification tasks.

