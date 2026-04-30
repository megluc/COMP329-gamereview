# COMP329-gamereview
This project applies Natural Language Processing (NLP) and machine learning techniques to analyze and classify video game reviews. The main objective is to predict whether a user recommends a game based on their written review, while also gaining insight into sentiment patterns expressed in player feedback.
The project compares three different approaches: a traditional machine learning baseline model using Logistic Regression, and two deep learning models using a Convolutional Neural Network (CNN) and a Long Short-Term Memory (LSTM) network. These models process and learn from textual game reviews to classify sentiment as either positive (recommended) or negative (not recommended).
Models Used
1. Logistic Regression (Baseline Model)
The baseline model uses TF-IDF vectorization with unigram and bigram features to convert text into numerical representations. Logistic Regression is trained on these features to perform binary classification. This model also provides interpretability by identifying important positive and negative words and phrases that influence prediction outcomes.
2. Convolutional Neural Network (CNN)
The CNN model is implemented using PyTorch and learns word embeddings directly from the data. It applies multiple convolutional filters of different sizes to capture local patterns and phrases within reviews. Max-pooling is used to extract the most important features from each filter, which are then passed through fully connected layers for final classification.
3. Long Short-Term Memory (LSTM)
The LSTM model is built using TensorFlow/Keras and is designed to capture sequential and contextual relationships in text. Reviews are tokenized and converted into padded sequences before being passed through an embedding layer and an LSTM layer. This allows the model to learn dependencies between words across the entire review, improving its ability to understand sentiment in context.
Objective
The goal of this project is to:
Analyze player sentiment in video game reviews
Classify reviews as positive or negative (recommended or not recommended)
Compare the performance of traditional machine learning and deep learning approaches on text classification tasks
Dataset
The dataset contains user-written reviews of video games along with a binary label indicating whether the user recommends the game. It is sourced from a public game review dataset and includes both training and test splits.
