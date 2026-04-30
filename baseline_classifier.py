#!/usr/bin/env python3
"""Baseline TF-IDF classifier for train.csv and test.csv."""

import argparse
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline


def load_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if 'user_suggestion' not in train_df.columns:
        raise ValueError('train.csv must contain a user_suggestion column.')

    train_df = train_df.fillna({'title': '', 'user_review': ''})
    test_df = test_df.fillna({'title': '', 'user_review': ''})

    train_df['text'] = train_df['title'].astype(str) + ' ' + train_df['user_review'].astype(str)
    test_df['text'] = test_df['title'].astype(str) + ' ' + test_df['user_review'].astype(str)

    return train_df, test_df


def build_pipeline():
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words='english',
    )

    classifier = LogisticRegression(
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
    )

    return Pipeline(
        [
            ('tfidf', vectorizer),
            ('clf', classifier),
        ]
    )


def main(train_path: str, test_path: str, output_path: str):
    train_df, test_df = load_data(train_path, test_path)
    X_train = train_df['text']
    y_train = train_df['user_suggestion'].astype(int)

    pipeline = build_pipeline()

    print('Training baseline TF-IDF classifier...')
    f1_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1', n_jobs=1)
    accuracy_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=1)
    print(f'5-fold F1 scores: {f1_scores}')
    print(f'Mean F1 score: {f1_scores.mean():.4f}')
    print(f'5-fold accuracy scores: {accuracy_scores}')
    print(f'Mean accuracy: {accuracy_scores.mean():.4f}')

    cv_preds = cross_val_predict(pipeline, X_train, y_train, cv=5, n_jobs=1)
    print('\nCross-validation classification report:')
    print(classification_report(y_train, cv_preds, digits=4))
    print(f'Overall cross-validation accuracy: {accuracy_score(y_train, cv_preds):.4f}')

    pipeline.fit(X_train, y_train)

    print('Predicting on test.csv...')
    test_df['prediction'] = pipeline.predict(test_df['text'])
    if hasattr(pipeline, 'predict_proba'):
        test_df['confidence'] = pipeline.predict_proba(test_df['text'])[:, 1]
    else:
        test_df['confidence'] = None

    output_df = test_df[['review_id', 'prediction', 'confidence']].copy()
    output_df.to_csv(output_path, index=False)
    print(f'Saved test predictions to: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline TF-IDF classifier for user review suggestion labels.')
    parser.add_argument('--train', default='train.csv', help='Path to the training CSV file.')
    parser.add_argument('--test', default='test.csv', help='Path to the test CSV file.')
    parser.add_argument('--output', default='test_predictions.csv', help='Output CSV path for test predictions.')

    args = parser.parse_args()

    if not os.path.exists(args.train):
        raise FileNotFoundError(f'Training file not found: {args.train}')
    if not os.path.exists(args.test):
        raise FileNotFoundError(f'Test file not found: {args.test}')

    main(args.train, args.test, args.output)
