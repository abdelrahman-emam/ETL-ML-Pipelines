import sys
import pandas as pd
import re
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet'])
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('table_name', 'sqlite:///' + str(database_filepath))
    
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

def tokenize(text):
    
    tokens = word_tokenize(text) # tokenize text
    
    lemmatizer = WordNetLemmatizer() # initiate lemmatizer
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
    # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('moc', MultiOutputClassifier(KNeighborsClassifier()))])
    
    # Splitting the data
    # X_train, X_test, y_train, y_test = train_test_split(X, Y)
    
    # Tunning our model
    parameters = {
    'tfidf__use_idf': (True, False)
#     'moc__n_jobs':(1, 2)
    }
    
    cv = GridSearchCV(pipeline, parameters, refit = True)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    for idx, val in enumerate(category_names):
        print(val)
        print(classification_report(Y_test.values[:,idx], y_pred[:,idx]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()