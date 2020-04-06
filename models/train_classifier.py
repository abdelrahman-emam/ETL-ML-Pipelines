import sys
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet'])
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''loading the database file
    
    Args:
        database_filepath: file path of the database to work on
        
    Returns:
        X : that resembles the input feature -messages-
        Y : that resembles the response vecotr
        category names: holds the names of the categories'''
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('table_name', 'sqlite:///' + str(database_filepath))
    
    X = df['message'] # assigning X to the messages col
    Y = df[df.columns[4:]] # assigning Y to response cols
    category_names = Y.columns.tolist() # getting column names
    
    return X, Y, category_names

def tokenize(text):
    '''Using nltk to normalize, lemmatize and tokenize text
    
    Args:
        text: to normalize, lemmatize, tokenize and apply tf-idf
        
    Returns:
        clean tokens from stopwords, ..etc.'''
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
    '''building our Machine learning pipline, 
    some parameters for the model to tune through
    
    Args:
        None
        
    Returns:
        Machine Learning model'''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('moc', MultiOutputClassifier(RandomForestClassifier()))])
   
    # Parameters to tune our model through
    parameters = {
    'moc__estimator__criterion': ['gini', 'entropy'],
    'moc__estimator__max_depth': [None, 20, 50],
    'moc__estimator__max_features': ['auto', 10, 20]
    }
    
    # Gridsearch the pipeline using the given parameters
    cv = GridSearchCV(pipeline, parameters, refit = True)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''Prints the f1 score, precision and recall 
    for the test set for each category using classification_report
    
    Args:
        model: Machine learning model
        X_test: input test portion of the data
        Y_test: output test portion of the data 
        category_names: names of the categories from the database
        
    Returns:
        None
    '''
    y_pred = model.predict(X_test) # predict on test data
    
    for idx, val in enumerate(category_names):
        print(val) # printing the name of each column
        print(classification_report(Y_test.values[:,idx], y_pred[:,idx]))
        # printing the classification report for each category


def save_model(model, model_filepath):
    '''Saving the model as .pkl file
    
    Args:
        model: machine learning model
        model_filepath: the file path to store the model in
    
    Returns:
        None
    '''
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
