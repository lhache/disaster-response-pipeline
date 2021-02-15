import sys
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import re
from  nltk import download as nltk_download
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, hamming_loss, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

nltk_download('punkt')
nltk_download('wordnet')
nltk_download('stopwords')

def load_data(database_filepath):
    '''
    Load data from database to dataframes
        
    Args:
        database_filepath (str)
    
    Returns:
        X (pandas.DataFrame)
        Y (pandas.DataFrame)
        category_names (Index)
    '''
    df = pd.read_sql_table('DisasterResponse', 'sqlite:///' + database_filepath)

    df_categories = df.loc[:, 'related':'direct_report']
    
    X = df['message'].values
    Y = df_categories.values
    category_names= list(df_categories.columns)

    return (X, Y, category_names)

def tokenize(text):
    '''
    Tokenizes and cleans text
        
    Args:
        text (str)
    
    Returns:
        clean_tokens (list)
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds a NLP multilabel classification pipeline, tuning hyperparameters with GridSearch
    
    Returns:
        cv (GridSearchCV object)
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        # ('clf', MultiOutputClassifier(RandomForestClassifier(max_depth=2, random_state=0)))
        # ('clf', MultiOutputClassifier(KNeighborsClassifier()))
        # ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        # ('clf', MultiOutputClassifier(OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))))
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ])
    
    parameters = {
        'vect__max_features':[1000, 10000, 40000],
        'tfidf__use_idf': [True, False]
    }
    
    cv = GridSearchCV(pipeline, parameters, cv=3, verbose=3, n_jobs=-1)    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates model performances on each category
    
    Args:
        model: classifier model
        X_test: test data
        Y_test: test labels
        category_names: names for test label columns

    '''
    Y_pred = model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean()
    
    for index, cat in enumerate(category_names):
        print("Category {}: {}".format(index, cat))
        print("Accuracy: {}     Precision: {}        Recall: {}        Hamming Loss: {}".format(
            "%.3f" % accuracy_score(y_true=Y_test[:, index], y_pred=Y_pred[:, index]),
            "%.3f" % precision_score(y_true=Y_test[:, index], y_pred=Y_pred[:, index], average="weighted", zero_division=0),
            "%.3f" % recall_score(y_true=Y_test[:, index], y_pred=Y_pred[:, index], average="weighted"),
            "%.3f" % hamming_loss(y_true=Y_test[:, index], y_pred=Y_pred[:, index]),
        ))
        print("\n")

    print("Overall accuracy: ", accuracy)

    return None
    
    

def save_model(model, model_filepath):
    '''
    Saves a model as a pickle file
    '''
    try:
        f = open(model_filepath, 'wb')
        pickle.dump(model, f)
        return True
    except IOError:
        return False


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        if save_model(model, model_filepath):
            print('Success saving model.')
        else:
            print('Error saving model')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()