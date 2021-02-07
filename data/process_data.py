import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=['id'])
    return df

def clean_categories(df):
    # get categories
    categories = df['categories'].str.split(';', expand=True)
    
    # clean category names
    first_row = categories.head(1)
    category_colnames = first_row.apply(lambda cell: cell.str.split('-')[0][0])
    categories.columns = category_colnames
    
    # clean values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-')[0][1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop categories column from original dataframe
    df = df.drop(['categories'], axis=1)

    ## concat the two dataframes
    df = pd.concat([df, categories], axis=1)

    return df

def remove_duplicates(df):
    df = df.drop_duplicates()
    return df


def clean_data(df):
    df = df.pipe(clean_categories).pipe(remove_duplicates)
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(df.head())
        
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()