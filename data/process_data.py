import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Loading, merging the data into a dataframe
     
    Args:
        messages_filepath:   path to messages csv file
        categories_filepath: path to categories csv file
        
    Returns:
        Dataframe created from merging both datasets
    '''
    messages = pd.read_csv(str(messages_filepath))
    categories = pd.read_csv(str(categories_filepath))
    df = messages.merge(categories, on = 'id')
                             
    return df

def clean_data(df):
    '''Cleaning teh data, & removing duplicate data
    
    Args:
        df: input merged dataframe
    
    Returns:
        df: Cleaned dataframe with no duplicates ready to be saved
    '''
    categories = df['categories'].str.split(pat = ';', expand = True) # Split on ;
     # Getting categories column names (unfiltered)
    categories.columns = categories.iloc[0]
    
    row = categories.iloc[0]
    # Getting categories column names (filtered)
    category_colnames = row.apply(lambda x: row[x][0:-2])
    categories.columns = category_colnames # Assigning the proper column names
    
    # converting column values from string to equivalent integers
    for column in categories:
    # set each value to be the equivalent value (str)
        categories[column] = categories[column].str[-1]
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
                             
    # Dropping old unfiltered categories column
    df.drop(columns = ['categories'], axis = 1, inplace = True)
                             
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], sort = False, axis = 1)
                             
    # Removing duplicates
    df = df[~df.duplicated(keep='first')]
    
    return df

def save_data(df, database_filename):
    '''Saving the data into sqlite database
    
    Args:
        df: input dataframe
        database_filename: database file name to be saved
    
    Returns:
        engine: sqlite database file
    '''
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('table_name', engine, index=False)
                             
    return engine


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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