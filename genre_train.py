import string 
import pickle
from sklearn.externals import joblib
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('words')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import sqlite3

print("training")

#import data from SQL database

def sql_data(db):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    query = """select  a.title as IMSDB_title, a.genres, a.script,
            b.TMDB_title, b.budget, b.runtime, b.TMDB_genres,b.vote_average,
                     b.TMDB_release_date, b.popularity,b.TMDB_original_title
                     ,b.imdb_id,b.TMDB_Rating,b.TMDB_revenue,b.TMDB_vote_count,b.TMDB_id
             from IMSDB a 
             inner join TMDB b
             on (trim(upper(a.title)) = trim(upper(b.TMDB_title)))
             """
    post = pd.read_sql_query(sql=query, con = conn)
    return post

df = sql_data(r"C:\\Users\\sohana\\Desktop\\DSI\\Module_3\\Movies\\scriptanalyser2.db")

#Function to remove stopwords, non-english words, convert to lowercase, stem words and tokenize words
def clean_script(script, stemmer = PorterStemmer(), 
                  stop_words = set(stopwords.words('english')), engwords = set(nltk.corpus.words.words())):
    
    #Converts to Lower Case and splits up the words
    words = word_tokenize(script.lower())
    
    filtered_words = []
    
    for word in words:
        # Removes the stop words and punctuation
        if word not in stop_words and word.isalpha() and word in engwords:
            filtered_words.append(stemmer.stem(word))
    
    
    return filtered_words

df['clean_script'] = df['script'].apply(clean_script)

#apply count vectorizer to entire dataset
vectorizer = CountVectorizer(analyzer = clean_script)
movie_countvectorizer = vectorizer.fit_transform(df['script'])

#clean the data - remove duplicates, convert vote_average, budget etc. to numeric, 
#put genres in columns, create length column from length of words in cleaned script
#exclude rows where there are zeroes

def data_clean(dataframe):
    dataframe = dataframe.drop_duplicates(subset=['IMSDB_title', 'TMDB_title'])
    dataframe['length'] = dataframe['clean_script'].apply(len)
    dataframe[["budget", "runtime", "vote_average", "popularity", "TMDB_revenue"]] = dataframe[["budget", "runtime", "vote_average", "popularity", "TMDB_revenue"]].apply(pd.to_numeric, errors = 'coerce')
    dataframe['vote_average'] = dataframe['vote_average'].astype(int)
    dataframe2 = dataframe[["IMSDB_title", "script", "clean_script", "length", "budget", "runtime", "vote_average", "popularity", "TMDB_revenue", "genres"]]
    gen_split = dataframe2['genres'].apply(lambda x: x.replace('[', '').replace(']', '').replace("'", '').split(',')).apply(pd.Series).stack()
    gens = pd.DataFrame(gen_split, columns=['movie_genre']).reset_index(0)
    df3 = dataframe2.merge(gens, left_index = True, right_on = 'level_0')
    #df3 = df3.drop(['level_0'], axis = 1)
    df3 = df3.drop_duplicates(subset=['IMSDB_title'])
    df4 = pd.crosstab(index = df3['IMSDB_title'], columns=df3['movie_genre']).reset_index()
    df5 = df4.merge(df3, left_on = 'IMSDB_title', right_on = 'IMSDB_title')
    df5 = df5.drop(df5[(df5['length'] < 5000) | (df5['budget'] == 0) | (df5['TMDB_revenue'] == 0)].index)
    return df5

movie_df = data_clean(df)

#create subsets of data for further analysis
movie_nlp = movie_df.iloc[0:580] #this dataset will be used for training and testing nlp model
movie_demo = movie_df.iloc[580:] #this dataset will be used for validation and demos

#create a new count_vect for the train/test dataset
vectorizer = CountVectorizer(analyzer = clean_script)
movie_countvectorizer_nlp = vectorizer.fit_transform(movie_nlp['script'])

#declare x and y variables

label_gen = movie_nlp['movie_genre'].values
X_gen = movie_countvectorizer_nlp
y_gen = label_gen

#define train and test sets

X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(X_gen, y_gen, test_size = 0.25)

#NLP model with Naive Bayes to predict rating that a movie script will receive

NB_classifier_gen = MultinomialNB()
NB_classifier_gen.fit(X_train_gen, y_train_gen)
NB_classifier_gen.score(X_test_gen, y_test_gen)

#serializing the model

joblib.dump(NB_classifier_gen, 'NB_movie_script_gen.pkl')
pickle.dump(vectorizer.vocabulary_, open('Vocab_gen.txt', 'wb'))


print("finished training")

