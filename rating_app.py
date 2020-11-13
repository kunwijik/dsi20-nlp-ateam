import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory
from werkzeug.utils import secure_filename

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
from sklearn.externals import joblib

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

vocabulary_to_load = pickle.load(open('Vocab_rating.txt', 'rb'))
vectorizer = CountVectorizer(analyzer = clean_script, vocabulary=vocabulary_to_load)
vectorizer._validate_vocabulary()

NB_rating_model = open('NB_movie_script_rating.pkl','rb')
clf_rating = joblib.load(NB_rating_model)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.txt']
app.config['UPLOAD_PATH'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

def predict(script):
    data = [script]
    vect = vectorizer.transform(data).toarray()
    my_prediction = clf_rating.predict(vect)
    return my_prediction

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    file = open(app.config['UPLOAD_PATH'] + '/' + filename)
    text = file.read()
    my_prediction = predict(text)
    return render_template('result.html',prediction = my_prediction)

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

if __name__ == '__main__':
    app.run(debug=True)