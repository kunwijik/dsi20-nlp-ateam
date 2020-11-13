import imghdr
import os
import jinja2
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
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('words')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import sqlite3
from sklearn.externals import joblib
import yagmail


# Define Function to send an email
yag = yagmail.SMTP(user='ateamdsiafrica@gmail.com', password='nadeemoozeer')
def send_mail(email_address,email_body):
    print(email_address)
    print(email_body)

    yag.send(to=email_address,\
             subject='Script Analyser Results',\
             contents=email_body)


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


#The model for rating
NB_rating_model = open('NB_movie_script_rating.pkl','rb')
clf_rating = joblib.load(NB_rating_model)

#The model for genre
NB_genre_model = open('NB_movie_script_gen.pkl', 'rb')
clf_genre = joblib.load(NB_genre_model)

#The model for revenue
Revenue_model = open('NB_movie_script_revenue_predict.pkl', 'rb')
clf_rev = joblib.load(Revenue_model)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.txt']
app.config['UPLOAD_PATH'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

def predict_review(script):
    data = [script]
    vect = vectorizer.transform(data).toarray()
    my_prediction = clf_rating.predict(vect)
    return my_prediction

def predict_genre(script):
    data_gen = [script]
    vect_gen = vectorizer.transform(data_gen).toarray()
    gen_prediction = clf_genre.predict(vect_gen)
    return gen_prediction

def predict_profit(budget):
    data_rev = np.array([budget])
    data_revenue = data_rev.reshape(-1, 1)
    profit = clf_rev.predict(data_revenue)
    return profit

@app.route('/results', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    budget = request.form['budget']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    file = open(app.config['UPLOAD_PATH'] + '/' + filename)
    text = file.read()
    review_prediction = predict_review(text)
    genre_prediction = predict_genre(text)
    profit_pred = predict_profit(budget)
    #email_body =  render_template("email.html",record = rows)   
    template = jinja2.Template("""
                               The script analysis tool has predicted that for script: {{ name }} 
                                the predicted rating would be: {{ prediction_review }} stars
                                with a revenue of $ {{prediction_revenue}} 
                                and would fall into the{{predict_genre[0]}} genre.""")
    email_body = template.render(name=filename,prediction_review=review_prediction[0], prediction_revenue =  round(profit_pred[0]), predict_genre = genre_prediction)
    email = request.form['email']
    send_mail(email_address=email,email_body=email_body)

    return render_template('result.html',revprediction = review_prediction, genprediction = genre_prediction, profit = round(profit_pred[0]))

# @app.route('/uploads/<filename>')
# def upload(filename):
#     return send_from_directory(app.config['UPLOAD_PATH'], filename)

if __name__ == '__main__':
    app.run(debug=True)