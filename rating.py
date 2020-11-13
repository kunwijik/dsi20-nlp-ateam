import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as numpy
from nltk.corpus import stopwords
import string 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


def message_cleaning(message):
  test_punc_removed = [char for char in message if char not in string.punctuation]
  test_punc_removed_join = ''.join(test_punc_removed)
  test_punc_removed_join_clean = [word for word in test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
  return test_punc_removed_join_clean

vocabulary_to_load = pickle.load(open('Vocab.txt', 'rb'))
vectorizer = CountVectorizer(analyzer = message_cleaning, vocabulary=vocabulary_to_load)
vectorizer._validate_vocabulary()

NB_spam_model = open('NB_spam_model.pkl','rb')
clf = joblib.load(NB_spam_model)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.txt']
app.config['UPLOAD_PATH'] = 'uploads'


@app.route('/')
def index():
    return render_template('index.html')

def predict(message):
    data = [message]
    vect = vectorizer.transform(data).toarray()
    my_prediction = clf.predict(vect)
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