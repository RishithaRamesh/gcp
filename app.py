#Importing Libraries
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import string
import nltk
# nltk.download('wordnet')
from nltk.stem import PorterStemmer,WordNetLemmatizer

import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama

colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

with open("questions_final.json") as file:
    data = json.load(file)

from flask import Flask,render_template,request, redirect, url_for
 
app = Flask(__name__)
arr = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index',methods=['POST','GET'])
def chat():
    # load trained model
    print("Loading Saved Model...")
    model = keras.models.load_model('chat_model.h5')

    # load tokenizer object
    print("Loading Tokenizer Model...")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    print("Loading Encoder Model...")
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    while True:
        if request.method == 'POST':
            # print(arr)
            inp = request.form['Question']
            # if inp.lower() == "quit":
            #     break
            result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),truncating='post', maxlen=max_len))

            if np.amax(result) < 0.8:
                pred = "ChatBot: Sorry I didn't understand!"
                # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, "Sorry I didn't understand!")
                
            else:
                tag = lbl_encoder.inverse_transform([np.argmax(result)])
                print(np.amax(result))
                for i in data['FAQ']:
                    if i['tag'] == tag:
                        pred = "CHATBOT: "+ np.random.choice(i['responses'])
                        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, np.random.choice(i['responses']))

            # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))
            arr.append([inp, pred])
            return render_template('index.html',pred=pred,arr=arr)
            # return redirect(url_for('index',pred=pred,arr=arr),code = 302)

if __name__ == '__main__':
    app.run(debug=True)