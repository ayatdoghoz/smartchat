from flask import Flask, request, jsonify
import json
import pickle
import nltk
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from keras.models import load_model

app = Flask(__name__)

file = open(r"C:\Users\asus\Desktop\chatbot\arabicdata.json", encoding='utf-8').read()
data = json.loads(file)
word = pickle.load(open("C:\\Users\\asus\\Desktop\\chatbot\\word.pkl", "rb"))
words=sorted(word)
classes = pickle.load(open("C:\\Users\\asus\\Desktop\\chatbot\\classes.pkl", "rb"))
corpus = pickle.load(open("C:\\Users\\asus\\Desktop\\chatbot\\corpus.pkl", "rb"))
model = load_model("C:\\Users\\asus\\Desktop\\chatbot\\model.h5")

def sen_word(sen):
    sword = nltk.word_tokenize(sen)
    ignore=["!","?"]
    sword = [w for w in sword if w not in ignore]

    sword = nltk.word_tokenize(sen)
    sword = [w for w in sword if w not in ignore]
    return sword

def digitize(sen, words):
    input_vec = [0] * len(words)
    sword = sen_word(sen)
    for sw in sword:
        if sw in words:
            indx = words.index(sw)
            input_vec[indx] = 1
    return input_vec

def predect_class(sen, words, classes, model):
    sample = digitize(sen, words)
    value = model.predict(np.array([sample]))[0]

    indx = list(value).index(max(value))
    return classes[indx]

def get_response(sen, words, classes, model):
    tag = predect_class(sen, words, classes, model)
    for record in data['info']:
        if record['title'] == tag:
            return random.choice(record['responses']).encode('utf-8').decode('utf-8')

@app.route("/api", methods=["POST"])
def get_bot_response():
    user_txt = request.json["message"]
    response = get_response(user_txt, words, classes, model)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run("0.0.0.0")