import numpy as np
import gradio as gr
import tensorflow
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.math import argmax
import os
import pandas as pd
import json

df = pd.read_json('News_Category_Dataset_v2.json', lines = True)

with open('model_json.json', 'r') as f:
    model_json = f.read()

model = model_from_json(model_json)
model.load_weights('model.h5')
model.compile(loss='SparseCategoricalCrossentropy', optimizer='adam', metrics=['accuracy'])


category_label_enc = {key: value for key, value in enumerate(df.category.unique())}

df = pd.read_json('News_Category_Dataset_v2.json', lines = True)
token = Tokenizer(num_words=10000)
token.fit_on_texts(df['headline'])

def get_seq(t, token, max_seq_length=36):
    seq = token.texts_to_sequences(t)
    seq = pad_sequences(seq, maxlen=max_seq_length, padding='post')
    return seq

def modelfn(text):
    text = text.split()
    text = get_seq(t = text, token = token)
    y = model.predict(text)
    y = np.argmax(y, axis=-1)
    out_cat = category_label_enc[y[0]]
    return out_cat, y[0]

interface = gr.Interface(fn = modelfn, inputs = ['text'], outputs = ['text', 'text'])

interface.launch()
