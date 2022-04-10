# Standar Library
import re
import string
from tkinter import Image
from unittest import result

# Third-party Library
import streamlit as st
import tensorflow as tf
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#Load Model
mymodel = tf.keras.models.load_model("mymodel", compile=False)

punctuations = re.sub(r"[!<_>#:)\.]", "", string.punctuation)

def punct2wspace(text):
    return re.sub(r"[{}]+".format(punctuations), " ", text)

def normalize_wspace(text):
    return re.sub(r"\s+", " ", text)

def casefolding(text):
    return text.lower()

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    text = punct2wspace(text)
    text = normalize_wspace(text)
    text = casefolding(text)
    return text

def main():

    review_text = st.text_input('Enter Your Review Summary Here')

    if st.button('Predict'):
        result_text = preprocess_text(review_text)
        contoh_review = [result_text]
        prediksi = mymodel.predict(contoh_review) # Probabilitas
        prediksi.squeeze()
        if prediksi.squeeze()>0.5:
            st.write("Review positif")
            st.image(image='good-review.png',  width=50)
        else:
            st.write("Review negatif")
            st.image(image='bad-review.png',  width=50)
        
if __name__=='__main__':
    main()