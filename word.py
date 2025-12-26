import os
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'next_word_lstm.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pickle')

# Load model and tokenizer safely with clear errors
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Place 'next_word_lstm.h5' in this folder.")
    st.stop()
model = load_model(MODEL_PATH)

if not os.path.exists(TOKENIZER_PATH):
    st.error(f"Tokenizer file not found: {TOKENIZER_PATH}. Place 'tokenizer.pickle' in this folder.")
    st.stop()
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')


# To run the app, use the command:
# ----- 
#$env:TF_ENABLE_ONEDNN_OPTS=0
#C:\Users\USER\Desktop\ANN\meenv\python.exe -m streamlit run LSTM-RNN\word.py