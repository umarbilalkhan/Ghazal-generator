import streamlit as st
import tensorflow as tf
import numpy as np
import os
import pickle  # To load the tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths to model and tokenizer
MODEL_PATH = "Ghazal-generator/urdu_poetry_model.h5"
TOKENIZER_PATH = "Ghazal-generator/tokenizer.pkl"


# Load the trained model (check if it exists first)
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
else:
    st.error(f"Model file '{MODEL_PATH}' not found. Please upload it.")
    model = None

# Load the tokenizer (check if it exists first)
if os.path.exists(TOKENIZER_PATH):
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
else:
    st.error(f"Tokenizer file '{TOKENIZER_PATH}' not found. Please upload it.")
    tokenizer = None

# Define the maximum sequence length (same as used in training)
MAX_SEQ_LENGTH = 50  # Update if different in training

# Function to generate poetry
def generate_poetry(seed_text, next_words=20, temperature=1.0, num_lines=1):
    poetry = ""
    
    first_line = True  # Ensures seed text appears only in the first line
    
    for _ in range(num_lines):
        generated_line = seed_text if first_line else ""  # Use seed text only once
        first_line = False  # Disable for subsequent lines
        
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([generated_line])[0]
            token_list = pad_sequences([token_list], maxlen=MAX_SEQ_LENGTH - 1, padding='pre')

            predicted_probs = model.predict(token_list, verbose=0)[0]

            # Apply temperature scaling
            predicted_probs = np.log(predicted_probs + 1e-7) / temperature
            predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))

            predicted_word_index = np.random.choice(len(predicted_probs), p=predicted_probs)

            # Find the corresponding word
            predicted_word = None
            for word, index in tokenizer.word_index.items():
                if index == predicted_word_index:
                    predicted_word = word
                    break

            if predicted_word:
                generated_line += " " + predicted_word
            else:
                break  # Stop if no valid word is found
        
        poetry += generated_line.strip() + "\n"
    
    return poetry.strip()

# Streamlit Interface
st.title('Urdu Poetry Generator')
st.write("Enter a seed stanza, specify the number of words, temperature, and lines.")

# User Inputs
seed_text = st.text_input('Enter First Stanza', "")  # Default empty
next_words = st.number_input('Words per Line', min_value=1, value=10)  
temperature = st.slider('Temperature (Higher = More Creative)', min_value=0.5, max_value=2.0, value=1.0, step=0.1)
num_lines = st.number_input('Number of Lines', min_value=1, value=1)  

# Button to Generate Poetry
if st.button('Generate Poetry'):
    if not seed_text:
        st.error("Please enter a valid seed text.")
    elif model is None or tokenizer is None:
        st.error("Model or tokenizer not loaded correctly.")
    else:
        generated_poetry = generate_poetry(seed_text, next_words, temperature, num_lines)
        st.subheader("Generated Poetry:")
        st.text_area("", generated_poetry, height=200)  # Text area for better formatting
