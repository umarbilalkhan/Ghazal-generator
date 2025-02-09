import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle  # To load the tokenizer

# Load the trained model
model = tf.keras.models.load_model("urdu_poetry_model.h5")

# Load the trained tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Set max sequence length (based on your training configuration)
max_seq_length = 50  # Replace with the value used during training

# Function to generate poetry
def generate_poetry(seed_text, next_words=20, temperature=1.0, num_lines=1):
    poetry = ""
    
    # Flag to ensure the seed word is only added at the start of the first line
    first_line = True
    
    for _ in range(num_lines):
        generated_line = ""
        
        if first_line:
            generated_line = seed_text  # Start with seed text for the first line
            first_line = False  # Disable the seed text for subsequent lines
        
        # Generate the next words after the seed word
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([generated_line])[0]
            token_list = pad_sequences([token_list], maxlen=max_seq_length-1, padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)[0]
            
            # Apply temperature sampling
            predicted_probs = np.log(predicted_probs + 1e-7) / temperature
            predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))
            predicted_word_index = np.random.choice(len(predicted_probs), p=predicted_probs)

            for word, index in tokenizer.word_index.items():
                if index == predicted_word_index:
                    generated_line += " " + word
                    break
        
        poetry += generated_line + "\n"  # Add the line with seed word at the start of the first line only
    
    return poetry

# Streamlit interface
st.title('Urdu Poetry Generation')
st.write("Enter a seed text, specify the number of words, temperature, and the number of lines.")

# Inputs from the user
seed_text = st.text_input('Seed Text', 'ahmed')  # Default seed text
next_words = st.number_input('Number of Words per Line', min_value=1, value=10)  # Default 10 words per line
temperature = st.slider('Temperature', 0.1, 2.0, 1.0, 0.1)  # Default temperature is 1.0
num_lines = st.number_input('Number of Lines', min_value=1, value=1)  # Default 1 line

# Button to generate poetry
if st.button('Generate Poetry'):
    if seed_text:
        generated_poetry = generate_poetry(seed_text, next_words, temperature, num_lines)
        st.write("Generated Poetry:")
        st.text(generated_poetry)  # Use `st.text()` to maintain the format
    else:
        st.write("Please provide a valid seed text.")
