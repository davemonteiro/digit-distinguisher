import numpy as np
import random
import cv2
import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from tensorflow.keras.models import load_model

import build_model

dir_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.isdir(dir_path + '\model'):
    build_model.main(dir_path)
    
model = load_model(dir_path + '\model')

@st.cache() 
def get_nums():
    # Return 2 random numbers for game

    num1 = -1
    num2 = -1
    while num1 == num2:
        num1 = random.randint(0,9)
        num2 = random.randint(0,9)

    return num1, num2

num1, num2 = get_nums()

st.title('Digit Distinguisher')

if st.button('Get a new pair of numbers', key=1):
    st.caching.clear_cache()
    num1, num2 = get_nums()

st.write('Challenge: can you draw a number that looks somewhat like', min(num1, num2), 'and', max(num1, num2), '?')

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=256,
    height=256,
    drawing_mode='freedraw',
    key='canvas')

if st.button('Predict digit', key=2):
    canvas_result = canvas_result.image_data.astype('float32') / 255
    canvas_result = cv2.cvtColor(canvas_result, cv2.COLOR_RGBA2GRAY)
    canvas_result = cv2.resize(canvas_result, (28, 28), interpolation=cv2.INTER_NEAREST)

    st.image(canvas_result, caption='Input to model')

    probs = model.predict(canvas_result.reshape(1, 28, 28, 1))[0]
    st.write('Results:')
    st.bar_chart(probs)

    if probs[num1] > 0.2 and probs[num2] > 0.2:
        st.write('Nice!')
    elif probs[num1] > 0.2:
        st.write('Your number doesn\'t look enough like', num2)
    elif probs[num2] > 0.2:
        st.write('Your number doesn\'t look enough like', num1)
    else:
        st.write('Your number doesn\'t really look like', min(num1, num2), 'or', max(num1, num2))
