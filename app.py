from os import write
import sys
from sklearn import model_selection  
sys.path.insert(0, './scripts')
from model_inference import ModelInference
import streamlit as st
import numpy as np
import pandas as pd
import json

# import modeling
# import visualize
import pickle
import librosa
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from six.moves import xrange as range
# import json
# from python_speech_features import mfcc


# import glob


def file_uploader():
    uploaded_file = st.file_uploader("Upload Files",type=['wav'])
    return uploaded_file

@st.cache()  
# defining the function which will make 
# the prediction using data about the users 
def prediction(file):   
    y,sr = librosa.load(file)
    mi = ModelInference(y,sr)
    result = mi.get_prediction()
    return result

    
def main_page():
    st.write('Amharic Speech to Text')
    Store = st.write("Upload your audio file")
    file = file_uploader()

    result = ""
    
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(file)
        st.write(result)
        # st.success('The user is {}'.format(result))     
  


     
if __name__=='__main__': 
    main_page()