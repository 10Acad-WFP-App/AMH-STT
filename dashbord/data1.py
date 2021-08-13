import streamlit as st
import pandas as pd 
import awesome_streamlit as ast


# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Data ..."):
        ast.shared.components.title_awesome("Speech to text for Amharic language ")
        st.title('Data description')
        st.write("""
                 We used the Amharic reading speech collected for speech recognition purposes in the conventional ASR approaches , around 2 hour reading speech containing 1000 sentences was used.
                 These reading speech corpora were collected from different sources to maintain variety, such as political, economic, sport, health news,and fiction.
                 All contents of a corpus were set up using morphological concepts. Reading speech corpora of the corresponding texts were prepared using a 22500 Hz sampling frequency,
                 22.5 bit sample size and a stereo channel.

        """)
        na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        train = pd.read_csv('../data/train/trsTrain.txt', na_values=na_value)
         # pd.read_csv('../data//store.csv', na_values=na_value)
        st.write(train.sample(50))