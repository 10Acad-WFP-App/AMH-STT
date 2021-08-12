import streamlit as st
import awesome_streamlit as ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.insert(0,"../scripts")
from audio_explorer import AudioExplorer

def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Plots ..."):
        st.title('Raw Data Visualisation  📈 📊')
    
    train_audio_explorer = AudioExplorer(directory='../data/train')
    train_info_df = train_audio_explorer.get_audio_info()   
   
    st.write( train_info_df.sample(10))
    st.line_chart(train_info_df.Channel.value_counts().plot(kind='bar', title='Channel Types',
                                          ylabel='Number of Audio-Files', xlabel='Channel', figsize=(7, 5))
)
   