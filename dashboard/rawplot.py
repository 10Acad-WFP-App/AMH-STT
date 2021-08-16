import streamlit as st
import awesome_streamlit as ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
# sys.path.insert(0,"../scripts")
from matplotlib import pyplot as plt
from scripts.audio_explorer import AudioExplorer
from PIL import Image
image = Image.open("./dashboard/8.png")


def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Plots ..."):
        st.title('Raw Data Visualisation  📈 📊')

    train_audio_explorer = AudioExplorer(directory='./data/train')
    train_info_df = train_audio_explorer.get_audio_info()
    bins = pd.cut(train_info_df['Duration(sec)'], np.arange(
        0, int(max(train_info_df['Duration(sec)'].tolist())) + 1))

    data = st.selectbox("select data to visualize",
                        ('data info', 'Channel', 'Audio_duration'))
    if data == "data info":
        st.write(train_info_df.sample(10))
    elif data == "Channel":
        y = train_info_df.Channel.value_counts()
        st.bar_chart(y)

    elif data == 'Audio_duration ':
        st.image(image, width=1500)
