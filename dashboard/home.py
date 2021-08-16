
''' This is the home/index/introductory page'''

# Libraries
import streamlit as st
import awesome_streamlit as ast


# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        # ast.shared.components.title_awesome("Rossmann Pharmaceuticals 💊🩸🩺🩹💉 ")
        st.title('Amharic speech to text')
        st.write(
            """
            Spoken language is the primary method of human to human communication. 
            But this app is about building only the speech recognition (Speech to Text) system, specifically for Amharic language. Amharic language has more than 200 characters but the standard keyboard is made for English alphabet. This limited number of keys has imposed the need of 2 – 4 key strokes to write a single Amharic letter. 
            The practical project of this thesis is to develop functional software
            with speech to text capabilities for Amharic language. 
                """
        )
        st.image("../dashboard/speech.png", use_column_width=True)

        