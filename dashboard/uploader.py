from enum import Enum
from io import BytesIO, StringIO
from typing import SupportsRound, Union

import pandas as pd
import streamlit as st
import streamlit as st
from dashboard.transcribe import *
import time


def write():

    STYLE = """
    <style>
    img {
        max-width: 100%;
    }
    </style>
    """

    FILE_TYPES = ["csv", "py",  "wav"]

    class FileType(Enum):
        """Used to distinguish between file types"""

        CSV = "csv"
        PYTHON = "Python"
        SOUND = "Voice"

    st.title('SPEECH TO TEXT FOR AMHARIC')

    def get_file_type(file: Union[BytesIO, StringIO]) -> FileType:
        """The file uploader widget does not provide information on the type of file uploaded so we have
        to guess using rules or ML. See
        [Issue 896](https://github.com/streamlit/streamlit/issues/896)
        I've implemented rules for now :-)
        Arguments:
            file {Union[BytesIO, StringIO]} -- The file uploaded
        Returns:
            FileType -- A best guess of the file type
        """

        if isinstance(file, BytesIO):
            return FileType.SOUND
        content = file.getvalue()
        if (
            content.startswith('"""')
            or "import" in content
            or "from " in content
            or "def " in content
            or "class " in content
            or "print(" in content
        ):
            return FileType.PYTHON

        return FileType.CSV

    def main():
        """Run this function to display the Streamlit app"""
        st.info(
            "This webpage lets you upload wav audio file and transribe it to Amharic, CHECK THAT OUT !!")
        st.markdown(STYLE, unsafe_allow_html=True)
        st.header("Upload audio file")
        file = st.file_uploader("Audio file", type=FILE_TYPES)
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " +
                           ", ".join(FILE_TYPES))
            return

        file_type = get_file_type(file)
        if file_type == FileType.PYTHON:
            st.code(file.getvalue())

        elif file_type == FileType.SOUND:
            # st.code(file.getvalue())
            audio_bytes = file.read()
            st.audio(audio_bytes,  format="audio/ogg")

        else:
            data = pd.read_csv(file)
            st.dataframe(data.head(10))

        with open(os.path.join("./tempfile", file.name), "wb") as f:
            f.write(file.getbuffer())
        st.success("Processing File..")

        st.header("Transcribe audio")
        if st.button('Transcribe'):
            st.write("")
            with st.spinner('wait for it ...'):
                time.sleep(60)
            st.success('Done!')
        else:
            st.write('')

        # if file:
        #     token, t_id = upload_file(file)
        #     result = {}
        #     #polling
        #     sleep_duration = 1
        #     percent_complete = 0
        #     progress_bar = st.progress(percent_complete)
        #     st.text("Currently in queue")
        #     while result.get("status") != "processing":
        #         percent_complete += sleep_duration
        #         time.sleep(sleep_duration)
        #         progress_bar.progress(percent_complete/10)
        #         result = get_text(token,t_id)

        #     sleep_duration = 0.01

        #     for percent in range(percent_complete,101):
        #         time.sleep(sleep_duration)
        #         progress_bar.progress(percent)

        #     with st.spinner("Processing....."):
        #         while result.get("status") != 'completed':
        #             result = get_text(token,t_id)

        #     st.balloons()
        #     st.header("Transcribed Text")
        #     st.subheader(result['text'])

        file.close()

    main()
