import streamlit as st
from infrance_model import inf_model
import os
from whisper_utils import WhisperModel



st.sidebar.header(' Summarizer')
st.sidebar.write("")
Text = ""

sidebar_option = st.sidebar.selectbox('Select Data Type:', ('Document', 'Video File'))


if sidebar_option == 'Document':
    uploaded_file = st.sidebar.file_uploader("Choose a text file")
    if uploaded_file is not None:
        try:
            Text = uploaded_file.read().decode("utf-8")
            st.sidebar.write("Text file loaded successfully.")
        except Exception as e:
            st.write(f"An error occurred while reading the TXT file: {e}")
    else:
        st.write("Please upload a TXT file.")

elif sidebar_option == 'Video File':
    audio_file = st.sidebar.file_uploader("Upload Audio File", type=["mp4", "mkv", "mov", "mp3"])
    if audio_file is not None:
        wh_model = WhisperModel()

        audio_path = "uploaded_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        Text = wh_model.transcribe(audio_path)
        st.sidebar.write("Video file processed successfully.")
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
            st.sidebar.write(f"Temporary file {audio_path} deleted.")

st.title("💬 Summarizer")




if st.button("Summarize"):
    model = inf_model()


    llm_summary = model.llm_summarize(Text)


    llm_summary = llm_summary.split("### Response:")[1]


    st.chat_message("assistant").write(llm_summary)