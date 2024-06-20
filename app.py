import streamlit as st
import whisper
from st_audiorec import st_audiorec
import io
import numpy as np
import librosa
st.title("🎙️Multilingual Speech Recognition Model Using Open AI Whishper")
# Record audio
#with st.sidebar:
audio= st_audiorec()
# Load the Whisper model
model = whisper.load_model("base")
if audio:
    audio_file = io.BytesIO(audio)
    audio_data, sr = librosa.load(audio_file, sr=16000)  
    audio_data = whisper.pad_or_trim(audio_data)
    mel = whisper.log_mel_spectrogram(audio_data).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    st.write("output:")
    st.write(result.text)
