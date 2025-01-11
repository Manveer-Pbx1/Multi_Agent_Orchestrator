import streamlit as st
import whisper
import os
from pathlib import Path
import numpy as np
import librosa
import io
import sounddevice as sd
import time

st.title("Audio to Text Transcription App")
st.sidebar.title("Options")

st.sidebar.write("Select transcription mode:")
mode = st.sidebar.radio("Mode", ("Upload Audio File", "Real-Time Transcription"))

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

def transcribe_audio(audio_file_path):
    try:
        audio_path = str(Path(audio_file_path).absolute())
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"The file does not exist: {audio_path}")
        
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

if mode == "Upload Audio File":
    st.header("Upload an Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

    if uploaded_file is not None:
        try:
            audio_bytes = uploaded_file.read()
            
            try:
                audio_io = io.BytesIO(audio_bytes)
                audio_array, sr = librosa.load(audio_io, sr=16000)
                
                st.write("Debug - Audio array shape:", audio_array.shape)
                st.write("Debug - Sample rate:", sr)
                
                st.audio(audio_bytes)
                
                st.write("Starting transcription...")
                transcript = model.transcribe(audio_array)
                
                if transcript and "text" in transcript:
                    st.subheader("Transcription:")
                    st.write(transcript["text"])
                    
                    st.download_button(
                        label="Download Transcription as TXT",
                        data=transcript["text"],
                        file_name="transcription.txt",
                        mime="text/plain",
                    )
            except Exception as e:
                st.error(f"Audio processing error: {str(e)}")
                st.write(f"Audio format: {uploaded_file.type}")
                st.write(f"Audio size: {len(audio_bytes)} bytes")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.write(f"Full error details: {repr(e)}")

elif mode == "Real-Time Transcription":
    st.header("Real-Time Voice Transcription")
    
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'audio_chunks' not in st.session_state:
        st.session_state.audio_chunks = []

    st.sidebar.write("Debug Info:")
    st.sidebar.write(f"Recording: {st.session_state.recording}")
    st.sidebar.write(f"Has Audio: {st.session_state.audio_data is not None}")
    st.sidebar.write(f"Processed: {st.session_state.processed}")

    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 1024
    
    def record_audio_continuous():
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.float32) as stream:
            while st.session_state.recording:
                audio_chunk, _ = stream.read(CHUNK_SIZE)
                st.session_state.audio_chunks.append(audio_chunk)
                time.sleep(0.01)  
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not st.session_state.recording:
            if st.button("Start Recording"):
                st.session_state.recording = True
                st.session_state.audio_data = None
                st.session_state.processed = False
                st.session_state.audio_chunks = []
                st.rerun()
    
    with col2:
        if st.session_state.recording:
            if st.button("Stop Recording"):
                st.session_state.recording = False
                st.session_state.audio_data = np.concatenate(st.session_state.audio_chunks)
                st.rerun()
    
    with col3:
        if (not st.session_state.recording and 
            st.session_state.audio_data is not None and 
            not st.session_state.processed):
            if st.button("Process Recording"):
                st.session_state.processed = True
                st.rerun()
    
    if st.session_state.recording:
        st.warning("Recording in progress... Press 'Stop Recording' when finished.")
        record_audio_continuous()
    elif st.session_state.audio_data is not None and not st.session_state.processed:
        st.info("Recording complete. Click 'Process Recording' to transcribe.")

    if st.session_state.processed and st.session_state.audio_data is not None:
        try:
            audio_data = st.session_state.audio_data.flatten()
            
            st.subheader("Audio Waveform:")
            st.line_chart(audio_data[:1000])
            
            st.info("Transcribing audio...")
            transcript = model.transcribe(audio_data)
            
            if transcript and "text" in transcript:
                st.subheader("Transcription:")
                st.write(transcript["text"])
                
                st.download_button(
                    label="Download Transcription as TXT",
                    data=transcript["text"],
                    file_name="transcription.txt",
                    mime="text/plain",
                )
        
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            st.error("Full error:", exc_info=True)
