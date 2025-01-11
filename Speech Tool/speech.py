import streamlit as st
import os
from pathlib import Path
import numpy as np
import librosa
import io
import sounddevice as sd
import time
import openai

st.title("Audio to Text Transcription App")
st.sidebar.title("Options")

st.sidebar.write("Select transcription mode:")
mode = st.sidebar.radio("Mode", ("Upload Audio File", "Real-Time Transcription"))

st.sidebar.write("OpenAI Configuration")
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

def transcribe_audio_file(audio_file):
    try:
        response = openai.Audio.transcribe(
            "whisper-1",
            audio_file
        )
        return response["text"]
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def analyze_text(text):
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to enable analysis.")
        return None
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes transcribed text."},
                {"role": "user", "content": f"Please analyze this text and provide a brief summary and key points:\n{text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

if mode == "Upload Audio File":
    st.header("Upload an Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

    if uploaded_file is not None and api_key:
        try:
            st.audio(uploaded_file)
            st.write("Starting transcription...")
            
            transcript = transcribe_audio_file(uploaded_file)
            
            if transcript:
                st.subheader("Transcription:")
                st.write(transcript)
                
                if api_key:
                    st.subheader("Analysis:")
                    analysis = analyze_text(transcript)
                    if analysis:
                        st.write(analysis)
                
                st.download_button(
                    label="Download Transcription as TXT",
                    data=transcript,
                    file_name="transcription.txt",
                    mime="text/plain",
                )
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.write(f"Full error details: {repr(e)}")
    elif not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to enable transcription.")

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

    if st.session_state.processed and st.session_state.audio_data is not None and api_key:
        try:
            audio_data = st.session_state.audio_data.flatten()
            
            st.subheader("Audio Waveform:")
            st.line_chart(audio_data[:1000])
            
            st.info("Transcribing audio...")
            
            # Save audio to temporary file and transcribe
            temp_path = "temp_recording.wav"
            import soundfile as sf
            sf.write(temp_path, audio_data, SAMPLE_RATE)
            
            with open(temp_path, "rb") as audio_file:
                transcript = transcribe_audio_file(audio_file)
            
            os.remove(temp_path)
            
            if transcript:
                st.subheader("Transcription:")
                st.write(transcript)
                
                if api_key:
                    st.subheader("Analysis:")
                    analysis = analyze_text(transcript)
                    if analysis:
                        st.write(analysis)
                
                st.download_button(
                    label="Download Transcription as TXT",
                    data=transcript,
                    file_name="transcription.txt",
                    mime="text/plain",
                )
        
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            st.error("Full error:", exc_info=True)
    elif not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to enable transcription.")
