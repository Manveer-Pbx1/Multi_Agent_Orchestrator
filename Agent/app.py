import os
import logging  # Add logging import
import streamlit as st
from main import process_query
from agent_manager import AgentManager
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO
from together import Together

# Configure logger
logging.basicConfig(level=logging.ERROR)  # Set to ERROR to avoid debug logs
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'audio_chunks' not in st.session_state:
    st.session_state.audio_chunks = []
if 'temp_audio_path' not in st.session_state:
    st.session_state.temp_audio_path = None

if 'together_api_key' not in st.session_state:
    st.session_state.together_api_key = os.getenv('TOGETHER_API_KEY')
if 'model_name' not in st.session_state:
    st.session_state.model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# Initialize AgentManager and other components
agent_manager = AgentManager()
chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.5)

# Page configuration
st.set_page_config(page_title="AI Agent Chat", page_icon="🤖", layout="wide")

# Title (optional, can be removed for a clean chat UI)
st.title("AI Agent Orchestrator")

# Create a temp directory if it doesn't exist
if 'temp_dir' not in st.session_state:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    st.session_state.temp_dir = temp_dir

# Define supported file types
SUPPORTED_FORMATS = ['pdf', 'docx', 'xlsx', 'csv', 'png', 'jpg', 'jpeg', 
                    'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'mp3', 'wav', 'm4a', 'flac', 'aac']

# Input box fixed at the bottom
prompt = st.chat_input("Type your message here (e.g., 'Convert this PDF to DOCX' or 'Convert this video to MP4')...")
uploaded_file = st.file_uploader("Upload your file", type=SUPPORTED_FORMATS)

# Add format hint
if uploaded_file:
    file_type = os.path.splitext(uploaded_file.name)[1][1:].lower()
    
    # Define all video formats
    VIDEO_FORMATS = ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv']
    
    supported_conversions = {
        'pdf': ['docx'],
        'docx': ['pdf'],
        'xlsx': ['csv'],
        'csv': ['xlsx'],
        'png': ['txt'],  # OCR
        'jpg': ['txt'],  # OCR
        'jpeg': ['txt']  # OCR
    }
    
    # Add video format conversions dynamically
    for video_format in VIDEO_FORMATS:
        supported_conversions[video_format] = [fmt for fmt in VIDEO_FORMATS if fmt != video_format]
    
    if file_type in supported_conversions:
        formats = supported_conversions[file_type]
        if file_type in VIDEO_FORMATS:
            st.info(f"💡 You can convert this video to any of these formats: {', '.join(formats).upper()}")
        else:
            st.info(f"💡 You can convert this {file_type.upper()} file to: {', '.join(formats).upper()}")
    
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    st.session_state.uploaded_file = uploaded_file

    # Add this hint after file upload for data files
    if uploaded_file and uploaded_file.name.endswith(('.csv', '.xlsx', '.xls')):
        st.info("💡 You can ask me to create visualizations from this data. Try asking something like: \n\n" +
                "- 'Create a visualization showing the distribution of values'\n" +
                "- 'Show me a graph comparing the different categories'\n" +
                "- 'Generate a plot analyzing the trends in this data'")

col1, col2 = st.columns(2)

with col1:
    if not st.session_state.recording:
        if st.button("🎤 Start Recording"):
            st.session_state.recording = True
            st.session_state.audio_chunks = []
            st.rerun()

with col2:
    if st.session_state.recording:
        if st.button("⏹️ Stop Recording"):
            st.session_state.recording = False
            if st.session_state.audio_chunks:
                audio_data = np.concatenate(st.session_state.audio_chunks)
                temp_path = os.path.join(st.session_state.temp_dir, "temp_recording.wav")
                sf.write(temp_path, audio_data, 16000)
                st.session_state.temp_audio_path = temp_path
                st.session_state.audio_data = audio_data
            st.rerun()

if st.session_state.recording:
    st.warning("🎙️ Recording in progress... Press 'Stop Recording' when finished.")
    try:
        with sd.InputStream(channels=1, samplerate=16000, dtype=np.float32) as stream:
            while st.session_state.recording:
                audio_chunk, _ = stream.read(1024)
                st.session_state.audio_chunks.append(audio_chunk)
                time.sleep(0.01)
    except Exception as e:
        st.error(f"Recording error: {str(e)}")
        st.session_state.recording = False

if st.session_state.temp_audio_path and os.path.exists(st.session_state.temp_audio_path):
    st.audio(st.session_state.temp_audio_path)
    st.info("💡 Type 'transcribe this audio' or similar to process the recording")

# Display chat messages above the input box
with st.container():
    st.subheader("Chat Messages")
    for message in reversed(st.session_state.messages):  # Reverse the order of messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Add MIME type mapping
MIME_TO_EXTENSION = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'text/csv': '.csv',
    'text/plain': '.txt',
    'video/mp4': '.mp4',
    'video/x-msvideo': '.avi',
    'video/x-matroska': '.mkv',
    'video/quicktime': '.mov',
    'video/x-ms-wmv': '.wmv',
    'video/x-flv': '.flv'
}


def display_response(response):
    try:
        if isinstance(response, dict):
            response_type = response.get('type')
            
            # Enhanced visualization handling
            if response_type == 'visualization':
                if 'image_data' in response:
                    try:
                        # Display the visualization
                        st.image(response['image_data'])
                        
                        # Show interpretation if available
                        if 'interpretation' in response:
                            st.markdown("### Interpretation")
                            st.markdown(response['interpretation'])
                        
                        # Add download button
                        st.download_button(
                            label="📥 Download Visualization",
                            data=response['image_data'],
                            file_name="visualization.png",
                            mime="image/png"
                        )
                        
                        return "Visualization generated successfully"
                    except Exception as viz_error:
                        st.error(f"Error displaying visualization: {str(viz_error)}")
                        return "Failed to display visualization"

            # Handle visualization responses first
            if response_type == 'visualization':
                if 'image_data' in response:
                    try:
                        # Display the visualization
                        st.image(response['image_data'])
                        
                        # Add download button
                        st.download_button(
                            label="📥 Download Visualization",
                            data=response['image_data'],
                            file_name="visualization.png",
                            mime="image/png"
                        )
                        return "Visualization generated successfully"
                    except Exception as viz_error:
                        st.error(f"Error displaying visualization: {str(viz_error)}")
                        return "Failed to display visualization"

             
            # Handle image generation responses
            if response_type == 'image':
                if 'url' in response:
                    st.image(response['url'], caption="Generated by DALL-E", use_container_width=True)
                    # Display the URL below the image
                    st.markdown(f"**Image URL**: {response['url']}")
                    return response.get('content', "Image generated successfully")
             

            # Handle visualization responses
            if response_type == 'visualization':
                if 'image_data' in response:
                    # Display the image
                    st.image(response['image_data'], use_column_width=True)
                    
                    # Add download button
                    st.download_button(
                        label="📥 Download Visualization",
                        data=response['image_data'],
                        file_name=response.get('filename', 'visualization.png'),
                        mime=response.get('mime_type', 'image/png')
                    )
                    return "Visualization generated successfully"
            
            # Handle file conversion responses
            if response_type == 'file' and 'data' in response and 'mime_type' in response:
                if not isinstance(response['data'], bytes):
                    st.error("Invalid file data format")
                    return "File conversion failed: Invalid data format"
                
                extension = MIME_TO_EXTENSION.get(response['mime_type'], '.txt')
                filename = f"converted_file{extension}"
                
                # Create a unique key for each download button
                download_key = f"download_{filename}_{st.session_state.messages.__len__()}"
                
                # Create download button
                st.download_button(
                    label=f"📥 Download {extension.upper()} File",
                    data=response['data'],
                    file_name=filename,
                    mime=response['mime_type'],
                    key=download_key  # Ensure unique key
                )
                
                # Show success message
                st.success(f"✅ File successfully converted to {extension.upper()}")
                # Avoid returning early to allow further processing
                # return f"File converted to {extension.upper()}. Click the download button above to save your file."

             
            # Handle image generation responses
            if response_type == 'image':
                if 'url' in response:
                    image_url = response['url']
                    if image_url.startswith(('http://', 'https://')):
                         
                        # For DALL-E images (direct URLs)
                        st.image(image_url, caption="Generated by DALL-E", use_container_width=True)
                    elif image_url.startswith('file://'):
                         
                        # For FLUX images (local files)
                        file_path = response.get('file_path')
                        if file_path and os.path.exists(file_path):
                            with open(file_path, 'rb') as f:
                                st.image(f.read(), caption="Generated by FLUX.1-Schnell", use_container_width=True)
                    st.markdown(f"**Image URL**: {image_url}")
                    return response.get('content', "Image generated successfully")

             
            # Handle transcription responses
            if response_type == 'text' and "Transcription" in response.get('content', ''):
                st.markdown(f"**Transcription:** {response['content']}")
                return response['content']

             
            # Handle visualization responses
            if response_type == 'visualization' and 'plot_data' in response:
                 
                # Display the base64 image
                st.image(response['plot_data'], use_column_width=True)
                
                 
                # Show interpretation if available
                if response.get('explanation'):
                    st.markdown("### Analysis")
                    st.markdown(response['explanation'])
                
                return response.get('content', "Visualization generated successfully")

            # Handle non-audio query responses
            if response.get('agent_info'):
                st.info(response['agent_info'])

            # Handle text responses
            elif response_type == 'text' or 'content' in response:
                content = response.get('content', '')
                if content:
                    st.markdown(content)
                    return content
            
            # Default handling
            if 'content' in response:
                st.markdown(response['content'])
                return response['content']
                
        elif isinstance(response, str):
            st.markdown(response)
            return response
            
        return "Unsupported response format"
        
    except Exception as e:
        error_msg = f"Error displaying response: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg)
        return error_msg

# Handle input processing
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                # Check if there's recorded audio and the prompt is about transcription
                if (st.session_state.temp_audio_path and 
                    os.path.exists(st.session_state.temp_audio_path) and 
                    any(word in prompt.lower() for word in ['transcribe', 'transcription', 'convert to text'])):
                    response = process_query(
                        file_path=st.session_state.temp_audio_path,
                        query="Please transcribe this audio file"
                    )
                elif st.session_state.get("uploaded_file"):
                    file_path = os.path.join(st.session_state.temp_dir, st.session_state.uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(st.session_state.uploaded_file.getbuffer())
                    response = process_query(file_path=file_path, query=prompt)
                else:
                    response = process_query(file_path=None, query=prompt)

                # After processing, clean up the temporary audio file
                if st.session_state.temp_audio_path and os.path.exists(st.session_state.temp_audio_path):
                    try:
                        os.remove(st.session_state.temp_audio_path)
                        st.session_state.temp_audio_path = None
                    except Exception as e:
                        logger.error(f"Error cleaning up temp audio file: {str(e)}")

                # Display and format the response
                chat_message = display_response(response)
                
                # Add assistant's response to chat history
                if chat_message:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": chat_message
                    })
                    
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                st.error(error_msg)
                logger.error(f"Processing error: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Sidebar with agent information
with st.sidebar:
    st.header("Available Agents")
    agents = agent_manager.get_agents()
    for agent in agents:
        st.subheader(agent['name'])
        st.write(f"Description: {agent['description']}")
        st.write(f"Tools: {', '.join(agent['tools'])}")
        st.divider()

# Add this function after your imports
def chat_with_llm(df: pd.DataFrame, user_message: str):
    try:
        # Add column info to the prompt
        column_info = f"\nAvailable columns: {', '.join(df.columns.tolist())}"
        system_prompt = f"""You're a Python data scientist and data visualization expert. Given a pandas DataFrame 'df', generate Python code to create visualizations using matplotlib, seaborn, or pandas plotting functions. 

The data fields should be clearly visible. Focus on creating clear and informative visualizations that answer the user's query.

{column_info}

Important: Your response MUST include Python code wrapped in ```python``` code blocks.
Example format:
Brief interpretation of what the visualization will show.
```python
# Your Python code here
```"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        with st.spinner('Creating visualization...'):
            # Debug: Print API key (masked)
            api_key = st.session_state.together_api_key
            if api_key:
                logger.info(f"API Key found: {api_key[:4]}...{api_key[-4:]}")
            else:
                st.error("No Together API key found!")
                return {"success": False, "type": "text", "content": "API key missing"}

            client = Together(api_key=st.session_state.together_api_key)
            
            # Debug: Print DataFrame info
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            logger.info(f"DataFrame shape: {df.shape}")

            response = client.chat.completions.create(
                model=st.session_state.model_name,
                messages=messages,
            )

            response_message = response.choices[0].message
            
            # Debug: Print raw response
            logger.info(f"Raw response: {response_message.content}")

            # Extract code with better error handling
            if '```python' in response_message.content:
                python_code = response_message.content.split('```python')[1].split('```')[0]
                logger.info(f"Extracted code: {python_code}")
            else:
                st.error("No Python code found in response")
                return {"success": False, "type": "text", "content": "No visualization code generated"}

            interpretation = response_message.content.split('```')[0].strip()
            
            if python_code:
                try:
                    # Clear any existing plots
                    plt.close('all')
                    plt.figure(figsize=(10, 6))
                    
                    
                    # Execute the visualization code
                    local_vars = {'df': df, 'plt': plt, 'sns': sns, 'pd': pd}
                    exec(python_code, globals(), local_vars)
                    
                    
                    # Save the plot to bytes
                    buf = BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                    buf.seek(0)
                    image_data = buf.getvalue()
                    plt.close()
                    
                    return {
                        "success": True,
                        "type": "visualization",
                        "image_data": image_data,
                        "interpretation": interpretation
                    }
                except Exception as viz_error:
                    st.error(f"Visualization execution error: {str(viz_error)}")
                    st.error(f"Problematic code:\n{python_code}")
                    return {
                        "success": False,
                        "type": "text",
                        "content": f"Error executing visualization: {str(viz_error)}"
                    }
            
            return {
                "success": False,
                "type": "text",
                "content": "No valid visualization code generated"
            }
            
    except Exception as e:
        logger.error(f"General error in chat_with_llm: {str(e)}")
        return {
            "success": False,
            "type": "text",
            "content": f"Error generating visualization: {str(e)}"
        }


# Update the visualization handling section
if uploaded_file is not None and uploaded_file.name.endswith(('.csv', '.xlsx', '.xls')):
    try:
        
        # Read the data file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        
        # Display data preview and column names
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        st.write("Available columns:", ', '.join(df.columns.tolist()))

        
        # Add visualization prompt
        if prompt and any(word in prompt.lower() for word in 
                         ['visualize', 'plot', 'graph', 'chart', 'show', 'create', 'distribution', 'trend']):
            with st.spinner("Generating visualization..."):
                result = chat_with_llm(df, prompt)
                
                if result.get('success'):
                    
                    # Display the visualization
                    st.image(result['image_data'])
                    
                    if result.get('interpretation'):
                        st.markdown("### Interpretation")
                        st.markdown(result['interpretation'])
                    
                    st.download_button(
                        label="📥 Download Visualization",
                        data=result['image_data'],
                        file_name="visualization.png",
                        mime="image/png"
                    )
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Visualization generated successfully"
                    })
                else:
                    st.error(result.get('content', 'Failed to generate visualization'))

    except Exception as e:
        st.error(f"Error processing data file: {str(e)}")
        logger.error(f"Error processing data file: {str(e)}")


