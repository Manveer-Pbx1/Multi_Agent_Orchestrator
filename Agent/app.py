import os
import streamlit as st
from main import process_query
from agent_manager import AgentManager
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize AgentManager and other components
agent_manager = AgentManager()
chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.5)

# Page configuration
st.set_page_config(page_title="AI Agent Chat", page_icon="🤖", layout="wide")

# Title (optional, can be removed for a clean chat UI)
st.title("AI Agent Chat Interface")

# Create a temp directory if it doesn't exist
if 'temp_dir' not in st.session_state:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    st.session_state.temp_dir = temp_dir

# Input box fixed at the bottom
with st.container():
    # Chat input and file uploader section
    col1, col2 = st.columns([3, 1])

    # Chat input box
    with col1:
        prompt = st.chat_input("Type your message here...")

    # Upload button
    with col2:
        if st.button("Upload File"):
            st.session_state.show_uploader = True

# Independent file uploader section
if st.session_state.get("show_uploader", False):
    st.markdown("### Upload your file below:")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'txt', 'pdf'])
    if uploaded_file:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.session_state.show_uploader = False
        st.session_state.uploaded_file = uploaded_file
    else:
        st.session_state.uploaded_file = None

# Display chat messages above the input box
with st.container():
    st.subheader("Chat Messages")
    for message in reversed(st.session_state.messages):  # Reverse the order of messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Handle input processing
if prompt:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            # Handle queries with or without uploaded files
            if st.session_state.get("uploaded_file"):
                file_path = os.path.abspath(os.path.join(st.session_state.temp_dir, st.session_state.uploaded_file.name))
                with open(file_path, "wb") as f:
                    f.write(st.session_state.uploaded_file.getbuffer())
                response = process_query(file_path=file_path, query=prompt)
            else:
                response = process_query(file_path=None, query=prompt)

            # Save and display the assistant's response
            content = response if isinstance(response, str) else str(response)
            st.write(content)
            st.session_state.messages.append({"role": "assistant", "content": content})

# Sidebar with agent information
with st.sidebar:
    st.header("Available Agents")
    agents = agent_manager.get_agents()
    for agent in agents:
        st.subheader(agent['name'])
        st.write(f"Description: {agent['description']}")
        st.write(f"Tools: {', '.join(agent['tools'])}")
        st.divider()
