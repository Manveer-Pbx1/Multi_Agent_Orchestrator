import streamlit as st
from index import process_prompt, handle_conversation
import uuid
import os
from datetime import datetime
from document_processor import store_document, query_document
import tempfile

# Create prompts directory if it doesn't exist
PROMPTS_DIR = "prompts"
os.makedirs(PROMPTS_DIR, exist_ok=True)

def save_prompt(user_id, prompt):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(PROMPTS_DIR, f"{user_id}_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"User ID: {user_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Prompt: {prompt}\n")
    return filename

# Update LLM Model configurations with more options
LLM_MODELS = {
    "OpenAI": [
        "openai/gpt-4",
        "openai/gpt-4o-mini",
    ],
    "Anthropic": [
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-3-5-haiku-20241022"
    ],
    "Groq": [
        "groq/llama-3.1-8b-instant",
        "groq/llama3-70b-8192",
        "groq/llama3-8b-8192",
    ]
}

def main():
    st.title("Conversational AI Agent")
    
    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_created" not in st.session_state:
        st.session_state.agent_created = False
    if "crew" not in st.session_state:
        st.session_state.crew = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()  # Track uploaded files

    # Sidebar tabs
    tab1, tab2 = st.sidebar.tabs(["Agent Configuration", "Document Upload"])

    with tab2:
        st.header("Document Upload (RAG)")
        st.write("Upload documents for the agent to reference during conversations.")
        uploaded_file = st.file_uploader(
            "Upload PDF or DOCX file",
            type=["pdf", "docx"],
            help="Upload a document for the agent to use as reference"
        )
        
        enable_rag = st.checkbox("Enable RAG-based responses", 
                               value=st.session_state.rag_enabled)
        
        if enable_rag != st.session_state.rag_enabled:
            st.session_state.rag_enabled = enable_rag

        # Show currently uploaded files
        if st.session_state.uploaded_files:
            st.subheader("Uploaded Documents")
            for filename in st.session_state.uploaded_files:
                st.write(f"- {filename}")

        if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner("Processing document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        file_path = tmp_file.name

                    # Store document and get chunk info
                    result = store_document(file_path, st.session_state.user_id)
                    st.session_state.uploaded_files.add(uploaded_file.name)
                    
                    # Display chunk information
                    st.success(f"""Document processed successfully!
                    - Filename: {result['filename']}
                    - Total chunks: {result['chunk_count']}
                    - Average chunk size: {500} characters
                    """)
                    
                    os.unlink(file_path)
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

    with tab1:
        # Existing model selection and agent creation code
        st.header("Model Selection")
        provider = st.selectbox(
            "Select Provider",
            options=list(LLM_MODELS.keys()),
            key="provider_select"
        )
        model = st.selectbox(
            "Select Model",
            options=LLM_MODELS[provider],
            key="model_select"
        )

        st.header("Create Agent")
        user_prompt = st.text_area("Enter agent description", height=150,
                                placeholder="Example: I need an agent that acts as a professional food critic...")
        
        if st.button("Create Agent"):
            if not user_prompt:
                st.error("Please enter a prompt first!")
                return
                
            with st.spinner("Creating agent..."):
                try:
                    # Save prompt to file
                    saved_file = save_prompt(st.session_state.user_id, user_prompt)
                    st.success(f"Prompt saved to: {saved_file}")
                    
                    # Create agent with selected model
                    result = process_prompt(user_prompt, provider, model)
                    st.session_state.crew = result["crew"]
                    st.session_state.agent_created = True
                    
                    # Display agent properties in sidebar
                    st.success("Agent created successfully!")
                    st.subheader("Agent Properties")
                    st.write(f"**Role:** {result['agent'].role}")
                    st.write(f"**Goal:** {result['agent'].goal}")
                    st.write(f"**Backstory:** {result['agent'].backstory}")
                    
                    # Display tasks from crew
                    st.write("**Tasks:**")
                    for task in result['crew'].tasks:
                        st.write("**Description:**")
                        st.write(f"- {task.description}")
                        st.write("**Expected Output:**")
                        st.write(f"- {task.expected_output}")
                    
                except Exception as e:
                    st.error(f"Error creating agent: {str(e)}")

    # Main chat interface
    if st.session_state.agent_created:
        # Display current model info
        st.info(f"Currently using: {model}")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Chat with your agent..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Agent is thinking..."):
                if st.session_state.rag_enabled and st.session_state.uploaded_files:
                    try:
                        # Get relevant document chunks
                        relevant_chunks = query_document(prompt, st.session_state.user_id)
                        
                        if not relevant_chunks:
                            response = "No relevant information found in the uploaded documents."
                        else:
                            # Display retrieved chunks info
                            with st.expander("View source context", expanded=True):
                                st.markdown("### Retrieved Document Chunks")
                                for i, chunk in enumerate(relevant_chunks, 1):
                                    st.markdown(f"""
                                    **Chunk {i}**
                                    - Source: {chunk['metadata']['source']}
                                    - Chunk: {chunk['metadata']['chunk_index'] + 1} of {chunk['metadata']['total_chunks']}
                                    - Relevance Score: {chunk['score']:.3f}
                                    ```
                                    {chunk['content']}
                                    ```
                                    """)
                            
                            response = handle_conversation(
                                st.session_state.crew,
                                prompt,
                                provider,
                                model,
                                is_rag=True,
                                context=relevant_chunks
                            )
                    except Exception as e:
                        response = f"Error querying documents: {str(e)}"
                        print(f"RAG error: {str(e)}")
                else:
                    response = handle_conversation(
                        st.session_state.crew,
                        prompt,
                        provider,
                        model
                    )
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    else:
        st.info("Create an agent using the sidebar to start chatting!")

if __name__ == "__main__":
    main()
