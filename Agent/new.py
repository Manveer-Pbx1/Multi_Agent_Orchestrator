import os
import json
import re
import sys
import warnings
from typing import Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for inline plotting
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from together import Together
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""

def execute_visualization(code: str, df: pd.DataFrame) -> Optional[plt.Figure]:
    try:
        # Clear any existing plots
        plt.close('all')
        
        # Create a new figure
        plt.figure(figsize=(10, 6))
        
        # Add df to local namespace
        local_vars = {'df': df, 'plt': plt, 'sns': sns, 'pd': pd}
        
        # Execute the code
        exec(code, globals(), local_vars)
        
        # Get the current figure and tight layout
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error during visualization: {str(e)}")
        return None

def chat_with_llm(df: pd.DataFrame, user_message: str) -> Tuple[Optional[plt.Figure], str]:
    try:
        system_prompt = """You're a Python data scientist and data visualization expert. Given a pandas DataFrame 'df', generate Python code to create visualizations using matplotlib, seaborn, or pandas plotting functions. The data fields should be clearly visible. Focus on creating clear and informative visualizations that answer the user's query. Only use columns that exist in the DataFrame. Provide ONLY a very brief interpretation of the visualization without showing any code."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        with st.spinner('Creating visualization...'):
            client = Together(api_key=st.session_state.together_api_key)
            response = client.chat.completions.create(
                model=st.session_state.model_name,
                messages=messages,
            )

            response_message = response.choices[0].message
            python_code = match_code_blocks(response_message.content)
            
            # Extract only the interpretation part (text before the code block)
            interpretation = response_message.content.split('```')[0].strip()
            
            if python_code:
                fig = execute_visualization(python_code, df)
                return fig, interpretation
            else:
                st.warning("Unable to generate visualization")
                return None, interpretation
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        return None, f"Error: {str(e)}"

def main():
    """Main Streamlit application."""
    st.title("📊 AI Data Visualization Agent")
    st.write("Upload your dataset and ask questions about it!")

    # Get API key from environment variable
    together_api_key = os.getenv('TOGETHER_API_KEY')
    if not together_api_key:
        st.error("TOGETHER_API_KEY environment variable is not set")
        return

    if 'together_api_key' not in st.session_state:
        st.session_state.together_api_key = together_api_key
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''

    with st.sidebar:
        st.header("Model Configuration")
        model_options = {
            "Meta-Llama 3.1 405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        }
        st.session_state.model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0  
        )
        st.session_state.model_name = model_options[st.session_state.model_name]

    uploaded_file = st.file_uploader("Choose a data file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("Dataset Preview:")
            st.dataframe(df.head())

            query = st.text_area("What would you like to visualize?",
                               "Can you compare the average values between different categories?")
            
            if st.button("Analyze"):
                fig, interpretation = chat_with_llm(df, query)
                
                if fig:
                    st.pyplot(fig)
                    if interpretation:
                        st.write("**Interpretation:**")
                        st.write(interpretation)

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return

if __name__ == "__main__":
    main()