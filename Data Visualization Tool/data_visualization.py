import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
from PIL import Image
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from together import Together
from e2b_code_interpreter import Sandbox

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

#TOGEHTER_API_KEY=19f0d27a6c1fb6f895945660a5083f0c35b92381fff8f90c32650f783c3a30e0
#E2B: e2b_9cc533cc6f421e8d3e5758f1c96d713e92a8c74c

def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    try:
        with st.spinner('Executing code in E2B sandbox...'):
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec = e2b_code_interpreter.run_code(code)

            if stderr_capture.getvalue():
                print("[Code Interpreter Warnings/Errors]", file=sys.stderr)
                print(stderr_capture.getvalue(), file=sys.stderr)

            if stdout_capture.getvalue():
                print("[Code Interpreter Output]", file=sys.stdout)
                print(stdout_capture.getvalue(), file=sys.stdout)

            if exec.error:
                st.error(f"Code execution error: {exec.error}")
                return None
            return exec.results
    except Exception as e:
        st.error(f"Error during code execution: {str(e)}")
        return None

def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""

def chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    try:
        system_prompt = f"""You're a Python data scientist and data visualization expert. You are given a dataset at path '{dataset_path}' and also the user's query. Rephrase natural language queries into data visualization prompts that strictly and explicitly reference the exact column names from the dataset. DO NOT infer or create columns that are not explicitly mentioned in the dataset. Check the user query for hints as they always contain the column name. 
You need to analyze the dataset and answer the user's query with a response and you run Python code to solve them. 
IMPORTANT: Always use the dataset path variable '{dataset_path}' in your code when reading the CSV file."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        with st.spinner('Getting response from Together AI LLM model...'):
            client = Together(api_key=st.session_state.together_api_key)
            response = client.chat.completions.create(
                model=st.session_state.model_name,
                messages=messages,
            )

            response_message = response.choices[0].message
            python_code = match_code_blocks(response_message.content)
            
            if python_code:
                code_interpreter_results = code_interpret(e2b_code_interpreter, python_code)
                return code_interpreter_results, response_message.content
            else:
                st.warning(f"Failed to match any Python code in model's response")
                return None, response_message.content
    except Exception as e:
        st.error(f"Error communicating with LLM: {str(e)}")
        return None, f"Error: {str(e)}"

def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    
    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error


def main():
    """Main Streamlit application."""
    st.title("📊 AI Data Visualization Agent")
    st.write("Upload your dataset and ask questions about it!")

    if 'together_api_key' not in st.session_state:
        st.session_state.together_api_key = ''
    if 'e2b_api_key' not in st.session_state:
        st.session_state.e2b_api_key = ''
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''

    with st.sidebar:
        st.header("API Keys and Model Configuration")
        st.session_state.together_api_key = st.sidebar.text_input("Together AI API Key", type="password")
        st.sidebar.info("💡 Everyone gets a free $1 credit by Together AI - AI Acceleration Cloud platform")
        st.sidebar.markdown("[Get Together AI API Key](https://api.together.ai/signin)")
        
        st.session_state.e2b_api_key = st.sidebar.text_input("Enter E2B API Key", type="password")
        st.sidebar.markdown("[Get E2B API Key](https://e2b.dev/docs/legacy/getting-started/api-key)")
        
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
            
            if file_extension not in ['csv', 'xlsx', 'xls']:
                st.error("Only CSV and Excel files (xlsx, xls) are supported.")
                return
                
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            else:  # Excel files (xlsx, xls)
                df = pd.read_excel(uploaded_file)
            
            st.write("Dataset:")
            show_full = st.checkbox("Show full dataset")
            if show_full:
                st.dataframe(df)
            else:
                st.write("Preview (first 5 rows):")
                st.dataframe(df.head())

            # For sandbox: convert Excel to CSV if needed
            if file_extension in ['xlsx', 'xls']:
                buffer = BytesIO()
                df.to_csv(buffer, index=False)
                buffer.seek(0)
                uploaded_file = buffer
                uploaded_file.name = uploaded_file.name.rsplit('.', 1)[0] + '.csv'
            else:
                uploaded_file.seek(0) 

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return

        query = st.text_area("What would you like to know about your data?",
                            "Can you compare the average cost for two people between different categories?")
        
        if st.button("Analyze"):
            if not st.session_state.together_api_key:
                st.error("Please enter your Together AI API key in the sidebar.")
                return
            if not st.session_state.e2b_api_key:
                st.error("Please enter your E2B API key in the sidebar.")
                return
            
            try:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)
                    code_results, llm_response = chat_with_llm(code_interpreter, query, dataset_path)
                    
                    st.write("AI Response:")
                    st.write(llm_response)
                    
                    if code_results:
                        for result in code_results:
                            try:
                                if hasattr(result, 'png') and result.png:
                                    png_data = base64.b64decode(result.png)
                                    image = Image.open(BytesIO(png_data))
                                    st.image(image, caption="Generated Visualization", use_container_width=False)
                                elif hasattr(result, 'figure'):
                                    st.pyplot(result.figure)
                                elif hasattr(result, 'show'):
                                    st.plotly_chart(result)
                                elif isinstance(result, (pd.DataFrame, pd.Series)):
                                    st.dataframe(result)
                                else:
                                    st.write(result)
                            except Exception as e:
                                st.error(f"Error displaying result: {str(e)}")
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()