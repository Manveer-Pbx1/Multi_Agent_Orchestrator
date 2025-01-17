import os
import json
import re
import sys
import io
import contextlib
import warnings
import base64
import uuid
from typing import Optional, List, Any, Tuple
import pandas as pd
from together import Together
from e2b_code_interpreter import Sandbox
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

class DataVisualizer:
    def __init__(self, 
                 together_api_key: str = "19f0d27a6c1fb6f895945660a5083f0c35b92381fff8f90c32650f783c3a30e0", 
                 e2b_api_key: str = "e2b_9cc533cc6f421e8d3e5758f1c96d713e92a8c74c",
                 model_name: str = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"):
        self.together_api_key = together_api_key
        self.e2b_api_key = e2b_api_key
        self.model_name = model_name
        self.pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
        self.plots_dir = "static/plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def save_plot_to_base64(self, plt):
        try:
            # Save plot to a bytes buffer
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            # Convert to base64
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()  # Clean up
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"Error saving plot: {e}")
            return None
        
    def code_interpret(self, e2b_code_interpreter: Sandbox, code: str) -> Optional[str]:
        try:
            # Extract the code for execution
            if 'python' in code:
                _, code = code.split('python', 1)

            # Get absolute path for plots directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            plots_dir = os.path.join(base_dir, self.plots_dir)
            os.makedirs(plots_dir, exist_ok=True)

            # Create the complete code with necessary setup
            complete_code = f"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
plt.style.use('seaborn')
plt.figure(figsize=(10, 6))

# Run the visualization code
try:
    {code}
    
    # Save the plot
    plt.tight_layout()
    plot_filename = '{uuid.uuid4()}.png'
    plot_path = os.path.join(r"{plots_dir}", plot_filename)
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(plot_path)
except Exception as e:
    print(f"Error in visualization: {{e}}")
"""

            # Execute in sandbox
            result = e2b_code_interpreter.run_code(complete_code)

            if result.error:
                print(f"Code execution error: {result.error}")
                return None

            # Extract the plot path from the result
            plot_path = result.stdout.strip() if isinstance(result.stdout, str) else None  # Modified line
            plot_path = os.path.abspath(plot_path) if plot_path else None  # Added line to ensure absolute path
            print(f"Debug - Plot path: {plot_path}")  # Debug line

            if plot_path and os.path.exists(plot_path):
                print(f"Plot successfully saved at: {plot_path}")  # Debug line
                return plot_path
            else:
                print(f"Plot was not saved or plot_path is invalid. Check sandbox execution logs.")  # Modified line
                return None

        except Exception as e:
            print(f"Error during code execution: {str(e)}")
            return None

    def match_code_blocks(self, llm_response: str) -> str:
        match = self.pattern.search(llm_response)
        if (match):
            code = match.group(1)
            return code
        return ""

    def chat_with_llm(self, e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[str], str]:
        try:
            system_prompt = f"""You are a data visualization expert. Create a visualization for the dataset at {dataset_path}.
Follow these rules strictly:
1. ALWAYS use plt.figure(figsize=(10, 6)) at the start
2. ALWAYS read the CSV file using: df = pd.read_csv('{dataset_path}')
3. Create clear, readable visualizations
4. Use appropriate chart types based on the data
5. Include proper labels and titles
6. Return ONLY the Python code wrapped in ```python ``` tags
7. Do not include any explanations
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            client = Together(api_key=self.together_api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )

            response_message = response.choices[0].message
            python_code = self.match_code_blocks(response_message.content)
            
            if python_code:
                code_interpreter_results = self.code_interpret(e2b_code_interpreter, python_code)
                return code_interpreter_results, response_message.content
            else:
                print(f"Failed to match any Python code in model's response")
                return None, response_message.content
        except Exception as e:
            print(f"Error communicating with LLM: {str(e)}")
            return None, f"Error: {str(e)}"

    def analyze_dataset(self, file_path: str, query: str) -> Tuple[Optional[str], str]:
        """
        Analyze a dataset with the given query.
        
        Args:
            file_path (str): Path to the CSV or Excel file
            query (str): Query about the dataset
            
        Returns:
            Tuple[Optional[str], str]: Results and LLM response
        """
        try:
            with Sandbox(api_key=self.e2b_api_key) as code_interpreter:
                # Read and convert file if needed
                if file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                    temp_csv = file_path.rsplit('.', 1)[0] + '.csv'
                    df.to_csv(temp_csv, index=False)
                    file_path = temp_csv

                # Upload file to sandbox
                with open(file_path, 'rb') as f:
                    dataset_path = f"./{os.path.basename(file_path)}"
                    code_interpreter.files.write(dataset_path, f)

                # Process the query
                return self.chat_with_llm(code_interpreter, query, dataset_path)  # Modified line
                
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return None, f"Error: {str(e)}"

# Example usage:
# visualizer = DataVisualizer(together_api_key="your_key", e2b_api_key="your_key")
# results, response = visualizer.analyze_dataset("path/to/dataset.csv", "Compare the average cost between categories")