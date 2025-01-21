import os
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for inline plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from io import BytesIO
import base64
from together import Together

class DataVisualizer:
    def __init__(self):
        self.pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

    def match_code_blocks(self, llm_response: str) -> str:
        match = self.pattern.search(llm_response)
        if match:
            code = match.group(1)
            return code
        return ""

    def execute_visualization(self, code: str, df: pd.DataFrame):
        try:
            plt.close('all')
            plt.figure(figsize=(10, 6))
            local_vars = {'df': df, 'plt': plt, 'sns': sns, 'pd': pd}
            exec(code, globals(), local_vars)
            fig = plt.gcf()
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return image_png
            
        except Exception as e:
            return None

    def analyze_dataset(self, file_path: str, query: str) -> dict:
        try:
            # Read the dataset
            file_extension = file_path.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # System prompt for the visualization
            system_prompt = """You're a Python data scientist and data visualization expert. Given a pandas DataFrame 'df', generate Python code to create visualizations using matplotlib, seaborn, or pandas plotting functions. The data fields should be clearly visible. Focus on creating clear and informative visualizations that answer the user's query. Only use columns that exist in the DataFrame. Provide ONLY a very brief interpretation of the visualization without showing any code."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            # Get API key from environment
            together_api_key = os.getenv('TOGETHER_API_KEY')
            client = Together(api_key=together_api_key)
            
            # Get model response
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=messages,
            )

            response_message = response.choices[0].message
            python_code = self.match_code_blocks(response_message.content)
            interpretation = response_message.content.split('```')[0].strip()

            if python_code:
                image_data = self.execute_visualization(python_code, df)
                if image_data:
                    return {
                        "success": True,
                        "image_data": image_data,
                        "interpretation": interpretation
                    }

            return {
                "success": False,
                "error": "Failed to generate visualization"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
