from Tools.sentiment import analyze_sentiment
from Tools.data_visualization import DataVisualizer
from Tools.file_converter import convert_file
from Tools.image_gen import generate_image, edit_image
from Tools.speech import launch_speech_app
from Tools.index import web_analysis_tool
from crewai.tools import BaseTool

class SentimentAnalyzerTool(BaseTool):
    name: str = "sentiment_analyzer"
    description: str = "Analyzes the emotional tone and sentiment in text"

    def _run(self, text: str) -> dict:
        try:
            analyze_sentiment(text)  # Execute but don't return result
            return {"notification": "Sentiment analysis tool has been invoked. Analyzing your text."}
        except Exception as e:
            return {"error": f"Failed to analyze sentiment: {str(e)}"}

class DataVisualizationTool(BaseTool):
    name: str = "data_visualization"
    description: str = "Creates visual representations of data through graphs, charts, and plots"

    def _run(self, file_path: str, query: str) -> dict:
        try:
            visualizer = DataVisualizer()
            results, response = visualizer.analyze_dataset(file_path, query)
            
            if results:
                return {"results": results, "explanation": "Visualization generated successfully"}
            else:
                return {"results": None, "explanation": f"Failed to generate visualization: {response}"}
        except Exception as e:
            return {"results": None, "explanation": f"Error: {str(e)}"}

class ImageGeneratorTool(BaseTool):
    name: str = "image_generator"
    description: str = "Generates and processes images"

    def _run(self, prompt: str, **kwargs) -> dict:
        try:
            generate_image(prompt)  # Execute but don't return result
            return {"notification": "Image generation tool has been invoked. Your image will be generated shortly."}
        except Exception as e:
            return {"error": f"Failed to generate image: {str(e)}"}

class WebAnalyzerTool(BaseTool):
    name: str = "web_analyzer"
    description: str = "Analyzes web content and performs web-related tasks"

    def _run(self, action: str, params: dict) -> dict:
        try:
            web_analysis_tool(action, params)  # Execute but don't return result
            return {"notification": "Web analysis tool has been invoked. Processing your request."}
        except Exception as e:
            return {"error": f"Web analysis failed: {str(e)}"}

class FileConverterTool(BaseTool):
    name: str = "file_converter"
    description: str = "Converts files between different formats"

    def _run(self, file_bytes: bytes, input_format: str, target_format: str) -> dict:
        try:
            result = convert_file(file_bytes, input_format, target_format)
            return {"converted_file": result}
        except Exception as e:
            return {"error": f"File conversion failed: {str(e)}"}

class SpeechProcessorTool(BaseTool):
    name: str = "speech_processor"
    description: str = "Processes and analyzes speech and audio content"

    def _run(self, **kwargs) -> dict:
        # Instead of launching the app directly, just return a notification
        return {"notification": "Speech processing tool has been invoked. A new window will open for audio interaction."}

class ToolManager:
    def __init__(self):
        self.data_visualizer = DataVisualizer()
        self.tools = {
            'sentiment_analyzer': SentimentAnalyzerTool(),
            'data_visualization': DataVisualizationTool(),
            'file_converter': FileConverterTool(),
            'image_generator': ImageGeneratorTool(),
            'web_analyzer': WebAnalyzerTool(),
            'speech_processor': SpeechProcessorTool()
        }
    
    def get_tool(self, tool_name):
        return self.tools.get(tool_name)

    def get_tools(self):
        return list(self.tools.values())
