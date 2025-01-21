import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
from venv import logger
from Tools.sentiment import analyze_sentiment
from Tools.data_visualization import DataVisualizer
from Tools.file_converter import convert_file
from Tools.image_gen import _run, generate_image, edit_image, generate_image_flux
from Tools.speech import launch_speech_app, transcribe_audio_file
from Tools.index import web_analysis_tool
from crewai.tools import BaseTool
import base64

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SentimentAnalyzerTool(BaseTool):
    name: str = "sentiment_analyzer"
    description: str = "Analyzes the emotional tone and sentiment in text"

    def _run(self, text: str) -> dict:
        try:
            result = analyze_sentiment(text)
            if result["success"]:
                formatted_response = (
                    f"Sentiment Analysis Result:\n"
                    f"• Sentiment: {result['sentiment']}\n"
                    f"• Confidence: {result['confidence'] * 100:.1f}%"
                )
                return {
                    "success": True,
                    "type": "text",
                    "content": formatted_response
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Failed to analyze sentiment")
                }
        except Exception as e:
            return {
                "success": False,
                "type": "text",
                "content": f"Failed to analyze sentiment: {str(e)}"
            }

class DataVisualizationTool(BaseTool):
    name: str = "data_visualization"
    description: str = "Creates visual representations of data"

    def _run(self, file_path: str, query: str) -> dict:
        try:
            visualizer = DataVisualizer()
            result = visualizer.analyze_dataset(file_path, query)
            
            if result.get('success'):
                return {
                    "success": True,
                    "type": "visualization",
                    "image_data": result['image_data'],
                    "interpretation": result.get('interpretation', '')
                }
            
            return {
                "success": False,
                "type": "text",
                "content": result.get('error', 'Failed to generate visualization')
            }
            
        except Exception as e:
            return {
                "success": False,
                "type": "text",
                "content": f"Visualization failed: {str(e)}"
            }

class ImageGeneratorTool(BaseTool):
    name: str = "image_generator"
    description: str = "Generates and processes images"

    def _run(self, prompt: str, model: str = "DALL-E") -> dict:
        try:
            result = _run(prompt, model)
            
            # Format response consistently
            if result.get("success"):
                if "url" in result:  # For both DALL-E and FLUX results
                    return {
                        "success": True,
                        "type": "image",
                        "url": result["url"],
                        "content": result["content"]
                    }
            return result
        except Exception as e:
            return {
                "success": False,
                "type": "text",
                "content": f"Failed to generate image: {str(e)}"
            }

class WebAnalyzerTool(BaseTool):
    name: str = "web_analyzer"
    description: str = "Analyzes web content and performs web-related tasks"

    def _run(self, action: str, params: dict) -> dict:
        try:
            result = web_analysis_tool(action, params)
            # Ensure proper formatting of search results
            if isinstance(result, dict):
                return result
            else:
                return {
                    "success": True,
                    "type": "text",
                    "content": str(result)
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Web analysis failed: {str(e)}"
            }

class FileConverterTool(BaseTool):
    name: str = "file_converter"
    description: str = "Converts files between different formats"

    def _run(self, file_bytes: bytes, input_format: str, target_format: str) -> dict:
        try:
            if not isinstance(file_bytes, bytes):
                raise ValueError("Invalid file data: expected bytes")
                
            converted_bytes, mime_type = convert_file(file_bytes, input_format, target_format)
            if not converted_bytes:
                raise ValueError("No converted data received")
            
            # Avoid logging binary data
            logger.debug(f"Successfully converted {input_format} to {target_format}")
            
            return {
                "success": True,
                "type": "file",
                "data": converted_bytes,
                "mime_type": mime_type,
                "content": f"Successfully converted {input_format.upper()} to {target_format.upper()}"
            }
        except Exception as e:
            logger.error(f"File conversion failed: {str(e)}")
            return {
                "success": False,
                "type": "text",
                "content": f"File conversion failed: {str(e)}"
            }

    def __str__(self):
        return f"FileConverterTool(name={self.name})"

class SpeechProcessorTool(BaseTool):
    name: str = "speech_processor"
    description: str = "Processes and analyzes speech and audio content"
    
    def _run(self, audio_file: bytes) -> dict:
        try:
            if not client.api_key:
                return {
                    "success": False,
                    "type": "text",
                    "content": "OpenAI API key not found in environment variables"
                }
            
            from io import BytesIO
            audio_stream = BytesIO(audio_file)
            audio_stream.name = "audio.wav"  # Required by OpenAI API
            transcript = transcribe_audio_file(audio_stream)
            
            if not transcript:
                return {
                    "success": False,
                    "type": "text",
                    "content": "No transcription generated"
                }
                
            return {
                "success": True,
                "type": "text",
                "content": f"Transcription: {transcript}"
            }
        except Exception as e:
            return {
                "success": False,
                "type": "text",
                "content": f"Failed to process speech: {str(e)}"
            }

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
