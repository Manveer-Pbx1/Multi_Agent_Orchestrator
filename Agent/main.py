import io
from typing import Optional
from crewai import Agent, Task, Crew, Process, LLM
from langchain.chat_models import ChatOpenAI
import openai
from agent_manager import AgentManager
import os
import re  # Add re import here
from dotenv import load_dotenv
from langchain.schema import HumanMessage
import json  # Import json module
import logging  # Import logging module
load_dotenv()
import warnings
warnings.filterwarnings("ignore")

# Configure logger
logging.basicConfig(level=logging.ERROR)  # Set to ERROR to minimize logs
logger = logging.getLogger(__name__)  # Initialize logger

llm = LLM(model="gpt-4", temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))
agent_manager = AgentManager()

# Update AVAILABLE_TOOLS with descriptions
AVAILABLE_TOOLS = {
    "data_visualization": "Creates visual representations of data through graphs, charts, and plots",
    "file_converter": "Converts files between different formats",
    "image_generator": "Generates and processes images",
    "web_analyzer": "Analyzes web content and performs web-related tasks",
    "sentiment_analyzer": "Analyzes the emotional tone and sentiment in text",
    "speech_processor": "Processes and analyzes speech and audio content"
}

# Create orchestrator agent
orchestrator = Agent(
    role="Agent Manager",
    goal="Analyze user queries to create or find appropriate agents with required capabilities",
    backstory="""I am an expert at understanding user requirements and creating specialized agents.
    I can determine the agent's name, description, and required tools based on the user's query.""",
    llm=llm,
    tools=[],  # No direct tools, just agent management
    verbose=True
)

# Initialize ChatOpenAI instance
chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.5)

def get_agent_identity(query):
    prompt = f"""Based on the query: '{query}'
    Create a JSON object with ONLY the name and description of the agent.
    Use these standard names for common tasks:
    - "Web Analysis Agent" for any web searching, scraping, or analysis tasks
    - "Image Generation Agent" for any image creation tasks
    - "Data Visualization Agent" for data visualization
    - "Sentiment Analysis Agent" for sentiment analysis
    - "File Conversion Agent" for file conversion
    
    {{
        "name": "Web Analysis Agent",
        "description": "Specializes in web-related tasks including search and analysis"
    }}
    """
    
    try:
        message = [HumanMessage(content=prompt)]
        response = chat_model.predict_messages(message)
        response_text = response.content.strip()
        return json.loads(response_text)
    except Exception as e:
        print(f"Error getting agent identity: {e}")
        return None

def create_agent_config(identity, tools):
    return {
        "name": identity["name"],
        "description": identity["description"],
        "tools": tools
    }

def determine_required_tools(query):
    prompt = f"""Based on this user query: '{query}'
    Identify the most appropriate tool from the available tools, even if the query uses different phrasing.
    For example:
    - "Create a graph" → data_visualization
    - "Show me a plot" → data_visualization
    - "Visualize this data" → data_visualization
    - "Generate a chart" → data_visualization
    
    Available tools and their purposes:
    {json.dumps(AVAILABLE_TOOLS, indent=2)}
    
    Common synonyms and phrases to consider:
    - data_visualization: visualization, graph, chart, plot, trend, distribution, compare, analyze data
    # ...existing other tool mappings...
    
    Respond with ONLY the tool name in a JSON array format.
    Example: ["data_visualization"]
    """
    
    try:
        message = [HumanMessage(content=prompt)]
        response = chat_model.predict_messages(message)
        # Extract content correctly from the response
        response_text = response.content.strip()
        return json.loads(response_text)
    except Exception as e:
        print(f"Error determining required tools: {e}")
        return None

from tool_manager import ToolManager

tool_manager = ToolManager()

def create_specialized_agent(agent_details):
    # Get the appropriate tool for the agent
    tool = tool_manager.get_tool(agent_details['tools'][0])
    if not tool:
        raise ValueError(f"Tool {agent_details['tools'][0]} not found")
        
    return Agent(
        role=agent_details["name"],
        goal=agent_details["description"],
        backstory=f"I am specialized in {agent_details['description']}",
        tools=[tool],
        llm=llm
    )

def execute_agent_task(agent_config, user_input):
    try:
        tool_name = agent_config['tools'][0]
        logger.debug(f"Executing task with tool: {tool_name}")
        
        tool = tool_manager.get_tool(tool_name)
        if not tool:
            return {"error": f"Tool {tool_name} not found"}

        if tool_name == "speech_processor":
            audio_file = user_input.get('file_bytes')
            result = tool._run(audio_file)
        else:
            parsed_input = parse_input_for_tool(tool_name, user_input)
            logger.debug(f"Parsed input: {parsed_input}")
            result = tool._run(**parsed_input)
        
        logger.debug(f"Tool result: {result}")

        # Always return a properly formatted response
        if isinstance(result, dict):
            if not result.get('success', True):
                return {
                    "success": False,
                    "type": "text",
                    "content": result.get('content', 'Unknown error occurred')
                }
            
            # Ensure text content is properly formatted
            if 'content' in result:
                response = {
                    "type": result.get('type', 'text'),
                    "content": result['content'],
                    "success": True
                }
                # Include additional fields based on type
                if result['type'] == 'file':
                    response['data'] = result.get('data')
                    response['mime_type'] = result.get('mime_type')
                return response
            
        # If result is a string, wrap it properly
        return {
            "type": "text",
            "content": str(result),
            "success": True
        }

    except Exception as e:
        logger.error(f"Error in execute_agent_task: {str(e)}")
        return {
            "type": "text",
            "content": f"Error executing tool: {str(e)}",
            "success": False
        }

def parse_input_for_tool(tool_name, user_input):
    """Parse user input based on tool requirements"""
    query = user_input.get('query') if isinstance(user_input, dict) else user_input
    file_path = user_input.get('file_path') if isinstance(user_input, dict) else None

    if tool_name == "web_analyzer":
        logger.debug(f"Parsing web analyzer input: {query}")
        query_lower = query.lower()
        
        # Detect URLs in the query
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        urls = url_pattern.findall(query)
        
        # Enhanced scraping keywords
        scrape_keywords = [
            "scrape", "extract", "analyze url", "analyze website", 
            "read webpage", "get content from", "fetch from",
            "get information from", "extract data from"
        ]
        is_scrape_command = any(keyword in query_lower for keyword in scrape_keywords)
        
        # If URL found or scrape command, use scrape
        if urls or is_scrape_command:
            url = urls[0] if urls else None
            if not url:
                # Try to extract URL from the query even if it's not properly formatted
                possible_url = re.search(r'(?:scrape|extract|analyze|read|from)\s+(.+?)(?:\s|$)', query_lower)
                if possible_url:
                    url = f"https://{possible_url.group(1)}" if not possible_url.group(1).startswith(('http://', 'https://')) else possible_url.group(1)
            
            if url:
                logger.debug(f"Using scrape for URL: {url}")
                return {
                    "action": "scrape",
                    "params": {
                        "url": url,
                        "prompt": query,
                        "openai_api_key": os.getenv("OPENAI_API_KEY")
                    }
                }
        
        # If no URL or scrape command, use search
        logger.debug(f"Using search for query: {query}")
        return {
            "action": "search",
            "params": {
                "query": query,
                "serper_api_key": os.getenv("SERPER_API_KEY")
            }
        }

    elif tool_name == "sentiment_analyzer":
        logger.debug(f"Parsing sentiment input: {query}")
        return {"text": query.strip()}
    elif tool_name == "data_visualization":
        if file_path:
            return {
                "file_path": file_path,
                "query": query
            }
        elif " with query " in query:
            file_path, viz_query = query.split(" with query ", 1)
            return {
                "file_path": file_path.strip(),
                "query": viz_query.strip()
            }
        else:
            return {
                "file_path": query,
                "query": "Analyze the data and create a suitable visualization"
            }
    elif tool_name == "image_generator":
        # Only use DALL-E if explicitly requested, otherwise use FLUX.1-Schnell
        query_lower = query.lower()
        use_dalle = any(term in query_lower for term in ["dall-e", "dalle", "dall e", "use dall"])
        return {
            "prompt": query,
            "model": "DALL-E" if use_dalle else "FLUX.1-Schnell"
        }
    elif tool_name == "speech_processor":
        return {"text": query}  # Pass the transcribed text for further processing
    elif tool_name == "file_converter":
        file_bytes = None
        input_format = None
        target_format = None

        # Video formats that can be converted between each other
        VIDEO_FORMATS = ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv']

        if file_path:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            input_format = os.path.splitext(file_path)[1][1:].lower()
            logger.debug(f"Detected input format: {input_format}")

        # Try to find target format in query
        query_lower = query.lower()
        target_format = None

        # First try to find explicit format mention
        for format in VIDEO_FORMATS + ['pdf', 'docx', 'csv', 'xlsx', 'txt']:
            if f"to {format}" in query_lower or f"as {format}" in query_lower:
                target_format = format
                break

        # If no target format found and it's a video file
        if not target_format and input_format in VIDEO_FORMATS:
            # Choose the next format in the list
            current_index = VIDEO_FORMATS.index(input_format)
            target_format = VIDEO_FORMATS[(current_index + 1) % len(VIDEO_FORMATS)]
            logger.debug(f"No target format specified, converting to {target_format}")

        # For other file types, use default mappings
        if not target_format:
            format_pairs = {
                'pdf': 'docx',
                'docx': 'pdf',
                'xlsx': 'csv',
                'csv': 'xlsx',
                'png': 'txt',
                'jpg': 'txt',
                'jpeg': 'txt'
            }
            target_format = format_pairs.get(input_format)

        if not target_format:
            raise ValueError(f"Unsupported conversion for format: {input_format}")

        logger.debug(f"Converting from {input_format} to {target_format}")

        return {
            "file_bytes": file_bytes,
            "input_format": input_format,
            "target_format": target_format
        }
        
    # Add other tool parsing as needed
    return {"text": query}  # Default case

# Task
orchestrator_task = Task(
    description="Process user query: {inputs}, find or create appropriate agent, and store new agent details if needed",
    expected_output="Agent configuration and task results",
    agent=orchestrator
)

def find_suitable_agent(query, required_tools):
    """Find an existing agent that can handle the task based on tools and capabilities"""
    if not required_tools:
        return None
        
    needed_tool = required_tools[0]  # Get the primary tool needed
    
    # Get all agents
    agents = agent_manager.get_agents()
    
    for agent in agents:
        # Check if agent has the required tool
        if needed_tool in agent['tools']:
            return agent
            
    return None

def main():
    user_query = "I want an agent for sentiment analysis."
    
    # First, determine required tools
    required_tools = determine_required_tools(user_query)
    if not required_tools:
        print("Could not determine required tools.")
        return
        
    # Find suitable existing agent
    existing_agent = find_suitable_agent(user_query, required_tools)
    
    if existing_agent:
        print(f"Using existing agent: {existing_agent['name']}")
        result = execute_agent_task(existing_agent, user_query)
        print(result)
        return
    
    # Only create new agent if no suitable agent exists
    agent_identity = get_agent_identity(user_query)
    if agent_identity:
        agent_config = create_agent_config(agent_identity, required_tools)
        if agent_config:
            agent_manager.save_agent(agent_config)
            specialized_agent = create_specialized_agent(agent_config)
            print(f"Created new agent: {agent_config['name']}")
            
            crew = Crew(
                agents=[orchestrator, specialized_agent],
                tasks=[orchestrator_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff(inputs={'inputs': user_query})
            print(result)
        else:
            print("Failed to create agent configuration.")
    else:
        print("Could not determine agent identity.")

# Update app.py integration
def process_non_audio_query(file_path: str, query: str) -> dict:
    """
    Process queries that do not involve audio files.
    
    Args:
        file_path (str): The path to the uploaded file.
        query (str): The user's query.
        
    Returns:
        dict: The response from executing the agent task.
    """
    # Example implementation; modify as needed based on application logic
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        
        # Determine the required tool based on the query
        required_tools = determine_required_tools(query)
        if not required_tools:
            return {"success": False, "type": "text", "content": "Could not determine required tools."}
        
        # Find a suitable agent
        existing_agent = find_suitable_agent(query, required_tools)
        if not existing_agent:
            return {"success": False, "type": "text", "content": "No suitable agent found for this task."}
        
        # Execute the task
        response = execute_agent_task(existing_agent, {"file_path": file_path, "file_bytes": file_bytes, "query": query})
        return response
    except Exception as e:
        logger.error(f"Error in process_non_audio_query: {str(e)}")
        return {"success": False, "type": "text", "content": f"Error processing non-audio query: {str(e)}"}

def process_query(file_path: Optional[str], query: str):
    try:
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            return {
                "success": False,
                "type": "text",
                "content": "OpenAI API key not found in environment variables. Please set OPENAI_API_KEY."
            }
        
        # Ensure OpenAI API key is set for the module
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Remove or comment out print statements
        # print(f"Debug - Received file_path: {file_path}")  # Remove this line
        # print(f"Debug - Received query: {query}")  # Remove this line
        
        # Check if query is about creating a new agent
        if any(phrase in query.lower() for phrase in ["create an agent", "make an agent", "create agent", "make agent"]):
            required_tools = determine_required_tools(query)
            if not required_tools:
                return "Could not determine required tools."
                
            existing_agent = find_suitable_agent(query, required_tools)
            if existing_agent:
                return f"An agent with the required capabilities already exists: {existing_agent['name']}. You can use it directly."
                
            # Create new agent
            agent_identity = get_agent_identity(query)
            if agent_identity:
                agent_config = create_agent_config(agent_identity, required_tools)
                if agent_config:
                    agent_manager.save_agent(agent_config)
                    return f"Successfully created a new agent: {agent_config['name']} with tools: {', '.join(agent_config['tools'])}. You can now use this agent for tasks."
                return "Failed to create agent configuration."
            return "Could not determine agent identity."

        # For non-creation queries, proceed with task execution
        required_tools = determine_required_tools(query)
        if not required_tools:
            return "Could not determine required tools."
            
        existing_agent = find_suitable_agent(query, required_tools)
        if not existing_agent:
            return "No suitable agent found for this task. Try creating one first."
        
        # Add this line to inform about which agent is being used
        agent_info = f"🤖 Using agent: {existing_agent['name']} for this task."
            
        # Execute task with existing agent
        if file_path:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Don't log binary data
            logger.debug(f"Processing file: {os.path.basename(file_path)}")
            
            file_extension = os.path.splitext(file_path)[1][1:].lower()
            audio_formats = ["mp3", "wav", "m4a", "flac", "aac"]
            if file_extension in audio_formats:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                response = execute_agent_task(existing_agent, {
                    "file_path": file_path,
                    "file_bytes": file_bytes,
                    "query": query
                })
            else:
                response = process_non_audio_query(file_path, query)  # Updated to use the new function
        else:
            response = execute_agent_task(existing_agent, {
                "file_path": None,
                "query": query
            })

        # Regular response handling using response instead of result
        if isinstance(response, dict):
            response['agent_info'] = agent_info
        else:
            response = {
                'type': 'text',
                'content': str(response),
                'agent_info': agent_info
            }
        return response

    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        return {
            "success": False,
            "type": "text",
            "content": f"Error processing query: {str(e)}"
        }

if __name__ == "__main__":
    main()