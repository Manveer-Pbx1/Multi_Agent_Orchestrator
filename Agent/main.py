from typing import Optional
from crewai import Agent, Task, Crew, Process, LLM
from langchain.chat_models import ChatOpenAI
from agent_manager import AgentManager
import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage
import json  # Import json module
import logging  # Import logging module
load_dotenv()
import warnings
warnings.filterwarnings("ignore")

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
    - "I want to create a speech agent" → speech_processor
    - "I need an agent for images" → image_generator
    - "Create an agent for web analysis" → web_analyzer
    
    Available tools and their purposes:
    {json.dumps(AVAILABLE_TOOLS, indent=2)}
    
    Common synonyms and phrases to consider:
    - speech_processor: speech, voice, audio, talk, speaking
    - image_generator: image, picture, photo, drawing, illustration
    - web_analyzer: web, website, internet, search, browse
    - sentiment_analyzer: sentiment, emotion, feeling, mood
    - data_visualization: visualization, graph, chart, plot, data
    - file_converter: convert, transformation, change format
    
    Respond with ONLY the tool name in a JSON array format.
    Example: ["speech_processor"]
    """
    
    try:
        message = [HumanMessage(content=prompt)]
        response = chat_model.predict_messages(message)
        # Extract content correctly from the response
        response_text = response.content.strip()
        print(f"Debug - Tools response: {response_text}")  # Debug line
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
            
        # Add notification-only behavior for all tools
        notification_tools = {
            'file_converter': "File conversion tool has been invoked. Please check the downloads folder for the converted file.",
            'data_visualization': "Data visualization tool has been invoked. The visualization will be displayed when ready.",
            'image_generator': "Image generation tool has been invoked. The image will be generated shortly.",
            'sentiment_analyzer': "Sentiment analysis tool has been invoked. Analyzing the text sentiment.",
            'web_analyzer': "Web analysis tool has been invoked. Processing your web-related request."
        }
        
        # Handle speech processor separately since it returns its own notification
        if tool_name == 'speech_processor':
            result = tool._run(**parse_input_for_tool(tool_name, user_input))
            # Launch the speech app in the background if needed
            try:
                import threading
                from Tools.speech import launch_speech_app
                threading.Thread(target=launch_speech_app, daemon=True).start()
            except Exception as e:
                logger.error(f"Background launch error: {str(e)}")
            return result

        # For all other tools
        if tool_name in notification_tools:
            # Execute the tool in the background
            try:
                tool._run(**parse_input_for_tool(tool_name, user_input))
            except Exception as e:
                logger.error(f"Background execution error: {str(e)}")
            
            # Return notification message
            return {"notification": notification_tools[tool_name]}
            
        # This section should now rarely be used as all tools are handled above
        # Keeping it for future tools or special cases
        parsed_input = parse_input_for_tool(tool_name, user_input)
        if isinstance(user_input, dict) and user_input.get('file_path'):
            parsed_input['file_path'] = user_input['file_path']
            
        try:
            result = tool._run(**parsed_input)
            return result
        except Exception as e:
            logger.error(f"Error executing {tool_name}: {str(e)}")
            return {"error": f"Error executing {tool_name}: {str(e)}"}

    except Exception as e:
        logger.error(f"Error in execute_agent_task: {str(e)}")
        return {"error": f"Error executing tool: {str(e)}"}

def parse_input_for_tool(tool_name, user_input):
    """Parse user input based on tool requirements"""
    if tool_name == "web_analyzer":
        logger.debug(f"Parsing web analyzer input: {user_input}")
        
        import re
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        urls = url_pattern.findall(user_input)
        
        if urls:
            # Always use scrape for URLs
            return {
                "action": "scrape",
                "params": {
                    "url": urls[0],
                    "prompt": user_input,
                    "openai_api_key": os.getenv("OPENAI_API_KEY")
                }
            }
        else:
            # No URL found, use search
            return {
                "action": "search",
                "params": {
                    "query": user_input
                }
            }
    elif tool_name == "sentiment_analyzer":
        return {"text": user_input}
    elif tool_name == "data_visualization":
        # Split the input into file path and query
        if " with query " in user_input:
            file_path, query = user_input.split(" with query ", 1)
        else:
            file_path = user_input
            query = "Analyze the data and create a suitable visualization"
            
        return {
            "file_path": file_path.strip(),
            "query": query.strip()
        }
    elif tool_name == "image_generator":
        # Check if user specifically requests DALL-E
        use_dalle = "dall-e" in user_input.lower() or "dalle" in user_input.lower()
        return {
            "prompt": user_input,
            "model": "DALL-E" if use_dalle else "FLUX.1-Schnell"
        }
    # Add other tool parsing as needed
    return {"text": user_input}  # Default case

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
def process_query(file_path: Optional[str], query: str):
    """Process query for app.py integration with separate file_path and query"""
    print(f"Debug - Received file_path: {file_path}")  # Debug line
    print(f"Debug - Received query: {query}")  # Debug line
    
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
        
    # Execute task with existing agent
    if file_path:
        return execute_agent_task(existing_agent, {"file_path": file_path, "query": query})
    else:
        return execute_agent_task(existing_agent, {"file_path": None, "query": query})

if __name__ == "__main__":
    main()