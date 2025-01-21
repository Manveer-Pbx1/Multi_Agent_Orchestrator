from crewai import Agent, Task, Crew, Process, LLM
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import groq

load_dotenv()

# Initialize API clients
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
groq_client = groq.Client(api_key=os.environ.get("GROQ_API_KEY"))

def get_llm_config(provider, model_name):
    config = {
        "temperature": 0.5,
    }
    
    if provider == "Anthropic":
        config["api_key"] = os.environ.get("ANTHROPIC_API_KEY")
    elif provider == "Groq":
        config["api_key"] = os.environ.get("GROQ_API_KEY")
    else:  # OpenAI
        config["api_key"] = os.environ.get("OPENAI_API_KEY")
    
    # Use the full model name with provider prefix
    config["model"] = model_name
    
    return config

def extract_agent_data(prompt):
    """Use LLM to extract structured data for agent and task creation."""
    system_message = """You are a helper tool that creates AI agent specifications. 
    Format all responses in second-person perspective as if directly addressing the agent with 'you'."""
    
    user_message = f"""Given the following prompt, extract the following details in JSON format:
    - role: The agent's role (in second person, e.g., "You are a professional chef")
    - goal: The agent's primary goal (in second person, e.g., "Your goal is to...")
    - backstory: Additional context or backstory (in second person, e.g., "You have spent years...")
    - task_description: A detailed description of what you need to do
    - task_expected_output: The specific format or type of output you need to provide
    
    Prompt: {prompt}"""
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        max_tokens=300
    )
    
    extracted_data = response.choices[0].message.content
    return json.loads(extracted_data)

def process_prompt(prompt, provider, model_name):
    # Configure LLM based on selected provider and model
    llm_config = get_llm_config(provider, model_name)
    llm = LLM(**llm_config)
    
    # Extract agent data using LLM
    extracted_data = extract_agent_data(prompt)
    
    # Create Agent and Task objects with configured LLM
    agent = Agent(
        role=extracted_data.get("role", ""),
        goal=extracted_data.get("goal", ""),
        backstory=extracted_data.get("backstory", ""),
        llm=llm
    )
    
    task = Task(
        description=extracted_data.get("task_description", ""),
        expected_output=extracted_data.get("task_expected_output", ""),
        agent=agent
    )
    
    # Create Crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    return {
        "agent": agent,
        "task": task,
        "crew": crew
    }

def handle_conversation(crew, user_message, provider, model_name, is_rag=False, context=None):
    """Handle conversation with the created agent, optionally using RAG context."""
    llm_config = get_llm_config(provider, model_name)
    crew.agents[0].llm = LLM(**llm_config)
    
    if is_rag and context:
        # Format context with relevance scores and content
        context_blocks = []
        for chunk in context:
            context_blocks.append(
                f"[Content] (Relevance: {chunk['score']:.3f}):\n{chunk['content']}\n"
                f"[Source]: {chunk['metadata']['source']}"
            )
        
        context_text = "\n\n---\n\n".join(context_blocks)
        
        task_description = f"""SYSTEM: You are a document-focused AI assistant. 
        The user needs you to provide a fully detailed plan including technology stacks, 
        a step-by-step process, and clear milestones. Answer ONLY using the supplied context.

        GIVEN CONTEXT:
        {context_text}

        USER QUESTION: {user_message}

        STRICT INSTRUCTIONS:
        1. Include specific steps, tech stack, and metrics for success.
        2. If the context lacks info, say: "I cannot find this information in the provided documents."
        3. Provide references and citations from the context.
        """
    else:
        task_description = (
            "Respond with a detailed plan including tech stack, steps, and milestones: "
            f"{user_message}"
        )

    task = Task(
        description=task_description,
        expected_output="A response based strictly on provided document context" if is_rag else "A helpful response",
        agent=crew.agents[0]
    )
    
    crew.tasks = [task]
    return crew.kickoff()
