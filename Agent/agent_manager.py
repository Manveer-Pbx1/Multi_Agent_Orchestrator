import json
from json import tool
import os

class AgentManager:
    def __init__(self, agents_file="agents.txt"):
        self.agents_file = agents_file
        self._ensure_agents_file()

    def _ensure_agents_file(self):
        if not os.path.exists(self.agents_file):
            with open(self.agents_file, 'w') as f:
                json.dump([], f)

    def save_agent(self, agent_details):
        # Validate agent configuration
        required_fields = ['name', 'description', 'tools']
        if not all(field in agent_details for field in required_fields):
            raise ValueError("Invalid agent configuration. Missing required fields.")
            
        try:
            agents = self.get_agents()  # This now safely handles empty files
            
            # Check for duplicate agents
            if not any(agent['name'] == agent_details['name'] for agent in agents):
                agents.append(agent_details)
                with open(self.agents_file, 'w') as f:
                    json.dump(agents, f, indent=2)
                return True
            return False
        except Exception as e:
            print(f"Error saving agent: {str(e)}")
            return False

    def get_agents(self):
        try:
            with open(self.agents_file, 'r') as f:
                content = f.read().strip()
                if not content:  # If file is empty
                    return []
                return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
        except Exception as e:
            print(f"Error reading agents file: {str(e)}")
            return []

    def find_agent_by_name(self, name):
        """Find agent by generic name and capability"""
        agents = self.get_agents()
        name_lower = name.lower()
        
        # First try exact match
        for agent in agents:
            if agent['name'].lower() == name_lower:
                return agent
                
        # Then try partial match on core capability
        core_types = {
            "image": ["image", "picture", "photo", "drawing"],
            "sentiment": ["sentiment", "emotion", "feeling"],
            "data": ["data", "visualization", "chart", "graph"],
            "web": ["web", "internet", "website"],
            "file": ["file", "convert", "conversion"],
            "speech": ["speech", "audio", "voice"]
        }
        
        # Find the relevant core type from the name
        matched_type = None
        for core_type, keywords in core_types.items():
            if any(keyword in name_lower for keyword in keywords):
                matched_type = core_type
                break
        
        if matched_type:
            # Find an agent matching the core type
            for agent in agents:
                agent_name_lower = agent['name'].lower()
                if any(keyword in agent_name_lower for keyword in core_types[matched_type]):
                    return agent
                    
        return None

    def find_agent(self, query, required_tools=None):
        """Find agent by tools and fuzzy matching"""
        agents = self.get_agents()
        query = query.lower()
        
        # First try exact name match from the query
        for agent in agents:
            if agent['name'].lower() in query:
                if not required_tools or all(tool in agent['tools'] for tool in required_tools):
                    return agent
        
        # Then try tool matching
        for agent in agents:
            if required_tools and all(tool in agent['tools'] for required_tool in required_tools):
                return agent
        
        return None
