from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import os
from dotenv import load_dotenv
load_dotenv()

llm = LLM(model="gpt-4", temperature=0.5, api_key=os.environ.get("OPENAI_API_KEY"))

researcher = Agent(
    role="{topic} Senior Researcher",
    goal="Uncover groundbreaking technologies in {topic} for year 2025.",
    backstory="Driven by curiosity, you explore and share the latest innovations.",
    tools=[SerperDevTool()],
    llm=llm
)

research_task= Task(
    description="Identify the next big trend in {topic} with pros and cons.",
    expected_output="A 3-paragraph report on emerging {topic} technologies.",
    agent=researcher
)

def main():
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff(inputs={'topic': 'AI Agents'})
    print(result)

if __name__ == "__main__":
    main()