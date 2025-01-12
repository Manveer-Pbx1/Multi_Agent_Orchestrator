from scrapegraphai.graphs import SmartScraperGraph
import gradio as gr
import json
from openai import OpenAI
import requests

def process_with_gpt(scraped_data, user_prompt, api_key):
    client = OpenAI(api_key=api_key)
    
    system_prompt = """You are an assistant that helps analyze scraped website data. 
    Use the provided JSON data to answer the user's query in a natural, informative way."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the scraped data: {scraped_data}\n\nBased on this data, {user_prompt}"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )
    
    return response.choices[0].message.content

def scrape_website(url, prompt, api_key):
    # Define the configuration for the scraping pipeline
    graph_config = {
        "llm": {
            "api_key": api_key,
            "model": "openai/gpt-4o-mini",
        },
        "verbose": True,
        "headless": False,
    }

    # Create the SmartScraperGraph instance
    smart_scraper_graph = SmartScraperGraph(
        prompt=prompt,
        source=url,
        config=graph_config
    )

    # Run the pipeline
    result = smart_scraper_graph.run()
    
    # Process the results with GPT-4
    try:
        gpt_response = process_with_gpt(
            json.dumps(result),
            prompt,
            api_key
        )
        return f"AI Analysis:\n\n{gpt_response}\n\nRaw Data:\n{json.dumps(result, indent=4)}"
    except Exception as e:
        return f"Error processing results: {str(e)}\n\nRaw Data:\n{json.dumps(result, indent=4)}"

def search_with_serper(query, serper_api_key):
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    
    payload = {
        "q": query,
        "num": 5  # Number of results
    }
    
    response = requests.post(
        'https://google.serper.dev/search',
        headers=headers,
        json=payload
    )
    
    results = response.json()
    
    # Format results in a readable way
    formatted_results = "Search Results:\n\n"
    for idx, item in enumerate(results.get('organic', []), 1):
        formatted_results += f"{idx}. {item.get('title', 'No title')}\n"
        formatted_results += f"Link: {item.get('link', 'No link')}\n"
        formatted_results += f"Snippet: {item.get('snippet', 'No snippet')}\n\n"
    
    return formatted_results

# Create scraping interface
scraping_interface = gr.Interface(
    fn=scrape_website,
    inputs=[
        gr.Textbox(label="Website URL", placeholder="https://example.com"),
        gr.Textbox(label="What would you like to know about this website?", 
                   placeholder="E.g., 'What does this company do and who are their key team members?'",
                   lines=3),
        gr.Textbox(label="OpenAI API Key", type="password")
    ],
    outputs=gr.Textbox(label="Results", lines=15),
    title="Website Analysis",
    description="Enter a URL and your question to get an AI-powered analysis of the website."
)

# Create search interface
search_interface = gr.Interface(
    fn=search_with_serper,
    inputs=[
        gr.Textbox(label="Search Query", placeholder="Enter your search query..."),
        gr.Textbox(label="Serper API Key", type="password")
    ],
    outputs=gr.Textbox(label="Search Results", lines=15),
    title="Web Search",
    description="SERPER API KEY: 2fdb52d81ddbb785cc764c014766df9a871c9652"
)

# Create tabbed interface
demo = gr.TabbedInterface(
    [scraping_interface, search_interface],
    ["Website Scraper", "Web Search"],
    title="Web Analysis Tools",
    
)

if __name__ == "__main__":
    demo.launch()