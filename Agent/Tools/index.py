from scrapegraphai.graphs import SmartScraperGraph
import json
from openai import OpenAI
import requests
import os
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def scrape_website(url, prompt, api_key):
    logger.debug(f"Starting scrape for URL: {url}")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            
            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
                time.sleep(2)  # Give JS time to load
                content = page.content()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract text content
                text_content = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])])
                
                # Process with GPT
                try:
                    gpt_response = process_with_gpt(
                        json.dumps({"content": text_content}),
                        prompt,
                        api_key
                    )
                    return f"Analysis Result:\n{gpt_response}"
                except Exception as e:
                    logger.error(f"GPT processing error: {str(e)}")
                    return f"Successfully scraped but error in analysis: {str(e)}"
            finally:
                context.close()
                browser.close()
                
    except Exception as e:
        logger.error(f"Scraping error: {str(e)}")
        return f"Error scraping website: {str(e)}\nPlease make sure the URL is accessible and try again."

def search_with_serper(query, serper_api_key=None):
    api_key = serper_api_key or os.getenv('SERPER_API_KEY')
    if not api_key:
        raise ValueError("Serper API key not provided and SERPER_API_KEY environment variable not set")
        
    headers = {
        'X-API-KEY': api_key,
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

def web_analysis_tool(action, params):
    try:
        logger.debug(f"Web analysis tool called with action: {action}, params: {params}")
        
        if action == 'scrape':
            if not params.get('url'):
                return "Error: No URL provided for scraping"
            
            url = params['url']
            if not url.startswith(('http://', 'https://')):
                return "Error: Invalid URL. Must start with http:// or https://"
            
            try:
                return scrape_website(
                    url,
                    params.get('prompt', 'Extract and summarize the main content'),
                    params.get('openai_api_key')
                )
            except Exception as e:
                logger.error(f"Scraping error: {str(e)}")
                return f"Error scraping website: {str(e)}\nPlease check if the URL is valid and accessible."
        
        elif action == 'search':
            if not params.get('query'):
                return "Error: No search query provided"
                
            try:
                result = search_with_serper(
                    params['query'],
                    params.get('serper_api_key')
                )
                logger.debug(f"Search completed for query: {params['query']}")
                return result
            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                return f"Error performing search: {str(e)}"
        else:
            return "Invalid action. Please use 'scrape' or 'search'."
    except Exception as e:
        logger.error(f"General error in web_analysis_tool: {str(e)}")
        return f"Error: {str(e)}"

# Example usage:
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Web Analysis Tool')
    parser.add_argument('action', choices=['scrape', 'search'], help='Action to perform')
    parser.add_argument('--url', help='URL to scrape (required for scrape action)')
    parser.add_argument('--prompt', '--query', help='Prompt for scraping or query for searching')
    args = parser.parse_args()

    if args.action == 'scrape':
        if not args.url or not args.prompt:
            print("Usage: python index.py scrape --url <url> --prompt <prompt>")
            sys.exit(1)
            
        scrape_params = {
            'url': args.url,
            'prompt': args.prompt,
            'openai_api_key': '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'
        }
        print(web_analysis_tool('scrape', scrape_params))
        
    elif args.action == 'search':
        if not args.prompt:
            print("Usage: python index.py search --query <query>")
            sys.exit(1)
            
        search_params = {
            'query': args.prompt,
            'serper_api_key': '2fdb52d81ddbb785cc764c014766df9a871c9652'
        }
        print(web_analysis_tool('search', search_params))