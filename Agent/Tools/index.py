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
import subprocess  # Add this import at the top
import asyncio
import nest_asyncio
from asyncio import WindowsSelectorEventLoopPolicy, set_event_loop_policy

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

# Set up asyncio for Windows
if os.name == 'nt':
    set_event_loop_policy(WindowsSelectorEventLoopPolicy())
nest_asyncio.apply()

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
        # First try SmartScraperGraph
        try:
            scraper_config = {
                "api_key": api_key,
                "max_depth": 1,
                "extract_text": True,
                "extract_links": True,
                "extract_structured": True
            }
            
            scraper = SmartScraperGraph(
                prompt="Extract and analyze the main content of this webpage",
                source=url,
                config=scraper_config
            )
            logger.debug("Initialized SmartScraperGraph")
            
            scraped_data = scraper.scrape()
            logger.debug(f"Raw scraped data: {scraped_data}")
            
            # Ensure data is valid before accessing
            if not scraped_data or not isinstance(scraped_data, dict):
                logger.warning("SmartScraperGraph returned None or invalid data")
                raise ValueError("Invalid scrape result from SmartScraperGraph")

            formatted_data = {
                'title': scraped_data.get('title', 'No title available'),
                'content': scraped_data.get('content') or scraped_data.get('text', 'No content available'),
                'links': scraped_data.get('links', []),
                'metadata': scraped_data.get('metadata', {}),
                'structured_data': scraped_data.get('structured_data', {})
            }
            
            # Ensure we have some content to process
            if formatted_data['content'] != 'No content available':
                try:
                    gpt_response = process_with_gpt(
                        json.dumps(formatted_data),
                        prompt or "Analyze and summarize the main content",
                        api_key
                    )
                    return {
                        "success": True,
                        "type": "text",
                        "content": gpt_response,
                        "url": url,
                        "source": "scrapegraphai"
                    }
                except Exception as e:
                    logger.error(f"GPT processing error with SmartScraperGraph: {str(e)}")
                    raise  # Let the fallback handle it
            else:
                logger.warning("No content found in SmartScraperGraph response")
                raise ValueError("No content found in scraping response")
            
        except Exception as e:
            logger.error(f"SmartScraperGraph failed: {str(e)}")
            
            # Initialize Playwright with error handling and proper event loop
            try:
                import subprocess
                try:
                    # Install Playwright browsers if needed
                    subprocess.run(['playwright', 'install', 'chromium'], check=True)
                except Exception as install_error:
                    logger.error(f"Failed to install Playwright: {install_error}")
                
                # Create and set event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        headless=True,
                        args=['--no-sandbox']  # Add this for better compatibility
                    )
                    context = browser.new_context(
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    )
                    page = context.new_page()
                    
                    try:
                        page.goto(url, wait_until="networkidle", timeout=30000)
                        time.sleep(2)
                        content = page.content()
                        
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Enhanced content extraction
                        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'header']):
                            element.decompose()  # Ensure elements are removed
                        
                        # Prioritize content extraction
                        content_areas = [
                            soup.find('main'),
                            soup.find('article'),
                            soup.find('div', {'class': ['content', 'main', 'article', 'post']}),
                            soup.find('div', {'id': ['content', 'main', 'article', 'post']})
                        ]
                        
                        text_content = ''
                        for area in content_areas:
                            if area:
                                text_content = ' '.join(area.stripped_strings)
                                break  # Stop after finding the first relevant area
                        
                        if not text_content:
                            text_content = ' '.join([
                                p.get_text().strip() 
                                for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                                if len(p.get_text().strip()) > 50  # Filter out short snippets
                            ])
                        
                        # Process with GPT
                        try:
                            gpt_response = process_with_gpt(
                                json.dumps({"content": text_content}),
                                prompt or "Summarize the main content",
                                api_key
                            )
                            return {
                                "success": True,
                                "type": "scrape",
                                "content": gpt_response,
                                "url": url
                            }
                        except Exception as e:
                            logger.error(f"GPT processing error with Playwright: {str(e)}")
                            return {
                                "success": False,
                                "error": f"Successfully scraped but error in analysis: {str(e)}"
                            }
                    finally:
                        context.close()
                        browser.close()
                        
            except NotImplementedError:
                logger.error("Playwright not properly initialized. Trying alternative approach...")
                # Fallback to direct requests if Playwright fails
                try:
                    import requests
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(url, headers=headers, timeout=30)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract text content
                    text_content = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])])
                    
                    # Process with GPT
                    gpt_response = process_with_gpt(
                        json.dumps({"content": text_content}),
                        prompt or "Summarize the main content",
                        api_key
                    )
                    return {
                        "success": True,
                        "type": "text",
                        "content": gpt_response,
                        "url": url
                    }
                except Exception as req_error:
                    logger.error(f"Fallback request failed: {str(req_error)}")
                    raise
            # ...existing error handling code...

    except Exception as e:
        logger.error(f"Scraping error: {str(e)}")
        return {
            "success": False,
            "error": f"Error scraping website: {str(e)}\nPlease make sure the URL is accessible and try again."
        }

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
                return {
                    "success": False,
                    "error": "Error: No URL provided for scraping"
                }
            
            url = params['url']
            if not url.startswith(('http://', 'https://')):
                return {
                    "success": False,
                    "error": "Error: Invalid URL. Must start with http:// or https://"
                }
            
            try:
                scrape_result = scrape_website(
                    url,
                    params.get('prompt', 'Extract and summarize the main content'),
                    params.get('openai_api_key')
                )
                return {
                    "success": True,
                    "type": "text",
                    "content": scrape_result.get('content', ''),
                    "url": url
                }
            except Exception as e:
                logger.error(f"Scraping error: {str(e)}")
                return {
                    "success": False,
                    "error": f"Error scraping website: {str(e)}"
                }
        
        elif action == 'search':
            if not params.get('query'):
                return {
                    "success": False,
                    "error": "Error: No search query provided"
                }
                
            try:
                search_results = search_with_serper(
                    params['query'],
                    params.get('serper_api_key')
                )
                return {
                    "success": True,
                    "type": "text",
                    "content": search_results
                }
            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                return {
                    "success": False,
                    "error": f"Error performing search: {str(e)}"
                }
        else:
            return {
                "success": False,
                "error": "Invalid action. Please use 'scrape' or 'search'."
            }
    except Exception as e:
        logger.error(f"General error in web_analysis_tool: {str(e)}")
        return {
            "success": False,
            "error": f"Error: {str(e)}"
        }

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
            'openai_api_key': os.getenv('OPENAI_API_KEY')
        }
        print(web_analysis_tool('scrape', scrape_params))
        
    elif args.action == 'search':
        if not args.prompt:
            print("Usage: python index.py search --query <query>")
            sys.exit(1)
            
        search_params = {
            'query': args.prompt,
            'serper_api_key': os.getenv('SERPER_API_KEY')
        }
        print(web_analysis_tool('search', search_params))