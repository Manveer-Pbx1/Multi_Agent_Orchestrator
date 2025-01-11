import streamlit as st
import requests
import openai
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def fetch_webpage_content(url):
    """
    Fetch and parse webpage content
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Error fetching webpage content: {e}")
        return None

def extract_relevant_content(html_content, user_query):
    """
    Extract relevant content from HTML based on user query
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Extract text content from the webpage
        text_content = soup.get_text(separator=' ', strip=True)
        return text_content
    except Exception as e:
        st.error(f"Error extracting content from HTML: {e}")
        return None

def analyze_content_with_llm(content, user_query):
    """
    Use GPT to analyze extracted content based on user's query
    """
    try:
        truncated_content = truncate_content(content)
        llm_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts specific information from web content based on user queries. The content might be truncated, so focus on the available information."},
                {"role": "user", "content": f"From this webpage content: {truncated_content}\n\nUser wants to know: {user_query}\n\nPlease extract and summarize the relevant information. If the content appears truncated, mention that in your response."}
            ],
            max_tokens=None,
        )
        return llm_response["choices"][0]["message"]["content"]
    except openai.error.RateLimitError:
        return "Error: The webpage content is too large to process. Please try a smaller webpage or a more specific section."
    except Exception as e:
        return f"Error processing request: {e}"

def truncate_content(content, max_chars=100000):
    """
    Truncate content to a reasonable size while keeping meaningful content
    """
    if len(content) > max_chars:
        print(content[:max_chars] + "... (content truncated)")
        return content[:max_chars] + "... (content truncated)"
    return content

def search_with_serper(query):
    """
    Perform a search using Serper API
    """
    try:
        url = "https://google.serper.dev/search"
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        payload = {
            'q': query,
            'num': 5  # Number of results to return
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error performing search: {e}")
        return None

def display_search_results(results):
    """
    Display search results in a formatted way
    """
    if not results or 'organic' not in results:
        st.error("No results found")
        return

    for result in results['organic']:
        st.write("---")
        st.write(f"### [{result['title']}]({result['link']})")
        st.write(result['snippet'])
        st.write(f"URL: {result['link']}")

def main():
    st.title("Web Content Analysis Tools")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Web Scraper", "Search Tool"])
    
    # Web Scraper Tab
    with tab1:
        st.header("Smart Web Content Extractor")
        url = st.text_input("Enter the URL of the webpage:")
        user_query = st.text_area("What information would you like to extract from this webpage?")

        if st.button("Extract Information"):
            if not url or not user_query:
                st.error("Please provide both the URL and your query.")
                return

            with st.spinner("Fetching and analyzing webpage content..."):
                html_content = fetch_webpage_content(url)
                if html_content:
                    text_content = extract_relevant_content(html_content, user_query)
                    if text_content:
                        result = analyze_content_with_llm(text_content, user_query)
                        st.write("### Results")
                        st.write(result)
    
    # Search Tool Tab
    with tab2:
        st.header("Web Search Tool")
        search_query = st.text_input("Enter your search query:")
        
        if st.button("Search"):
            if not search_query:
                st.error("Please enter a search query.")
                return
            
            with st.spinner("Searching..."):
                search_results = search_with_serper(search_query)
                if search_results:
                    display_search_results(search_results)

if __name__ == "__main__":
    main()
