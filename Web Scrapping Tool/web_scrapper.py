import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

st.title("Web Scraping & Web Search Tool")

# Tabs for functionality
tab1, tab2 = st.tabs(["Scrape Data", "Web Search"])

# Tab 1: Scrape Data
with tab1:
    # User inputs
    webpage_url = st.text_input(
        "Enter webpage URL:", 
        "https://www.amazon.in/s?k=gaming+controller+for+pc+wired&crid=Z8HELQ4IDU2D&sprefix=gaming+controller+for+pc+wir%2Caps%2C264&ref=nb_sb_noss_2"
    )
    api_method_name = st.text_input("Enter API method name:", "getItemDetails")

    # Create dynamic fields for response structure
    st.subheader("Define Response Structure")
    response_structure = {}
    num_fields = st.number_input("Number of fields", min_value=1, value=5)

    for i in range(num_fields):
        col1, col2 = st.columns(2)
        with col1:
            key = st.text_input(f"Field {i+1} name", key=f"key_{i}")
        with col2:
            value = st.text_input(f"Field {i+1} description", key=f"value_{i}", placeholder="<description>")
        if key and value:
            response_structure[key] = value

    if st.button("Scrape Data"):
        url = "https://instantapi.ai/api/retrieve/"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "webpage_url": webpage_url,
            "api_method_name": api_method_name,
            "api_response_structure": json.dumps(response_structure),
            "api_key": os.getenv("INSTANTAPI_KEY")
        }

        with st.spinner("Fetching data..."):
            try:
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    st.success("Data fetched successfully!")
                    st.json(response.json())
                else:
                    st.error(f"Error: {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")

# Tab 2: Web Search with DuckDuckGo
with tab2:
    st.subheader("Web Search Tool (DuckDuckGo)")
    search_query = st.text_input("Enter your search query:", "Python web search library")

    if st.button("Search DuckDuckGo"):
        with st.spinner("Searching..."):
            try:
                # Call DuckDuckGo API
                url = f"https://api.duckduckgo.com/?q={search_query}&format=json"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    related_topics = data.get("RelatedTopics", [])
                    
                    if related_topics:
                        st.success("Search results:")
                        for i, topic in enumerate(related_topics):
                            if "Text" in topic and "FirstURL" in topic:
                                st.write(f"{i+1}. [{topic['Text']}]({topic['FirstURL']})")
                            elif "Name" in topic:
                                st.write(f"{i+1}. {topic['Name']}")
                    else:
                        st.info("No results found.")
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
