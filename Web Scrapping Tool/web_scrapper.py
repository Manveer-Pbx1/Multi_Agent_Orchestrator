import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

st.title("Web Scraping Tool")

# User inputs
webpage_url = st.text_input("Enter webpage URL:", "https://www.myntra.com/casual-shoes/nike/nike-men-dunk-low-shoes/30936230/buy")
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