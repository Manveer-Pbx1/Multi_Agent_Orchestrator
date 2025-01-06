import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

url = "https://instantapi.ai/api/retrieve/"
headers = {
    "Content-Type": "application/json"
}
data = {
    "webpage_url": "https://www.myntra.com/casual-shoes/nike/nike-men-dunk-low-shoes/30936230/buy",
    "api_method_name": "getItemDetails",
    "api_response_structure": json.dumps({
        "item_name": "<the item name>",
        "item_price": "<the item price>",
        "item_image": "<the absolute URL of the first item image>",
        "item_url": "<the absolute URL of the item>",
        "item_type": "<the item type>",
        "item_weight": "<the item weight>",
        "item_main_feature": "<the main feature of this item that would most appeal to its target audience>",
        "item_review_summary": "<a summary of the customer reviews received for this item>",
        "item_available_colors": "<the available colors of the item, converted to closest primary colors>",
        "item_materials": "<the materials used in the item>",
        "item_shape": "<the shape of the item>"
    }),
    "api_key": os.getenv("INSTANTAPI_KEY")
}

response = requests.post(url, headers=headers, json=data)
print(response.json())