from venv import logger
import PIL
import streamlit as st
from openai import OpenAI
from PIL import Image, UnidentifiedImageError
import io
import os
import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients with environment variables
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
hf_api_key = os.getenv('HF_API_KEY')

# Verify API keys are present
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not os.getenv('HF_API_KEY'):
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")

# Streamlit app

def generate_image(prompt, size="1024x1024"):
    """Generate an image using DALL-E"""
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size=size,
            quality="standard",
            response_format="url"
        )
        
        if hasattr(response, 'data') and len(response.data) > 0:
            return {
                "success": True,
                "type": "image",
                "url": response.data[0].url,
                "content": "Image generated successfully"
            }
        else:
            return {
                "success": False,
                "type": "text",
                "content": "Failed to generate image: No image data received"
            }
            
    except Exception as e:
        logger.error(f"DALL-E generation error: {str(e)}")
        return {
            "success": False,
            "type": "text",
            "content": f"Failed to generate image: {str(e)}"
        }

def generate_image_flux(prompt, size="1024x1024"):
    """Generate an image using FLUX.1-Schnell model from HuggingFace"""
    try:
        if not hf_api_key:
            return "HuggingFace API key is not configured"
            
        client = InferenceClient(token=hf_api_key)
        width, height = map(int, size.split('x'))
        
        image_bytes = client.post(
            model="stabilityai/stable-diffusion-xl-base-1.0",
            data={
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": "blurry, bad quality, distorted",
                    "width": width,
                    "height": height,
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                }
            }
        )
        
        if isinstance(image_bytes, bytes):
            # Save image bytes to a temporary file and get its URL
            import tempfile
            import os
            
            temp_dir = os.path.join(os.getcwd(), "temp_images")
            os.makedirs(temp_dir, exist_ok=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=temp_dir) as temp_file:
                temp_file.write(image_bytes)
                temp_file_path = temp_file.name
                
            # Convert file path to relative URL
            relative_path = os.path.relpath(temp_file_path, os.getcwd())
            url = f"file://{os.path.abspath(temp_file_path)}"
            
            return {
                "success": True,
                "type": "image",
                "url": url,
                "file_path": temp_file_path,  # Include file path for cleanup later
                "content": "Image generated successfully with FLUX.1-Schnell"
            }
        else:
            raise ValueError("Invalid response from image generation API")
            
    except Exception as e:
        logger.error(f"Error in generate_image_flux: {str(e)}")
        return {
            "success": False,
            "type": "text",
            "content": f"Failed to generate image: {str(e)}"
        }

def edit_image(image_file, prompt, size="1024x1024"):
    try:
        if not image_file:
            return "No image file provided"
            
        image = Image.open(image_file)
        image = image.convert("RGBA")
        
        if image.size[0] * image.size[1] > 4194304:
            return "Image is too large. Please use a smaller image."
            
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        
        response = client.images.edit(
            image=buffered,
            prompt=prompt,
            n=1,
            size=size
        )
        edited_image_url = response.data[0].url
        return edited_image_url
    except UnidentifiedImageError:
        return "Uploaded file is not a valid image."
    except PIL.Image.DecompressionBombError:
        return "Image is too large to process."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# Remove the 'self' parameter since this is not a class method
def _run(prompt, model="FLUX.1-Schnell"):
    """Main entry point for the tool"""
    try:
        use_dalle = any(term in prompt.lower() for term in ["dall-e", "dalle", "dall e", "use dall"])
        
        if use_dalle:
            result = generate_image(prompt)
        else:
            result = generate_image_flux(prompt)
        
        # Print the URL for debugging
        if result.get("success") and "url" in result:
            print(f"Generated image URL: {result['url']}")
            
        return result
    except Exception as e:
        return {
            "success": False,
            "type": "text",
            "content": f"Failed to generate image: {str(e)}"
        }

def launch_image_gen_app():
    st.title("AI Image Generator and Editor")

    tab1, tab2 = st.tabs(["Generate Image", "Edit Image"])

    with tab1:
        st.header("Generate Image")
        st.write("Enter a description below, and AI will generate an image for you! (Add 'use DALL-E' in your prompt to use DALL-E)")

        try:
            max_prompt_length = 1000
            user_prompt = st.text_area("Enter your image description:", "")
            if len(user_prompt) > max_prompt_length:
                st.warning(f"Prompt is too long. Maximum length is {max_prompt_length} characters.")
                user_prompt = user_prompt[:max_prompt_length]
                
            image_size = st.selectbox("Select image size:", ["256x256", "512x512", "1024x1024"])

            if st.button("Generate Image", key="generate"):
                if not user_prompt.strip():
                    st.warning("Please enter a valid description.")
                else:
                    with st.spinner("Generating image..."):
                        use_dalle = "use dall-e" in user_prompt.lower()
                        result = _run(user_prompt)  # Use _run instead of direct generation
                        
                        if result["success"]:
                            if "url" in result:  # DALL-E result
                                st.image(result["url"], caption="Generated by DALL-E", use_container_width=True)
                            elif "url" in result:  # FLUX result
                                st.image(result["url"], caption="Generated by FLUX.1-Schnell", use_container_width=True)
                            st.success(result["content"])
                        else:
                            st.error(result["content"])
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    with tab2:
        st.header("Edit Image")
        st.write("Upload an image and describe the edits you'd like DALL-E to make.")

        uploaded_image = st.file_uploader("Upload an image:", type=["jpeg", "jpg", "png"])
        edit_prompt = st.text_area("Describe the edits you want to make:", "")
        edit_size = st.selectbox("Select output image size:", ["256x256", "512x512", "1024x1024"])

        if st.button("Edit Image", key="edit"):
            if uploaded_image is None:
                st.warning("Please upload an image.")
            elif edit_prompt.strip() == "":
                st.warning("Please describe the edits you want to make.")
            else:
                with st.spinner("Editing image..."):
                    edited_image_url = edit_image(uploaded_image, edit_prompt, edit_size)
                if edited_image_url.startswith("http"):
                    st.image(edited_image_url, caption="Edited by DALL-E", use_container_width=True)
                else:
                    st.error(f"Error: {edited_image_url}")

    st.markdown("---")
    st.write("Powered by OpenAI's DALL-E and Black Forest Labs' FLUX.1-Schnell.")

if __name__ == "__main__":
    launch_image_gen_app()
