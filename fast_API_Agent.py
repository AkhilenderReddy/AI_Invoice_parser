import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import base64

# Load API key from .env
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

app = FastAPI()

def encode_image_to_base64(image_bytes):
    """Encodes image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")

def extract_invoice_data_from_image_bytes(image_bytes):
    """Extracts invoice data from image bytes using OpenAI Vision API."""
    try:
        base64_image = encode_image_to_base64(image_bytes)
        response = client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=[
                {
                    "role": "user",
                    "content": [
        {
                            "type": "text",
                            "text": "Please extract the following fields from this invoice image and respond only in the given JSON format. If any field is missing, return null for that field.\n\n"
                                    "{\n"
                                    "  \"order_number\": null,\n"
                                    "  \"invoice_number\": null,\n"
                                    "  \"order_date\": null,\n"
                                    "  \"invoice_id\": null,\n"
                                    "  \"invoice_date\": null,\n"
                                    "  \"transaction_id\": null,\n"
                                    "  \"date_time\": null,\n"
                                    "  \"invoice_value\": null,\n"
                                    "  \"mode_of_payment\": null,\n"
                                    "  \"place_of_supply\": null,\n"
                                    "  \"place_of_delivery\": null,\n"
                                    "  \"seller\": {\n"
                                    "    \"name\": null,\n"
                                    "    \"gst\": null,\n"
                                    "    \"pan\": null,\n"
                                    "    \"address\": null,\n"
                                    "    \"state\": null,\n"
                                    "    \"pincode\": null,\n"
                                    "    \"country\": null,\n"
                                    "    \"bank_details\": {\n"
                                    "      \"account_name\": null,\n"
                                    "      \"account_number\": null,\n"
                                    "      \"bank_name\": null,\n"
                                    "      \"branch\": null,\n"
                                    "      \"ifsc\": null\n"
                                    "    },\n"
                                    "    \"contact_details\": {\n"
                                    "      \"phone\": null,\n"
                                    "      \"email\": null\n"
                                    "    }\n"
                                    "  },\n"
                                    "  \"buyer\": {\n"
                                    "    \"name\": null,\n"
                                    "    \"gst\": null,\n"
                                    "    \"pan\": null,\n"
                                    "    \"address\": null,\n"
                                    "    \"state\": null,\n"
                                    "    \"pincode\": null,\n"
                                    "    \"country\": null,\n"
                                    "    \"contact_details\": {\n"
                                    "      \"email\": null,\n"
                                    "      \"phone\": null\n"
                                    "    },\n"
                                    "    \"billing_address\": null,\n"
                                    "    \"shipping_address\": null\n"
                                    "  },\n"
                                    "  \"invoice_items\": [\n"
                                    "    {\n"
                                    "      \"sl.no\": null,\n"
                                    "      \"hsn\": null,\n"
                                    "      \"description\": null,\n"
                                    "      \"unit_price\": null,\n"
                                    "      \"qty\": null,\n"
                                    "      \"net_amount\": null,\n"
                                    "      \"tax\": [\n"
                                    "        {\n"
                                    "          \"tax_type\": null,\n"
                                    "          \"tax_rate\": null,\n"
                                    "          \"tax_amount\": null\n"
                                    "        }\n"
                                    "      ],\n"
                                    "      \"total_amount\": null\n"
                                    "    }\n"
                                    "  ]\n"
                                    "}"
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=2000, # Increased max_tokens to handle potentially larger JSON responses
        )

        # Log the raw response to debug the structure
        print("Raw Response:", response.choices[0].message.content)

        # Clean the response if wrapped in code block format
        cleaned_response = re.sub(r"```(?:json)?\s*(.*?)```", r"\1", response.choices[0].message.content, flags=re.DOTALL).strip()

        # Try to parse the JSON data
        try:
            parsed_data = json.loads(cleaned_response)
            return parsed_data
        except json.JSONDecodeError as e:
            # If JSON parsing fails, print the error and the response content
            print(f"Error decoding JSON: {e}")
            print(f"Response Content: {cleaned_response}")
            return None

    except Exception as e:
        # Catch other exceptions (like network issues or API errors)
        print(f"An error occurred: {e}")
        return None

@app.post("/extract_invoice_data/")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image and extract invoice data.

    Args:
        file (UploadFile): The image file to upload.

    Returns:
        JSONResponse: A JSON response containing the extracted invoice data,
                        or an error message if extraction fails.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()
        result = extract_invoice_data_from_image_bytes(image_bytes)
        if result:
            return result
        else:
            raise HTTPException(status_code=500, detail="Failed to extract invoice data from the image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during image processing: {e}")
