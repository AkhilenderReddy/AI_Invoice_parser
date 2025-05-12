import os
import json
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import base64
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    message = "OPENAI_API_KEY is not set in .env file or environment variables."
    logger.critical(message)
    raise ValueError(message)

# Initialize OpenAI client for OpenRouter
def get_openai_client() -> OpenAI:
    """Initializes and returns an OpenAI client for OpenRouter."""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENAI_API_KEY,
        )
        return client
    except Exception as e:
        message = f"Failed to initialize OpenAI client: {e}"
        logger.error(message)
        raise RuntimeError(message) from e

# Define Output Schema for Buyer
buyer_schema = [
    ResponseSchema(name="name", description="Name of the buyer"),
    ResponseSchema(name="gst", description="Buyer's GST number"),
    ResponseSchema(name="pan", description="Buyer's PAN number"),
    ResponseSchema(name="address", description="Buyer's address"),
    ResponseSchema(name="state", description="Buyer's state"),
    ResponseSchema(name="pincode", description="Buyer's pincode"),
    ResponseSchema(name="country", description="Buyer's country"),
    ResponseSchema(
        name="contact_details",
        description="Buyer's contact details, including email and phone number.",
        schema={
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Buyer's email address"},
                "phone": {"type": "string", "description": "Buyer's phone number"},
            },
            "required": ["email", "phone"],
        },
    ),
    ResponseSchema(name="billing_address", description="Buyer's billing address"),
    ResponseSchema(name="shipping_address", description="Buyer's shipping address"),
]

def create_buyer_image_prompt_and_extract(image_base64: str) -> str:
    """
    Creates the prompt for extracting buyer details and returns the expected JSON string.
    """
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                """
Extract buyer details from the provided image. Provide the response in JSON format, strictly adhering to the schema. If a field is not present, return an empty string. For contact_details, if email or phone is not found, return an empty string for that specific field.

{format_instructions}
"""
            ),
            HumanMessagePromptTemplate.from_template(
                f"""
<image>data:image/jpeg;base64,{image_base64}</image>
"""
            ),
        ],
        input_variables=["image_base64"],
        partial_variables={"format_instructions": StructuredOutputParser.from_response_schemas(buyer_schema).get_format_instructions()}
    )

    model = ChatOpenAI(
        model_name="google/gemma-3-27b-it",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_API_KEY"),  # Ensure API key is loaded
        max_tokens=3000,
    )
    parser = StructuredOutputParser.from_response_schemas(buyer_schema)
    messages = prompt.format_messages(image_base64=image_base64)
    logger.info("Extracting buyer details from image.")

    try:
        response = model.invoke(messages)
        logger.info(f"Raw LLM response (buyer): {response.content}")
        buyer_data = parser.parse(response.content)
        logger.info(f"Parsed buyer details: {buyer_data}")
        buyer_json = json.dumps({"buyer": buyer_data})
        return buyer_json
    except Exception as e:
        logger.error(f"Error during buyer details extraction: {e}")
        return json.dumps({"buyer": {
            "error": str(e),
            "message": "Failed to extract buyer details from the image."
        }})

app = FastAPI()

@app.post("/extract_buyer_data/")
async def upload_image_buyer(file: UploadFile = File(...), openai_client: OpenAI = Depends(get_openai_client)) -> str:
    """
    Endpoint to upload an image and extract buyer data as a JSON string.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        buyer_json_string = create_buyer_image_prompt_and_extract(base64_image)
        return buyer_json_string
    except Exception as e:
        error_msg = f"An error occurred during buyer data extraction: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)