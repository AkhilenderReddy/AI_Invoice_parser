import os
import json
import re
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union, Any  # Import Any
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from pydantic import BaseModel, Field, validator
from langgraph.graph import StateGraph, END
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from PIL import Image
from io import BytesIO
from openai import OpenAI
import logging  # Import the logging module

# Set up logging (very helpful for debugging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Get a specific logger


# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    message = "OPENAI_API_KEY is not set in .env file or environment variables."
    logger.critical(message)
    raise ValueError(message)  # Raise an exception to stop execution

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
        raise RuntimeError(message) from e  # Re-raise with original exception as context

# 1. Define the State (Data Structure)
class InvoiceData(BaseModel):
    order_number: Optional[str] = Field(default=None, description="Order number")
    invoice_number: Optional[str] = Field(default=None, description="Invoice number")
    order_date: Optional[str] = Field(default=None, description="Date of the order")
    invoice_id: Optional[str] = Field(default=None, description="Invoice ID")
    invoice_date: Optional[str] = Field(default=None, description="Date of the invoice")
    transaction_id: Optional[str] = Field(default=None, description="Transaction ID")
    date_time: Optional[str] = Field(default=None, description="Date and time of the invoice")
    invoice_value: Optional[str] = Field(default=None, description="Total value of the invoice")
    mode_of_payment: Optional[str] = Field(default=None, description="Mode of payment")
    place_of_supply: Optional[str] = Field(default=None, description="Place where goods/services are supplied")
    place_of_delivery: Optional[str] = Field(default=None, description="Place where goods are delivered")
    seller: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Seller information")  # More specific type hint
    buyer: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Buyer information") # More specific
    invoice_items: List[Dict[str, Any]] = Field(default_factory=list, description="List of invoice items") # More specific
    image_base64: Optional[str] = Field(description="Base64 encoded image of the invoice")

    @validator('*')
    def set_none_for_empty_string(cls, value: Optional[str]) -> Optional[str]: # Added type hint
        """Converts empty strings to None (null equivalent in JSON)."""
        if value == "":
            return None
        return value



# 2. Define Output Schemas
order_invoice_schema = [
    ResponseSchema(name="order_number", description="Order number"),
    ResponseSchema(name="invoice_number", description="Invoice number"),
    ResponseSchema(name="order_date", description="Date of the order"),
    ResponseSchema(name="invoice_id", description="Invoice ID"),
    ResponseSchema(name="invoice_date", description="Date of the invoice"),
    ResponseSchema(name="transaction_id", description="Transaction ID"),
    ResponseSchema(name="date_time", description="Date and time of the invoice"),
    ResponseSchema(name="invoice_value", description="Total value of the invoice"),
    ResponseSchema(name="mode_of_payment", description="Mode of payment"),
    ResponseSchema(name="place_of_supply", description="Place where goods/services are supplied"),
    ResponseSchema(name="place_of_delivery", description="Place where goods are delivered"),
]

seller_schema = [
    ResponseSchema(name="name", description="Name of the seller"),
    ResponseSchema(name="gst", description="Seller's GST number"),
    ResponseSchema(name="pan", description="Seller's PAN number"),
    ResponseSchema(name="address", description="Seller's address"),
    ResponseSchema(name="state", description="Seller's state"),
    ResponseSchema(name="pincode", description="Seller's pincode"),
    ResponseSchema(name="country", description="Seller's country"),
    ResponseSchema(name="account_name", description="Seller's bank account name"),
    ResponseSchema(name="account_number", description="Seller's bank account number"),
    ResponseSchema(name="bank_name", description="Seller's bank name"),
    ResponseSchema(name="branch", description="Seller's bank branch"),
    ResponseSchema(name="ifsc", description="Seller's bank IFSC code"),
    ResponseSchema(name="phone", description="Seller's phone number"),
    ResponseSchema(name="email", description="Seller's email address"),
]

buyer_schema = [
    ResponseSchema(name="name", description="Name of the buyer"),
    ResponseSchema(name="gst", description="Buyer's GST number"),
    ResponseSchema(name="pan", description="Buyer's PAN number"),
    ResponseSchema(name="address", description="Buyer's address"),
    ResponseSchema(name="state", description="Buyer's state"),
    ResponseSchema(name="pincode", description="Buyer's pincode"),
    ResponseSchema(name="country", description="Buyer's country"),
    ResponseSchema(name="email", description="Buyer's email address"),
    ResponseSchema(name="phone", description="Buyer's phone number"),
    ResponseSchema(name="billing_address", description="Buyer's billing address"),
    ResponseSchema(name="shipping_address", description="Buyer's shipping address"),
]

item_schema = [
    ResponseSchema(name="sl_no", description="Serial number of the item"),
    ResponseSchema(name="hsn", description="HSN code of the item"),
    ResponseSchema(name="description", description="Description of the item"),
    ResponseSchema(name="unit_price", description="Unit price of the item"),
    ResponseSchema(name="qty", description="Quantity of the item"),
    ResponseSchema(name="net_amount", description="Net amount for the item"),
    ResponseSchema(name="tax", description="List of taxes applied to the item"),
    ResponseSchema(name="total_amount", description="Total amount for the item"),
]

tax_schema = [
    ResponseSchema(name="tax_type", description="Type of tax (e.g., CGST, SGST, IGST)"),
    ResponseSchema(name="tax_rate", description="Tax rate"),
    ResponseSchema(name="tax_amount", description="Amount of tax"),
]



# 3. Define Prompts for Image Input
def create_order_invoice_prompt() -> ChatPromptTemplate:
    """Creates the prompt for extracting order and invoice details from an image."""
    return ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                """
Extract order and invoice details from the provided image. Provide the response in JSON format, strictly adhering to the schema. If a field is not present, return null.

{format_instructions}
"""
            ),
            HumanMessagePromptTemplate.from_template(
                """
<image>data:image/jpeg;base64,{image_base64}</image>
"""
            ),
        ],
        input_variables=["image_base64"],
        partial_variables={"format_instructions": StructuredOutputParser.from_response_schemas(order_invoice_schema).get_format_instructions()}
    )



def create_seller_image_prompt() -> ChatPromptTemplate:
    """Creates the prompt for extracting seller details from an image."""
    return ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                """
Extract seller details from the provided image. Provide the response in JSON format, strictly adhering to the schema. If a field is not present, return null.

{format_instructions}
"""
            ),
            HumanMessagePromptTemplate.from_template(
                """
<image>data:image/jpeg;base64,{image_base64}</image>
"""
            ),
        ],
        input_variables=["image_base64"],
        partial_variables={"format_instructions": StructuredOutputParser.from_response_schemas(seller_schema).get_format_instructions()}
    )



def create_buyer_image_prompt() -> ChatPromptTemplate:
    """Creates the prompt for extracting buyer details from an image."""
    return ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                """
Extract buyer details from the provided image. Provide the response in JSON format, strictly adhering to the schema. If a field is not present, return null.

{format_instructions}
"""
            ),
            HumanMessagePromptTemplate.from_template(
                """
<image>data:image/jpeg;base64,{image_base64}</image>
"""
            ),
        ],
        input_variables=["image_base64"],
        partial_variables={"format_instructions": StructuredOutputParser.from_response_schemas(buyer_schema).get_format_instructions()}
    )



def create_items_image_prompt() -> ChatPromptTemplate:
    """Creates the prompt for extracting invoice items details from an image."""
    return ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                """
Extract invoice items and their details from the provided image. Provide the response in JSON format, strictly adhering to the schema. If a field is not present, return null. Ensure that the 'tax' field is a list of tax objects, even if there is only one tax.

{format_instructions}
"""
            ),
            HumanMessagePromptTemplate.from_template(
                """
<image>data:image/jpeg;base64,{image_base64}</image>
"""
            ),
        ],
        input_variables=["image_base64"],
        partial_variables={"format_instructions": StructuredOutputParser.from_response_schemas(item_schema).get_format_instructions()}
    )


def extract_order_invoice_details_from_image(state: InvoiceData, openai_client: OpenAI = Depends(get_openai_client)) -> Dict[str, Any]:
    """Extracts order and invoice details from an image."""
    model = ChatOpenAI(
        model_name="google/gemma-3-27b-it",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENAI_API_KEY,
        max_tokens=1000,  # Adjust as needed
    )
    prompt = create_order_invoice_prompt()
    parser = StructuredOutputParser.from_response_schemas(order_invoice_schema)
    messages = prompt.format_messages(image_base64=state.image_base64)
    logger.info(f"Extracting order/invoice details from image. Messages: {messages}")
    try:
        response = model.invoke(messages)
        logger.info(f"Raw LLM response (order/invoice): {response.content}")  # ADDED HERE
        order_invoice_data = parser.parse(response.content)
        logger.info(f"Parsed order/invoice details: {order_invoice_data}")
        return {"order_invoice_data": order_invoice_data}
    except Exception as e:
        logger.error(f"Error parsing order/invoice details: {e}")
        return {"order_invoice_data": {}}



def extract_seller_details_from_image(state: InvoiceData, openai_client: OpenAI = Depends(get_openai_client)) -> Dict[str, Any]:
    """Extracts seller details from an image."""
    model = ChatOpenAI(
        model_name="google/gemma-3-27b-it",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENAI_API_KEY,
        max_tokens=3000,
    )
    prompt = create_seller_image_prompt()
    parser = StructuredOutputParser.from_response_schemas(seller_schema)
    messages = prompt.format_messages(image_base64=state.image_base64)
    logger.info(f"Extracting seller details from image. Messages: {messages}")
    try:
        response = model.invoke(messages)
        logger.info(f"Raw LLM response (seller): {response.content}")  # ADDED HERE
        seller_data = parser.parse(response.content)
        logger.info(f"Parsed seller details: {seller_data}")
        return {"seller_data": seller_data}
    except Exception as e:
        logger.error(f"Error parsing seller details: {e}")
        return {"seller_data": {}}
    

def extract_buyer_details_from_image(state: InvoiceData, openai_client: OpenAI = Depends(get_openai_client)) -> Dict[str, Any]:
    """Extracts buyer details from an image."""
    model = ChatOpenAI(
        model_name="google/gemma-3-27b-it",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENAI_API_KEY,
        max_tokens=3000,
    )
    prompt = create_buyer_image_prompt()
    parser = StructuredOutputParser.from_response_schemas(buyer_schema)
    messages = prompt.format_messages(image_base64=state.image_base64)
    logger.info(f"Extracting buyer details from image. Messages: {messages}")
    try:
        response = model.invoke(messages)
        logger.info(f"Raw LLM response (buyer): {response.content}")  # ADDED HERE
        buyer_data = parser.parse(response.content)
        logger.info(f"Parsed buyer details: {buyer_data}")
        return {"buyer_data": buyer_data}
    except Exception as e:
        logger.error(f"Error parsing buyer details: {e}")
        return {"buyer_data": {}}
    


def extract_items_details_from_image(state: InvoiceData, openai_client: OpenAI = Depends(get_openai_client)) -> Dict[str, Any]:
    """Extracts invoice items details from an image."""
    model = ChatOpenAI(
        model_name="google/gemma-3-27b-it",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENAI_API_KEY,
        max_tokens=3000,
    )
    prompt = create_items_image_prompt()
    parser = StructuredOutputParser.from_response_schemas(item_schema)
    messages = prompt.format_messages(image_base64=state.image_base64)
    logger.info(f"Extracting items details from image. Messages: {messages}")
    try:
        response = model.invoke(messages)
        logger.info(f"Raw LLM response (items): {response.content}")  # ADDED HERE
        items_data = parser.parse(response.content)
        logger.info(f"Parsed item details: {items_data}")
        return {"items_data": items_data}
    except Exception as e:
        logger.error(f"Error parsing items details: {e}")
        return {"items_data": []}





# 4. Define Functions for Agent Actions (Image Input)
# def extract_order_invoice_details_from_image(state: InvoiceData, openai_client: OpenAI = Depends(get_openai_client)) -> Dict[str, Any]:
#     """Extracts order and invoice details from an image."""
#     model = ChatOpenAI(
#         model_name="google/gemma-3-27b-it",
#         base_url="https://openrouter.ai/api/v1",
#         api_key=OPENAI_API_KEY,
#         max_tokens=3000,  # Adjust as needed
#     )
#     prompt = create_order_invoice_prompt()
#     parser = StructuredOutputParser.from_response_schemas(order_invoice_schema)
#     messages = prompt.format_messages(image_base64=state.image_base64)
#     logger.info(f"Extracting order/invoice details from image. Messages: {messages}")
#     try:
#         response = model.invoke(messages)
#         logger.info(f"LLM response for order/invoice details: {response.content}")
#         order_invoice_data = parser.parse(response.content)
#         logger.info(f"Parsed order/invoice details: {order_invoice_data}")
#         return {"order_invoice_data": order_invoice_data}
#     except Exception as e:
#         message = f"Error parsing order/invoice details: {e} - Response: {response.content}"
#         logger.error(message)
#         return {"order_invoice_data": {}}  # Return empty dict, don't raise here, handle in graph



# def extract_seller_details_from_image(state: InvoiceData, openai_client: OpenAI = Depends(get_openai_client)) -> Dict[str, Any]:
#     """Extracts seller details from an image."""
#     model = ChatOpenAI(
#         model_name="google/gemma-3-27b-it",
#         base_url="https://openrouter.ai/api/v1",
#         api_key=OPENAI_API_KEY,
#         max_tokens=3000,
#     )
#     prompt = create_seller_image_prompt()
#     parser = StructuredOutputParser.from_response_schemas(seller_schema)
#     messages = prompt.format_messages(image_base64=state.image_base64)
#     logger.info(f"Extracting seller details from image. Messages: {messages}")
#     try:
#         response = model.invoke(messages)
#         logger.info(f"LLM response for seller details: {response.content}")
#         seller_data = parser.parse(response.content)
#         logger.info(f"Parsed seller details: {seller_data}")
#         return {"seller_data": seller_data}
#     except Exception as e:
#         message = f"Error parsing seller details: {e} - Response: {response.content}"
#         logger.error(message)
#         return {"seller_data": {}}



# def extract_buyer_details_from_image(state: InvoiceData, openai_client: OpenAI = Depends(get_openai_client)) -> Dict[str, Any]:
#     """Extracts buyer details from an image."""
#     model = ChatOpenAI(
#         model_name="google/gemma-3-27b-it",
#         base_url="https://openrouter.ai/api/v1",
#         api_key=OPENAI_API_KEY,
#         max_tokens=3000,
#     )
#     prompt = create_buyer_image_prompt()
#     parser = StructuredOutputParser.from_response_schemas(buyer_schema)
#     messages = prompt.format_messages(image_base64=state.image_base64)
#     logger.info(f"Extracting buyer details from image. Messages: {messages}")
#     try:
#         response = model.invoke(messages)
#         logger.info(f"LLM response for buyer details: {response.content}")
#         buyer_data = parser.parse(response.content)
#         logger.info(f"Parsed buyer details: {buyer_data}")
#         return {"buyer_data": buyer_data}
#     except Exception as e:
#         message = f"Error parsing buyer details: {e} - Response: {response.content}"
#         logger.error(message)
#         return {"buyer_data": {}}



# def extract_items_details_from_image(state: InvoiceData, openai_client: OpenAI = Depends(get_openai_client)) -> Dict[str, Any]:
#     """Extracts invoice items details from an image."""
#     model = ChatOpenAI(
#         model_name="google/gemma-3-27b-it",
#         base_url="https://openrouter.ai/api/v1",
#         api_key=OPENAI_API_KEY,
#         max_tokens=3000,
#     )
#     prompt = create_items_image_prompt()
#     parser = StructuredOutputParser.from_response_schemas(item_schema)
#     messages = prompt.format_messages(image_base64=state.image_base64)
#     logger.info(f"Extracting items details from image. Messages: {messages}")
#     try:
#         response = model.invoke(messages)
#         logger.info(f"LLM response for item details: {response.content}")
#         items_data = parser.parse(response.content)
#         logger.info(f"Parsed item details: {items_data}")
#         return {"items_data": items_data}
#     except Exception as e:
#         message = f"Error parsing items details: {e} - Response: {response.content}"
#         logger.error(message)
#         return {"items_data": []}  # Return empty list, handle in graph



def combine_results(state: InvoiceData) -> Dict[str, Any]:
    """Combines the extracted information into the final JSON output."""
    logger.info(f"Combining results. Current state: {state}")
    final_result: Dict[str, Any] = {
        "order_number": state.order_invoice_data.get("order_number", None) if hasattr(state, 'order_invoice_data') and state.order_invoice_data else None,
        "invoice_number": state.order_invoice_data.get("invoice_number", None) if hasattr(state, 'order_invoice_data') and state.order_invoice_data else None,
        "order_date": state.order_invoice_data.get("order_date", None) if hasattr(state, 'order_invoice_data') and state.order_invoice_data else None,
        "invoice_id": state.order_invoice_data.get("invoice_id", None) if hasattr(state, 'order_invoice_data') and state.order_invoice_data else None,
        "invoice_date": state.order_invoice_data.get("invoice_date", None) if hasattr(state, 'order_invoice_data') and state.order_invoice_data else None,
        "transaction_id": state.order_invoice_data.get("transaction_id", None) if hasattr(state, 'order_invoice_data') and state.order_invoice_data else None,
        "date_time": state.order_invoice_data.get("date_time", None) if hasattr(state, 'order_invoice_data') and state.order_invoice_data else None,
        "invoice_value": state.order_invoice_data.get("invoice_value", None) if hasattr(state, 'order_invoice_data') and state.order_invoice_data else None,
        "mode_of_payment": state.order_invoice_data.get("mode_of_payment", None) if hasattr(state, 'order_invoice_data') and state.order_invoice_data else None,
        "place_of_supply": state.order_invoice_data.get("place_of_supply", None) if hasattr(state, 'order_invoice_data') and state.order_invoice_data else None,
        "place_of_delivery": state.order_invoice_data.get("place_of_delivery", None) if hasattr(state, 'order_invoice_data') and state.order_invoice_data else None,
        "seller": state.seller if hasattr(state, 'seller') and state.seller else {},
        "buyer": state.buyer if hasattr(state, 'buyer') and state.buyer else {},
        "invoice_items": state.invoice_items if hasattr(state, 'invoice_items') and state.invoice_items else [],
    }
    logger.info(f"Combined result: {final_result}")
    return {"result": final_result}



# 5. Define the Graph for Image Input
def create_image_graph() -> StateGraph:
    """Creates the LangGraph graph for image input."""
    graph_builder = StateGraph(InvoiceData)

    graph_builder.add_node("extract_order_invoice_details", extract_order_invoice_details_from_image)
    graph_builder.add_node("extract_seller_details", extract_seller_details_from_image)
    graph_builder.add_node("extract_buyer_details", extract_buyer_details_from_image)
    graph_builder.add_node("extract_items_details", extract_items_details_from_image)
    graph_builder.add_node("combine_results", combine_results)

    graph_builder.set_entry_point("extract_order_invoice_details")
    graph_builder.add_edge("extract_order_invoice_details", "extract_seller_details")
    graph_builder.add_edge("extract_seller_details", "extract_buyer_details")
    graph_builder.add_edge("extract_buyer_details", "extract_items_details")
    graph_builder.add_edge("extract_items_details", "combine_results")
    graph_builder.add_edge("combine_results", END)

    return graph_builder.compile()



app = FastAPI()

# 6. Define the FastAPI endpoint
@app.post("/extract_invoice_data/")
async def upload_image(file: UploadFile = File(...), openai_client: OpenAI = Depends(get_openai_client)) -> Dict[str, Any]: # Added type hint
    """
    Endpoint to upload an image and extract invoice data using LangGraph.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()
        graph = create_image_graph()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        # Initialize InvoiceData with image_base64
        initial_state = InvoiceData(image_base64=base64_image)
        logger.info(f"Initial state: {initial_state}")
        result = graph.invoke(initial_state)
        logger.info(f"Graph result: {result}")
        return result["result"]
    except Exception as e:
        message = f"An error occurred during image processing: {e}"
        logger.error(message)
        raise HTTPException(status_code=500, detail=message)