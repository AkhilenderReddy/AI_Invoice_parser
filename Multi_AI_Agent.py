import os
import json
import re
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union, Any
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
import logging

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
    seller_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Seller information")
    buyer_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Buyer information")
    items_data: List[Dict[str, Any]] = Field(default_factory=list, description="List of invoice items")
    order_invoice_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Order and invoice information")
    image_base64: Optional[str] = Field(description="Base64 encoded image of the invoice")

    @validator('*')
    def set_none_for_empty_string(cls, value: Optional[str]) -> Optional[str]:
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

# 4. Define Functions for Agent Actions (Image Input)
def extract_order_invoice_details_from_image(state: InvoiceData, openai_client: OpenAI = Depends(get_openai_client)) -> Dict[str, Any]:
    """Extracts order and invoice details from an image."""
    model = ChatOpenAI(
        model_name="google/gemma-3-27b-it",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENAI_API_KEY,
        max_tokens=1000,
    )
    prompt = create_order_invoice_prompt()
    parser = StructuredOutputParser.from_response_schemas(order_invoice_schema)
    messages = prompt.format_messages(image_base64=state.image_base64)
    logger.info("Extracting order/invoice details from image.")
    
    try:
        response = model.invoke(messages)
        logger.info(f"Raw LLM response (order/invoice): {response.content}")
        
        # Try to parse the response
        try:
            order_invoice_data = parser.parse(response.content)
            logger.info(f"Parsed order/invoice details: {order_invoice_data}")
        except Exception as parse_error:
            logger.error(f"Error parsing LLM response for order/invoice: {parse_error}")
            order_invoice_data = {}  # Use empty dict if parsing fails
        
        # FIX: Use model_dump() for Pydantic v2 or dict() for v1
        try:
            # Try Pydantic v2 method first
            state_dict = state.model_dump()
        except AttributeError:
            # Fall back to Pydantic v1 method
            state_dict = state.dict()
            
        return {**state_dict, "order_invoice_data": order_invoice_data}
    except Exception as e:
        logger.error(f"Error in order/invoice extraction: {e}")
        
        # FIX: Use model_dump() for Pydantic v2 or dict() for v1
        try:
            # Try Pydantic v2 method first
            state_dict = state.model_dump()
        except AttributeError:
            # Fall back to Pydantic v1 method
            state_dict = state.dict()
            
        return {**state_dict, "order_invoice_data": {}}

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
    logger.info("Extracting seller details from image.")
    
    try:
        response = model.invoke(messages)
        logger.info(f"Raw LLM response (seller): {response.content}")
        
        # Try to parse the response
        try:
            seller_data = parser.parse(response.content)
            logger.info(f"Parsed seller details: {seller_data}")
        except Exception as parse_error:
            logger.error(f"Error parsing LLM response for seller: {parse_error}")
            seller_data = {}  # Use empty dict if parsing fails
        
        # FIX: Use model_dump() for Pydantic v2 or dict() for v1
        try:
            # Try Pydantic v2 method first
            state_dict = state.model_dump()
        except AttributeError:
            # Fall back to Pydantic v1 method
            state_dict = state.dict()
            
        return {**state_dict, "seller_data": seller_data}
    except Exception as e:
        logger.error(f"Error in seller details extraction: {e}")
        
        # FIX: Use model_dump() for Pydantic v2 or dict() for v1
        try:
            # Try Pydantic v2 method first
            state_dict = state.model_dump()
        except AttributeError:
            # Fall back to Pydantic v1 method
            state_dict = state.dict()
            
        return {**state_dict, "seller_data": {}}

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
    logger.info("Extracting buyer details from image.")
    
    try:
        response = model.invoke(messages)
        logger.info(f"Raw LLM response (buyer): {response.content}")
        
        # Try to parse the response
        try:
            buyer_data = parser.parse(response.content)
            logger.info(f"Parsed buyer details: {buyer_data}")
        except Exception as parse_error:
            logger.error(f"Error parsing LLM response for buyer: {parse_error}")
            buyer_data = {}  # Use empty dict if parsing fails
        
        # FIX: Use model_dump() for Pydantic v2 or dict() for v1
        try:
            # Try Pydantic v2 method first
            state_dict = state.model_dump()
        except AttributeError:
            # Fall back to Pydantic v1 method
            state_dict = state.dict()
            
        return {**state_dict, "buyer_data": buyer_data}
    except Exception as e:
        logger.error(f"Error in buyer details extraction: {e}")
        
        # FIX: Use model_dump() for Pydantic v2 or dict() for v1
        try:
            # Try Pydantic v2 method first
            state_dict = state.model_dump()
        except AttributeError:
            # Fall back to Pydantic v1 method
            state_dict = state.dict()
            
        return {**state_dict, "buyer_data": {}}

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
    logger.info("Extracting items details from image.")
    
    try:
        response = model.invoke(messages)
        logger.info(f"Raw LLM response (items): {response.content}")
        
        # FIX: More robust parsing with better error handling
        try:
            # First check if the response is a list or a single item
            content = response.content
            if "```json" in content:
                # Extract JSON content if it's in a code block
                content = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if content:
                    content = content.group(1)
                
            # Try to parse as JSON
            try:
                parsed_json = json.loads(content)
                
                # Handle both single item and array responses
                if isinstance(parsed_json, dict):
                    items_data = [parsed_json]  # Wrap single item in list
                elif isinstance(parsed_json, list):
                    items_data = parsed_json
                else:
                    # Fallback for unexpected format
                    items_data = []
                    logger.warning(f"Unexpected format in items response: {type(parsed_json)}")
            except json.JSONDecodeError:
                # If not valid JSON, use the parser
                items_data = parser.parse(content)
                if not isinstance(items_data, list):
                    items_data = [items_data]
                    
            logger.info(f"Parsed items details: {items_data}")
        except Exception as parse_error:
            logger.error(f"Error parsing LLM response for items: {parse_error}")
            items_data = []  # Use empty list if parsing fails
        
        # FIX: Use model_dump() for Pydantic v2 or dict() for v1
        try:
            # Try Pydantic v2 method first
            state_dict = state.model_dump()
        except AttributeError:
            # Fall back to Pydantic v1 method
            state_dict = state.dict()
            
        return {**state_dict, "items_data": items_data}
    except Exception as e:
        logger.error(f"Error in items details extraction: {e}")
        
        # FIX: Use model_dump() for Pydantic v2 or dict() for v1
        try:
            # Try Pydantic v2 method first
            state_dict = state.model_dump()
        except AttributeError:
            # Fall back to Pydantic v1 method
            state_dict = state.dict()
            
        return {**state_dict, "items_data": []}

def combine_results(state: InvoiceData) -> Dict[str, Any]:
    """Combines the extracted information into the final JSON output."""
    logger.info("Combining results to produce final output.")
    
    # Safely get values from state
    order_invoice_data = getattr(state, "order_invoice_data", {}) or {}
    seller_data = getattr(state, "seller_data", {}) or {}
    buyer_data = getattr(state, "buyer_data", {}) or {}
    items_data = getattr(state, "items_data", []) or []
    
    # For debugging
    logger.info(f"State contains order_invoice_data: {bool(order_invoice_data)}")
    logger.info(f"State contains seller_data: {bool(seller_data)}")
    logger.info(f"State contains buyer_data: {bool(buyer_data)}")
    logger.info(f"State contains items_data: {bool(items_data)}")
    
    # Build the final result object
    final_result = {
        "order_number": order_invoice_data.get("order_number"),
        "invoice_number": order_invoice_data.get("invoice_number"),
        "order_date": order_invoice_data.get("order_date"),
        "invoice_id": order_invoice_data.get("invoice_id"),
        "invoice_date": order_invoice_data.get("invoice_date"),
        "transaction_id": order_invoice_data.get("transaction_id"),
        "date_time": order_invoice_data.get("date_time"),
        "invoice_value": order_invoice_data.get("invoice_value"),
        "mode_of_payment": order_invoice_data.get("mode_of_payment"),
        "place_of_supply": order_invoice_data.get("place_of_supply"),
        "place_of_delivery": order_invoice_data.get("place_of_delivery"),
        "seller": seller_data,
        "buyer": buyer_data,
        "invoice_items": items_data,
    }
    
    logger.info(f"Final result: {final_result}")
    
    # Explicitly remove the image_base64 field from the result
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

# Create FastAPI app
app = FastAPI()

# Define the FastAPI endpoint
@app.post("/extract_invoice_data/")
async def upload_image(file: UploadFile = File(...), openai_client: OpenAI = Depends(get_openai_client)) -> Dict[str, Any]:
    """
    Endpoint to upload an image and extract invoice data using LangGraph.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Read the image file
        image_bytes = await file.read()
        
        # Create the graph
        graph = create_image_graph()
        
        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # Initialize InvoiceData with image_base64
        initial_state = InvoiceData(image_base64=base64_image)
        logger.info("Processing image through the extraction pipeline...")
        
        # Execute the graph
        final_state = graph.invoke(initial_state)
        logger.info("Graph execution completed")
        
        # Check if result is in the final state
        if "result" in final_state:
            result = final_state["result"]
            logger.info("Successfully extracted invoice data")
            return result
        else:
            error_msg = "Processing completed but no result field was found in the output"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
    except Exception as e:
        error_msg = f"An error occurred during image processing: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)