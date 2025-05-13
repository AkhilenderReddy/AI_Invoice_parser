# AI Invoice Parser API

This project provides an API endpoint to extract structured data from invoice documents (currently supporting PDF files). It leverages LangGraph and OpenAI (via OpenRouter) to intelligently parse and extract information such as order details, seller and buyer information, and individual invoice items.

## Features

- **PDF Invoice Processing:** Upload a PDF invoice and receive structured JSON output containing extracted data.
- **Comprehensive Data Extraction:** Extracts key information including:
    - Order and Invoice Numbers/Dates
    - Transaction and Invoice IDs
    - Total Invoice Value and Mode of Payment
    - Place of Supply and Delivery
    - Seller Details (Name, GST, PAN, Address, Bank Information, Contact Details)
    - Buyer Details (Name, GST, PAN, Address, Contact Details, Billing/Shipping Addresses)
    - Invoice Items (Serial Number, HSN, Description, Unit Price, Quantity, Net Amount, Taxes, Total Amount)
- **Robust Error Handling:** Provides informative error messages for invalid file types or processing issues.
- **Modular Design:** Utilizes LangGraph for a well-structured and maintainable extraction pipeline.

## Prerequisites

- **Python 3.10**
- **pip** (Python package installer)
- **OpenAI API Key:** You need an API key from [OpenRouter](https://openrouter.ai/). Sign up and obtain your API key for **gemma-3-27b-it**.


## 🚀 Installation

1.  **Clone the repository:**
    
    ```bash
    git clone https://github.com/AkhilenderReddy/AI_Invoice_parser.git
    cd AI_Invoice_parser
    ```
    
2.  **Create a virtual environment (recommended):**
    
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```
    
3.  **Install the required dependencies:**
    
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create `.env` file with your OpenRouter API key:**
    
    ```bash
    OPENAI_API_KEY=your_openrouter_api_key_here
    ```
    
5. **Run the FastAPI app (Multi_Agent_AI_pdf_parser.py):**
    
    ```bash
    uvicorn Multi_Agent_AI_pdf_parser:app --reload
    ```
    
6.  **Access the API docs:**
    
    -   Open your browser and go to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)





## Sending a Request with Postman

Follow these concise steps to process a PDF invoice:

1.  **Open Postman.** Launch the Postman application on your computer.

2.  **Create a New Request:** Click the "+" button to open a new tab or select "New" -> "HTTP Request".

3.  **Select POST Method:** In the dropdown menu next to the URL field, choose `POST`.

4.  **Enter API Endpoint URL:** In the URL field, type or paste: `http://127.0.0.1:8000/extract_invoice_data_pdf/`.

5.  **Navigate to Body:** Click on the "Body" tab below the URL field.

6.  **Choose form-data:** Select the `form-data` option.

7.  **Add PDF File:**
    -   In the "Key" column, enter `file`.
    -   In the "Value" column, click the **"File"** option (it appears when you hover).
    -   Browse to and select your PDF invoice file from your computer.

8.  **Send Request:** Click the "Send" button.

Upon successful processing, the API will return a JSON response containing the extracted invoice data. This response will be displayed in the "Body" section of the Postman window.
        

