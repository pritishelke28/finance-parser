import os
import json
import streamlit as st
from pydantic import BaseModel, Field
from typing import Optional
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv

# Load credentials
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Financial Report Parser", page_icon="📈", layout="centered")

st.title("📈 AI Financial Report Parser")
st.caption("Upload a financial PDF and extract structured data instantly using Groq & Llama-3.1")

# --- Financial Data Schema ---
class FinancialDataModel(BaseModel):
    company_name: str = Field(description="The name of the company or organization")
    fiscal_period: str = Field(description="The year or quarter of the data (e.g., FY2025, Q3 2026)")
    currency: str = Field(description="The currency code used (e.g., USD, EUR, INR)")
    total_revenue: float = Field(description="Total revenue or top-line sales reported")
    net_income: float = Field(description="Net income or net profit/loss for the period")
    cash_and_equivalents: Optional[float] = Field(description="Cash and cash equivalents on the balance sheet")

# --- Helper Functions ---
def extract_relevant_pages(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    keywords = ["revenue", "net income", "balance sheet", "income statement", "cash equivalents"]
    filtered_text = ""
    
    # Keep cover page
    first_page = reader.pages[0].extract_text()
    if first_page:
        filtered_text += first_page + "\n"

    for page in reader.pages[1:]:
        text = page.extract_text()
        if text and any(kw in text.lower() for kw in keywords):
            filtered_text += text + "\n"
            
    return filtered_text

def analyze_with_groq(document_text: str):
    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY missing from your .env file!")
        return None
        
    client = Groq(api_key=GROQ_API_KEY)
    schema_json = json.dumps(FinancialDataModel.model_json_schema(), indent=2)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[
            {
                "role": "system",
                "content": f"Extract financial data matching this JSON Schema exactly:\n{schema_json}\nReturn ONLY valid JSON."
            },
            {
                "role": "user",
                "content": document_text
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return response.choices[0].message.content

# --- UI Layout ---
uploaded_file = st.file_uploader("Choose your financial PDF report", type=["pdf"])

if uploaded_file is not None:
    st.success(f"📄 Connected to: {uploaded_file.name}")
    
    if st.button("🚀 Extract Financial Data", type="primary"):
        with st.spinner("Reading PDF and filtering pages..."):
            raw_text = extract_relevant_pages(uploaded_file)
            
        if not raw_text.strip():
            st.error("Could not find relevant text or financial keywords in this PDF.")
        else:
            with st.spinner("Analyzing tables with Groq AI..."):
                try:
                    json_output = analyze_with_groq(raw_text)
                    structured_data = json.loads(json_output)
                    
                    st.balloons()
                    st.subheader("📊 Extracted Results")
                    
                    # Display metrics beautifully in columns
                    col1, col2, col3 = st.columns(3)
                    curr = structured_data.get('currency', 'USD')
                    
                    col1.metric("Company", structured_data.get("company_name"))
                    col2.metric("Period", structured_data.get("fiscal_period"))
                    col3.metric("Currency", curr)
                    
                    col1.metric("Total Revenue", f"{structured_data.get('total_revenue'):,}")
                    col2.metric("Net Income", f"{structured_data.get('net_income'):,}")
                    col3.metric("Cash & Equivalents", f"{structured_data.get('cash_and_equivalents', 0):,}")
                    
                    # Show raw JSON code block
                    st.subheader("💾 Raw JSON Payload")
                    st.json(structured_data)
                    
                    # Add a download button for the JSON file
                    st.download_button(
                        label="📥 Download JSON File",
                        data=json.dumps(structured_data, indent=2),
                        file_name=uploaded_file.name.replace(".pdf", "_extracted.json"),
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")