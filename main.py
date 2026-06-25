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
def extract_relevant_pages(uploaded_file):
    """Reads PDF, checks if it is financial, and caps size to prevent 413 token errors."""
    reader = PdfReader(uploaded_file)
    keywords = ["revenue", "net income", "balance sheet", "income statement", "cash equivalents", "profit", "loss"]
    
    filtered_text = ""
    financial_content_found = False
    
    # Check the first few pages strictly to see if this document is even financial
    sample_text = ""
    for i in range(min(5, len(reader.pages))):
        text = reader.pages[i].extract_text()
        if text:
            sample_text += text.lower()
            
    if not any(kw in sample_text for kw in keywords):
        return None, False  # Fails financial validation guard

    # Always keep cover page for context
    first_page = reader.pages[0].extract_text()
    if first_page:
        filtered_text += first_page + "\n"

    for page in reader.pages[1:]:
        text = page.extract_text()
        if text and any(kw in text.lower() for kw in keywords):
            filtered_text += text + "\n"
            
        # ⚠️ BUG 2 FIX: Cap local context extraction size at roughly ~60,000 characters
        # This acts as a circuit-breaker so we never hit Groq's 413 token limits on huge files
        if len(filtered_text) > 60000:
            filtered_text = filtered_text[:60000] + "\n...[Text truncated to prevent API rate limits]..."
            break
            
    return filtered_text, True

def analyze_with_groq(document_text: str):
    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY missing from your environment setup!")
        return None
        
    client = Groq(api_key=GROQ_API_KEY)
    schema_json = json.dumps(FinancialDataModel.model_json_schema(), indent=2)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[
            {
                "role": "system",
                "content": f"Extract financial data matching this JSON Schema exactly:\n{schema_json}\nReturn ONLY valid JSON. Ensure strict compliance with standard JSON syntax."
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
        with st.spinner("Analyzing document structure and verifying contents..."):
            raw_text, is_financial = extract_relevant_pages(uploaded_file)
            
        # ⚠️ BUG 3 FIX: Reject non-financial files early
        if not is_financial:
            st.error("❌ Validation Error: This document does not appear to be a financial statement. Please upload an annual report, 10-K, 10-Q, or financial summary.")
        elif not raw_text or not raw_text.strip():
            st.error("❌ Error: Could not extract readable text from this file format.")
        else:
            with st.spinner("Analyzing tables with Groq AI..."):
                try:
                    json_output = analyze_with_groq(raw_text)
                    
                    # ⚠️ BUG 1 FIX: Load string into native python dict first to sanitize syntax, 
                    # then let Streamlit render it natively to avoid missing commas or raw formatting bugs.
                    structured_data = json.loads(json_output)
                    
                    st.balloons()
                    st.subheader("📊 Extracted Results")
                    
                    # Visual Metric Widgets
                    col1, col2, col3 = st.columns(3)
                    curr = structured_data.get('currency', 'USD')
                    
                    col1.metric("Company", structured_data.get("company_name", "N/A"))
                    col2.metric("Period", structured_data.get("fiscal_period", "N/A"))
                    col3.metric("Currency", curr)
                    
                    col1.metric("Total Revenue", f"{structured_data.get('total_revenue', 0.0):,}")
                    col2.metric("Net Income", f"{structured_data.get('net_income', 0.0):,}")
                    col3.metric("Cash & Equivalents", f"{structured_data.get('cash_and_equivalents', 0.0):,}")
                    
                    # Native Streamlit JSON output display
                    st.subheader("💾 Validated JSON Payload")
                    st.json(structured_data)
                    
                    st.download_button(
                        label="📥 Download Clean JSON File",
                        data=json.dumps(structured_data, indent=2),
                        file_name=uploaded_file.name.replace(".pdf", "_extracted.json"),
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Pipeline failed: {e}")