# 📈 AI Financial Report Parser

An interactive web application built with Python, Streamlit, and Groq that extracts structured financial metrics from complex corporate PDF reports (like SEC 10-K, 10-Q filings, or earnings summaries) using Llama 3 models.

## 🚀 Features

* **Smart Page Filtering**: Scans full PDFs locally to isolate only pages mentioning financial statements, saving token costs and staying within API limits.
* **Structured Data Extraction**: Utilizes Groq and Pydantic schemas to output precise, layout-aware financial JSON objects.
* **Beautiful Dashboard UI**: Upload reports via a drag-and-drop web page, view live metrics, and download extracted JSON data cleanly.
* **Production Ready**: Fully pre-configured for both local development and instant deployment to Streamlit Community Cloud.

---

