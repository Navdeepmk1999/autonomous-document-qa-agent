# Autonomous Clinical Document QA Agent

## Overview
This repository contains a production-ready, autonomous AI agent designed to extract precise factual information from clinical documents (PDFs) using Retrieval-Augmented Generation (RAG). Built specifically for enterprise environments, it utilizes LangChain for orchestration, ChromaDB for persistent vector storage, and the OpenAI API for reasoning and generation.

The agent is intentionally configured with a strict system prompt and a temperature of 0 to prioritize factual accuracy and eliminate hallucinations—a critical requirement for healthcare and clinical data workflows.

## Features
- **Intelligent Document Ingestion:** Processes complex PDFs by chunking text while maintaining semantic overlap to preserve context.
- **Persistent Vector Storage:** Utilizes ChromaDB to store document embeddings locally (`./chroma_db`), eliminating redundant API costs for repeated queries on the same document.
- **Zero-Hallucination Configuration:** Configured specifically for clinical data; if the answer is not within the provided document context, the agent explicitly refuses to generate a fabricated answer.
- **Modern LangChain Architecture:** Uses the updated `langchain-classic` and `langchain-community` modules to ensure compatibility with modern deployment standards.

## Prerequisites
- Python 3.10+
- An active [OpenAI API Key](https://platform.openai.com/)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Navdeepmk1999/autonomous-document-qa-agent.git
   cd autonomous-document-qa-agent
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
    ```

3. **Install the required dependencies::**
    ```bash
    pip install -r requirements.txt
    ```

4. **Environment Setup:**

    Create a .env file in the root directory and add your OpenAI API key:

    ```
    OPENAI_API_KEY="sk-your-api-key-here"
    ```

## Usage

1. **Add your document: Place the PDF you wish to query in the root directory of the project and ensure it is named sample_document.pdf (or update the target_pdf variable in main.py).**

2. **Execute the Agent:**

Run the main script from your terminal:

    ```bash
    python main.py
    ```
3. **Expected Output:**

The script will log the four stages of the RAG pipeline (Ingestion, Database Setup, Initialization, Execution) before returning the AI's factual answer based on the document provided.

## Project Structure

```
autonomous-document-qa-agent/
│
├── main.py                # The core execution script for the LangChain agent
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (not tracked in Git)
├── .gitignore             # Ignored files to protect API keys and local DBs
└── chroma_db/             # Local directory for persistent vector storage (auto-generated)
```

## Technical Details & MLOps

This project reflects a transition from monolithic script execution to structured, modular ML pipelines. It handles the deprecation cycles of modern LLM frameworks by cleanly managing virtual environments and separating core dependencies (langchain-core, langchain-classic). The architecture is designed to be easily containerized via Docker for deployment on AWS ECS or Azure Container Apps.