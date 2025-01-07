# QA Chatbot

## Overview
The QA Chatbot is designed to process PDF documents from a specified folder, store them in a Chroma vector store (ChromaDB), and answer user queries based on the documents using OpenAI's GPT-3.5 model.

## Folder Structure

    QA_chatbot/
    ├── Api_Based/
    │   ├── db/
    │   ├── Manuals/
    │   ├── Rag_pdf_loader_gpt_3.5.py
    └── Model_Based/

### Api_Based Folder
- **Manuals/**: This folder contains the PDF documents to be processed and stored in the Chroma vector store.
- **db/**: After generating the embeddings for each document, they are stored in the `chroma_manuals` directory inside the `db` folder.
- **Rag_pdf_loader_gpt_3.5.py**: The main script that processes the PDF files, stores the embeddings, and handles user queries.

## Script: Rag_pdf_loader_gpt_3.5.py

### Input:
- The script processes PDF files in the specified `Manuals` folder.
- Queries entered by the user in the terminal.

### Output:
- Relevant documents retrieved from the database.
- Answers to user queries based on the retrieved documents.

### Functionality:
1. **Load and Process PDF Documents**:
   - The script loads PDF documents from the specified folder.
   - Splits the documents into chunks for better processing and retrieval.
   
2. **Store Processed Documents in ChromaDB**:
   - Generates embeddings for each document chunk.
   - Stores the embeddings in ChromaDB for efficient retrieval.

3. **Answer User Queries Using OpenAI GPT-3.5 Model**:
   - The script allows users to enter queries in the terminal.
   - Retrieves relevant documents from ChromaDB based on the query.
   - Uses OpenAI GPT-3.5 model to generate answers based on the retrieved documents.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd QA_chatbot/Api_Based
   pip install -r requirements.txt - install the required dependecnies
   python Rag_pdf_loader_gpt_3_5.py - run the script
   

