# RAG Q&A Application

This RAG (Retrieval-Augmented Generation) Q&A app allows users to upload a document and query it for relevant answers. Users can input their OpenAI API key, upload a document (PDF or text file), and ask questions related to the document content. The app will generate precise answers based on the uploaded document using OpenAI's language models.

## Features

- **Document Upload**: Supports PDF and text file uploads.
- **OpenAI API Integration**: Users can input their own OpenAI API key for generating responses.
- **Interactive Q&A**: Allows users to ask questions about the content of the uploaded document and receive relevant answers.

## Setup and Installation

### Prerequisites

- Python 3.8 or above
- Git (optional if uploading directly)
- An OpenAI API key (required for querying the document)

## Usage

1. **Enter Your OpenAI API Key (Mandatory)**: The app requires an OpenAI API key to generate responses. Enter your key in the input field provided at the start. If you don't have an OpenAI API key, create one.
   
2. **Upload a Document**: Click the upload button to select a document (PDF or text file). The app will process the content for querying.

3. **Ask Questions**: Enter your question in the query box related to the content of the uploaded document. The app will generate and display relevant answers based on the document content.

## File Structure

```
RAG_Project
│
├── app.py                # Main Streamlit application
├── requirements.txt      # List of dependencies
└── README.md             # Documentation for the app (this file)
```

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the interactive app framework.
- [OpenAI](https://openai.com/) for providing the API and language models.

---
