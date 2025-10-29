---
title: NesyX
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
---

# 🧠 NesyX - Reasoning Researcher

A neurosymbolic assistant that combines a small knowledge graph with language reasoning using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).

## Features

- **Knowledge Graph**: Add facts and information to build a symbolic knowledge base
- **File Upload**: Upload PDF, DOCX, TXT, and CSV files to automatically extract knowledge
- **Neurosymbolic Reasoning**: Combines symbolic facts with LLM reasoning
- **Interactive Chat**: Chat interface with configurable parameters
- **Automatic Fallback**: Uses public models when authentication isn't available

## Authentication

This app uses Hugging Face OAuth (`hf_oauth: true`) to automatically authenticate with the Inference API. The app will:

1. **With Authentication**: Use `HuggingFaceH4/zephyr-7b-beta` for better reasoning
2. **Without Authentication**: Fall back to `microsoft/DialoGPT-medium` (public model)

## Usage

### Adding Knowledge
1. **Text Input**: Paste text in the "Add Knowledge" box to build your knowledge graph
2. **File Upload**: Upload PDF, DOCX, TXT, or CSV files to automatically extract and add knowledge
3. **View Graph**: Click "Show Knowledge Graph" to see all stored facts

### Chatting
- Ask questions that will be answered using both the knowledge graph and LLM reasoning
- The system combines your uploaded knowledge with AI reasoning for comprehensive answers

### Supported File Types
- **PDF**: Research papers, documents, reports
- **DOCX**: Word documents, articles
- **TXT**: Plain text files, notes
- **CSV**: Data files, spreadsheets
