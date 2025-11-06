"""
Document Text Extraction Module
================================

This module handles extracting raw text content from various file formats.
The extracted text is then passed to knowledge.py for fact extraction.

Supported Formats:
- PDF: Uses PyPDF2
- DOCX: Uses python-docx
- TXT: Direct file read
- CSV: Uses pandas (extracts text from cells)
- PPTX: Uses python-pptx (if available)

Flow:
1. User uploads file → api_server.py receives it
2. api_server.py calls handle_file_upload() → extracts text
3. Extracted text → knowledge.add_to_graph() → extracts facts

Key Functions:
- handle_file_upload(file_paths): Main entry point - processes multiple files
- process_uploaded_file(file): Processes single file, returns text
- extract_text_from_pdf(): PDF text extraction
- extract_text_from_docx(): DOCX text extraction
- extract_text_from_csv(): CSV text extraction

Author: Research Brain Team
Last Updated: 2025-01-15
"""

import os
from datetime import datetime
import pandas as pd
import PyPDF2
from docx import Document
from knowledge import add_to_graph

# ============================================================================
# GLOBAL STATE
# ============================================================================

last_extracted_text = ""  # Last extracted text (for debugging)
processed_files = []      # List of processed file names

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                text += (page_text or "") + "\n"
            return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = "".join(p.text + "\n" for p in doc.paragraphs)
        return text.strip()
    except Exception as e:
        return f"Error reading DOCX: {e}"

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        return f"Error reading TXT: {e}"

def extract_text_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        text = f"CSV Data with {len(df)} rows and {len(df.columns)} columns:\n\n"
        text += f"Columns: {', '.join(df.columns)}\n\n"
        text += "Sample data:\n"
        for i, row in df.head(5).iterrows():
            text += f"Row {i+1}: {dict(row)}\n"
        return text.strip()
    except Exception as e:
        return f"Error reading CSV: {e}"

def update_extracted_text(text):
    global last_extracted_text
    last_extracted_text = text

def show_extracted_text():
    global last_extracted_text
    if not last_extracted_text:
        return " No file has been processed yet.\n\nUpload a file and process it to see the extracted text here."
    preview = last_extracted_text[:1000]
    if len(last_extracted_text) > 1000:
        preview += "\n\n... (truncated, showing first 1000 characters)"
    return f" **Last Extracted Text:**\n\n{preview}"

def process_uploaded_file(file):
    if file is None:
        return "No file uploaded."
    # Handle both string paths and file objects
    if isinstance(file, str):
        file_path = file
    else:
        file_path = file.name if hasattr(file, 'name') else str(file)
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        extracted_text = extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        extracted_text = extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        extracted_text = extract_text_from_txt(file_path)
    elif file_extension == '.csv':
        extracted_text = extract_text_from_csv(file_path)
    else:
        return f" Unsupported file type: {file_extension}\n\nSupported formats: PDF, DOCX, TXT, CSV"
    if extracted_text.startswith("Error"):
        return f" {extracted_text}"
    update_extracted_text(extracted_text)
    preview = extracted_text[:300] + "..." if len(extracted_text) > 300 else extracted_text
    result = add_to_graph(extracted_text)
    file_size = len(extracted_text)
    return f" Successfully processed {os.path.basename(file_path)}!\n\n📊 File stats:\n• Size: {file_size:,} characters\n• Type: {file_extension.upper()}\n\n Text preview:\n{preview}\n\n{result}"

def handle_file_upload(files):
    global processed_files
    if not files or len(files) == 0:
        return "Please select at least one file to process."
    results = []
    new_processed = []
    for file in files:
        if file is None:
            continue
        try:
            if isinstance(file, str):
                file_path = file
                file_name = os.path.basename(file)
            else:
                file_path = file.name
                file_name = os.path.basename(file.name)
            if any(f['name'] == file_name for f in processed_files):
                results.append(f"SKIP: {file_name} - Already processed, skipping")
                continue
            result = process_uploaded_file(file)
            results.append(f"SUCCESS: {file_name} - {result}")
            new_processed.append({
                'name': file_name,
                'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
        except Exception as e:
            file_name = os.path.basename(file) if isinstance(file, str) else os.path.basename(file.name) if hasattr(file, 'name') else str(file)
            results.append(f"ERROR: {file_name} - Error: {e}")
    processed_files.extend(new_processed)
    total_files = len(files)
    successful = len([r for r in results if r.startswith("SUCCESS")])
    skipped = len([r for r in results if r.startswith("SKIP")])
    failed = len([r for r in results if r.startswith("ERROR")])
    summary = f"**Upload Summary:**\n"
    summary += f"• Total files: {total_files}\n"
    summary += f"• Successfully processed: {successful}\n"
    summary += f"• Already processed: {skipped}\n"
    summary += f"• Failed: {failed}\n\n"
    summary += "**File Results:**\n"
    for result in results:
        summary += f"{result}\n"
    return summary

def show_processed_files():
    global processed_files
    if not processed_files:
        return "**No files processed yet.**\n\n**Start building your knowledge base:**\n1. Select one or more files (PDF, DOCX, TXT, CSV)\n2. Click 'Process Files' to extract knowledge\n3. View your processed files here\n4. Upload more files to expand your knowledge base!"
    result = f"**Processed Files ({len(processed_files)}):**\n\n"
    for i, file_info in enumerate(processed_files, 1):
        result += f"**{i}. {file_info['name']}**\n"
        result += f"   • Size: {file_info['size']:,} bytes\n"
        result += f"   • Processed: {file_info['processed_at']}\n\n"
    return result

def clear_processed_files():
    global processed_files
    processed_files = []
    return "Processed files list cleared. You can now re-upload previously processed files."

def simple_test():
    return " Event handler is working! Button clicked successfully!"


