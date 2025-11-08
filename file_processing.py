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

def process_uploaded_file(file, original_filename=None):
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
    # Use original filename if provided, otherwise use basename of file path
    if original_filename:
        filename = original_filename
    else:
        filename = os.path.basename(file_path)
    
    display_name = original_filename if original_filename else os.path.basename(file_path)
    
    # Debug: Check if text was extracted
    if not extracted_text or len(extracted_text.strip()) < 10:
        print(f"⚠️  WARNING: Extracted text is too short or empty for {display_name}")
        print(f"   Text length: {len(extracted_text) if extracted_text else 0}")
        return f"⚠️  Warning: Could not extract meaningful text from {display_name}. File may be empty or in an unsupported format."
    
    print(f"📄 Processing {display_name}: Extracted {len(extracted_text)} characters")
    print(f"   Text preview: {extracted_text[:200]}...")
    
    timestamp = datetime.now().isoformat()
    result = add_to_graph(extracted_text, source_document=filename, uploaded_at=timestamp)
    
    # Debug: Check if facts were added
    from knowledge import graph as kb_graph
    fact_count = len(kb_graph)
    print(f"✅ After processing {display_name}: Graph now has {fact_count} facts")
    print(f"   Result message: {result[:200]}...")
    
    file_size = len(extracted_text)
    return f" Successfully processed {display_name}!\n\n📊 File stats:\n• Size: {file_size:,} characters\n• Type: {file_extension.upper()}\n\n Text preview:\n{preview}\n\n{result}"

def handle_file_upload(files, original_filenames=None):
    """
    Process multiple files.
    
    Args:
        files: List of file paths (strings) or file objects
        original_filenames: Optional dict mapping file paths to original filenames
                          {file_path: original_filename}
    """
    global processed_files
    if not files or len(files) == 0:
        return "Please select at least one file to process."
    
    # Create mapping if not provided
    if original_filenames is None:
        original_filenames = {}
    
    results = []
    new_processed = []
    for file in files:
        if file is None:
            continue
        try:
            if isinstance(file, str):
                file_path = file
                file_name = original_filenames.get(file_path, os.path.basename(file))
            else:
                file_path = file.name if hasattr(file, 'name') else str(file)
                file_name = original_filenames.get(file_path, os.path.basename(file_path))
            if any(f['name'] == file_name for f in processed_files):
                results.append(f"SKIP: {file_name} - Already processed, skipping")
                continue
            result = process_uploaded_file(file, original_filename=file_name)
            results.append(f"SUCCESS: {file_name} - {result}")
            new_processed.append({
                'name': file_name,
                'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
        except Exception as e:
            file_name = original_filenames.get(file_path, os.path.basename(file)) if isinstance(file, str) else original_filenames.get(file_path, os.path.basename(file.name)) if hasattr(file, 'name') else str(file)
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


