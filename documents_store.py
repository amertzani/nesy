"""
Document Metadata Management Module
====================================

This module manages metadata about uploaded documents (name, size, type, 
number of facts extracted). It stores this information in documents_store.json.

Key Features:
- Only stores documents that contributed facts (facts_extracted > 0)
- Automatically cleans up documents without facts
- Tracks document statistics

Key Functions:
- add_document(): Add new document metadata (only if facts_extracted > 0)
- get_all_documents(): Get all document metadata
- cleanup_documents_without_facts(): Remove documents with 0 facts
- delete_document(): Delete a specific document

Storage:
- documents_store.json: JSON file with document metadata array

Author: Research Brain Team
Last Updated: 2025-01-15
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional

DOCUMENTS_FILE = "documents_store.json"

def load_documents() -> List[Dict]:
    """Load documents from storage"""
    try:
        if os.path.exists(DOCUMENTS_FILE):
            with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('documents', [])
        return []
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def save_documents(documents: List[Dict]):
    """Save documents to storage"""
    try:
        data = {
            'last_updated': datetime.now().isoformat(),
            'total_documents': len(documents),
            'documents': documents
        }
        with open(DOCUMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving documents: {e}")
        return False

def add_document(name: str, size: int, file_type: str, facts_extracted: int = 0) -> Dict:
    """
    Add a new document to storage.
    IMPORTANT: Only saves documents with facts_extracted > 0.
    Documents with 0 facts are NEVER saved.
    """
    # NEVER save documents with 0 facts
    if facts_extracted <= 0:
        print(f"⚠️  Skipping document {name}: has 0 facts (not saved)")
        return None
    
    documents = load_documents()
    
    # Check if document already exists
    existing = next((d for d in documents if d['name'] == name), None)
    if existing:
        # Update existing document
        existing['size'] = size
        existing['type'] = file_type
        existing['uploaded_at'] = datetime.now().isoformat()
        existing['facts_extracted'] = facts_extracted
        existing['status'] = 'completed'
        save_documents(documents)
        return existing
    
    # Create new document (only if facts_extracted > 0)
    new_doc = {
        'id': str(len(documents) + 1),
        'name': name,
        'type': file_type,
        'size': size,
        'uploaded_at': datetime.now().isoformat(),
        'status': 'completed',
        'facts_extracted': facts_extracted
    }
    
    documents.append(new_doc)
    save_documents(documents)
    return new_doc

def get_document_by_name(name: str) -> Optional[Dict]:
    """Get a document by name"""
    documents = load_documents()
    return next((d for d in documents if d['name'] == name), None)

def get_all_documents() -> List[Dict]:
    """Get all documents"""
    return load_documents()


def delete_document(document_id: str) -> bool:
    """Delete a document by ID"""
    documents = load_documents()
    original_count = len(documents)
    documents = [d for d in documents if d.get('id') != document_id]
    
    if len(documents) < original_count:
        save_documents(documents)
        return True
    return False

def cleanup_documents_without_facts() -> int:
    """
    PERMANENTLY remove documents that have facts_extracted=0 OR documents whose
    claimed facts don't actually exist in the knowledge graph.
    
    This ensures documents from previous sessions that don't have facts are removed,
    and also removes documents that claim to have facts but those facts were deleted
    or never actually added to the graph.
    
    Returns:
        Number of documents removed
    """
    documents = load_documents()
    original_count = len(documents)
    
    # Import knowledge graph to verify facts exist
    try:
        from knowledge import graph as kb_graph
        graph_fact_count = len(kb_graph)
    except:
        # If we can't load the graph, assume it's empty
        graph_fact_count = 0
        print("⚠️  Warning: Could not load knowledge graph for verification")
    
    # PERMANENTLY REMOVE all documents with facts_extracted = 0
    # OR documents that claim facts but the graph is empty/has fewer facts
    cleaned_documents = []
    removed_count = 0
    
    for doc in documents:
        facts_claimed = doc.get('facts_extracted', 0)
        
        if facts_claimed <= 0:
            # Document has 0 facts - PERMANENTLY REMOVE IT
            removed_count += 1
            print(f"   🗑️  Removing {doc.get('name', 'unknown')}: claims 0 facts")
        elif graph_fact_count == 0:
            # Graph is empty but document claims facts - REMOVE IT
            removed_count += 1
            print(f"   🗑️  Removing {doc.get('name', 'unknown')}: claims {facts_claimed} facts but graph is empty")
        else:
            # Document has facts - KEEP IT (we trust the count if graph has facts)
            cleaned_documents.append(doc)
    
    if removed_count > 0:
        save_documents(cleaned_documents)
        print(f"🧹 PERMANENTLY removed {removed_count} documents without valid facts (from {original_count} total)")
        print(f"✅ {len(cleaned_documents)} documents with facts remain")
        print(f"📊 Knowledge graph has {graph_fact_count} facts")
    
    return removed_count

