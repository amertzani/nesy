
# app.py - AsiminaM
import gradio as gr
import rdflib
import re
import os
import tempfile
from huggingface_hub import InferenceClient
import PyPDF2
from docx import Document
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import plotly.graph_objects as go
import plotly.express as px
from file_processing import handle_file_upload as fp_handle_file_upload
from knowledge import (
    show_graph_contents as kb_show_graph_contents,
    visualize_knowledge_graph as kb_visualize_knowledge_graph,
    import_knowledge_from_json_file as kb_import_json,
    save_knowledge_graph as kb_save_knowledge_graph,
    load_knowledge_graph as kb_load_knowledge_graph,
    graph as kb_graph,
    delete_all_knowledge as kb_delete_all_knowledge,
    add_to_graph as kb_add_to_graph
)
from knowledge import create_comprehensive_backup as kb_create_comprehensive_backup, BACKUP_FILE
from responses import respond as rqa_respond

# ==========================================================
#  🧠 1. Global Knowledge Graph with Persistent Storage
# ==========================================================
import json
import pickle
from datetime import datetime

# Storage file paths
KNOWLEDGE_FILE = "knowledge_graph.pkl"
BACKUP_FILE = "knowledge_backup.json"

graph = rdflib.Graph()

# Mapping of fact IDs to triples for editing operations
fact_index = {}

def import_knowledge_from_json_file(file):
    """Import knowledge facts from a JSON file (backup format or simple list).
    Supported formats:
    - { "metadata": {...}, "facts": [{subject,predicate,object,...}, ...] }
    - { "facts": [{subject,predicate,object}, ...] }
    - [ {subject,predicate,object}, ... ]
    Returns a status message about counts imported.
    """
    try:
        if file is None:
            return "⚠️ No file selected."

        file_path = file.name if hasattr(file, 'name') else str(file)
        if not os.path.exists(file_path):
            return f"⚠️ File not found: {file_path}"

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Normalize to list of fact dicts
        if isinstance(data, dict) and 'facts' in data:
            facts = data['facts']
        elif isinstance(data, list):
            facts = data
        else:
            return "❌ Unsupported JSON structure. Expect an object with 'facts' or a list of facts."

        added = 0
        skipped = 0
        for fact in facts:
            try:
                subject = fact.get('subject') or fact.get('full_subject')
                predicate = fact.get('predicate') or fact.get('full_predicate')
                obj = fact.get('object') or fact.get('full_object')
                if not subject or not predicate or obj is None:
                    skipped += 1
                    continue
                # Use short forms; ensure URNs
                s_ref = rdflib.URIRef(subject if str(subject).startswith('urn:') else f"urn:{subject}")
                p_ref = rdflib.URIRef(predicate if str(predicate).startswith('urn:') else f"urn:{predicate}")
                o_lit = rdflib.Literal(obj)
                graph.add((s_ref, p_ref, o_lit))
                added += 1
            except Exception:
                skipped += 1

        save_knowledge_graph()
        return f"✅ Imported {added} facts. Skipped {skipped}. Total facts: {len(graph)}."
    except Exception as e:
        return f"❌ Import failed: {e}"

def handle_import_json(file):
    """Gradio handler: import JSON knowledge and report status"""
    status = import_knowledge_from_json_file(file)
    return status

def save_knowledge_graph():
    """Save the knowledge graph to persistent storage"""
    try:
        # Save as pickle for RDF graph
        with open(KNOWLEDGE_FILE, 'wb') as f:
            pickle.dump(graph, f)
        
        # Also save a human-readable backup
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "total_facts": len(graph),
            "facts": []
        }
        
        for i, (s, p, o) in enumerate(graph):
            backup_data["facts"].append({
                "id": i+1,
                "subject": str(s),
                "predicate": str(p), 
                "object": str(o)
            })
        
        with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(graph)} facts to persistent storage")
        return f" Saved {len(graph)} facts to storage"
        
    except Exception as e:
        error_msg = f" Error saving knowledge: {e}"
        print(error_msg)
        return error_msg

def load_knowledge_graph():
    """Load the knowledge graph from persistent storage"""
    global graph
    
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, 'rb') as f:
                graph = pickle.load(f)
            print(f"📂 Loaded {len(graph)} facts from storage")
            return f"📂 Loaded {len(graph)} facts from storage"
        else:
            print("📂 No existing knowledge file found, starting fresh")
            return "📂 No existing knowledge file found, starting fresh"
            
    except Exception as e:
        error_msg = f" Error loading knowledge: {e}"
        print(error_msg)
        return error_msg

def create_and_get_backup():
    """Create a comprehensive backup and return the file path"""
    try:
        print(f"Creating backup for graph with {len(graph)} facts")
        
        # Create comprehensive backup
        create_comprehensive_backup()
        
        # Verify the backup was created and contains data
        if os.path.exists(BACKUP_FILE):
            with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
                backup_content = json.load(f)
                fact_count = backup_content.get('metadata', {}).get('total_facts', 0)
                print(f" Knowledge backup created with {fact_count} facts")
                
                if fact_count == 0:
                    print("⚠️ Warning: Backup file created but contains no facts")
                    # Create a backup even if empty to show the structure
                    create_empty_backup_structure()
                    
                # Return both the file path and status message
                return BACKUP_FILE, f" Backup created successfully with {fact_count} facts!"
        else:
            print(" Backup file was not created")
            return None, " Failed to create backup file"
        
    except Exception as e:
        print(f" Error creating backup: {e}")
        # Create a minimal backup file even if there's an error
        create_error_backup(str(e))
        return BACKUP_FILE, f"⚠️ Backup created with errors: {e}"

def verify_backup_contents():
    """Verify and display backup file contents"""
    try:
        if not os.path.exists(BACKUP_FILE):
            return " No backup file found. Click 'Create Knowledge Backup' first."
        
        with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        metadata = backup_data.get('metadata', {})
        facts = backup_data.get('facts', [])
        
        result = f"📊 **Backup File Verification:**\n\n"
        result += f"**File:** `{BACKUP_FILE}`\n"
        result += f"**Size:** {os.path.getsize(BACKUP_FILE):,} bytes\n"
        result += f"**Created:** {metadata.get('timestamp', 'Unknown')}\n"
        result += f"**Total Facts:** {metadata.get('total_facts', 0)}\n"
        result += f"**Backup Type:** {metadata.get('backup_type', 'Unknown')}\n\n"
        
        if facts:
            result += f"**Sample Facts (first 5):**\n"
            for i, fact in enumerate(facts[:5]):
                result += f"{i+1}. {fact.get('subject')} {fact.get('predicate')} {fact.get('object')}\n"
            
            if len(facts) > 5:
                result += f"\n... and {len(facts) - 5} more facts\n"
        else:
            result += "**⚠️ No facts found in backup file!**\n"
        
        return result
        
    except Exception as e:
        return f" Error verifying backup: {e}"

def get_knowledge_file():
    """Return the knowledge backup file for download (legacy function)"""
    file_path, status = create_and_get_backup()
    return file_path

def create_comprehensive_backup():
    """Create a comprehensive backup file with all knowledge facts"""
    global graph
    
    try:
        print(f"Creating backup for graph with {len(graph)} facts")
        
        # Create detailed backup data
        backup_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_facts": len(graph),
                "backup_type": "comprehensive_knowledge_base",
                "graph_size": len(graph)
            },
            "facts": []
        }
        
        # Add all facts from the graph
        fact_count = 0
        for i, (s, p, o) in enumerate(graph):
            # Clean up the subject, predicate, object for better readability
            subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
            predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
            object_val = str(o)
            
            backup_data["facts"].append({
                "id": i + 1,
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "full_subject": str(s),
                "full_predicate": str(p),
                "full_object": str(o)
            })
            fact_count += 1
        
        # Update the fact count in metadata
        backup_data["metadata"]["total_facts"] = fact_count
        
        # Save as JSON
        with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        # Also create a human-readable text version
        create_readable_backup()
        
        print(f"Created comprehensive backup with {fact_count} facts")
        
    except Exception as e:
        print(f" Error creating comprehensive backup: {e}")
        # Create a minimal backup even if there's an error
        create_error_backup(str(e))

def create_empty_backup_structure():
    """Create a backup file structure even when no facts exist"""
    try:
        backup_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_facts": 0,
                "backup_type": "empty_knowledge_base",
                "message": "No facts found in knowledge graph"
            },
            "facts": [],
            "instructions": {
                "how_to_add_knowledge": [
                    "1. Add text directly using the 'Add Knowledge from Text' box",
                    "2. Upload documents (PDF, DOCX, TXT, CSV) using the file upload",
                    "3. Process files to extract knowledge automatically",
                    "4. Use 'Save Knowledge' to persist your data"
                ]
            }
        }
        
        with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        print(" Created empty backup structure")
        
    except Exception as e:
        print(f" Error creating empty backup: {e}")

def create_error_backup(error_message):
    """Create a backup file when there's an error"""
    try:
        backup_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_facts": 0,
                "backup_type": "error_backup",
                "error": error_message
            },
            "facts": [],
            "note": "An error occurred while creating the backup. Please try again."
        }
        
        with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        print(f" Created error backup: {error_message}")
        
    except Exception as e:
        print(f" Error creating error backup: {e}")

def create_readable_backup():
    """Create a human-readable text backup"""
    global graph
    
    try:
        print(f"Creating readable backup for {len(graph)} facts")
        
        # Create readable text file
        readable_text = f"# Knowledge Base Backup\n"
        readable_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        readable_text += f"Total Facts: {len(graph)}\n\n"
        
        if len(graph) == 0:
            readable_text += "No facts in knowledge base.\n\n"
            readable_text += "## How to Add Knowledge:\n"
            readable_text += "1. Add text directly using the 'Add Knowledge from Text' box\n"
            readable_text += "2. Upload documents (PDF, DOCX, TXT, CSV) using the file upload\n"
            readable_text += "3. Process files to extract knowledge automatically\n"
            readable_text += "4. Use 'Save Knowledge' to persist your data\n"
        else:
            # Group facts by subject for better organization
            facts_by_subject = {}
            fact_count = 0
            
            for s, p, o in graph:
                subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
                predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
                object_val = str(o)
                
                if subject not in facts_by_subject:
                    facts_by_subject[subject] = []
                facts_by_subject[subject].append(f"{predicate}: {object_val}")
                fact_count += 1
            
            # Add organized facts
            for subject, facts in facts_by_subject.items():
                readable_text += f"## {subject}\n"
                for fact in facts:
                    readable_text += f"- {fact}\n"
                readable_text += "\n"
            
            readable_text += f"\n## Summary\n"
            readable_text += f"Total facts processed: {fact_count}\n"
            readable_text += f"Unique subjects: {len(facts_by_subject)}\n"
        
        # Save readable version
        with open("knowledge_readable.txt", 'w', encoding='utf-8') as f:
            f.write(readable_text)
        
        print(f" Created readable backup: knowledge_readable.txt with {len(graph)} facts")
        
    except Exception as e:
        print(f" Error creating readable backup: {e}")
        # Create a minimal readable backup even if there's an error
        try:
            error_text = f"# Knowledge Base Backup (Error)\n"
            error_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            error_text += f"Error: {e}\n"
            error_text += f"Total Facts: {len(graph)}\n"
            
            with open("knowledge_readable.txt", 'w', encoding='utf-8') as f:
                f.write(error_text)
            print(" Created error-readable backup")
        except:
            print(" Failed to create even error-readable backup")

def debug_backup_process():
    """Debug function to help troubleshoot backup issues"""
    global graph
    
    debug_info = f" **Backup Debug Information:**\n\n"
    
    # Check graph state
    debug_info += f"**Graph State:**\n"
    debug_info += f"• Graph length: {len(graph)}\n"
    debug_info += f"• Graph type: {type(graph)}\n"
    debug_info += f"• Graph empty: {len(graph) == 0}\n\n"
    
    # Check files
    debug_info += f"**File Status:**\n"
    debug_info += f"• Knowledge file exists: {os.path.exists(KNOWLEDGE_FILE)}\n"
    debug_info += f"• Backup file exists: {os.path.exists(BACKUP_FILE)}\n"
    debug_info += f"• Readable file exists: {os.path.exists('knowledge_readable.txt')}\n\n"
    
    # Show sample facts if any exist
    if len(graph) > 0:
        debug_info += f"**Sample Facts (first 5):**\n"
        fact_count = 0
        for s, p, o in graph:
            if fact_count >= 5:
                break
            debug_info += f"• {s} {p} {o}\n"
            fact_count += 1
        debug_info += "\n"
    else:
        debug_info += f"**No facts in graph**\n\n"
    
    # Test backup creation
    debug_info += f"**Testing Backup Creation:**\n"
    try:
        create_comprehensive_backup()
        debug_info += f"• Backup creation:  Success\n"
        
        if os.path.exists(BACKUP_FILE):
            with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
                fact_count = backup_data.get('metadata', {}).get('total_facts', 0)
                debug_info += f"• Facts in backup: {fact_count}\n"
                debug_info += f"• Backup metadata: {backup_data.get('metadata', {})}\n"
        else:
            debug_info += f"• Backup file:  Not created\n"
            
    except Exception as e:
        debug_info += f"• Backup creation:  Error: {e}\n"
    
    return debug_info

def show_storage_info():
    """Show information about where files are stored"""
    info = f"📁 **Storage Information:**\n\n"
    
    # Check if files exist
    pkl_exists = os.path.exists(KNOWLEDGE_FILE)
    json_exists = os.path.exists(BACKUP_FILE)
    
    info += f"**Primary Storage:** `{KNOWLEDGE_FILE}` {' Exists' if pkl_exists else ' Not found'}\n"
    info += f"**Backup Storage:** `{BACKUP_FILE}` {' Exists' if json_exists else ' Not found'}\n"
    info += f"**Readable Backup:** `knowledge_readable.txt` {' Exists' if os.path.exists('knowledge_readable.txt') else ' Not found'}\n\n"
    
    if pkl_exists:
        file_size = os.path.getsize(KNOWLEDGE_FILE)
        info += f"**File Size:** {file_size:,} bytes\n"
    
    info += f"**Total Facts:** {len(graph)}\n\n"
    
    info += "**How to Access:**\n"
    info += "• On Hugging Face Spaces: Files are in `/home/user/app/`\n"
    info += "• On Local Machine: Files are in your project folder\n"
    info += "• Use ' Download Knowledge' button to get the JSON backup\n"
    
    return info


def extract_triples(text):
    """
    Enhanced extraction for better knowledge extraction from documents.
    Uses improved pattern matching and entity recognition.
    """
    triples = []
    
    print(f"Extracting knowledge from {len(text)} characters...")
    
    # Extract entities (people, organizations, locations, dates)
    entities = extract_entities(text)
    for entity in entities:
        triples.append((entity, 'type', 'entity'))
    
    # Extract structured data (key-value pairs)
    triples.extend(extract_structured_triples(text))
    
    # Extract regular sentences with improved parsing
    triples.extend(extract_regular_triples_improved(text, entities))
    
    # Also try original extraction as backup for coverage
    triples.extend(extract_regular_triples(text))
    
    # Remove duplicates and validate
    unique_triples = []
    for s, p, o in triples:
        if s and p and o and len(s) > 2 and len(p) > 1 and len(o) > 2:
            # Clean and validate
            s = s.strip()[:100]  # Limit length
            p = p.strip()[:50]
            o = o.strip()[:200]
            if (s, p, o) not in unique_triples:
                unique_triples.append((s, p, o))
    
    print(f"Total extracted {len(unique_triples)} unique triples")
    for i, (s, p, o) in enumerate(unique_triples[:10]):
        print(f"  {i+1}. {s} {p} {o}")
    
    return unique_triples

def extract_entities(text):
    """Extract named entities (people, organizations, locations, etc.)"""
    entities = []
    
    # Capitalized word patterns (likely proper nouns)
    capitalized_words = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', text)
    entities.extend(capitalized_words)
    
    # Extract organizations (typical suffixes)
    org_patterns = [
        r'([A-Z][a-zA-Z\s]+)\s+(Inc|Ltd|LLC|Corp|Corporation|Company|Co\.|Ltd\.)',
        r'([A-Z][a-zA-Z\s]+)\s+(University|Institute|Lab|Laboratory)',
    ]
    for pattern in org_patterns:
        matches = re.findall(pattern, text)
        entities.extend([m[0].strip() for m in matches])
    
    # Extract locations (cities, countries)
    location_keywords = ['in ', 'at ', 'near ', 'from ']
    for keyword in location_keywords:
        pattern = f'{keyword}([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        matches = re.findall(pattern, text)
        entities.extend(matches)
    
    # Extract dates
    dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', text)
    entities.extend(dates)
    
    # Remove duplicates and clean
    entities = list(set([e.strip() for e in entities if len(e.strip()) > 3]))
    return entities[:50]  # Limit to top 50

def extract_regular_triples_improved(text, entities):
    """Improved extraction with better sentence parsing and entity linking"""
    triples = []
    
    # Split into sentences
    sentences = re.split(r'[.!?\n]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:  # Skip very short sentences
            continue
        
        # Try improved patterns
        improved_patterns = [
            # Subject-Verb-Object patterns
            (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(is|are|was|were|becomes|represents|means|refers to|denotes)\s+(.+)', 'relates to'),
            (r'([A-Z][a-zA-Z\s]+)\s+(uses|employs|utilizes|applies)\s+(.+)', 'uses'),
            (r'([A-Z][a-zA-Z\s]+)\s+(develops|created|designed|implemented)\s+(.+)', 'creates'),
            (r'([A-Z][a-zA-Z\s]+)\s+(requires|needs|demands)\s+(.+)', 'requires'),
            (r'([A-Z][a-zA-Z\s]+)\s+(enables|allows|permits)\s+(.+)', 'enables'),
            (r'([A-Z][a-zA-Z\s]+)\s+(affects|impacts|influences|affects)\s+(.+)', 'affects'),
            
            # Research/technical patterns
            (r'([A-Z][a-zA-Z\s]+)\s+(found|discovered|identified|observed|detected)\s+(.+)', 'discovered'),
            (r'([A-Z][a-zA-Z\s]+)\s+(studies|analyzes|examines|investigates)\s+(.+)', 'studies'),
            (r'([A-Z][a-zA-Z\s]+)\s+(proposes|suggests|recommends)\s+(.+)', 'proposes'),
            (r'([A-Z][a-zA-Z\s]+)\s+(results in|leads to|causes)\s+(.+)', 'causes'),
            
            # Relationships
            (r'([A-Z][a-zA-Z\s]+)\s+(works with|collaborates with|partnered with)\s+(.+)', 'works with'),
            (r'([A-Z][a-zA-Z\s]+)\s+(located in|based in|situated in)\s+(.+)', 'located in'),
        ]
        
        for pattern, predicate in improved_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                groups = match.groups()
                subject = groups[0].strip() if len(groups) > 0 else ''
                object_val = groups[-1].strip() if len(groups) > 1 else ''
                
                # Clean up
                subject = re.sub(r'^(the|a|an)\s+', '', subject, flags=re.IGNORECASE).strip()
                object_val = re.sub(r'^(the|a|an)\s+', '', object_val, flags=re.IGNORECASE).strip()
                
                if subject and object_val and len(subject) > 3 and len(object_val) > 3:
                    triples.append((subject, predicate, object_val))
                    break
        
        # Also extract simple clauses with 'that', 'which', 'who'
        clause_patterns = [
            r'([A-Z][a-zA-Z\s]+)\s+which\s+(.+)',
            r'([A-Z][a-zA-Z\s]+)\s+that\s+(.+)',
            r'([A-Z][a-zA-Z\s]+)\s+who\s+(.+)',
        ]
        for pattern in clause_patterns:
            match = re.search(pattern, sentence)
            if match:
                subject = match.group(1).strip()
                description = match.group(2).strip()
                if subject and description and len(subject) > 3 and len(description) > 3:
                    triples.append((subject, 'has property', description[:150]))
    
    return triples

def extract_structured_triples(text):
    """Extract triples from structured data (key-value pairs, tables, etc.)"""
    triples = []
    lines = text.split('\n')
    
    # General patterns for structured data extraction
    patterns = [
        # Date patterns
        (r'date\s*:?\s*([0-9\/\-\.]+)', 'date', 'is'),
        (r'time\s*:?\s*([0-9:]+)', 'time', 'is'),
        (r'created\s*:?\s*([0-9\/\-\.]+)', 'created_date', 'is'),
        (r'modified\s*:?\s*([0-9\/\-\.]+)', 'modified_date', 'is'),
        
        # ID and reference patterns
        (r'id\s*:?\s*([A-Z0-9\-]+)', 'id', 'is'),
        (r'number\s*:?\s*([A-Z0-9\-]+)', 'number', 'is'),
        (r'code\s*:?\s*([A-Z0-9\-]+)', 'code', 'is'),
        (r'reference\s*:?\s*([A-Z0-9\-]+)', 'reference', 'is'),
        
        # Name and title patterns
        (r'name\s*:?\s*([A-Za-z\s&.,]+)', 'name', 'is'),
        (r'title\s*:?\s*([A-Za-z\s&.,]+)', 'title', 'is'),
        (r'company\s*:?\s*([A-Za-z\s&.,]+)', 'company', 'is'),
        (r'organization\s*:?\s*([A-Za-z\s&.,]+)', 'organization', 'is'),
        
        # Contact patterns
        (r'email\s*:?\s*([A-Za-z0-9@\.\-]+)', 'email', 'is'),
        (r'phone\s*:?\s*([0-9\s\-\+\(\)]+)', 'phone', 'is'),
        (r'address\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'address', 'is'),
        
        # Description patterns
        (r'description\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'description', 'is'),
        (r'type\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'type', 'is'),
        (r'category\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'category', 'is'),
        (r'status\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'status', 'is'),
        
        # Location patterns
        (r'location\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'location', 'is'),
        (r'department\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'department', 'is'),
        (r'section\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'section', 'is'),
        
        # Amount patterns
        (r'amount\s*:?\s*\$?([0-9,]+\.?[0-9]*)', 'amount', 'is'),
        (r'total\s*:?\s*\$?([0-9,]+\.?[0-9]*)', 'total', 'is'),
        (r'price\s*:?\s*\$?([0-9,]+\.?[0-9]*)', 'price', 'is'),
        (r'cost\s*:?\s*\$?([0-9,]+\.?[0-9]*)', 'cost', 'is'),
    ]
    
    for line in lines:
        line = line.strip()
        if len(line) < 5:
            continue
            
        for pattern, subject, predicate in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value and len(value) > 1:
                    triples.append((subject, predicate, value))
                    break  # Only one match per line
    
    # General key-value pair extraction
    kv_patterns = [
        # Standard colon format
        r'([A-Za-z\s]+):\s*([A-Za-z0-9\s\$\-\.\/,]+)',
        # Equals format
        r'([A-Za-z\s]+)\s*=\s*([A-Za-z0-9\s\$\-\.\/,]+)',
        # Dash format
        r'([A-Za-z\s]+)\s*-\s*([A-Za-z0-9\s\$\-\.\/,]+)',
    ]
    
    for line in lines:
        for pattern in kv_patterns:
            match = re.search(pattern, line)
            if match:
                key = match.group(1).strip().lower().replace(' ', '_')
                value = match.group(2).strip()
                if len(key) > 2 and len(value) > 1:
                    triples.append((key, 'is', value))
    
    # Extract any line that looks like "Label: Value" or "Label Value"
    for line in lines:
        line = line.strip()
        if ':' in line and len(line) > 10:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                if len(key) > 2 and len(value) > 1 and not key.isdigit():
                    # Clean the key
                    clean_key = re.sub(r'[^A-Za-z0-9\s]', '', key).strip().lower().replace(' ', '_')
                    if clean_key:
                        triples.append((clean_key, 'is', value))
    
        print(f"Structured extraction found {len(triples)} triples")
    return triples

def extract_regular_triples(text):
    """Extract triples using regular sentence patterns"""
    triples = []
    
    # Clean and split text into sentences
    sentences = re.split(r"[.?!\n]", text)
    print(f" Found {len(sentences)} sentences for regular extraction")
    
    # English extraction patterns
    patterns = [
        # Basic patterns
        r"\s+(is|are|was|were)\s+",
        r"\s+(has|have|had)\s+",
        r"\s+(uses|used|using)\s+",
        r"\s+(creates|created|creating)\s+",
        r"\s+(develops|developed|developing)\s+",
        r"\s+(leads|led|leading)\s+",
        r"\s+(affects|affected|affecting)\s+",
        r"\s+(contains|contained|containing)\s+",
        r"\s+(includes|included|including)\s+",
        r"\s+(involves|involved|involving)\s+",
        r"\s+(requires|required|requiring)\s+",
        r"\s+(produces|produced|producing)\s+",
        r"\s+(causes|caused|causing)\s+",
        r"\s+(results|resulted|resulting)\s+",
        r"\s+(enables|enabled|enabling)\s+",
        r"\s+(provides|provided|providing)\s+",
        r"\s+(supports|supported|supporting)\s+",
        r"\s+(allows|allowed|allowing)\s+",
        r"\s+(helps|helped|helping)\s+",
        r"\s+(improves|improved|improving)\s+",
        r"\s+(located|situated|found)\s+",
        r"\s+(consists|composed|made)\s+",
        r"\s+(operates|functions|works)\s+",
        r"\s+(generates|creates|produces)\s+",
        r"\s+(transforms|converts|changes)\s+",
        r"\s+(connects|links|relates)\s+",
        r"\s+(influences|impacts|affects)\s+",
        r"\s+(depends|relies|based)\s+",
        r"\s+(represents|symbolizes|stands)\s+",
        r"\s+(describes|explains|defines)\s+",
        r"\s+(refers|referring|referenced)\s+",
        r"\s+(concerns|concerning|concerned)\s+",
        r"\s+(relates|relating|related)\s+",
        r"\s+(analyzes|analyzing|analyzed)\s+",
        r"\s+(examines|examining|examined)\s+",
        r"\s+(studies|studying|studied)\s+",
        r"\s+(checks|checking|checked)\s+",
        r"\s+(manages|managing|managed)\s+",
        r"\s+(organizes|organizing|organized)\s+",
        r"\s+(coordinates|coordinating|coordinated)\s+",
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short sentences
            continue
            
        # Try each pattern
        for pattern in patterns:
            parts = re.split(pattern, sentence, maxsplit=1)
            if len(parts) == 3:
                subj, pred, obj = parts
                subj = subj.strip()
                pred = pred.strip()
                obj = obj.strip()
                
                # Clean up the parts
                if subj and pred and obj and len(subj) > 2 and len(obj) > 2:
                    # Remove common prefixes/suffixes
                    subj = re.sub(r'^(the|a|an)\s+', '', subj, flags=re.IGNORECASE)
                    obj = re.sub(r'^(the|a|an)\s+', '', obj, flags=re.IGNORECASE)
                    
                    triples.append((subj, pred, obj))
                    break  # Found a match, move to next sentence
    
    print(f"Regular extraction found {len(triples)} triples")
    return triples


def add_to_graph(text):
    """
    Parse text into triples and add them to the RDF graph.
    """
    new_triples = extract_triples(text)
    for s, p, o in new_triples:
        graph.add((rdflib.URIRef(f"urn:{s}"), rdflib.URIRef(f"urn:{p}"), rdflib.Literal(o)))
    
    # Automatically save after adding knowledge
    save_result = save_knowledge_graph()
    
    return f" Added {len(new_triples)} new triples. Total facts stored: {len(graph)}.\n{save_result}"


def retrieve_context(question, limit=10):
    """
    Retrieve RDF facts related to keywords in the question with better matching.
    """
    matches = []
    qwords = question.lower().split()
    
    # Remove common words that don't add meaning
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why', 'who'}
    qwords = [w for w in qwords if w not in stop_words and len(w) > 2]
    
    print(f"Searching for: {qwords}")
    
    # Score matches by relevance
    scored_matches = []
    
    for s, p, o in graph:
        subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
        predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
        object_val = str(o)
        
        fact_text = f"{subject} {predicate} {object_val}".lower()
        
        # Calculate relevance score
        score = 0
        for word in qwords:
            if word in fact_text:
                score += 1
                # Bonus for exact matches
                if word == subject.lower() or word == predicate.lower():
                    score += 2
        
        if score > 0:
            scored_matches.append((score, f"{subject} {predicate} {object_val}"))
    
    # Sort by relevance score (highest first)
    scored_matches.sort(key=lambda x: x[0], reverse=True)
    
    # Take top matches
    matches = [match[1] for match in scored_matches[:limit]]
    
    print(f"Found {len(matches)} relevant facts")
    
    if matches:
        result = "**Relevant Knowledge:**\n"
        for i, match in enumerate(matches, 1):
            result += f"{i}. {match}\n"
        return result
    else:
        return "**No directly relevant facts found.**\n\nTry asking about topics that might be in your knowledge base, or add more knowledge first!"

def handle_add_knowledge(text):
    """Handle adding knowledge from text input"""
    if not text or text.strip() == "":
        return "Please enter some text to extract knowledge from.", ""
    
    print(f"Adding knowledge from text input: {text[:1000]}...")
    result = kb_add_to_graph(text)
    print(f"Knowledge added: {result}")
    
    # Return enhanced status with current knowledge count
    total_facts = len(kb_graph)
    status = f"**Knowledge Extracted Successfully!**\n\n{result}\n\n**Current Knowledge Base:** {total_facts} facts"
    
    # Return status and empty string to clear the input box
    return status, ""

def show_graph_contents():
    """
    Return all current triples as readable text with better formatting.
    """
    print(f"Showing graph contents. Total triples: {len(graph)}")
    
    if len(graph) == 0:
        return "**Knowledge Graph Status: EMPTY**\n\n**How to build your knowledge base:**\n1. **Add text directly** - Paste any text in the 'Add Knowledge from Text' box above\n2. **Upload documents** - Use the file upload to process PDF, DOCX, TXT, CSV files\n3. **Extract facts** - The system will automatically extract knowledge from your content\n4. **Build knowledge** - Add more text or files to expand your knowledge base\n5. **Save knowledge** - Use 'Save Knowledge' to persist your data\n\n**Start by adding some text or uploading a document!**"
    
    # Organize facts by subject for better readability
    facts_by_subject = {}
    all_facts = []
    
    for s, p, o in graph:
        subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
        predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
        object_val = str(o)
        
        fact_text = f"{subject} {predicate} {object_val}"
        all_facts.append(fact_text)
        
        if subject not in facts_by_subject:
            facts_by_subject[subject] = []
        facts_by_subject[subject].append(f"{predicate} {object_val}")
    
    # Create organized display
    result = f"**Knowledge Graph Overview**\n"
    result += f"**Total Facts:** {len(graph)}\n"
    result += f"**Unique Subjects:** {len(facts_by_subject)}\n\n"
    
    # Show facts organized by subject
    result += "## **Knowledge by Subject:**\n\n"
    
    for i, (subject, facts) in enumerate(facts_by_subject.items()):
        if i >= 10:  # Limit to first 10 subjects for readability
            remaining = len(facts_by_subject) - 10
            result += f"... and {remaining} more subjects\n"
            break
            
        result += f"**{subject}:**\n"
        for fact in facts:
            result += f"  • {fact}\n"
        result += "\n"
    
    # Show all facts in a simple list
    result += "## **All Facts:**\n\n"
    for i, fact in enumerate(all_facts[:20]):  # Show first 20 facts
        result += f"{i+1}. {fact}\n"
    
    if len(all_facts) > 20:
        result += f"\n... and {len(all_facts) - 20} more facts"
    
    # Intentionally omit search suggestions to keep the view focused on facts
    
    return result

def list_facts_for_editing():
    """Return a dropdown update with choices and build index"""
    from knowledge import fact_index
    options = []
    for i, (s, p, o) in enumerate(list(kb_graph), start=1):
        subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
        predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
        object_val = str(o)
        label = f"{i}. {subject} {predicate} {object_val}"
        options.append(label)
        fact_index[i] = (s, p, o)
    status = f"Loaded {len(options)} facts for editing"
    return gr.update(choices=options, value=None), status

def load_fact_fields(fact_label):
    """Given a dropdown label, return subject, predicate, object fields"""
    from knowledge import load_fact_by_label
    if not fact_label:
        return "", "", ""
    triple = load_fact_by_label(fact_label)
    if not triple:
        return "", "", ""
    s, p, o = triple
    subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
    predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
    object_val = str(o)
    return subject, predicate, object_val

def update_fact(fact_label, new_subject, new_predicate, new_object):
    """Update a single fact by ID and persist changes"""
    from knowledge import fact_index
    if not fact_label:
        return "⚠️ Select a fact first.", gr.update()
    try:
        fact_id = int(fact_label.split('.', 1)[0].strip())
        old = fact_index.get(fact_id)
        if not old:
            return "⚠️ Fact not found. Click Refresh Facts and try again.", gr.update()
        s_old, p_old, o_old = old
        # Remove old triple
        kb_graph.remove((s_old, p_old, o_old))
        # Add new triple
        s_new = rdflib.URIRef(f"urn:{new_subject.strip()}")
        p_new = rdflib.URIRef(f"urn:{new_predicate.strip()}")
        o_new = rdflib.Literal(new_object.strip())
        kb_graph.add((s_new, p_new, o_new))
        # Persist
        kb_save_knowledge_graph()
        # Refresh list
        options_update, _ = list_facts_for_editing()
        return "✅ Fact updated and saved.", options_update
    except Exception as e:
        return f"❌ Update failed: {e}", gr.update()

def delete_fact(fact_label):
    """Delete a single fact by ID and persist changes"""
    from knowledge import fact_index
    if not fact_label:
        return "⚠️ Select a fact first.", gr.update()
    try:
        fact_id = int(fact_label.split('.', 1)[0].strip())
        old = fact_index.get(fact_id)
        if not old:
            return "⚠️ Fact not found. Click Refresh Facts and try again.", gr.update()
        kb_graph.remove(old)
        kb_save_knowledge_graph()
        options_update, _ = list_facts_for_editing()
        return "🗑️ Fact deleted.", options_update
    except Exception as e:
        return f"❌ Delete failed: {e}", gr.update()

def visualize_knowledge_graph():
    """Create an interactive network visualization of the knowledge graph"""
    global graph
    
    if len(graph) == 0:
        return "<p>No knowledge in graph. Add some text or upload a document first!</p>"
    
    try:
        print(f"Creating interactive network visualization for {len(graph)} facts...")
        
        # Create a NetworkX graph
        G = nx.Graph()
        fact_data = {}
        
        # Add nodes and edges from RDF triples
        for s, p, o in graph:
            subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
            predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
            object_val = str(o)
            
            # Truncate for display
            subject_short = (subject[:30] + "...") if len(subject) > 30 else subject
            object_short = (object_val[:30] + "...") if len(object_val) > 30 else object_val
            
            # Add nodes
            if subject not in G:
                G.add_node(subject, display=subject_short, node_type='subject')
            if object_val not in G:
                G.add_node(object_val, display=object_short, node_type='object')
            
            # Add edge
            G.add_edge(subject, object_val, label=predicate)
            fact_data[(subject, object_val)] = f"{subject} {predicate} {object_val}"
        
            print(f"NetworkX graph created with {len(G.nodes())} nodes")
        
        # Limit to top 40 nodes by degree for better visualization
        if len(G.nodes()) > 40:
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:40]
            top_node_names = [node[0] for node in top_nodes]
            G = G.subgraph(top_node_names)
            print(f"Showing top 40 nodes out of {len(graph)} total")
        
        # Get spring layout
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        
        # Normalize positions to fit in canvas
        import numpy as np
        x_positions = [pos[n][0] for n in G.nodes()]
        y_positions = [pos[n][1] for n in G.nodes()]
        
        x_min, x_max = min(x_positions), max(x_positions)
        y_min, y_max = min(y_positions), max(y_positions)
        
        # Scale to fit
        scale = min(500 / (x_max - x_min), 400 / (y_max - y_min)) if (x_max - x_min) > 0 and (y_max - y_min) > 0 else 50
        offset_x = 350
        offset_y = 300
        
        # Create SVG visualization
        svg_elements = []
        
        # Add edges first (so they appear behind nodes)
        for edge in G.edges():
            x1 = pos[edge[0]][0] * scale + offset_x
            y1 = pos[edge[0]][1] * scale + offset_y
            x2 = pos[edge[1]][0] * scale + offset_x
            y2 = pos[edge[1]][1] * scale + offset_y
            
            edge_data = G[edge[0]][edge[1]]
            label = edge_data.get('label', 'has')
            fact = fact_data.get((edge[0], edge[1]), f"{edge[0]} {label} {edge[1]}")
            
            svg_elements.append(f"""
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" 
                  stroke="#999" stroke-width="2" opacity="0.5" 
                  data-label="{label}"
                  onmouseover="this.style.stroke='#2196F3'; this.style.strokeWidth='3'; this.style.opacity='0.8'"
                  onmouseout="this.style.stroke='#999'; this.style.strokeWidth='2'; this.style.opacity='0.5'">
                <title>{label}</title>
            </line>
            """)
        
        # Add nodes
        node_info = []
        for i, node in enumerate(G.nodes()):
            x = pos[node][0] * scale + offset_x
            y = pos[node][1] * scale + offset_y
            display_name = G.nodes[node].get('display', node)
            node_type = G.nodes[node].get('node_type', 'unknown')
            
            # Color by type
            if node_type == 'subject':
                color = '#4CAF50'
            elif node_type == 'object':
                color = '#2196F3'
            else:
                color = '#546E7A'  # blue-grey
            
            # Get neighbors count
            neighbors = list(G.neighbors(node))
            neighbor_count = len(neighbors)
            
            node_info.append(f"""
            <circle cx="{x}" cy="{y}" r="{max(40, min(30, neighbor_count * 2 + 20))}" 
                    fill="{color}" stroke="#fff" stroke-width="2"
                    data-node="{i}" data-name="{display_name}" data-count="{neighbor_count}"
                    onmouseover="showNodeInfo(this)"
                    onmouseout="hideNodeInfo(this)"
                    onclick="showNodeDetails('{node}', '{display_name}', {neighbor_count})">
                <title>{display_name} ({neighbor_count} connections)</title>
            </circle>
            <text x="{x}" y="{y+6}" text-anchor="middle" font-size="15" font-weight="bold" fill="#000" 
                  pointer-events="none">{display_name[:15]}</text>
            """)
        
        # Combine all elements
        svg_content = '\n'.join(svg_elements + node_info)
        
        # Create complete HTML with interactive features
        html = f"""
        <div style="width: 100%; min-height: 700px; max-height: 800px; background: white; border: 2px solid #ddd; border-radius: 10px; padding: 20px; position: relative; overflow: auto;">
            <div style="position: absolute; top: 10px; left: 10px; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); z-index: 10;">
                <h3 style="margin: 0; font-size: 14px;">📊 Knowledge Network</h3>
                <p style="margin: 5px 0; font-size: 11px; color: #666;">Facts: {len(graph)} | Nodes: {len(G.nodes())} | Links: {len(G.edges())}</p>
            </div>
            
            <div id="nodeInfo" style="position: absolute; top: 10px; right: 10px; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); z-index: 10; display: none; max-width: 250px;">
                <div id="nodeInfoContent"></div>
            </div>
            
            <div style="position: absolute; bottom: 10px; left: 10px; background: #e3f2fd; padding: 8px; border-radius: 5px; font-size: 11px;">
                💡 Hover over nodes for details | Click to explore relationships
            </div>
            
            <div style="position: absolute; bottom: 50px; left: 10px; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); font-size: 10px; min-width: 150px;">
                <strong>Node Colors:</strong><br>
                <span style="color: #4CAF50;">●</span> Green = Subjects<br>
                <span style="color: #2196F3;">●</span> Blue = Objects<br>
                <span style="color: #546E7A;">●</span> Blue-Grey = Unknown
            </div>
            
            <svg width="100%" height="550" style="border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9; display: block;">
                {svg_content}
            </svg>
            
            <script>
                function showNodeInfo(element) {{
                    var name = element.getAttribute('data-name');
                    var count = element.getAttribute('data-count');
                    var infoDiv = document.getElementById('nodeInfo');
                    var infoContent = document.getElementById('nodeInfoContent');
                    
                    infoContent.innerHTML = '<strong>' + name + '</strong><br>Connections: ' + count;
                    infoDiv.style.display = 'block';
                }}
                
                function hideNodeInfo(element) {{
                    document.getElementById('nodeInfo').style.display = 'none';
                }}
                
                function showNodeDetails(nodeName, displayName, count) {{
                    var fullText = nodeName.length > 100 ? nodeName.substring(0, 150) + '...' : nodeName;
                    alert('📊 Research Entity: ' + displayName + '\\n🔗 Relationships: ' + count + '\\n\\n📝 Full Entity: ' + fullText);
                }}
            </script>
        </div>
        """
        
        print(f" Interactive network visualization created successfully")
        return html
        
    except Exception as e:
        print(f" Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return f"<p style='color: red; padding: 20px;'>Error creating visualization: {e}</p>"

# =========================================================
#  📁 File Processing Functions
# =========================================================

def extract_text_from_pdf(file_path):
    """Extract text from PDF file with better error handling"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            print(f" PDF has {len(pdf_reader.pages)} pages")
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n"
                print(f" Page {i+1}: {len(page_text)} characters")
            
            extracted_text = text.strip()
            print(f" Total extracted: {len(extracted_text)} characters")
            print(f" First 200 chars: {extracted_text[:200]}...")
            
            return extracted_text
    except Exception as e:
        error_msg = f"Error reading PDF: {e}"
        print(f" {error_msg}")
        return error_msg

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading DOCX: {e}"

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        return f"Error reading TXT: {e}"

def extract_text_from_csv(file_path):
    """Extract text from CSV file"""
    try:
        df = pd.read_csv(file_path)
        # Convert DataFrame to readable text
        text = f"CSV Data with {len(df)} rows and {len(df.columns)} columns:\n\n"
        text += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Add first few rows as examples
        text += "Sample data:\n"
        for i, row in df.head(5).iterrows():
            text += f"Row {i+1}: {dict(row)}\n"
        
        return text.strip()
    except Exception as e:
        return f"Error reading CSV: {e}"

def process_uploaded_file(file):
    """Process uploaded file and extract text"""
    if file is None:
        return "No file uploaded."
    
    file_path = file.name
    file_extension = os.path.splitext(file_path)[1].lower()
    
    print(f"📁 Processing file: {file_path} (type: {file_extension})")
    
    # Extract text based on file type
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
    
    # Store extracted text for debugging
    update_extracted_text(extracted_text)
    
    # Show preview of extracted text
    preview = extracted_text[:300] + "..." if len(extracted_text) > 300 else extracted_text
    print(f" Extracted text preview: {preview}")
    
    # Add extracted text to knowledge graph
    result = add_to_graph(extracted_text)
    
    # Return detailed summary
    file_size = len(extracted_text)
    return f" Successfully processed {os.path.basename(file_path)}!\n\n📊 File stats:\n• Size: {file_size:,} characters\n• Type: {file_extension.upper()}\n\n Text preview:\n{preview}\n\n{result}"

def handle_file_upload(files):
    """Handle multiple file uploads and processing"""
    global processed_files
    
    if not files or len(files) == 0:
        return "Please select at least one file to process."
    
    results = []
    new_processed = []
    
    for file in files:
        if file is None:
            continue
            
        try:
            # Handle both file objects and string paths
            if isinstance(file, str):
                file_path = file
                file_name = os.path.basename(file)
            else:
                file_path = file.name
                file_name = os.path.basename(file.name)
            
            # Check if file was already processed
            if any(f['name'] == file_name for f in processed_files):
                results.append(f"SKIP: {file_name} - Already processed, skipping")
                continue
            
            # Process the file
            result = process_uploaded_file(file)
            results.append(f"SUCCESS: {file_name} - {result}")
            
            # Add to processed files list
            new_processed.append({
                'name': file_name,
                'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'facts_added': len(graph) - sum(f.get('facts_count', 0) for f in processed_files)
            })
            
        except Exception as e:
            # Handle both file objects and string paths for error reporting
            if isinstance(file, str):
                file_name = os.path.basename(file)
            else:
                file_name = os.path.basename(file.name) if hasattr(file, 'name') else str(file)
            
            error_msg = f"ERROR: {file_name} - Error: {e}"
            print(error_msg)
            results.append(error_msg)
    
    # Update processed files list
    processed_files.extend(new_processed)
    
    # Create summary
    total_files = len(files)
    successful = len([r for r in results if r.startswith("SUCCESS")])
    skipped = len([r for r in results if r.startswith("SKIP")])
    failed = len([r for r in results if r.startswith("ERROR")])
    
    summary = f"**Upload Summary:**\n"
    summary += f"• Total files: {total_files}\n"
    summary += f"• Successfully processed: {successful}\n"
    summary += f"• Already processed: {skipped}\n"
    summary += f"• Failed: {failed}\n"
    summary += f"• Total facts in knowledge base: {len(graph)}\n\n"
    
    # Add individual results
    summary += "**File Results:**\n"
    for result in results:
        summary += f"{result}\n"
    
    # Return single status message
    return summary

def show_processed_files():
    """Show list of processed files"""
    global processed_files
    
    if not processed_files:
        return "**No files processed yet.**\n\n**Start building your knowledge base:**\n1. Select one or more files (PDF, DOCX, TXT, CSV)\n2. Click 'Process Files' to extract knowledge\n3. View your processed files here\n4. Upload more files to expand your knowledge base!"
    
    result = f"**Processed Files ({len(processed_files)}):**\n\n"
    
    for i, file_info in enumerate(processed_files, 1):
        result += f"**{i}. {file_info['name']}**\n"
        result += f"   • Size: {file_info['size']:,} bytes\n"
        result += f"   • Processed: {file_info['processed_at']}\n"
        result += f"   • Facts added: {file_info.get('facts_added', 'Unknown')}\n\n"
    
    result += f"**Total Knowledge Base:** {len(graph)} facts\n"
    result += f"**Ready for more uploads!**"
    
    return result

def clear_processed_files():
    """Clear the processed files list"""
    global processed_files
    processed_files = []
    return "Processed files list cleared. You can now re-upload previously processed files."


def simple_test():
    """Simple test function to verify event handlers work"""
    print("🔔 Simple test function called!")
    return " Event handler is working! Button clicked successfully!"

# Global variable to store last extracted text
last_extracted_text = ""

# Global variable to track processed files
processed_files = []

def show_extracted_text():
    """Show the last extracted text from file processing"""
    global last_extracted_text
    
    if not last_extracted_text:
        return " No file has been processed yet.\n\nUpload a file and process it to see the extracted text here."
    
    # Show first 1000 characters
    preview = last_extracted_text[:1000]
    if len(last_extracted_text) > 1000:
        preview += "\n\n... (truncated, showing first 1000 characters)"
    
    return f" **Last Extracted Text:**\n\n{preview}"

def update_extracted_text(text):
    """Update the global variable with extracted text"""
    global last_extracted_text
    last_extracted_text = text

def delete_all_knowledge():
    """Delete all knowledge from the graph"""
    global graph
    count = len(graph)
    graph = rdflib.Graph()  # Create a new empty graph
    save_knowledge_graph()  # Save the empty graph
    return f"🗑️ Deleted all {count} facts from the knowledge graph. Graph is now empty."

def handle_delete_all(confirm_text):
    """Validate confirmation and delete all knowledge"""
    if not confirm_text or confirm_text.strip().upper() != "DELETE":
        return "⚠️ Type DELETE to confirm full deletion."
    return kb_delete_all_knowledge()

def delete_knowledge_by_keyword(keyword):
    """Delete knowledge containing a specific keyword"""
    global graph
    if not keyword or keyword.strip() == "":
        return "⚠️ Please enter a keyword to search for."
    
    keyword = keyword.strip().lower()
    deleted_count = 0
    facts_to_remove = []
    
    # Find facts containing the keyword
    for s, p, o in graph:
        fact_text = f"{s} {p} {o}".lower()
        if keyword in fact_text:
            facts_to_remove.append((s, p, o))
    
    # Remove the facts
    for fact in facts_to_remove:
        graph.remove(fact)
        deleted_count += 1
    
    if deleted_count > 0:
        save_knowledge_graph()  # Save after deletion
        return f"🗑️ Deleted {deleted_count} facts containing '{keyword}'"
    else:
        return f"ℹ️ No facts found containing '{keyword}'"

def delete_recent_knowledge(count=5):
    """Delete the most recently added knowledge"""
    global graph
    if len(graph) == 0:
        return "ℹ️ Knowledge graph is already empty."
    
    # Convert graph to list to get order
    facts = list(graph)
    facts_to_remove = facts[-count:] if count < len(facts) else facts
    
    # Remove the facts
    for fact in facts_to_remove:
        graph.remove(fact)
    
    save_knowledge_graph()  # Save after deletion
    return f"🗑️ Deleted {len(facts_to_remove)} most recent facts"


# =========================================================
#  🤖 2. Intelligent Response Generation
# =========================================================

def generate_intelligent_response(message, context, system_message):
    """Generate intelligent responses based on available facts"""
    message_lower = message.lower()
    
    # Document understanding questions
    if any(phrase in message_lower for phrase in [
        'what is the document about', 'whats the document about', 'what is this about', 'whats this about', 
        'describe the document', 'summarize the document', 'what does this contain', 'what is this about'
    ]):
        return generate_document_summary(context)
    
    # General "what" questions
    elif message_lower.startswith('what'):
        return generate_what_response(message, context)
    
    # "Who" questions
    elif message_lower.startswith('who'):
        return generate_who_response(message, context)
    
    # "When" questions
    elif message_lower.startswith('when'):
        return generate_when_response(message, context)
    
    # "Where" questions
    elif message_lower.startswith('where'):
        return generate_where_response(message, context)
    
    # "How much" or amount questions
    elif any(phrase in message_lower for phrase in [
        'how much', 'amount', 'total', 'cost', 'price'
    ]):
        return generate_amount_response(message, context)
    
    # Default intelligent response
    else:
        return generate_general_response(message, context)

def generate_document_summary(context):
    """Generate a summary of what the document is about"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have enough information about this document to provide a summary. Please add more knowledge to the knowledge base first."
    
    # Extract key information from context
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            facts.append(line.strip())
    
    # Analyze the facts to understand document type
    document_type = "document"
    key_info = []
    
    for fact in facts:
        fact_lower = fact.lower()
        if 'invoice' in fact_lower or 'bill' in fact_lower:
            document_type = "invoice"
        elif 'contract' in fact_lower or 'agreement' in fact_lower:
            document_type = "contract"
        elif 'report' in fact_lower or 'analysis' in fact_lower:
            document_type = "report"
        elif 'company' in fact_lower or 'organization' in fact_lower or 'name' in fact_lower:
            key_info.append(fact)
        elif 'amount' in fact_lower or 'total' in fact_lower or 'cost' in fact_lower or 'price' in fact_lower:
            key_info.append(fact)
        elif 'date' in fact_lower or 'time' in fact_lower:
            key_info.append(fact)
        elif 'address' in fact_lower or 'location' in fact_lower:
            key_info.append(fact)
        elif 'description' in fact_lower or 'type' in fact_lower:
            key_info.append(fact)
        elif 'id' in fact_lower or 'number' in fact_lower or 'code' in fact_lower:
            key_info.append(fact)
    
    # Generate summary
    summary = f"Based on the information in my knowledge base, this appears to be a **{document_type}** document. "
    
    if key_info:
        summary += "Here are the key details I found:\n\n"
        for info in key_info[:5]:  # Limit to 5 most relevant facts
            summary += f"• {info}\n"
    else:
        summary += "However, I don't have enough specific details to provide a comprehensive summary."
    
    return summary

def generate_what_response(message, context):
    """Generate responses for 'what' questions"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have information about that topic in my knowledge base. Try asking about specific details that might be in the document."
    
    # Extract relevant facts
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            facts.append(line.strip())
    
    if not facts:
        return "I don't have specific information about that in my knowledge base."
    
    # Generate contextual response
    response = "Based on my knowledge base, here's what I can tell you:\n\n"
    for fact in facts[:3]:  # Show top 3 most relevant facts
        response += f"• {fact}\n"
    
    if len(facts) > 3:
        response += f"\nI have {len(facts)} total facts about this topic in my knowledge base."
    
    return response

def generate_who_response(message, context):
    """Generate responses for 'who' questions"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have information about people or entities in my knowledge base."
    
    # Look for person/company related facts
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            if any(keyword in line.lower() for keyword in ['company', 'name', 'person', 'επωνυμία', 'εταιρεία']):
                facts.append(line.strip())
    
    if not facts:
        return "I don't have specific information about people or companies in my knowledge base."
    
    response = "Here's what I know about people/entities:\n\n"
    for fact in facts:
        response += f"• {fact}\n"
    
    return response

def generate_when_response(message, context):
    """Generate responses for 'when' questions"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have date information in my knowledge base."
    
    # Look for date related facts
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            if any(keyword in line.lower() for keyword in ['date', 'ημερομηνία', 'due', 'προθεσμία']):
                facts.append(line.strip())
    
    if not facts:
        return "I don't have specific date information in my knowledge base."
    
    response = "Here's the date information I have:\n\n"
    for fact in facts:
        response += f"• {fact}\n"
    
    return response

def generate_where_response(message, context):
    """Generate responses for 'where' questions"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have location information in my knowledge base."
    
    # Look for address/location related facts
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            if any(keyword in line.lower() for keyword in ['address', 'διεύθυνση', 'location', 'place']):
                facts.append(line.strip())
    
    if not facts:
        return "I don't have specific location information in my knowledge base."
    
    response = "Here's the location information I have:\n\n"
    for fact in facts:
        response += f"• {fact}\n"
    
    return response

def generate_amount_response(message, context):
    """Generate responses for amount/money questions"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have financial information in my knowledge base."
    
    # Look for amount/money related facts
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            if any(keyword in line.lower() for keyword in ['amount', 'total', 'price', 'cost', 'σύνολο', 'φόρος', '€', '$']):
                facts.append(line.strip())
    
    if not facts:
        return "I don't have specific financial information in my knowledge base."
    
    response = "Here's the financial information I have:\n\n"
    for fact in facts:
        response += f"• {fact}\n"
    
    return response

def generate_general_response(message, context):
    """Generate general intelligent responses"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have specific information about that topic in my knowledge base. Try asking about details that might be in the uploaded document, like company names, dates, amounts, or addresses."
    
    # Extract facts and provide intelligent response
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            facts.append(line.strip())
    
    if not facts:
        return "I don't have relevant information about that in my knowledge base."
    
    response = "Based on my knowledge base, here's what I can tell you:\n\n"
    for fact in facts[:4]:  # Show top 4 most relevant facts
        response += f"• {fact}\n"
    
    if len(facts) > 4:
        response += f"\nI have {len(facts)} total relevant facts about this topic."
    
    return response

# =========================================================
#  🤖 3. Reasoning Function (LLM + Symbolic Context)
# =========================================================
def respond(message, history, system_message="You are an intelligent assistant that answers questions based on factual information from a knowledge base. You provide clear, accurate, and helpful responses. When you have relevant information, you share it directly. When you don't have enough information, you clearly state this limitation. You always stay grounded in the facts provided and never hallucinate information.", max_tokens=256, temperature=0.7, top_p=0.9):
    # Step 1: retrieve context from symbolic KB
    context = retrieve_context(message)

    # Step 2: Try intelligent response generation first
    try:
        intelligent_response = generate_intelligent_response(message, context, system_message)
        print(f"🧠 Generated intelligent response for: {message[:50]}...")
        return intelligent_response
    except Exception as e:
        print(f"⚠️ Intelligent response failed: {e}")
        # Fall back to AI model approach

    # Step 3: Fallback to AI models if intelligent response fails
    # Enhanced prompt for better responses
    prompt = (
        f"{system_message}\n\n"
        f"Context from knowledge base:\n{context}\n\n"
        f"User Question: {message}\n\n"
        f"Instructions:\n"
        f"- Answer based ONLY on the facts provided above\n"
        f"- Be specific and factual\n"
        f"- If you don't have enough information, say so clearly\n"
        f"- Provide a helpful and informative response\n"
        f"- Keep your answer concise but complete\n\n"
        f"Answer:"
    )

    try:
        # Try to get HF token from environment variables
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        # Enhanced model list - more powerful free models, ordered by quality
        models_to_try = [
            # High-quality free models (no token required)
            ("microsoft/DialoGPT-medium", None),  # Conversational AI, good for Q&A
            ("facebook/blenderbot-400M-distill", None),  # Facebook's conversational model
            ("microsoft/DialoGPT-small", None),  # Smaller but reliable DialoGPT
            ("distilgpt2", None),  # Fast and reliable
            ("gpt2", None),  # Most reliable fallback
        ]
        
        # Add authenticated models if token is available (these are usually better)
        if hf_token:
            # Insert powerful authenticated models at the beginning
            authenticated_models = [
                ("HuggingFaceH4/zephyr-7b-beta", hf_token),  # High-quality instruction following
                ("microsoft/DialoGPT-large", hf_token),  # Large conversational model
                ("facebook/blenderbot-1B-distill", hf_token),  # Large Facebook model
                ("EleutherAI/gpt-neo-125M", hf_token),  # GPT-Neo model
            ]
            models_to_try = authenticated_models + models_to_try
        
        # Try each model
        for model, token in models_to_try:
            try:
                print(f"🔄 Attempting to use model: {model}")
                
                # Create client
                if token:
                    client = InferenceClient(model=model, token=token)
                else:
                    client = InferenceClient(model=model)
                
                # Try to generate response with optimized parameters
                result = client.text_generation(
                    prompt=prompt,
                    max_new_tokens=min(int(max_tokens), 150),  # Optimized for speed
                    temperature=min(float(temperature), 0.8),  # Cap temperature for consistency
                    top_p=min(float(top_p), 0.9),  # Cap top_p for better quality
                    repetition_penalty=1.1,  # Slightly higher to avoid repetition
                    do_sample=True,  # Enable sampling for better responses
                    stream=False,
                    return_full_text=False,
                )
                
                print(f" Successfully generated response using: {model}")
                return result.strip()
                
            except Exception as model_error:
                print(f" Model {model} failed: {model_error}")
                continue  # Try next model
        
        # If all models failed, provide intelligent fallback
        print("⚠️ All models failed, providing intelligent fallback")
        fallback_response = generate_intelligent_response(message, context, system_message)
        return fallback_response
        
    except Exception as e:
        # Ultimate fallback - even if everything fails
        print(f"💥 Complete failure: {e}")
        return f"🤖 I'm having trouble connecting to AI models right now, but I can still help!\n\nBased on your knowledge graph, I found these relevant facts:\n{context}\n\nFor your question '{message}', I'd suggest checking the facts above. Try adding more information to the knowledge graph or check back later when the AI models are working properly."

def generate_mock_response(message, context, system_message):
    """Generate a helpful response even when AI models fail"""
    
    # Simple keyword-based responses
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return f"👋 Hello! I'm your reasoning assistant. I found these facts in your knowledge base:\n\n{context}\n\nHow can I help you today?"
    
    elif any(word in message_lower for word in ['what', 'who', 'when', 'where', 'how', 'why']):
        return f"🤔 Great question! Based on your knowledge graph, here's what I found:\n\n{context}\n\nWhile I can't provide a full AI-generated answer right now, these facts from your knowledge base should help you understand the topic better."
    
    elif any(word in message_lower for word in ['help', 'assist', 'support']):
        return f"🆘 I'm here to help! Your knowledge graph contains:\n\n{context}\n\nYou can:\n• Add more information to the knowledge graph\n• Ask specific questions about the facts\n• Try again later when AI models are working"
    
    else:
        return f"💭 Interesting question! From your knowledge base, I found:\n\n{context}\n\nWhile I'm having technical difficulties with AI models, I can still help you explore the information you've added to the knowledge graph. Try asking more specific questions or adding more context!"

# =========================================================
#  💬 3. Gradio Interface Definition
# =========================================================

def save_and_backup():
    """Save knowledge and create backup"""
    save_result = kb_save_knowledge_graph()
    kb_create_comprehensive_backup()
    return BACKUP_FILE, save_result

def refresh_visualization(*args):
    """Wrapper to refresh visualization, ignoring any arguments from previous handlers"""
    return kb_visualize_knowledge_graph()

# =========================================================
#  🧩 4. Interface Layout
# =========================================================
with gr.Blocks(title="Research Brain") as demo:
    # Add custom CSS for blue-grey theme - remove all orange!
    demo.css = """
    <style>
        /* Hide window control buttons (minimize, maximize, close) */
        button[aria-label*="close"],
        button[aria-label*="Close"],
        button[aria-label*="minimize"],
        button[aria-label*="Minimize"],
        button[aria-label*="maximize"],
        button[aria-label*="Maximize"],
        .gradio-container button[aria-label],
        .gradio-container .toolbar button,
        .gradio-container header button,
        div[class*="window-controls"],
        div[class*="title-bar"] button,
        .control-buttons,
        .window-controls {
            display: none !important;
            visibility: hidden !important;
        }
        
        /* Override all button colors - remove orange completely */
        button { 
            background-color: #546E7A !important; 
            border-color: #546E7A !important; 
            color: white !important;
        }
        
        button.primary { 
            background-color: #546E7A !important; 
            border-color: #546E7A !important; 
            color: white !important;
        }
        
        button.primary:hover,
        button:hover { 
            background-color: #455A64 !important; 
            border-color: #455A64 !important; 
        }
        
        button.secondary { 
            background-color: #78909C !important; 
            border-color: #78909C !important; 
            color: white !important;
        }
        
        button.secondary:hover { 
            background-color: #607D8B !important; 
            border-color: #607D8B !important; 
        }
        
        button:focus,
        button.primary:focus { 
            border-color: #546E7A !important; 
            box-shadow: 0 0 0 2px rgba(84, 110, 122, 0.2) !important; 
        }
        
        /* Make white boxes light grey */
        textarea, 
        input[type="text"],
        .wrap,
        .output-text,
        .panel,
        .chatbot,
        .chat,
        .message {
            background-color: #F5F5F5 !important;
        }
        
        textarea:focus,
        input[type="text"]:focus,
        textarea:focus {
            background-color: #FFFFFF !important;
            border-color: #546E7A !important;
        }
        
        /* Chat interface styling */
        .chatbot-message,
        .conversation-box,
        [class*="chat"],
        [class*="message"] {
            background-color: #F8F8F8 !important;
        }
        
        /* File upload drag area - ALWAYS light grey background */
        [class*="file-drop"],
        [class*="upload"],
        input[type="file"],
        .gradio-file,
        .gradio-file [class*="component"],
        [class*="file-drop"]:hover,
        [class*="upload"]:hover,
        [class*="file-drop"]:focus,
        [class*="upload"]:focus,
        [class*="file-drop"]:active,
        [class*="upload"]:active,
        div[role="button"],
        [data-testid="file-upload"],
        [class*="drag"],
        div[class*="wrap"][class*="file"],
        div[class*="component"][class*="file"] {
            background-color: #F5F5F5 !important;
            background: #F5F5F5 !important;
            border-color: #546E7A !important;
        }
        
        /* Force light grey background for Gradio file components */
        div[class*="wrap"].gradio-file,
        div[class*="wrap"][class*="file"],
        [class*="wrap"][class*="gradio"] {
            background-color: #F5F5F5 !important;
            background: #F5F5F5 !important;
        }
        
        [class*="file-drop"]:hover *,
        [class*="upload"]:hover *,
        [class*="file-drop"]:focus *,
        [class*="upload"]:focus * {
            background-color: #F5F5F5 !important;
        }
        
        /* Make download file component compact */
        .gradio-file .component {
            min-height: 60px !important;
            max-height: 70px !important;
        }
        
        .gradio-file button {
            padding: 8px !important;
            font-size: 12px !important;
        }
        
        /* Specifically target Gradio file upload components */
        .upload-component,
        [class*="component"],
        .rounded,
        .border,
        div.wrap,
        div[class*="wrap"] {
            transition: none !important;
        }
        
        .upload-component:hover,
        [class*="component"]:hover,
        div.wrap:hover,
        div[class*="wrap"]:hover {
            background-color: #F5F5F5 !important;
            border-color: #546E7A !important;
        }
        
        /* Target all child elements within file upload */
        [class*="file"]:hover,
        [class*="File"]:hover,
        [id*="upload"]:hover,
        [id*="Upload"]:hover {
            background-color: #F5F5F5 !important;
        }
        
        /* Override any SVG or icon colors */
        [class*="file"] svg:hover,
        [class*="upload"] svg:hover {
            fill: #546E7A !important;
        }
        
        /* Nuclear option - target EVERYTHING that could be file upload */
        div[id*="file"],
        div[id*="File"],
        div[id*="upload"],
        div[id*="Upload"],
        .file,
        div.file,
        div.upload,
        .gradio-file *,
        div.w-full *,
        div[class*="wrap"] *,
        div[class*="wrap"][class*="file"] * {
            background: #F5F5F5 !important;
            background-color: #F5F5F5 !important;
        }
        
        /* Remove hover transitions completely for file upload */
        div:hover[class*="file"],
        div:hover[class*="upload"],
        div:hover[id*="file"],
        div:hover[id*="upload"] {
            background: #F5F5F5 !important;
            background-color: #F5F5F5 !important;
        }
        
        /* Force no background change on hover for upload areas specifically */
        div[class*="upload"]:hover,
        span[class*="upload"]:hover,
        button[class*="upload"]:hover,
        form[class*="upload"]:hover,
        div[role="button"][class*="file"]:hover,
        div[class*="wrap"][class*="file"]:hover {
            background-color: #F5F5F5 !important;
            border-color: #546E7A !important;
            opacity: 1 !important;
        }
        
        /* Disable hover animation for file upload */
        div[class*="file"]:hover,
        div[class*="upload"]:hover {
            transform: none !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        }
        
        /* Remove any orange/red accent colors */
        * {
            --tw-gradient-from: transparent !important;
            --tw-gradient-to: transparent !important;
        }
        
        /* Override any Gradio default colors */
        .gradio-container,
        .gradio-button-primary,
        button[class*="primary"],
        button[class*="button"] {
            background-color: #546E7A !important;
            border-color: #546E7A !important;
            color: white !important;
        }
        
        button[class*="primary"]:hover,
        button[class*="button"]:hover {
            background-color: #455A64 !important;
            border-color: #455A64 !important;
        }
        
        /* Style chat send button - blue-grey instead of grey */
        button[aria-label*="send"],
        [role="button"][aria-label*="Send"],
        button[class*="send-button"],
        button[aria-label*="Send message"],
        svg[class*="send"],
        button svg,
        .chat button svg {
            background-color: #546E7A !important;
            color: white !important;
        }
        
        /* Remove grey container around chat send button */
        button[aria-label*="send"],
        [role="button"][aria-label*="Send"],
        .chat input[type="submit"],
        .chat button {
            border: none !important;
            box-shadow: none !important;
        }
        
        /* Style the circular send button background */
        button[aria-label*="send"],
        button[aria-label*="Send message"] {
            background-color: #546E7A !important;
            border: none !important;
        }
        
        button[aria-label*="send"]:hover,
        button[aria-label*="Send message"]:hover {
            background-color: #455A64 !important;
        }
        
        /* Make Save Knowledge button compact */
        button[data-testid*="save"],
        .gradio-button {
            max-width: 200px !important;
        }
    </style>
    
    <script>
    // Force file upload area to stay light grey
    setInterval(function() {
        var fileElements = document.querySelectorAll('[class*="file"], [class*="upload"], [id*="file"], [id*="upload"], input[type="file"]');
        fileElements.forEach(function(el) {
            var parent = el.closest('div');
            if (parent) {
                parent.style.backgroundColor = '#F5F5F5';
                parent.style.background = '#F5F5F5';
            }
        });
        
        // Style chat send button with blue-grey
        var chatButtons = document.querySelectorAll('button[aria-label*="send"], button[aria-label*="Send"], button[class*="send-button"]');
        chatButtons.forEach(function(btn) {
            btn.style.border = 'none';
            btn.style.background = '#546E7A';
            btn.style.backgroundColor = '#546E7A';
            btn.style.color = 'white';
            btn.style.boxShadow = 'none';
            
            // Style on hover
            if (!btn.hasAttribute('data-styled')) {
                btn.setAttribute('data-styled', 'true');
                btn.addEventListener('mouseenter', function() {
                    this.style.backgroundColor = '#455A64';
                });
                btn.addEventListener('mouseleave', function() {
                    this.style.backgroundColor = '#546E7A';
                });
            }
        });
        
        // Hide window control buttons
        var hideWindowControls = function() {
            var controls = document.querySelectorAll('button[aria-label*="close"], button[aria-label*="Close"], button[aria-label*="minimize"], button[aria-label*="Minimize"], button[aria-label*="maximize"], button[aria-label*="Maximize"], div[class*="window-controls"], div[class*="title-bar"] button');
            controls.forEach(function(el) {
                el.style.display = 'none';
                el.style.visibility = 'hidden';
            });
            
            // Hide any buttons in the top-right corner that look like window controls
            var headerButtons = document.querySelectorAll('header button, .gradio-container > div:first-child button');
            headerButtons.forEach(function(btn) {
                if (btn.offsetParent !== null && (btn.getAttribute('aria-label') || btn.textContent.trim() === '')) {
                    var rect = btn.getBoundingClientRect();
                    if (rect.top < 50 && rect.right > window.innerWidth - 100) {
                        btn.style.display = 'none';
                        btn.style.visibility = 'hidden';
                    }
                }
            });
        };
        
        hideWindowControls();
        setInterval(hideWindowControls, 500);
    }, 100);
    </script>
    """
    
    # Header with logo in top right
    logo_path = None
    for ext in [".jpeg", ".jpg", ".png"]:
        path = f"logo_G{ext}"
        if os.path.exists(path):
            logo_path = path
            break
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("## Research Brain\nBuild and explore knowledge graphs from research documents, publications, and datasets.")
        with gr.Column(scale=1, min_width=100):
            if logo_path:
                gr.Image(value=logo_path, label="", show_label=False, container=False, min_width=100, height=100)

    with gr.Row():
        # Sidebar: all controls grouped in sections
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### Controls")
            with gr.Accordion("Data Ingestion", open=True):
                upload_box = gr.Textbox(
                    lines=5,
                    placeholder="Paste research text, abstracts, findings, or any content to extract knowledge...",
                    label="Add Research Content",
                )
                add_button = gr.Button("Extract Knowledge", variant="primary")
                file_upload = gr.File(
                    label="Upload Research Documents (PDF, DOCX, TXT, CSV)",
                    file_types=[".pdf", ".docx", ".txt", ".csv"],
                    file_count="multiple"
                )
                upload_file_button = gr.Button("Process Documents", variant="primary")

            with gr.Accordion("Knowledge Base Management", open=True):
                save_button = gr.Button("Save Knowledge", variant="secondary")
                download_button = gr.File(label="Download Backup", visible=True)
                json_upload = gr.File(label="Upload Knowledge JSON", file_types=[".json"], file_count="single")
                import_json_button = gr.Button("Import Knowledge JSON", variant="secondary")
                delete_confirm = gr.Textbox(label="Type DELETE to confirm", placeholder="DELETE")
                delete_all_btn = gr.Button("Delete All Knowledge", variant="secondary")
                show_button = gr.Button("View Knowledge Base", variant="secondary")
                graph_view = gr.Textbox(label="Knowledge Contents", visible=True, lines=3, max_lines=4)

            with gr.Accordion("Edit or Remove Facts", open=False):
                refresh_facts_btn = gr.Button("Refresh Facts", variant="secondary")
                fact_selector = gr.Dropdown(label="Select Fact", choices=[], interactive=True, multiselect=False)
                subj_box = gr.Textbox(label="Subject")
                pred_box = gr.Textbox(label="Predicate")
                obj_box = gr.Textbox(label="Object", lines=2)
                with gr.Row():
                    update_fact_btn = gr.Button("Update Fact", variant="primary")
                    delete_fact_btn = gr.Button("Delete Fact", variant="secondary")
                fact_edit_status = gr.Textbox(label="Edit Status", interactive=False)

            graph_info = gr.Textbox(label="Status", interactive=False, visible=True, lines=1, max_lines=2)

        # Main content: Knowledge graph (large) and chat (smaller below)
        with gr.Column(scale=3):
            gr.Markdown("### Knowledge Graph Network")
            graph_plot = gr.HTML(label="Knowledge Graph", visible=True, min_height=600)
            
            gr.Markdown("### Research Assistant")
            chatbot = gr.ChatInterface(
                fn=lambda message, history: rqa_respond(message, history),
                title="Query Knowledge Base",
                description="Ask questions about your research data. Explore findings, relationships, and insights.",
                examples=[
                    "What are the key research findings?",
                    "Summarize the methodologies",
                    "What relationships exist in the data?",
                    "What are the important timelines?",
                    "What datasets were used?"
                ]
            )
            
    # Auto-load visualization on page load
    demo.load(
        fn=kb_visualize_knowledge_graph,
        inputs=[],
        outputs=[graph_plot]
    )
    
    # Event handlers for simplified UI
    add_button.click(
        fn=handle_add_knowledge, 
        inputs=upload_box, 
        outputs=[graph_info, upload_box]
    ).then(
        fn=refresh_visualization,
        outputs=[graph_plot]
    )
    
    upload_file_button.click(
        fn=fp_handle_file_upload, 
        inputs=file_upload, 
        outputs=graph_info
    ).then(
        fn=refresh_visualization,
        outputs=[graph_plot]
    )
    
    show_button.click(
        fn=kb_show_graph_contents, 
        inputs=[], 
        outputs=[graph_view]
    )
    
    save_button.click(
        fn=save_and_backup,
        outputs=[download_button, graph_info]
    ).then(
        fn=refresh_visualization,
        outputs=[graph_plot]
    )

    import_json_button.click(
        fn=kb_import_json,
        inputs=json_upload,
        outputs=graph_info
    ).then(
        fn=refresh_visualization,
        outputs=[graph_plot]
    )

    delete_all_btn.click(
        fn=handle_delete_all,
        inputs=delete_confirm,
        outputs=graph_info
    ).then(
        fn=refresh_visualization,
        outputs=[graph_plot]
    )

    # Fact editor events
    refresh_facts_btn.click(
        fn=list_facts_for_editing,
        outputs=[fact_selector, fact_edit_status]
    )
    fact_selector.change(
        fn=load_fact_fields,
        inputs=fact_selector,
        outputs=[subj_box, pred_box, obj_box]
    )
    update_fact_btn.click(
        fn=update_fact,
        inputs=[fact_selector, subj_box, pred_box, obj_box],
        outputs=[fact_edit_status, fact_selector]
    ).then(
        fn=refresh_visualization,
        outputs=[graph_plot]
    )
    delete_fact_btn.click(
        fn=delete_fact,
        inputs=fact_selector,
        outputs=[fact_edit_status, fact_selector]
    ).then(
        fn=refresh_visualization,
        outputs=[graph_plot]
    )

# =========================================================
#  🚀 5. Initialize Sample Data and Launch
# =========================================================


if __name__ == "__main__":
    # Fix Windows console encoding issue with emojis
    import sys
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # Initialize knowledge graph and load existing data
    print("Initializing knowledge graph...")
    load_result = kb_load_knowledge_graph()
    print(f"Startup: {load_result}")
    print(f"Knowledge graph ready with {len(kb_graph)} facts")
    
    # Launch the Gradio app
    # For Hugging Face Spaces, the platform handles the launch automatically
    # If running directly, we provide explicit parameters
    print("Launching Gradio application...")
    
    # Check if we're in Hugging Face Spaces (has SPACE_ID env var)
    is_hf_space = os.getenv("SPACE_ID") is not None
    
    if is_hf_space:
        # On Hugging Face Spaces, just launch with defaults
        # The platform will handle port binding
        print("Detected Hugging Face Spaces environment")
        demo.launch(server_name="0.0.0.0")
    else:
        # Local development - use explicit settings
        port = int(os.getenv("PORT", 7860))
        print(f"Local development mode - starting on http://127.0.0.1:{port}")
        # Bind to loopback so browsers can open localhost directly
        demo.launch(server_name="127.0.0.1", server_port=port, share=False)

# """
# Research Brain - STEP 6: Properly Exposed Gradio API
# Uses correct /api/predict/{api_name} endpoints
# """

# import gradio as gr
# import pickle
# import os
# import rdflib
# import json

# # Files for storing data
# STORAGE_FILE = "knowledge_base.pkl"
# RDF_FILE = "knowledge_graph.rdf"

# # Initialize RDF graph
# graph = rdflib.Graph()

# # Load existing knowledge base
# def load_knowledge():
#     global graph
    
#     facts = []
#     if os.path.exists(STORAGE_FILE):
#         try:
#             with open(STORAGE_FILE, 'rb') as f:
#                 facts = pickle.load(f)
#         except:
#             facts = []
    
#     if os.path.exists(RDF_FILE):
#         try:
#             graph.parse(RDF_FILE, format="turtle")
#         except:
#             pass
    
#     return facts

# # Save knowledge base
# def save_knowledge(kb):
#     global graph
#     with open(STORAGE_FILE, 'wb') as f:
#         pickle.dump(kb, f)
#     try:
#         graph.serialize(destination=RDF_FILE, format="turtle")
#     except:
#         pass

# # Initialize knowledge base
# knowledge_base = load_knowledge()

# # ==========================================================
# #  API FUNCTIONS - Return data directly (not JSON strings)
# # ==========================================================

# def api_get_knowledge_base():
#     """API: Get all facts"""
#     return {"facts": knowledge_base}

# def api_create_fact(subject, predicate, obj, source="API"):
#     """API: Create a new fact"""
#     if not subject.strip() or not predicate.strip() or not obj.strip():
#         return {"success": False, "error": "Missing required fields"}
    
#     fact = {
#         "id": str(len(knowledge_base) + 1),
#         "subject": subject.strip(),
#         "predicate": predicate.strip(),
#         "object": obj.strip(),
#         "source": source
#     }
#     knowledge_base.append(fact)
    
#     # Add to RDF graph
#     subj_uri = rdflib.URIRef(f"urn:{subject.strip().replace(' ', '_')}")
#     pred_uri = rdflib.URIRef(f"urn:{predicate.strip().replace(' ', '_')}")
#     obj_literal = rdflib.Literal(obj.strip())
#     graph.add((subj_uri, pred_uri, obj_literal))
    
#     save_knowledge(knowledge_base)
    
#     return {"success": True, "fact": fact}

# def api_update_fact(fact_id, subject="", predicate="", obj=""):
#     """API: Update a fact"""
#     for fact in knowledge_base:
#         if isinstance(fact, dict) and str(fact.get("id")) == str(fact_id):
#             if subject:
#                 fact["subject"] = subject
#             if predicate:
#                 fact["predicate"] = predicate
#             if obj:
#                 fact["object"] = obj
            
#             # Rebuild RDF graph
#             global graph
#             graph = rdflib.Graph()
#             for f in knowledge_base:
#                 if isinstance(f, dict):
#                     s = rdflib.URIRef(f"urn:{f['subject'].replace(' ', '_')}")
#                     p = rdflib.URIRef(f"urn:{f['predicate'].replace(' ', '_')}")
#                     o = rdflib.Literal(f['object'])
#                     graph.add((s, p, o))
            
#             save_knowledge(knowledge_base)
#             return {"success": True, "fact": fact}
    
#     return {"success": False, "error": "Fact not found"}

# def api_delete_fact(fact_id):
#     """API: Delete a fact"""
#     global graph
    
#     for i, fact in enumerate(knowledge_base):
#         if isinstance(fact, dict) and str(fact.get("id")) == str(fact_id):
#             deleted_fact = knowledge_base.pop(i)
            
#             # Rebuild RDF graph
#             graph = rdflib.Graph()
#             for f in knowledge_base:
#                 if isinstance(f, dict):
#                     s = rdflib.URIRef(f"urn:{f['subject'].replace(' ', '_')}")
#                     p = rdflib.URIRef(f"urn:{f['predicate'].replace(' ', '_')}")
#                     o = rdflib.Literal(f['object'])
#                     graph.add((s, p, o))
            
#             save_knowledge(knowledge_base)
#             return {"success": True, "deleted": deleted_fact}
    
#     return {"success": False, "error": "Fact not found"}

# def api_get_graph():
#     """API: Get graph visualization data"""
#     nodes = []
#     edges = []
#     node_set = set()
    
#     for fact in knowledge_base:
#         if isinstance(fact, dict):
#             subj = fact.get("subject", "")
#             pred = fact.get("predicate", "")
#             obj = fact.get("object", "")
            
#             if subj and subj not in node_set:
#                 nodes.append({"id": subj, "label": subj, "type": "concept"})
#                 node_set.add(subj)
            
#             if obj and obj not in node_set:
#                 nodes.append({"id": obj, "label": obj, "type": "entity"})
#                 node_set.add(obj)
            
#             if subj and pred and obj:
#                 edges.append({
#                     "id": f"{subj}-{pred}-{obj}",
#                     "source": subj,
#                     "target": obj,
#                     "label": pred
#                 })
    
#     return {"nodes": nodes, "edges": edges}

# # ==========================================================
# #  UI FUNCTIONS
# # ==========================================================

# def add_fact(subject, predicate, obj):
#     """Add fact via UI"""
#     result = api_create_fact(subject, predicate, obj, "UI")
#     if result["success"]:
#         fact = result["fact"]
#         return f"✅ Added fact #{fact['id']}! Total: {len(knowledge_base)} facts", "", "", ""
#     return f"⚠️ {result.get('error', 'Unknown error')}", subject, predicate, obj

# def view_facts():
#     """View all facts"""
#     if not knowledge_base:
#         return "📭 No facts yet. Add some!"
    
#     result = f"📊 Knowledge Base ({len(knowledge_base)} facts, {len(graph)} RDF triples)\n\n"
#     for fact in knowledge_base:
#         if isinstance(fact, dict):
#             result += f"#{fact.get('id', '?')}: {fact.get('subject', '?')} → {fact.get('predicate', '?')} → {fact.get('object', '?')}\n"
#     return result

# def view_rdf_graph():
#     """View RDF graph"""
#     if len(graph) == 0:
#         return "📭 RDF graph is empty"
#     try:
#         turtle_data = graph.serialize(format="turtle")
#         return f"🌐 RDF Graph ({len(graph)} triples)\n\n{turtle_data}"
#     except Exception as e:
#         return f"❌ Error: {e}"

# def delete_all():
#     """Delete all knowledge"""
#     global graph
#     knowledge_base.clear()
#     graph = rdflib.Graph()
#     save_knowledge(knowledge_base)
#     return "🗑️ All knowledge deleted!"

# def get_stats():
#     """Get statistics"""
#     if not knowledge_base:
#         return "No facts yet"
    
#     subjects = set()
#     predicates = set()
#     objects = set()
    
#     for fact in knowledge_base:
#         if isinstance(fact, dict):
#             subjects.add(fact.get('subject', ''))
#             predicates.add(fact.get('predicate', ''))
#             objects.add(fact.get('object', ''))
    
#     return f"""
# 📊 Statistics:
# - Total facts: {len(knowledge_base)}
# - RDF triples: {len(graph)}
# - Unique subjects: {len(subjects)}
# - Unique predicates: {len(predicates)}
# - Unique objects: {len(objects)}
#     """.strip()

# # ==========================================================
# #  GRADIO INTERFACE
# # ==========================================================

# with gr.Blocks(title="Research Brain") as demo:
#     gr.Markdown("# 🧠 Research Brain - Step 6: Proper API Endpoints")
#     gr.Markdown("✅ Using /api/predict/{api_name} format!")
    
#     # Regular UI tabs
#     with gr.Tab("Add Fact"):
#         gr.Markdown("### Create a New Fact")
        
#         with gr.Row():
#             subject_input = gr.Textbox(label="Subject", placeholder="e.g., Machine Learning")
#             predicate_input = gr.Textbox(label="Predicate", placeholder="e.g., is part of")
#             object_input = gr.Textbox(label="Object", placeholder="e.g., Artificial Intelligence")
        
#         add_btn = gr.Button("Add Fact", variant="primary", size="lg")
#         status = gr.Textbox(label="Status", interactive=False)
        
#         add_btn.click(
#             fn=add_fact,
#             inputs=[subject_input, predicate_input, object_input],
#             outputs=[status, subject_input, predicate_input, object_input]
#         )
    
#     with gr.Tab("View Facts"):
#         with gr.Row():
#             view_btn = gr.Button("Refresh Facts", variant="secondary")
#             stats_btn = gr.Button("Show Statistics", variant="secondary")
        
#         output = gr.Textbox(label="Knowledge Base", lines=15)
#         stats_output = gr.Textbox(label="Statistics", lines=6)
        
#         view_btn.click(fn=view_facts, outputs=[output])
#         stats_btn.click(fn=get_stats, outputs=[stats_output])
    
#     with gr.Tab("RDF Graph"):
#         rdf_view_btn = gr.Button("View RDF Graph", variant="secondary")
#         rdf_output = gr.Textbox(label="RDF Graph (Turtle Format)", lines=15)
        
#         rdf_view_btn.click(fn=view_rdf_graph, outputs=[rdf_output])
    
#     with gr.Tab("🔌 API Testing"):
#         gr.Markdown("""
#         ### Test API Endpoints
#         These functions are exposed at `/api/predict/{api_name}`
#         """)
        
#         with gr.Accordion("Get Knowledge Base", open=True):
#             test_get_btn = gr.Button("Get All Facts")
#             test_get_output = gr.JSON(label="Result")
#             test_get_btn.click(
#                 fn=api_get_knowledge_base,
#                 outputs=[test_get_output],
#                 api_name="get_knowledge_base"
#             )
        
#         with gr.Accordion("Create Fact", open=False):
#             with gr.Row():
#                 test_subj = gr.Textbox(label="Subject", value="Test")
#                 test_pred = gr.Textbox(label="Predicate", value="relates_to")
#                 test_obj = gr.Textbox(label="Object", value="API")
#             test_source = gr.Textbox(label="Source", value="API", visible=False)
#             test_create_btn = gr.Button("Create Fact")
#             test_create_output = gr.JSON(label="Result")
#             test_create_btn.click(
#                 fn=api_create_fact,
#                 inputs=[test_subj, test_pred, test_obj, test_source],
#                 outputs=[test_create_output],
#                 api_name="create_fact"
#             )
        
#         with gr.Accordion("Update Fact", open=False):
#             with gr.Row():
#                 test_update_id = gr.Textbox(label="Fact ID", value="1")
#                 test_update_subj = gr.Textbox(label="Subject", value="Updated")
#             test_update_pred = gr.Textbox(label="Predicate", value="")
#             test_update_obj = gr.Textbox(label="Object", value="")
#             test_update_btn = gr.Button("Update Fact")
#             test_update_output = gr.JSON(label="Result")
#             test_update_btn.click(
#                 fn=api_update_fact,
#                 inputs=[test_update_id, test_update_subj, test_update_pred, test_update_obj],
#                 outputs=[test_update_output],
#                 api_name="update_fact"
#             )
        
#         with gr.Accordion("Delete Fact", open=False):
#             test_delete_id = gr.Textbox(label="Fact ID", value="1")
#             test_delete_btn = gr.Button("Delete Fact")
#             test_delete_output = gr.JSON(label="Result")
#             test_delete_btn.click(
#                 fn=api_delete_fact,
#                 inputs=[test_delete_id],
#                 outputs=[test_delete_output],
#                 api_name="delete_fact"
#             )
        
#         with gr.Accordion("Get Graph", open=False):
#             test_graph_btn = gr.Button("Get Graph Data")
#             test_graph_output = gr.JSON(label="Result")
#             test_graph_btn.click(
#                 fn=api_get_graph,
#                 outputs=[test_graph_output],
#                 api_name="get_graph"
#             )
    
#     with gr.Tab("Manage"):
#         delete_btn = gr.Button("Delete All Facts", variant="stop")
#         delete_status = gr.Textbox(label="Status", interactive=False)
#         delete_btn.click(fn=delete_all, outputs=[delete_status])

# print(f"📂 Loaded {len(knowledge_base)} facts")
# print(f"🌐 RDF graph has {len(graph)} triples")
# print(f"✅ API endpoints ready at /api/predict/{{api_name}}")

# # Enable queue for better API handling (optional but recommended)
# demo.queue()

# # Launch
# demo.launch()
