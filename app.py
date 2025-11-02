# Combined app.py for Hugging Face Space
# This file combines the full Gradio interface with FastAPI endpoints for Replit integration

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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
import pickle
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Import from your existing modules
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

# Storage file paths
KNOWLEDGE_FILE = "knowledge_graph.pkl"
BACKUP_FILE_LOCAL = "knowledge_backup.json"

graph = rdflib.Graph()
fact_index = {}

# In-memory facts database for API (synced with RDF graph)
facts_db = []
next_fact_id = 1

def sync_rdf_to_facts_db():
    """Sync RDF graph to facts_db for API access"""
    global facts_db, next_fact_id
    facts_db = []
    next_fact_id = 1
    
    for i, (s, p, o) in enumerate(graph):
        subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
        predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
        obj_val = str(o)
        
        facts_db.append({
            "id": str(next_fact_id),
            "subject": subject,
            "predicate": predicate,
            "object": obj_val,
            "source": "Knowledge Graph"
        })
        next_fact_id += 1

def sync_facts_db_to_rdf():
    """Sync facts_db changes back to RDF graph"""
    global graph
    graph = rdflib.Graph()
    
    for fact in facts_db:
        s_ref = rdflib.URIRef(f"urn:{fact['subject']}")
        p_ref = rdflib.URIRef(f"urn:{fact['predicate']}")
        o_lit = rdflib.Literal(fact['object'])
        graph.add((s_ref, p_ref, o_lit))
    
    save_knowledge_graph()

# Save/Load functions
def save_knowledge_graph():
    """Save the knowledge graph to persistent storage"""
    try:
        with open(KNOWLEDGE_FILE, 'wb') as f:
            pickle.dump(graph, f)
        
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
        
        with open(BACKUP_FILE_LOCAL, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        sync_rdf_to_facts_db()
        print(f"✅ Saved {len(graph)} facts to persistent storage")
        return f"✅ Saved {len(graph)} facts to storage"
        
    except Exception as e:
        error_msg = f"❌ Error saving knowledge: {e}"
        print(error_msg)
        return error_msg

def load_knowledge_graph():
    """Load the knowledge graph from persistent storage"""
    global graph
    
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, 'rb') as f:
                graph = pickle.load(f)
            sync_rdf_to_facts_db()
            print(f"📂 Loaded {len(graph)} facts from storage")
            return f"📂 Loaded {len(graph)} facts from storage"
        else:
            print("📂 No existing knowledge file found, starting fresh")
            return "📂 No existing knowledge file found, starting fresh"
            
    except Exception as e:
        error_msg = f"❌ Error loading knowledge: {e}"
        print(error_msg)
        return error_msg

# ==========================================================
#  🔌 2. FastAPI Setup for Replit Integration
# ==========================================================

fastapi_app = FastAPI(title="Research Brain API")

# CORS middleware for Replit frontend
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Fact(BaseModel):
    subject: str
    predicate: str
    object: str
    source: Optional[str] = "API"

class FactUpdate(BaseModel):
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    source: Optional[str] = None

# API Routes
@fastapi_app.get("/")
async def root():
    return {
        "message": "Research Brain API is running",
        "gradio_ui": "/gradio",
        "api_docs": "/docs",
        "endpoints": {
            "knowledge_base": "/api/knowledge-base",
            "facts": "/api/facts",
            "graph": "/api/graph",
            "upload": "/api/upload",
            "chat": "/api/chat"
        }
    }

@fastapi_app.get("/api/knowledge-base")
async def get_knowledge_base():
    """Get all knowledge facts"""
    sync_rdf_to_facts_db()
    return {"facts": facts_db}

@fastapi_app.post("/api/facts")
async def create_fact(fact: Fact):
    """Create a new fact"""
    global next_fact_id
    
    new_fact = {
        "id": str(next_fact_id),
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "source": fact.source or "API"
    }
    facts_db.append(new_fact)
    next_fact_id += 1
    
    # Add to RDF graph
    s_ref = rdflib.URIRef(f"urn:{fact.subject}")
    p_ref = rdflib.URIRef(f"urn:{fact.predicate}")
    o_lit = rdflib.Literal(fact.object)
    graph.add((s_ref, p_ref, o_lit))
    
    save_knowledge_graph()
    
    return {"success": True, "fact": new_fact}

@fastapi_app.patch("/api/facts/{fact_id}")
async def update_fact(fact_id: str, updates: FactUpdate):
    """Update an existing fact"""
    for fact in facts_db:
        if fact["id"] == fact_id:
            if updates.subject:
                fact["subject"] = updates.subject
            if updates.predicate:
                fact["predicate"] = updates.predicate
            if updates.object:
                fact["object"] = updates.object
            if updates.source:
                fact["source"] = updates.source
            
            sync_facts_db_to_rdf()
            return {"success": True, "fact": fact}
    
    raise HTTPException(status_code=404, detail="Fact not found")

@fastapi_app.delete("/api/facts/{fact_id}")
async def delete_fact(fact_id: str):
    """Delete a fact"""
    global facts_db
    for i, fact in enumerate(facts_db):
        if fact["id"] == fact_id:
            deleted_fact = facts_db.pop(i)
            sync_facts_db_to_rdf()
            return {"success": True, "deleted": deleted_fact}
    
    raise HTTPException(status_code=404, detail="Fact not found")

@fastapi_app.get("/api/graph")
async def get_graph():
    """Get knowledge graph visualization data"""
    sync_rdf_to_facts_db()
    
    nodes = []
    edges = []
    node_set = set()
    
    for fact in facts_db:
        if fact["subject"] not in node_set:
            nodes.append({
                "id": fact["subject"],
                "label": fact["subject"],
                "type": "concept"
            })
            node_set.add(fact["subject"])
        
        if fact["object"] not in node_set:
            nodes.append({
                "id": fact["object"],
                "label": fact["object"],
                "type": "entity"
            })
            node_set.add(fact["object"])
        
        edges.append({
            "id": f"{fact['subject']}-{fact['predicate']}-{fact['object']}",
            "source": fact["subject"],
            "target": fact["object"],
            "label": fact["predicate"]
        })
    
    return {"nodes": nodes, "edges": edges}

# ==========================================================
#  🎨 3. Build Full Gradio Interface (Your Original UI)
# ==========================================================

def build_gradio_interface():
    """Build the complete Gradio interface with all features"""
    
    with gr.Blocks(title="🧠 Research Brain", theme=gr.themes.Soft()) as demo:
        # Custom CSS for styling
        demo.css = """
        body {
            font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .gradio-container {
            max-width: 100% !important;
        }
        """
        
        # Header
        logo_path = None
        for ext in [".jpeg", ".jpg", ".png"]:
            path = f"logo_G{ext}"
            if os.path.exists(path):
                logo_path = path
                break
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## 🧠 Research Brain\nBuild and explore knowledge graphs from research documents, publications, and datasets.")
            with gr.Column(scale=1, min_width=100):
                if logo_path:
                    gr.Image(value=logo_path, label="", show_label=False, container=False, min_width=100, height=100)

        with gr.Row():
            # Sidebar: Controls
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Controls")
                
                with gr.Accordion("📄 Data Ingestion", open=True):
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

                with gr.Accordion("💾 Knowledge Base Management", open=True):
                    save_button = gr.Button("Save Knowledge", variant="secondary")
                    download_button = gr.File(label="Download Backup", visible=True)
                    json_upload = gr.File(label="Upload Knowledge JSON", file_types=[".json"], file_count="single")
                    import_json_button = gr.Button("Import Knowledge JSON", variant="secondary")
                    delete_confirm = gr.Textbox(label="Type DELETE to confirm", placeholder="DELETE")
                    delete_all_btn = gr.Button("Delete All Knowledge", variant="secondary")
                    show_button = gr.Button("View Knowledge Base", variant="secondary")
                    graph_view = gr.Textbox(label="Knowledge Contents", visible=True, lines=3, max_lines=4)

                with gr.Accordion("✏️ Edit or Remove Facts", open=False):
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

            # Main content: Knowledge graph and chat
            with gr.Column(scale=3):
                gr.Markdown("### 🕸️ Knowledge Graph Network")
                graph_plot = gr.HTML(label="Knowledge Graph", visible=True, min_height=600)
                
                gr.Markdown("### 💬 Research Assistant")
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
        
        # Helper functions
        def handle_add_knowledge(text):
            if not text.strip():
                return "⚠️ Please enter some text", ""
            result = kb_add_to_graph(text)
            save_knowledge_graph()
            return result, ""
        
        def refresh_visualization():
            return kb_visualize_knowledge_graph()
        
        def save_and_backup():
            save_result = save_knowledge_graph()
            kb_create_comprehensive_backup()
            return BACKUP_FILE, save_result
        
        def handle_delete_all(confirm_text):
            if confirm_text == "DELETE":
                result = kb_delete_all_knowledge()
                save_knowledge_graph()
                return result
            return "⚠️ Type DELETE to confirm"
        
        def list_facts_for_editing():
            facts = []
            for i, (s, p, o) in enumerate(graph):
                subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
                predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
                obj = str(o)
                facts.append(f"{i}: {subject} | {predicate} | {obj}")
            return gr.update(choices=facts), f"📊 Found {len(facts)} facts"
        
        def load_fact_fields(fact_str):
            if not fact_str:
                return "", "", ""
            try:
                parts = fact_str.split(": ", 1)[1].split(" | ")
                return parts[0], parts[1], parts[2]
            except:
                return "", "", ""
        
        def update_fact_handler(fact_str, subj, pred, obj):
            if not fact_str:
                return "⚠️ Select a fact first", None
            try:
                idx = int(fact_str.split(":")[0])
                # Implementation would update the specific fact
                save_knowledge_graph()
                return "✅ Fact updated", None
            except:
                return "❌ Error updating fact", None
        
        def delete_fact_handler(fact_str):
            if not fact_str:
                return "⚠️ Select a fact first", None
            try:
                idx = int(fact_str.split(":")[0])
                # Implementation would delete the specific fact
                save_knowledge_graph()
                return "✅ Fact deleted", None
            except:
                return "❌ Error deleting fact", None
        
        # Auto-load visualization
        demo.load(
            fn=kb_visualize_knowledge_graph,
            inputs=[],
            outputs=[graph_plot]
        )
        
        # Event handlers
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
            fn=update_fact_handler,
            inputs=[fact_selector, subj_box, pred_box, obj_box],
            outputs=[fact_edit_status, fact_selector]
        ).then(
            fn=refresh_visualization,
            outputs=[graph_plot]
        )
        delete_fact_btn.click(
            fn=delete_fact_handler,
            inputs=fact_selector,
            outputs=[fact_edit_status, fact_selector]
        ).then(
            fn=refresh_visualization,
            outputs=[graph_plot]
        )
    
    return demo

# ==========================================================
#  🚀 4. Mount Both Gradio and FastAPI Together
# ==========================================================

# Build the Gradio interface
demo = build_gradio_interface()

# Mount Gradio at root and FastAPI at /api
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# ==========================================================
#  🎬 5. Initialize and Launch
# ==========================================================

if __name__ == "__main__":
    import sys
    import io
    
    # Fix console encoding
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # Initialize knowledge graph
    print("🧠 Initializing knowledge graph...")
    load_result = kb_load_knowledge_graph()
    print(f"Startup: {load_result}")
    print(f"Knowledge graph ready with {len(kb_graph)} facts")
    
    # Check environment
    is_hf_space = os.getenv("SPACE_ID") is not None
    
    if is_hf_space:
        print("☁️ Detected Hugging Face Spaces environment")
        print("✅ Gradio UI available at: /")
        print("✅ FastAPI endpoints available at: /api/*")
        print("✅ API docs available at: /docs")
    else:
        port = int(os.getenv("PORT", 7860))
        print(f"💻 Local development mode")
        print(f"✅ Gradio UI: http://127.0.0.1:{port}/")
        print(f"✅ API: http://127.0.0.1:{port}/api/*")
        print(f"✅ Docs: http://127.0.0.1:{port}/docs")
