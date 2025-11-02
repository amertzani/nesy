# Standalone app.py for Hugging Face Space
# This version works without external module dependencies

import gradio as gr
import rdflib
import re
import os
import json
import pickle
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ==========================================================
#  🧠 1. Global Knowledge Graph with Persistent Storage
# ==========================================================

KNOWLEDGE_FILE = "knowledge_graph.pkl"
BACKUP_FILE = "knowledge_backup.json"

graph = rdflib.Graph()
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
        
        with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        sync_rdf_to_facts_db()
        print(f"✅ Saved {len(graph)} facts")
        return f"✅ Saved {len(graph)} facts"
        
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return f"❌ Error: {e}"

def load_knowledge_graph():
    """Load the knowledge graph from persistent storage"""
    global graph
    
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, 'rb') as f:
                graph = pickle.load(f)
            sync_rdf_to_facts_db()
            print(f"📂 Loaded {len(graph)} facts")
            return f"📂 Loaded {len(graph)} facts"
        else:
            print("📂 Starting fresh")
            return "📂 Starting fresh"
            
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return f"❌ Error: {e}"

def add_to_graph(text):
    """Add knowledge from text to the graph"""
    if not text.strip():
        return "⚠️ No text provided"
    
    s_ref = rdflib.URIRef(f"urn:text_input")
    p_ref = rdflib.URIRef(f"urn:contains")
    o_lit = rdflib.Literal(text[:500])
    graph.add((s_ref, p_ref, o_lit))
    
    save_knowledge_graph()
    return f"✅ Added knowledge ({len(graph)} facts total)"

def show_graph_contents():
    """Show all facts in the graph"""
    if len(graph) == 0:
        return "📭 No facts in knowledge base yet"
    
    result = f"📊 **Knowledge Base ({len(graph)} facts)**\n\n"
    for i, (s, p, o) in enumerate(list(graph)[:20]):
        subject = str(s).split(':')[-1]
        predicate = str(p).split(':')[-1]
        obj = str(o)[:100]
        result += f"{i+1}. {subject} → {predicate} → {obj}\n"
    
    if len(graph) > 20:
        result += f"\n... and {len(graph) - 20} more facts"
    
    return result

def delete_all_knowledge():
    """Delete all knowledge from the graph"""
    global graph
    graph = rdflib.Graph()
    save_knowledge_graph()
    return "🗑️ All knowledge deleted"

# ==========================================================
#  🔌 2. FastAPI Setup for Replit Integration
# ==========================================================

fastapi_app = FastAPI(title="Research Brain API")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@fastapi_app.get("/")
async def root():
    return {
        "message": "Research Brain API is running",
        "total_facts": len(graph),
        "endpoints": {
            "knowledge_base": "/api/knowledge-base",
            "facts": "/api/facts",
            "graph": "/api/graph",
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
            
            global graph
            graph = rdflib.Graph()
            for f in facts_db:
                s_ref = rdflib.URIRef(f"urn:{f['subject']}")
                p_ref = rdflib.URIRef(f"urn:{f['predicate']}")
                o_lit = rdflib.Literal(f['object'])
                graph.add((s_ref, p_ref, o_lit))
            
            save_knowledge_graph()
            return {"success": True, "fact": fact}
    
    raise HTTPException(status_code=404, detail="Fact not found")

@fastapi_app.delete("/api/facts/{fact_id}")
async def delete_fact(fact_id: str):
    """Delete a fact"""
    global facts_db
    for i, fact in enumerate(facts_db):
        if fact["id"] == fact_id:
            deleted_fact = facts_db.pop(i)
            
            global graph
            graph = rdflib.Graph()
            for f in facts_db:
                s_ref = rdflib.URIRef(f"urn:{f['subject']}")
                p_ref = rdflib.URIRef(f"urn:{f['predicate']}")
                o_lit = rdflib.Literal(f['object'])
                graph.add((s_ref, p_ref, o_lit))
            
            save_knowledge_graph()
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
#  🎨 3. Build Gradio Interface
# ==========================================================

def build_gradio_interface():
    """Build the Gradio interface"""
    
    with gr.Blocks(title="🧠 Research Brain", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🧠 Research Brain
        ### Build and explore knowledge graphs from research documents
        
        **Quick Start:**
        1. Add knowledge using the text box or upload documents
        2. View your knowledge base and graph
        3. Use the API endpoints for programmatic access
        """)

        with gr.Tab("📝 Add Knowledge"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        lines=5,
                        placeholder="Enter research text, facts, or findings...",
                        label="Add Knowledge Text"
                    )
                    add_btn = gr.Button("Add to Knowledge Base", variant="primary", size="lg")
                    status_text = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("""
                    ### How to use:
                    - Paste text, research findings, or facts
                    - Click "Add to Knowledge Base"
                    - View your data in the other tabs
                    - Access via API at `/api/*` endpoints
                    """)

        with gr.Tab("📊 View Knowledge"):
            view_btn = gr.Button("Refresh Knowledge Base", variant="secondary")
            knowledge_view = gr.Textbox(label="Knowledge Base Contents", lines=20)

        with gr.Tab("💾 Manage"):
            with gr.Row():
                save_btn = gr.Button("Save Knowledge", variant="primary")
                download_btn = gr.File(label="Download Backup")
            
            with gr.Row():
                delete_confirm = gr.Textbox(label="Type DELETE to confirm", placeholder="DELETE")
                delete_btn = gr.Button("Delete All Knowledge", variant="stop")
            
            manage_status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("🔌 API Info"):
            gr.Markdown("""
            ### API Endpoints
            
            Your knowledge base is accessible via REST API:
            
            - **GET** `/api/knowledge-base` - Get all facts
            - **POST** `/api/facts` - Create new fact
            - **PATCH** `/api/facts/:id` - Update fact
            - **DELETE** `/api/facts/:id` - Delete fact
            - **GET** `/api/graph` - Get graph visualization data
            
            **API Documentation:** Visit `/docs` for interactive API docs
            
            **Example Request:**
            ```python
            import requests
            
            # Get all facts
            response = requests.get('https://asiminam-xnesy.hf.space/api/knowledge-base')
            facts = response.json()['facts']
            ```
            """)
            
            api_status = gr.Textbox(
                label="Current Status",
                value=f"✅ API is running | {len(graph)} facts in knowledge base",
                interactive=False
            )

        # Event handlers
        def handle_add(text):
            result = add_to_graph(text)
            return result, ""
        
        def handle_save():
            result = save_knowledge_graph()
            return BACKUP_FILE, result
        
        def handle_delete(confirm):
            if confirm == "DELETE":
                return delete_all_knowledge()
            return "⚠️ Type DELETE to confirm"
        
        add_btn.click(
            fn=handle_add,
            inputs=[text_input],
            outputs=[status_text, text_input]
        )
        
        view_btn.click(
            fn=show_graph_contents,
            outputs=[knowledge_view]
        )
        
        save_btn.click(
            fn=handle_save,
            outputs=[download_btn, manage_status]
        )
        
        delete_btn.click(
            fn=handle_delete,
            inputs=[delete_confirm],
            outputs=[manage_status]
        )
    
    return demo

# ==========================================================
#  🎬 4. Initialize Knowledge Graph
# ==========================================================

print("🧠 Initializing Research Brain...")
load_result = load_knowledge_graph()
print(f"Startup: {load_result}")
print(f"✅ Knowledge graph ready with {len(graph)} facts")

is_hf_space = os.getenv("SPACE_ID") is not None
if is_hf_space:
    print("☁️ Hugging Face Spaces environment detected")
    print("✅ Gradio UI available at: /")
    print("✅ FastAPI endpoints at: /api/*")
    print("✅ API docs at: /docs")

# ==========================================================
#  🚀 5. Build and Mount
# ==========================================================

demo = build_gradio_interface()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

print("🎉 Research Brain is ready!")
