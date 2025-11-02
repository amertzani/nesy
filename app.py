"""
Research Brain - Knowledge Management System
CORRECT structure for Hugging Face Spaces deployment
Gradio-first with FastAPI mounted underneath
"""

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import rdflib
import os
import json
import pickle
from datetime import datetime

# ==========================================================
#  KNOWLEDGE GRAPH STORAGE
# ==========================================================

KNOWLEDGE_FILE = "knowledge_graph.pkl"
BACKUP_FILE = "knowledge_backup.json"

graph = rdflib.Graph()
facts_db = []
next_fact_id = 1

def sync_rdf_to_facts_db():
    global facts_db, next_fact_id
    facts_db = []
    next_fact_id = 1
    for s, p, o in graph:
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
    try:
        with open(KNOWLEDGE_FILE, 'wb') as f:
            pickle.dump(graph, f)
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "total_facts": len(graph),
            "facts": [{"id": i+1, "subject": str(s), "predicate": str(p), "object": str(o)} 
                     for i, (s, p, o) in enumerate(graph)]
        }
        with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        sync_rdf_to_facts_db()
        return f"✅ Saved {len(graph)} facts"
    except Exception as e:
        return f"❌ Error: {e}"

def load_knowledge_graph():
    global graph
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, 'rb') as f:
                graph = pickle.load(f)
            sync_rdf_to_facts_db()
            return f"📂 Loaded {len(graph)} facts"
        return "📂 Starting fresh"
    except Exception as e:
        return f"❌ Error: {e}"

def add_to_graph(text):
    if not text.strip():
        return "⚠️ No text provided"
    s_ref = rdflib.URIRef(f"urn:text_input_{len(graph)}")
    p_ref = rdflib.URIRef(f"urn:contains")
    o_lit = rdflib.Literal(text[:500])
    graph.add((s_ref, p_ref, o_lit))
    save_knowledge_graph()
    return f"✅ Added knowledge ({len(graph)} facts total)"

def show_graph_contents():
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
    global graph
    graph = rdflib.Graph()
    save_knowledge_graph()
    return "🗑️ All knowledge deleted"

# ==========================================================
#  FASTAPI REST API (will be mounted under /api)
# ==========================================================

api_app = FastAPI(title="Research Brain API", version="1.0.0")

api_app.add_middleware(
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

@api_app.get("/knowledge-base")
async def get_knowledge_base():
    sync_rdf_to_facts_db()
    return {"facts": facts_db}

@api_app.post("/facts")
async def create_fact(fact: Fact):
    global next_fact_id
    new_fact = {
        "id": str(next_fact_id),
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "source": fact.source
    }
    facts_db.append(new_fact)
    next_fact_id += 1
    
    s_ref = rdflib.URIRef(f"urn:{fact.subject}")
    p_ref = rdflib.URIRef(f"urn:{fact.predicate}")
    o_lit = rdflib.Literal(fact.object)
    graph.add((s_ref, p_ref, o_lit))
    save_knowledge_graph()
    
    return {"success": True, "fact": new_fact}

@api_app.patch("/facts/{fact_id}")
async def update_fact(fact_id: str, updates: FactUpdate):
    for fact in facts_db:
        if fact["id"] == fact_id:
            if updates.subject:
                fact["subject"] = updates.subject
            if updates.predicate:
                fact["predicate"] = updates.predicate
            if updates.object:
                fact["object"] = updates.object
            
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

@api_app.delete("/facts/{fact_id}")
async def delete_fact(fact_id: str):
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

@api_app.get("/graph")
async def get_graph():
    sync_rdf_to_facts_db()
    nodes = []
    edges = []
    node_set = set()
    
    for fact in facts_db:
        if fact["subject"] not in node_set:
            nodes.append({"id": fact["subject"], "label": fact["subject"], "type": "concept"})
            node_set.add(fact["subject"])
        if fact["object"] not in node_set:
            nodes.append({"id": fact["object"], "label": fact["object"], "type": "entity"})
            node_set.add(fact["object"])
        edges.append({
            "id": f"{fact['subject']}-{fact['predicate']}-{fact['object']}",
            "source": fact["subject"],
            "target": fact["object"],
            "label": fact["predicate"]
        })
    
    return {"nodes": nodes, "edges": edges}

# ==========================================================
#  GRADIO UI (Gradio-first approach for HF Spaces)
# ==========================================================

demo = gr.Blocks(title="🧠 Research Brain", theme=gr.themes.Soft())

with demo:
    gr.Markdown("""
    # 🧠 Research Brain
    ### Build and explore knowledge graphs from research documents
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
        ### REST API Endpoints
        
        Access your knowledge base via REST API at `/api/*`:
        
        - **GET** `/api/knowledge-base` - Get all facts
        - **POST** `/api/facts` - Create new fact
        - **PATCH** `/api/facts/:id` - Update fact
        - **DELETE** `/api/facts/:id` - Delete fact
        - **GET** `/api/graph` - Get graph data
        
        **Note:** API docs are available at `/api/docs`
        """)

    # Event handlers
    def handle_add(text):
        result = add_to_graph(text)
        return result, ""
    
    def handle_save():
        result = save_knowledge_graph()
        return BACKUP_FILE if os.path.exists(BACKUP_FILE) else None, result
    
    def handle_delete(confirm):
        if confirm == "DELETE":
            return delete_all_knowledge()
        return "⚠️ Type DELETE to confirm"
    
    add_btn.click(fn=handle_add, inputs=[text_input], outputs=[status_text, text_input])
    view_btn.click(fn=show_graph_contents, outputs=[knowledge_view])
    save_btn.click(fn=handle_save, outputs=[download_btn, manage_status])
    delete_btn.click(fn=handle_delete, inputs=[delete_confirm], outputs=[manage_status])

# ==========================================================
#  INITIALIZATION - Gradio-first structure for HF Spaces
# ==========================================================

# Load knowledge graph
print("🧠 Initializing Research Brain...")
load_result = load_knowledge_graph()
print(load_result)

# Get the underlying Gradio ASGI app and mount FastAPI on it
# This is the CORRECT way for HF Spaces
gradio_app = demo.queue().app  # Get the Starlette app from Gradio
gradio_app.mount("/api", api_app)  # Mount FastAPI under /api

# HF Spaces looks for 'app' variable - provide the combined app
app = gradio_app

print("✅ Research Brain ready!")
print("✅ Gradio UI at: /")
print("✅ FastAPI at: /api/*")
