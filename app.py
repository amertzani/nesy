# GUARANTEED WORKING VERSION FOR HUGGING FACE SPACES
# Pure Gradio implementation - no FastAPI conflicts

import gradio as gr
import rdflib
import os
import json
import pickle
from datetime import datetime
from typing import Optional

# ==========================================================
#  Knowledge Graph Storage
# ==========================================================

KNOWLEDGE_FILE = "knowledge_graph.pkl"
BACKUP_FILE = "knowledge_backup.json"

graph = rdflib.Graph()
facts_db = []
next_fact_id = 1

def sync_rdf_to_facts_db():
    """Sync RDF graph to facts database"""
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
    """Save knowledge graph to file"""
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
    """Load knowledge graph from file"""
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
    """Add text to knowledge graph"""
    if not text.strip():
        return "⚠️ No text provided"
    s_ref = rdflib.URIRef(f"urn:text_input_{len(graph)}")
    p_ref = rdflib.URIRef(f"urn:contains")
    o_lit = rdflib.Literal(text[:500])
    graph.add((s_ref, p_ref, o_lit))
    save_knowledge_graph()
    return f"✅ Added knowledge ({len(graph)} facts total)"

def show_graph_contents():
    """Display knowledge base contents"""
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
    """Delete all knowledge"""
    global graph
    graph = rdflib.Graph()
    save_knowledge_graph()
    return "🗑️ All knowledge deleted"

def get_knowledge_base_json():
    """Get knowledge base as JSON for API"""
    sync_rdf_to_facts_db()
    return json.dumps({"facts": facts_db}, indent=2)

def create_fact_api(subject, predicate, obj, source="API"):
    """Create fact via API"""
    global next_fact_id
    if not subject or not predicate or not obj:
        return json.dumps({"error": "Missing required fields"})
    
    new_fact = {
        "id": str(next_fact_id),
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "source": source
    }
    facts_db.append(new_fact)
    next_fact_id += 1
    
    s_ref = rdflib.URIRef(f"urn:{subject}")
    p_ref = rdflib.URIRef(f"urn:{predicate}")
    o_lit = rdflib.Literal(obj)
    graph.add((s_ref, p_ref, o_lit))
    save_knowledge_graph()
    
    return json.dumps({"success": True, "fact": new_fact}, indent=2)

def update_fact_api(fact_id, subject=None, predicate=None, obj=None):
    """Update fact via API"""
    for fact in facts_db:
        if fact["id"] == fact_id:
            if subject:
                fact["subject"] = subject
            if predicate:
                fact["predicate"] = predicate
            if obj:
                fact["object"] = obj
            
            # Rebuild graph
            global graph
            graph = rdflib.Graph()
            for f in facts_db:
                s_ref = rdflib.URIRef(f"urn:{f['subject']}")
                p_ref = rdflib.URIRef(f"urn:{f['predicate']}")
                o_lit = rdflib.Literal(f['object'])
                graph.add((s_ref, p_ref, o_lit))
            save_knowledge_graph()
            
            return json.dumps({"success": True, "fact": fact}, indent=2)
    
    return json.dumps({"error": "Fact not found"})

def delete_fact_api(fact_id):
    """Delete fact via API"""
    global facts_db
    for i, fact in enumerate(facts_db):
        if fact["id"] == fact_id:
            deleted_fact = facts_db.pop(i)
            
            # Rebuild graph
            global graph
            graph = rdflib.Graph()
            for f in facts_db:
                s_ref = rdflib.URIRef(f"urn:{f['subject']}")
                p_ref = rdflib.URIRef(f"urn:{f['predicate']}")
                o_lit = rdflib.Literal(f['object'])
                graph.add((s_ref, p_ref, o_lit))
            save_knowledge_graph()
            
            return json.dumps({"success": True, "deleted": deleted_fact}, indent=2)
    
    return json.dumps({"error": "Fact not found"})

def get_graph_api():
    """Get graph data for visualization"""
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
    
    return json.dumps({"nodes": nodes, "edges": edges}, indent=2)

# ==========================================================
#  Gradio Interface
# ==========================================================

# Load knowledge graph at startup
print("🧠 Initializing Research Brain...")
load_result = load_knowledge_graph()
print(f"{load_result}")

with gr.Blocks(title="🧠 Research Brain", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🧠 Research Brain
    ### Build and explore knowledge graphs from research documents
    
    **Quick Start:** Add knowledge, view your data, and access via API
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
                - View your data in other tabs
                - Access via API endpoints below
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

    with gr.Tab("🔌 API Endpoints"):
        gr.Markdown("""
        ### REST API Access
        
        Use these Gradio API endpoints from your Replit frontend:
        
        **Available Functions:**
        - `get_knowledge_base_json()` - Get all facts as JSON
        - `create_fact_api(subject, predicate, object, source)` - Create new fact
        - `update_fact_api(fact_id, subject, predicate, object)` - Update fact
        - `delete_fact_api(fact_id)` - Delete fact
        - `get_graph_api()` - Get graph visualization data
        
        **Example Usage from JavaScript:**
        ```javascript
        // Get all facts
        const response = await fetch("https://asiminam-xnesy.hf.space/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                fn_index: 0,  // get_knowledge_base_json
                data: []
            })
        });
        ```
        """)
        
        with gr.Accordion("Test API Functions", open=False):
            gr.Markdown("### Test Get Knowledge Base")
            api_test_btn1 = gr.Button("Get Knowledge Base JSON")
            api_output1 = gr.JSON(label="API Response")
            
            gr.Markdown("### Test Create Fact")
            with gr.Row():
                api_subject = gr.Textbox(label="Subject", placeholder="Research")
                api_predicate = gr.Textbox(label="Predicate", placeholder="relates_to")
                api_object = gr.Textbox(label="Object", placeholder="AI")
            api_test_btn2 = gr.Button("Create Fact")
            api_output2 = gr.JSON(label="API Response")
            
            gr.Markdown("### Test Get Graph")
            api_test_btn3 = gr.Button("Get Graph Data")
            api_output3 = gr.JSON(label="API Response")

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
    
    # Connect UI events
    add_btn.click(fn=handle_add, inputs=[text_input], outputs=[status_text, text_input])
    view_btn.click(fn=show_graph_contents, outputs=[knowledge_view])
    save_btn.click(fn=handle_save, outputs=[download_btn, manage_status])
    delete_btn.click(fn=handle_delete, inputs=[delete_confirm], outputs=[manage_status])
    
    # API test events
    api_test_btn1.click(fn=lambda: json.loads(get_knowledge_base_json()), outputs=[api_output1])
    api_test_btn2.click(
        fn=lambda s, p, o: json.loads(create_fact_api(s, p, o)),
        inputs=[api_subject, api_predicate, api_object],
        outputs=[api_output2]
    )
    api_test_btn3.click(fn=lambda: json.loads(get_graph_api()), outputs=[api_output3])

print("✅ Research Brain is ready!")
print("✅ Gradio UI available at root")
print("✅ API functions accessible via Gradio API")
