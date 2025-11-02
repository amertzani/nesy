"""
Research Brain - STEP 5: API Functions
Add API functions for Replit frontend integration
"""

import gradio as gr
import pickle
import os
import rdflib

# Files for storing data
STORAGE_FILE = "knowledge_base.pkl"
RDF_FILE = "knowledge_graph.rdf"

# Initialize RDF graph
graph = rdflib.Graph()

# Load existing knowledge base
def load_knowledge():
    global graph
    
    facts = []
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, 'rb') as f:
                facts = pickle.load(f)
        except:
            facts = []
    
    if os.path.exists(RDF_FILE):
        try:
            graph.parse(RDF_FILE, format="turtle")
        except:
            pass
    
    return facts

# Save knowledge base
def save_knowledge(kb):
    global graph
    with open(STORAGE_FILE, 'wb') as f:
        pickle.dump(kb, f)
    try:
        graph.serialize(destination=RDF_FILE, format="turtle")
    except:
        pass

# Initialize knowledge base
knowledge_base = load_knowledge()

# ==========================================================
#  API FUNCTIONS (callable from Replit frontend)
# ==========================================================

def api_get_knowledge_base():
    """API: Get all facts"""
    return {"facts": knowledge_base}

def api_create_fact(subject, predicate, obj, source="API"):
    """API: Create a new fact"""
    if not subject.strip() or not predicate.strip() or not obj.strip():
        return {"success": False, "error": "Missing required fields"}
    
    fact = {
        "id": str(len(knowledge_base) + 1),
        "subject": subject.strip(),
        "predicate": predicate.strip(),
        "object": obj.strip(),
        "source": source
    }
    knowledge_base.append(fact)
    
    # Add to RDF graph
    subj_uri = rdflib.URIRef(f"urn:{subject.strip().replace(' ', '_')}")
    pred_uri = rdflib.URIRef(f"urn:{predicate.strip().replace(' ', '_')}")
    obj_literal = rdflib.Literal(obj.strip())
    graph.add((subj_uri, pred_uri, obj_literal))
    
    save_knowledge(knowledge_base)
    
    return {"success": True, "fact": fact}

def api_update_fact(fact_id, subject="", predicate="", obj=""):
    """API: Update a fact"""
    for fact in knowledge_base:
        if isinstance(fact, dict) and str(fact.get("id")) == str(fact_id):
            # Update fields if provided
            if subject:
                fact["subject"] = subject
            if predicate:
                fact["predicate"] = predicate
            if obj:
                fact["object"] = obj
            
            # Rebuild RDF graph
            global graph
            graph = rdflib.Graph()
            for f in knowledge_base:
                if isinstance(f, dict):
                    s = rdflib.URIRef(f"urn:{f['subject'].replace(' ', '_')}")
                    p = rdflib.URIRef(f"urn:{f['predicate'].replace(' ', '_')}")
                    o = rdflib.Literal(f['object'])
                    graph.add((s, p, o))
            
            save_knowledge(knowledge_base)
            return {"success": True, "fact": fact}
    
    return {"success": False, "error": "Fact not found"}

def api_delete_fact(fact_id):
    """API: Delete a fact"""
    global graph
    
    for i, fact in enumerate(knowledge_base):
        if isinstance(fact, dict) and str(fact.get("id")) == str(fact_id):
            deleted_fact = knowledge_base.pop(i)
            
            # Rebuild RDF graph
            graph = rdflib.Graph()
            for f in knowledge_base:
                if isinstance(f, dict):
                    s = rdflib.URIRef(f"urn:{f['subject'].replace(' ', '_')}")
                    p = rdflib.URIRef(f"urn:{f['predicate'].replace(' ', '_')}")
                    o = rdflib.Literal(f['object'])
                    graph.add((s, p, o))
            
            save_knowledge(knowledge_base)
            return {"success": True, "deleted": deleted_fact}
    
    return {"success": False, "error": "Fact not found"}

def api_get_graph():
    """API: Get graph visualization data"""
    nodes = []
    edges = []
    node_set = set()
    
    for fact in knowledge_base:
        if isinstance(fact, dict):
            subj = fact.get("subject", "")
            pred = fact.get("predicate", "")
            obj = fact.get("object", "")
            
            if subj and subj not in node_set:
                nodes.append({"id": subj, "label": subj, "type": "concept"})
                node_set.add(subj)
            
            if obj and obj not in node_set:
                nodes.append({"id": obj, "label": obj, "type": "entity"})
                node_set.add(obj)
            
            if subj and pred and obj:
                edges.append({
                    "id": f"{subj}-{pred}-{obj}",
                    "source": subj,
                    "target": obj,
                    "label": pred
                })
    
    return {"nodes": nodes, "edges": edges}

# ==========================================================
#  UI FUNCTIONS
# ==========================================================

def add_fact(subject, predicate, obj):
    """Add fact via UI"""
    result = api_create_fact(subject, predicate, obj, "UI")
    if result["success"]:
        fact = result["fact"]
        return f"✅ Added fact #{fact['id']}! Total: {len(knowledge_base)} facts", "", "", ""
    return f"⚠️ {result.get('error', 'Unknown error')}", subject, predicate, obj

def view_facts():
    """View all facts"""
    if not knowledge_base:
        return "📭 No facts yet. Add some!"
    
    result = f"📊 Knowledge Base ({len(knowledge_base)} facts, {len(graph)} RDF triples)\n\n"
    for fact in knowledge_base:
        if isinstance(fact, dict):
            result += f"#{fact.get('id', '?')}: {fact.get('subject', '?')} → {fact.get('predicate', '?')} → {fact.get('object', '?')}\n"
    return result

def view_rdf_graph():
    """View RDF graph"""
    if len(graph) == 0:
        return "📭 RDF graph is empty"
    try:
        turtle_data = graph.serialize(format="turtle")
        return f"🌐 RDF Graph ({len(graph)} triples)\n\n{turtle_data}"
    except Exception as e:
        return f"❌ Error: {e}"

def delete_all():
    """Delete all knowledge"""
    global graph
    knowledge_base.clear()
    graph = rdflib.Graph()
    save_knowledge(knowledge_base)
    return "🗑️ All knowledge deleted!"

def get_stats():
    """Get statistics"""
    if not knowledge_base:
        return "No facts yet"
    
    subjects = set()
    predicates = set()
    objects = set()
    
    for fact in knowledge_base:
        if isinstance(fact, dict):
            subjects.add(fact.get('subject', ''))
            predicates.add(fact.get('predicate', ''))
            objects.add(fact.get('object', ''))
    
    return f"""
📊 Statistics:
- Total facts: {len(knowledge_base)}
- RDF triples: {len(graph)}
- Unique subjects: {len(subjects)}
- Unique predicates: {len(predicates)}
- Unique objects: {len(objects)}
    """.strip()

# ==========================================================
#  GRADIO INTERFACE
# ==========================================================

with gr.Blocks(title="Research Brain") as demo:
    gr.Markdown("# 🧠 Research Brain - Step 5: API Integration")
    gr.Markdown("✅ API functions ready for Replit frontend!")
    
    with gr.Tab("Add Fact"):
        gr.Markdown("### Create a New Fact")
        
        with gr.Row():
            subject_input = gr.Textbox(label="Subject", placeholder="e.g., Machine Learning")
            predicate_input = gr.Textbox(label="Predicate", placeholder="e.g., is part of")
            object_input = gr.Textbox(label="Object", placeholder="e.g., Artificial Intelligence")
        
        add_btn = gr.Button("Add Fact", variant="primary", size="lg")
        status = gr.Textbox(label="Status", interactive=False)
        
        add_btn.click(
            fn=add_fact,
            inputs=[subject_input, predicate_input, object_input],
            outputs=[status, subject_input, predicate_input, object_input]
        )
    
    with gr.Tab("View Facts"):
        with gr.Row():
            view_btn = gr.Button("Refresh Facts", variant="secondary")
            stats_btn = gr.Button("Show Statistics", variant="secondary")
        
        output = gr.Textbox(label="Knowledge Base", lines=15)
        stats_output = gr.Textbox(label="Statistics", lines=6)
        
        view_btn.click(fn=view_facts, outputs=[output])
        stats_btn.click(fn=get_stats, outputs=[stats_output])
    
    with gr.Tab("RDF Graph"):
        rdf_view_btn = gr.Button("View RDF Graph", variant="secondary")
        rdf_output = gr.Textbox(label="RDF Graph (Turtle Format)", lines=15)
        
        rdf_view_btn.click(fn=view_rdf_graph, outputs=[rdf_output])
    
    with gr.Tab("🔌 API Functions"):
        gr.Markdown("""
        ### API Functions for Replit Integration
        
        These functions are accessible via Gradio's API system:
        
        **Available Functions:**
        - `api_get_knowledge_base()` - Get all facts
        - `api_create_fact(subject, predicate, object, source)` - Create fact
        - `api_update_fact(fact_id, subject, predicate, object)` - Update fact
        - `api_delete_fact(fact_id)` - Delete fact
        - `api_get_graph()` - Get graph visualization data
        
        **Your Replit frontend is already configured to use these!**
        """)
        
        gr.Markdown("### Test API Functions")
        
        with gr.Accordion("Test Get Knowledge Base", open=False):
            test_get_btn = gr.Button("Get All Facts")
            test_get_output = gr.JSON(label="API Response")
            test_get_btn.click(fn=api_get_knowledge_base, outputs=[test_get_output])
        
        with gr.Accordion("Test Create Fact", open=False):
            with gr.Row():
                test_subj = gr.Textbox(label="Subject", value="Test")
                test_pred = gr.Textbox(label="Predicate", value="relates_to")
                test_obj = gr.Textbox(label="Object", value="API")
            test_create_btn = gr.Button("Create Fact")
            test_create_output = gr.JSON(label="API Response")
            test_create_btn.click(
                fn=api_create_fact,
                inputs=[test_subj, test_pred, test_obj],
                outputs=[test_create_output]
            )
        
        with gr.Accordion("Test Get Graph", open=False):
            test_graph_btn = gr.Button("Get Graph Data")
            test_graph_output = gr.JSON(label="API Response")
            test_graph_btn.click(fn=api_get_graph, outputs=[test_graph_output])
    
    with gr.Tab("Manage"):
        delete_btn = gr.Button("Delete All Facts", variant="stop")
        delete_status = gr.Textbox(label="Status", interactive=False)
        delete_btn.click(fn=delete_all, outputs=[delete_status])

print(f"📂 Loaded {len(knowledge_base)} facts")
print(f"🌐 RDF graph has {len(graph)} triples")
print(f"✅ API functions ready!")

demo.launch()
