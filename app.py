"""
Research Brain - STEP 5 FIXED: API Functions with Proper Gradio API Exposure
"""

import gradio as gr
import pickle
import os
import rdflib
import json

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
    """API: Get all facts - returns JSON string"""
    return json.dumps({"facts": knowledge_base})

def api_create_fact(subject, predicate, obj, source="API"):
    """API: Create a new fact - returns JSON string"""
    if not subject.strip() or not predicate.strip() or not obj.strip():
        return json.dumps({"success": False, "error": "Missing required fields"})
    
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
    
    return json.dumps({"success": True, "fact": fact})

def api_update_fact(fact_id, subject="", predicate="", obj=""):
    """API: Update a fact - returns JSON string"""
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
            return json.dumps({"success": True, "fact": fact})
    
    return json.dumps({"success": False, "error": "Fact not found"})

def api_delete_fact(fact_id):
    """API: Delete a fact - returns JSON string"""
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
            return json.dumps({"success": True, "deleted": deleted_fact})
    
    return json.dumps({"success": False, "error": "Fact not found"})

def api_get_graph():
    """API: Get graph visualization data - returns JSON string"""
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
    
    return json.dumps({"nodes": nodes, "edges": edges})

# ==========================================================
#  UI FUNCTIONS
# ==========================================================

def add_fact(subject, predicate, obj):
    """Add fact via UI"""
    result_str = api_create_fact(subject, predicate, obj, "UI")
    result = json.loads(result_str)
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
    gr.Markdown("# 🧠 Research Brain - Step 5: API Integration (FIXED)")
    gr.Markdown("✅ API functions properly exposed via Gradio API!")
    
    # Hidden API endpoints (these expose the functions via Gradio's API)
    with gr.Accordion("🔌 API Endpoints (Hidden - Used by Replit)", open=False, visible=True):
        gr.Markdown("""
        ### These components expose the API functions to external apps
        They are configured with `api_name` to be callable from your Replit frontend.
        """)
        
        # API endpoint: Get knowledge base
        api_get_kb_btn = gr.Button("Get Knowledge Base", visible=False)
        api_get_kb_output = gr.Textbox(visible=False)
        api_get_kb_btn.click(
            fn=api_get_knowledge_base,
            inputs=[],
            outputs=[api_get_kb_output],
            api_name="api_get_knowledge_base"
        )
        
        # API endpoint: Create fact
        api_create_subj = gr.Textbox(visible=False)
        api_create_pred = gr.Textbox(visible=False)
        api_create_obj = gr.Textbox(visible=False)
        api_create_source = gr.Textbox(visible=False, value="API")
        api_create_btn = gr.Button("Create Fact", visible=False)
        api_create_output = gr.Textbox(visible=False)
        api_create_btn.click(
            fn=api_create_fact,
            inputs=[api_create_subj, api_create_pred, api_create_obj, api_create_source],
            outputs=[api_create_output],
            api_name="api_create_fact"
        )
        
        # API endpoint: Update fact
        api_update_id = gr.Textbox(visible=False)
        api_update_subj = gr.Textbox(visible=False)
        api_update_pred = gr.Textbox(visible=False)
        api_update_obj = gr.Textbox(visible=False)
        api_update_btn = gr.Button("Update Fact", visible=False)
        api_update_output = gr.Textbox(visible=False)
        api_update_btn.click(
            fn=api_update_fact,
            inputs=[api_update_id, api_update_subj, api_update_pred, api_update_obj],
            outputs=[api_update_output],
            api_name="api_update_fact"
        )
        
        # API endpoint: Delete fact
        api_delete_id = gr.Textbox(visible=False)
        api_delete_btn = gr.Button("Delete Fact", visible=False)
        api_delete_output = gr.Textbox(visible=False)
        api_delete_btn.click(
            fn=api_delete_fact,
            inputs=[api_delete_id],
            outputs=[api_delete_output],
            api_name="api_delete_fact"
        )
        
        # API endpoint: Get graph
        api_graph_btn = gr.Button("Get Graph", visible=False)
        api_graph_output = gr.Textbox(visible=False)
        api_graph_btn.click(
            fn=api_get_graph,
            inputs=[],
            outputs=[api_graph_output],
            api_name="api_get_graph"
        )
    
    # Regular UI tabs
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
    
    with gr.Tab("Manage"):
        delete_btn = gr.Button("Delete All Facts", variant="stop")
        delete_status = gr.Textbox(label="Status", interactive=False)
        delete_btn.click(fn=delete_all, outputs=[delete_status])

print(f"📂 Loaded {len(knowledge_base)} facts")
print(f"🌐 RDF graph has {len(graph)} triples")
print(f"✅ API functions exposed with api_name!")

demo.launch()
