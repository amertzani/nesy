"""
Research Brain - STEP 4: RDF Knowledge Graph
Add RDFLib for proper knowledge graph storage
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
    
    # Load facts list
    facts = []
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, 'rb') as f:
                facts = pickle.load(f)
        except:
            facts = []
    
    # Load RDF graph
    if os.path.exists(RDF_FILE):
        try:
            graph.parse(RDF_FILE, format="turtle")
            print(f"📊 Loaded RDF graph with {len(graph)} triples")
        except:
            pass
    
    return facts

# Save knowledge base
def save_knowledge(kb):
    global graph
    
    # Save facts list
    with open(STORAGE_FILE, 'wb') as f:
        pickle.dump(kb, f)
    
    # Save RDF graph
    try:
        graph.serialize(destination=RDF_FILE, format="turtle")
        print(f"💾 Saved RDF graph with {len(graph)} triples")
    except Exception as e:
        print(f"⚠️ Error saving RDF: {e}")

# Initialize knowledge base
knowledge_base = load_knowledge()

def add_fact(subject, predicate, obj):
    """Add structured fact to knowledge base AND RDF graph"""
    if not subject.strip() or not predicate.strip() or not obj.strip():
        return "⚠️ Please fill in all fields", subject, predicate, obj
    
    # Create fact dictionary
    fact = {
        "id": len(knowledge_base) + 1,
        "subject": subject.strip(),
        "predicate": predicate.strip(),
        "object": obj.strip()
    }
    knowledge_base.append(fact)
    
    # Add to RDF graph
    subj_uri = rdflib.URIRef(f"urn:{subject.strip().replace(' ', '_')}")
    pred_uri = rdflib.URIRef(f"urn:{predicate.strip().replace(' ', '_')}")
    obj_literal = rdflib.Literal(obj.strip())
    graph.add((subj_uri, pred_uri, obj_literal))
    
    save_knowledge(knowledge_base)
    
    return f"✅ Added fact #{fact['id']} to RDF graph! Total: {len(knowledge_base)} facts, {len(graph)} triples", "", "", ""

def view_facts():
    """View all facts"""
    if not knowledge_base:
        return "📭 No facts yet. Add some!"
    
    result = f"📊 Knowledge Base ({len(knowledge_base)} facts, {len(graph)} RDF triples)\n\n"
    for fact in knowledge_base:
        if isinstance(fact, dict):
            result += f"#{fact.get('id', '?')}: {fact.get('subject', '?')} → {fact.get('predicate', '?')} → {fact.get('object', '?')}\n"
        else:
            result += f"Old format: {fact}\n"
    return result

def view_rdf_graph():
    """View RDF graph in turtle format"""
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
    graph = rdflib.Graph()  # Reset graph
    save_knowledge(knowledge_base)
    return "🗑️ All knowledge and RDF graph deleted!"

def get_stats():
    """Get knowledge base statistics"""
    if not knowledge_base:
        return "No facts yet"
    
    # Count unique subjects, predicates, objects
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

# Create Gradio interface
with gr.Blocks(title="Research Brain") as demo:
    gr.Markdown("# 🧠 Research Brain - Step 4: RDF Knowledge Graph")
    gr.Markdown("✅ Now using RDFLib for proper knowledge graph storage!")
    
    with gr.Tab("Add Fact"):
        gr.Markdown("### Create a New Fact")
        gr.Markdown("Facts are stored in both structured format AND as RDF triples")
        gr.Markdown("*Example: `Machine Learning` → `is part of` → `Artificial Intelligence`*")
        
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
        gr.Markdown("### View RDF Knowledge Graph")
        gr.Markdown("See your knowledge as RDF triples in Turtle format")
        
        rdf_view_btn = gr.Button("View RDF Graph", variant="secondary")
        rdf_output = gr.Textbox(label="RDF Graph (Turtle Format)", lines=15)
        
        rdf_view_btn.click(fn=view_rdf_graph, outputs=[rdf_output])
    
    with gr.Tab("Manage"):
        gr.Markdown("### Manage Knowledge Base")
        delete_btn = gr.Button("Delete All Facts", variant="stop")
        delete_status = gr.Textbox(label="Status", interactive=False)
        
        delete_btn.click(fn=delete_all, outputs=[delete_status])

print(f"📂 Loaded {len(knowledge_base)} facts from storage")
print(f"🌐 RDF graph has {len(graph)} triples")

# Launch
demo.launch()
