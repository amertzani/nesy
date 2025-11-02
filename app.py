"""
Research Brain - STEP 3: Structured Facts
Add subject-predicate-object fact structure
"""

import gradio as gr
import pickle
import os

# File for storing data
STORAGE_FILE = "knowledge_base.pkl"

# Load existing knowledge base
def load_knowledge():
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            return []
    return []

# Save knowledge base
def save_knowledge(kb):
    with open(STORAGE_FILE, 'wb') as f:
        pickle.dump(kb, f)

# Initialize knowledge base - now stores structured facts
knowledge_base = load_knowledge()

def add_fact(subject, predicate, obj):
    """Add structured fact to knowledge base"""
    if not subject.strip() or not predicate.strip() or not obj.strip():
        return "⚠️ Please fill in all fields", subject, predicate, obj
    
    fact = {
        "id": len(knowledge_base) + 1,
        "subject": subject.strip(),
        "predicate": predicate.strip(),
        "object": obj.strip()
    }
    knowledge_base.append(fact)
    save_knowledge(knowledge_base)
    return f"✅ Added fact #{fact['id']}! Total: {len(knowledge_base)} facts", "", "", ""

def view_facts():
    """View all facts"""
    if not knowledge_base:
        return "📭 No facts yet. Add some!"
    
    result = f"📊 Knowledge Base ({len(knowledge_base)} facts)\n\n"
    for fact in knowledge_base:
        if isinstance(fact, dict):
            result += f"#{fact.get('id', '?')}: {fact.get('subject', '?')} → {fact.get('predicate', '?')} → {fact.get('object', '?')}\n"
        else:
            # Handle old format from previous steps
            result += f"Old format: {fact}\n"
    return result

def delete_all():
    """Delete all knowledge"""
    knowledge_base.clear()
    save_knowledge(knowledge_base)
    return "🗑️ All knowledge deleted!"

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
- Unique subjects: {len(subjects)}
- Unique predicates: {len(predicates)}
- Unique objects: {len(objects)}
    """.strip()

# Create Gradio interface
with gr.Blocks(title="Research Brain") as demo:
    gr.Markdown("# 🧠 Research Brain - Step 3: Structured Facts")
    gr.Markdown("✅ Now supports subject-predicate-object fact structure!")
    
    with gr.Tab("Add Fact"):
        gr.Markdown("### Create a New Fact")
        gr.Markdown("Enter a fact in the form: **Subject** → **Predicate** → **Object**")
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
        stats_output = gr.Textbox(label="Statistics", lines=5)
        
        view_btn.click(fn=view_facts, outputs=[output])
        stats_btn.click(fn=get_stats, outputs=[stats_output])
    
    with gr.Tab("Manage"):
        gr.Markdown("### Manage Knowledge Base")
        delete_btn = gr.Button("Delete All Facts", variant="stop")
        delete_status = gr.Textbox(label="Status", interactive=False)
        
        delete_btn.click(fn=delete_all, outputs=[delete_status])

print(f"📂 Loaded {len(knowledge_base)} facts from storage")

# Launch
demo.launch()
