"""
Research Brain - STEP 2: Add Persistent Storage
Now data saves to a file and persists across restarts
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

# Initialize knowledge base
knowledge_base = load_knowledge()

def add_knowledge(text):
    """Add text to knowledge base"""
    if text.strip():
        knowledge_base.append(text)
        save_knowledge(knowledge_base)  # Save after adding
        return f"✅ Added! Total items: {len(knowledge_base)}", ""
    return "⚠️ Please enter some text", text

def view_knowledge():
    """View all knowledge"""
    if not knowledge_base:
        return "📭 No knowledge yet. Add some!"
    
    result = f"📊 Knowledge Base ({len(knowledge_base)} items)\n\n"
    for i, item in enumerate(knowledge_base, 1):
        result += f"{i}. {item[:100]}...\n" if len(item) > 100 else f"{i}. {item}\n"
    return result

def delete_all():
    """Delete all knowledge"""
    knowledge_base.clear()
    save_knowledge(knowledge_base)
    return "🗑️ All knowledge deleted!"

# Create Gradio interface
with gr.Blocks(title="Research Brain") as demo:
    gr.Markdown("# 🧠 Research Brain - Step 2: Persistent Storage")
    gr.Markdown("✅ Data now saves to disk and persists across restarts!")
    
    with gr.Tab("Add Knowledge"):
        text_input = gr.Textbox(lines=3, label="Enter text", placeholder="Type something...")
        add_btn = gr.Button("Add", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        
        add_btn.click(fn=add_knowledge, inputs=[text_input], outputs=[status, text_input])
    
    with gr.Tab("View Knowledge"):
        view_btn = gr.Button("Refresh")
        output = gr.Textbox(label="Knowledge Base", lines=10)
        
        view_btn.click(fn=view_knowledge, outputs=[output])
    
    with gr.Tab("Manage"):
        gr.Markdown("### Delete All Knowledge")
        delete_btn = gr.Button("Delete All", variant="stop")
        delete_status = gr.Textbox(label="Status", interactive=False)
        
        delete_btn.click(fn=delete_all, outputs=[delete_status])

print(f"📂 Loaded {len(knowledge_base)} items from storage")

# Launch
demo.launch()
