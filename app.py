"""
Research Brain - MINIMAL VERSION
Simplest possible Gradio app that will work on HF Spaces
Once this works, we'll add features step by step
"""

import gradio as gr

# Simple in-memory storage
knowledge_base = []

def add_knowledge(text):
    """Add text to knowledge base"""
    if text.strip():
        knowledge_base.append(text)
        return f"✅ Added! Total items: {len(knowledge_base)}"
    return "⚠️ Please enter some text"

def view_knowledge():
    """View all knowledge"""
    if not knowledge_base:
        return "📭 No knowledge yet. Add some!"
    
    result = f"📊 Knowledge Base ({len(knowledge_base)} items)\n\n"
    for i, item in enumerate(knowledge_base, 1):
        result += f"{i}. {item[:100]}...\n" if len(item) > 100 else f"{i}. {item}\n"
    return result

# Create Gradio interface
with gr.Blocks(title="Research Brain") as demo:
    gr.Markdown("# 🧠 Research Brain - Minimal Version")
    
    with gr.Tab("Add Knowledge"):
        text_input = gr.Textbox(lines=3, label="Enter text", placeholder="Type something...")
        add_btn = gr.Button("Add", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        
        add_btn.click(fn=add_knowledge, inputs=[text_input], outputs=[status])
    
    with gr.Tab("View Knowledge"):
        view_btn = gr.Button("Refresh")
        output = gr.Textbox(label="Knowledge Base", lines=10)
        
        view_btn.click(fn=view_knowledge, outputs=[output])

# Launch
demo.launch()
