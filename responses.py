from knowledge import retrieve_context
import os

# Try to import transformers for lightweight LLM
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    LLM_AVAILABLE = True
    LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    LLM_AVAILABLE = False
    LLM_DEVICE = "cpu"
    print("⚠️  Transformers not available. Using rule-based responses.")

# LLM configuration
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"  # Enable by default
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # Lightweight, fast model
LLM_PIPELINE = None

def load_llm_model():
    """Load lightweight LLM for RAG responses (lazy loading)"""
    global LLM_PIPELINE
    
    if not LLM_AVAILABLE or not USE_LLM:
        return False
    
    if LLM_PIPELINE is not None:
        return True
    
    try:
        print("🔄 Loading lightweight LLM for research assistant...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16 if LLM_DEVICE == "cuda" else torch.float32,
            device_map="auto" if LLM_DEVICE == "cuda" else None,
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention to avoid flash-attn issues
        )
        
        LLM_PIPELINE = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if LLM_DEVICE == "cuda" else -1,
            torch_dtype=torch.float16 if LLM_DEVICE == "cuda" else torch.float32,
        )
        print(f"✅ LLM loaded successfully on {LLM_DEVICE}")
        return True
    except Exception as e:
        print(f"⚠️  Failed to load LLM: {e}")
        print("   Falling back to rule-based responses")
        import traceback
        traceback.print_exc()
        LLM_PIPELINE = None
        return False

def generate_llm_response(message, context, history=None):
    """Generate response using lightweight LLM with RAG"""
    if not LLM_AVAILABLE or not USE_LLM:
        return None
    
    try:
        if LLM_PIPELINE is None:
            if not load_llm_model():
                return None
        
        # Build prompt with context from knowledge graph
        system_prompt = """You are a helpful research assistant that answers questions based on factual information from a knowledge base. 
You provide clear, accurate, and helpful responses. When you have relevant information from the knowledge base, you use it directly.
When you don't have enough information, you clearly state this limitation. You always stay grounded in the facts provided and never hallucinate information."""
        
        # Format context from knowledge graph
        if context and "No directly relevant facts found" not in context:
            context_text = context.replace("**Relevant Knowledge:**\n", "").strip()
        else:
            context_text = "No specific relevant facts found in the knowledge base."
        
        # Build conversation prompt
        prompt = f"""<|system|>
{system_prompt}

Knowledge Base Context:
{context_text}
<|user|>
{message}
<|assistant|>
"""
        
        # Generate response with fixed parameters
        # Use model.generate directly to have more control
        inputs = LLM_PIPELINE.tokenizer(prompt, return_tensors="pt")
        if LLM_DEVICE == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        generated_text = None
        try:
            outputs = LLM_PIPELINE.model.generate(
                **inputs,
                max_new_tokens=128,  # Reduced for faster generation
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=LLM_PIPELINE.tokenizer.eos_token_id,
                use_cache=False,  # Disable cache to avoid compatibility issues
                max_time=30.0,  # Maximum 30 seconds for generation
            )
            generated_text = LLM_PIPELINE.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            # Fallback to pipeline if direct generation fails
            print(f"⚠️  Direct generation failed: {e}, trying pipeline...")
            try:
                response = LLM_PIPELINE(
                    prompt,
                    max_new_tokens=128,  # Reduced for faster generation
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=LLM_PIPELINE.tokenizer.eos_token_id,
                    max_time=30.0,  # Maximum 30 seconds for generation
                )
                generated_text = response[0]['generated_text']
            except Exception as e2:
                print(f"⚠️  Pipeline generation also failed: {e2}")
                return None
        
        if not generated_text:
            return None
        
        # Remove the prompt part from generated text
        if "<|assistant|>" in generated_text:
            answer = generated_text.split("<|assistant|>")[-1].strip()
        else:
            # Remove prompt if it's at the start
            if generated_text.startswith(prompt):
                answer = generated_text[len(prompt):].strip()
            else:
                answer = generated_text.strip()
        
        # Clean up the response
        answer = answer.split("<|end|>")[0].strip()
        answer = answer.split("<|user|>")[0].strip()
        answer = answer.split("<|system|>")[0].strip()
        
        # Remove any remaining special tokens
        answer = answer.replace("<|endoftext|>", "").strip()
        
        return answer if answer and len(answer) > 10 else None  # Only return if we have meaningful content
        
    except Exception as e:
        print(f"⚠️  LLM generation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_document_summary(context):
    if not context or "No directly relevant facts found" in context:
        return "I don't have enough information about this document to provide a summary. Please add more knowledge to the knowledge base first."
    facts = []
    for line in context.split('\n'):
        if line.strip() and not line.startswith('**'):
            facts.append(line.strip())
    document_type = "document"
    key_info = []
    for fact in facts:
        fact_lower = fact.lower()
        if 'invoice' in fact_lower or 'bill' in fact_lower:
            document_type = "invoice"
        elif 'contract' in fact_lower or 'agreement' in fact_lower:
            document_type = "contract"
        elif 'report' in fact_lower or 'analysis' in fact_lower:
            document_type = "report"
        elif any(k in fact_lower for k in ['company','organization','name','amount','total','cost','price','date','time','address','location','description','type','id','number','code']):
            key_info.append(fact)
    summary = f"Based on the information in my knowledge base, this appears to be a **{document_type}** document. "
    if key_info:
        summary += "Here are the key details I found:\n\n"
        for info in key_info[:5]:
            summary += f"• {info}\n"
    else:
        summary += "However, I don't have enough specific details to provide a comprehensive summary."
    return summary

def _facts_from_context(context):
    facts = []
    for line in context.split('\n'):
        if line.strip() and not line.startswith('**'):
            facts.append(line.strip())
    return facts

def generate_what_response(message, context):
    facts = _facts_from_context(context)
    if not facts:
        return "I don't have specific information about that in my knowledge base."
    response = "Based on my knowledge base, here's what I can tell you:\n\n"
    for fact in facts[:3]:
        response += f"• {fact}\n"
    if len(facts) > 3:
        response += f"\nI have {len(facts)} total facts about this topic in my knowledge base."
    return response

def generate_who_response(message, context):
    facts = _facts_from_context(context)
    facts = [f for f in facts if any(k in f.lower() for k in ['company','name','person','επωνυμία','εταιρεία'])]
    if not facts:
        return "I don't have specific information about people or companies in my knowledge base."
    return "Here's what I know about people/entities:\n\n" + "\n".join(f"• {f}" for f in facts)

def generate_when_response(message, context):
    facts = _facts_from_context(context)
    facts = [f for f in facts if any(k in f.lower() for k in ['date','ημερομηνία','due','προθεσμία'])]
    if not facts:
        return "I don't have specific date information in my knowledge base."
    return "Here's the date information I have:\n\n" + "\n".join(f"• {f}" for f in facts)

def generate_where_response(message, context):
    facts = _facts_from_context(context)
    facts = [f for f in facts if any(k in f.lower() for k in ['address','διεύθυνση','location','place'])]
    if not facts:
        return "I don't have specific location information in my knowledge base."
    return "Here's the location information I have:\n\n" + "\n".join(f"• {f}" for f in facts)

def generate_amount_response(message, context):
    facts = _facts_from_context(context)
    facts = [f for f in facts if any(k in f.lower() for k in ['amount','total','price','cost','σύνολο','φόρος','€','$'])]
    if not facts:
        return "I don't have specific financial information in my knowledge base."
    return "Here's the financial information I have:\n\n" + "\n".join(f"• {f}" for f in facts)

def generate_general_response(message, context):
    facts = _facts_from_context(context)
    if not facts:
        return "I don't have relevant information about that in my knowledge base."
    response = "Based on my knowledge base, here's what I can tell you:\n\n"
    for fact in facts[:4]:
        response += f"• {fact}\n"
    if len(facts) > 4:
        response += f"\nI have {len(facts)} total relevant facts about this topic."
    return response

def generate_intelligent_response(message, context, system_message):
    message_lower = message.lower()
    if any(phrase in message_lower for phrase in [
        'what is the document about', 'whats the document about', 'what is this about', 'whats this about', 
        'describe the document', 'summarize the document', 'what does this contain', 'what is this about'
    ]):
        return generate_document_summary(context)
    elif message_lower.startswith('what'):
        return generate_what_response(message, context)
    elif message_lower.startswith('who'):
        return generate_who_response(message, context)
    elif message_lower.startswith('when'):
        return generate_when_response(message, context)
    elif message_lower.startswith('where'):
        return generate_where_response(message, context)
    elif any(phrase in message_lower for phrase in ['how much','amount','total','cost','price']):
        return generate_amount_response(message, context)
    else:
        return generate_general_response(message, context)

def respond(message, history, system_message="You are an intelligent assistant that answers questions based on factual information from a knowledge base. You provide clear, accurate, and helpful responses. When you have relevant information, you share it directly. When you don't have enough information, you clearly state this limitation. You always stay grounded in the facts provided and never hallucinate information."):
    # Retrieve relevant context from knowledge graph
    context = retrieve_context(message)
    
    # Try to use LLM for intelligent response
    llm_response = generate_llm_response(message, context, history)
    if llm_response:
        return llm_response
    
    # Fallback to rule-based response if LLM not available
    return generate_intelligent_response(message, context, system_message)


