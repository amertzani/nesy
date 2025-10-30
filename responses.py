from knowledge import retrieve_context

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
    context = retrieve_context(message)
    return generate_intelligent_response(message, context, system_message)


