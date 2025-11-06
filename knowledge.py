"""
Knowledge Graph Management Module
==================================

This module is the CORE of the knowledge extraction and management system.
It handles:
- RDF knowledge graph storage and retrieval
- Knowledge extraction from text (NLP-based)
- Fact management (add, delete, check duplicates)
- Graph visualization data generation

CRITICAL FOR KNOWLEDGE EXTRACTION IMPROVEMENTS:
The main function to improve is: add_to_graph(text) - Line ~400
This function extracts subject-predicate-object triples from raw text.

Data Storage:
- knowledge_graph.pkl: Pickled RDFLib Graph object (binary)
- knowledge_backup.json: JSON backup of all facts

Author: Research Brain Team
Last Updated: 2025-01-15
"""

import os
import json
import pickle
from datetime import datetime
import rdflib
import re
import networkx as nx

# ============================================================================
# CONFIGURATION
# ============================================================================

# Storage file paths
KNOWLEDGE_FILE = "knowledge_graph.pkl"  # Main persistent storage (RDF graph)
BACKUP_FILE = "knowledge_backup.json"   # JSON backup for recovery

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Global RDF graph - in-memory representation of all knowledge
# Structure: RDF triples (subject, predicate, object)
graph = rdflib.Graph()

# Mapping of fact IDs to triples for editing operations
fact_index = {}

def save_knowledge_graph():
    try:
        with open(KNOWLEDGE_FILE, 'wb') as f:
            pickle.dump(graph, f)
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "total_facts": len(graph),
            "facts": []
        }
        for i, (s, p, o) in enumerate(graph):
            backup_data["facts"].append({
                "id": i+1,
                "subject": str(s),
                "predicate": str(p),
                "object": str(o)
            })
        with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        return f" Saved {len(graph)} facts to storage"
    except Exception as e:
        return f" Error saving knowledge: {e}"

def load_knowledge_graph():
    global graph
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, 'rb') as f:
                loaded_graph = pickle.load(f)
            
            original_count = len(loaded_graph)
            print(f"🔍 DEBUG: Loaded {original_count} facts from pickle file")
            
            # Clean up any invalid URIs (with spaces) that might cause issues
            # This fixes old facts that were saved with spaces in URIs
            # Only clean if there are actually invalid URIs - don't remove valid facts!
            cleaned_graph = rdflib.Graph()
            invalid_count = 0
            
            # Use loaded_graph, not the global graph (which might be empty)
            for s, p, o in loaded_graph:
                s_str = str(s)
                p_str = str(p)
                
                # Check if URI has spaces (invalid) - look for "urn:" followed by space
                # Valid URIs shouldn't have spaces after "urn:"
                has_invalid_uri = False
                try:
                    # Check if subject has spaces after urn:
                    if 'urn:' in s_str:
                        subject_part = s_str.split('urn:')[-1]
                        if ' ' in subject_part:
                            has_invalid_uri = True
                    # Check if predicate has spaces after urn:
                    if 'urn:' in p_str:
                        predicate_part = p_str.split('urn:')[-1]
                        if ' ' in predicate_part:
                            has_invalid_uri = True
                except:
                    # If we can't parse, assume valid
                    pass
                
                if has_invalid_uri:
                    # Invalid URI - recreate with proper encoding
                    invalid_count += 1
                    from urllib.parse import quote
                    s_clean = s_str.split(':')[-1] if ':' in s_str else s_str
                    p_clean = p_str.split(':')[-1] if ':' in p_str else p_str
                    s_clean = s_clean.strip().replace(' ', '_')
                    p_clean = p_clean.strip().replace(' ', '_')
                    s_new = rdflib.URIRef(f"urn:{quote(s_clean, safe='')}")
                    p_new = rdflib.URIRef(f"urn:{quote(p_clean, safe='')}")
                    cleaned_graph.add((s_new, p_new, o))
                else:
                    # Valid URI - keep as is
                    cleaned_graph.add((s, p, o))
            
            # Verify cleaned_graph has facts
            cleaned_count = len(cleaned_graph)
            print(f"🔍 DEBUG: cleaned_graph has {cleaned_count} facts after processing")
            
            if cleaned_count == 0 and original_count > 0:
                print(f"⚠️  CRITICAL ERROR: cleaned_graph is empty but original had {original_count} facts!")
                print("⚠️  Using original graph without cleaning")
                # Clear the global graph and add all facts from loaded_graph
                graph.clear()
                for triple in loaded_graph:
                    graph.add(triple)
            else:
                # Always use cleaned_graph (it has all facts, cleaned or not)
                # Clear the global graph and add all facts from cleaned_graph
                # This ensures we update the actual graph object, not replace the reference
                graph.clear()
                for triple in cleaned_graph:
                    graph.add(triple)
            
            # Only update if we actually fixed something
            if invalid_count > 0:
                print(f"⚠️  Fixed {invalid_count} facts with invalid URIs")
                save_knowledge_graph()  # Save the cleaned version
            else:
                # No invalid URIs found - graph already has all facts
                print(f"✅ All {original_count} facts have valid URIs")
            
            final_count = len(graph)
            print(f"🔍 DEBUG: Final graph has {final_count} facts")
            
            if final_count != original_count:
                print(f"⚠️  ERROR: Graph count changed from {original_count} to {final_count}!")
                # This shouldn't happen - restore original
                print("⚠️  Restoring original graph...")
                graph = loaded_graph
                return f"📂 Loaded {len(graph)} facts from storage (restored original)"
            
            # Verify the graph actually has facts
            if len(graph) == 0 and original_count > 0:
                print(f"⚠️  CRITICAL ERROR: Graph is empty after loading {original_count} facts!")
                # Restore from file
                graph = loaded_graph
                return f"📂 Loaded {len(graph)} facts from storage (restored after empty graph error)"
            
            return f"📂 Loaded {len(graph)} facts from storage"
        else:
            return "📂 No existing knowledge file found, starting fresh"
    except Exception as e:
        return f" Error loading knowledge: {e}"

def create_comprehensive_backup():
    try:
        backup_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_facts": len(graph),
                "backup_type": "comprehensive_knowledge_base",
                "graph_size": len(graph)
            },
            "facts": []
        }
        for i, (s, p, o) in enumerate(graph):
            subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
            predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
            object_val = str(o)
            backup_data["facts"].append({
                "id": i + 1,
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "full_subject": str(s),
                "full_predicate": str(p),
                "full_object": str(o)
            })
        with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
    except Exception:
        create_error_backup("unknown")

def create_error_backup(error_message):
    try:
        backup_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_facts": 0,
                "backup_type": "error_backup",
                "error": error_message
            },
            "facts": []
        }
        with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def extract_entities(text):
    entities = []
    capitalized_words = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', text)
    entities.extend(capitalized_words)
    org_patterns = [
        r'([A-Z][a-zA-Z\s]+)\s+(Inc|Ltd|LLC|Corp|Corporation|Company|Co\.|Ltd\.)',
        r'([A-Z][a-zA-Z\s]+)\s+(University|Institute|Lab|Laboratory)',
    ]
    for pattern in org_patterns:
        matches = re.findall(pattern, text)
        entities.extend([m[0].strip() for m in matches])
    location_keywords = ['in ', 'at ', 'near ', 'from ']
    for keyword in location_keywords:
        pattern = f'{keyword}([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        matches = re.findall(pattern, text)
        entities.extend(matches)
    dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', text)
    entities.extend(dates)
    entities = list(set([e.strip() for e in entities if len(e.strip()) > 3]))
    return entities[:50]

def extract_regular_triples_improved(text, entities):
    triples = []
    sentences = re.split(r'[.!?\n]+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue
        improved_patterns = [
            (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(is|are|was|were|becomes|represents|means|refers to|denotes)\s+(.+)', 'relates to'),
            (r'([A-Z][a-zA-Z\s]+)\s+(uses|employs|utilizes|applies)\s+(.+)', 'uses'),
            (r'([A-Z][a-zA-Z\s]+)\s+(develops|created|designed|implemented)\s+(.+)', 'creates'),
            (r'([A-Z][a-zA-Z\s]+)\s+(requires|needs|demands)\s+(.+)', 'requires'),
            (r'([A-Z][a-zA-Z\s]+)\s+(enables|allows|permits)\s+(.+)', 'enables'),
            (r'([A-Z][a-zA-Z\s]+)\s+(affects|impacts|influences|affects)\s+(.+)', 'affects'),
            (r'([A-Z][a-zA-Z\s]+)\s+(found|discovered|identified|observed|detected)\s+(.+)', 'discovered'),
            (r'([A-Z][a-zA-Z\s]+)\s+(studies|analyzes|examines|investigates)\s+(.+)', 'studies'),
            (r'([A-Z][a-zA-Z\s]+)\s+(proposes|suggests|recommends)\s+(.+)', 'proposes'),
            (r'([A-Z][a-zA-Z\s]+)\s+(results in|leads to|causes)\s+(.+)', 'causes'),
            (r'([A-Z][a-zA-Z\s]+)\s+(works with|collaborates with|partnered with)\s+(.+)', 'works with'),
            (r'([A-Z][a-zA-Z\s]+)\s+(located in|based in|situated in)\s+(.+)', 'located in'),
        ]
        for pattern, predicate in improved_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                groups = match.groups()
                subject = groups[0].strip() if len(groups) > 0 else ''
                object_val = groups[-1].strip() if len(groups) > 1 else ''
                subject = re.sub(r'^(the|a|an)\s+', '', subject, flags=re.IGNORECASE).strip()
                object_val = re.sub(r'^(the|a|an)\s+', '', object_val, flags=re.IGNORECASE).strip()
                if subject and object_val and len(subject) > 3 and len(object_val) > 3:
                    triples.append((subject, predicate, object_val))
                    break
        clause_patterns = [
            r'([A-Z][a-zA-Z\s]+)\s+which\s+(.+)',
            r'([A-Z][a-zA-Z\s]+)\s+that\s+(.+)',
            r'([A-Z][a-zA-Z\s]+)\s+who\s+(.+)',
        ]
        for pattern in clause_patterns:
            match = re.search(pattern, sentence)
            if match:
                subject = match.group(1).strip()
                description = match.group(2).strip()
                if subject and description and len(subject) > 3 and len(description) > 3:
                    triples.append((subject, 'has property', description[:150]))
    return triples

def extract_structured_triples(text):
    triples = []
    lines = text.split('\n')
    patterns = [
        (r'date\s*:?\s*([0-9\/\-\.]+)', 'date', 'is'),
        (r'time\s*:?\s*([0-9:]+)', 'time', 'is'),
        (r'created\s*:?\s*([0-9\/\-\.]+)', 'created_date', 'is'),
        (r'modified\s*:?\s*([0-9\/\-\.]+)', 'modified_date', 'is'),
        (r'id\s*:?\s*([A-Z0-9\-]+)', 'id', 'is'),
        (r'number\s*:?\s*([A-Z0-9\-]+)', 'number', 'is'),
        (r'code\s*:?\s*([A-Z0-9\-]+)', 'code', 'is'),
        (r'reference\s*:?\s*([A-Z0-9\-]+)', 'reference', 'is'),
        (r'name\s*:?\s*([A-Za-z\s&.,]+)', 'name', 'is'),
        (r'title\s*:?\s*([A-Za-z\s&.,]+)', 'title', 'is'),
        (r'company\s*:?\s*([A-Za-z\s&.,]+)', 'company', 'is'),
        (r'organization\s*:?\s*([A-Za-z\s&.,]+)', 'organization', 'is'),
        (r'email\s*:?\s*([A-Za-z0-9@\.\-]+)', 'email', 'is'),
        (r'phone\s*:?\s*([0-9\s\-\+\(\)]+)', 'phone', 'is'),
        (r'address\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'address', 'is'),
        (r'description\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'description', 'is'),
        (r'type\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'type', 'is'),
        (r'category\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'category', 'is'),
        (r'status\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'status', 'is'),
        (r'location\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'location', 'is'),
        (r'department\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'department', 'is'),
        (r'section\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'section', 'is'),
        (r'amount\s*:?\s*\$?([0-9,]+\.?[0-9]*)', 'amount', 'is'),
        (r'total\s*:?\s*\$?([0-9,]+\.?[0-9]*)', 'total', 'is'),
        (r'price\s*:?\s*\$?([0-9,]+\.?[0-9]*)', 'price', 'is'),
        (r'cost\s*:?\s*\$?([0-9,]+\.?[0-9]*)', 'cost', 'is'),
    ]
    for line in lines:
        line = line.strip()
        if len(line) < 5:
            continue
        for pattern, subject, predicate in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value and len(value) > 1:
                    triples.append((subject, predicate, value))
                    break
    kv_patterns = [
        r'([A-Za-z\s]+):\s*([A-Za-z0-9\s\$\-\.\/,]+)',
        r'([A-Za-z\s]+)\s*=\s*([A-Za-z0-9\s\$\-\.\/,]+)',
        r'([A-Za-z\s]+)\s*-\s*([A-Za-z0-9\s\$\-\.\/,]+)',
    ]
    for line in lines:
        for pattern in kv_patterns:
            match = re.search(pattern, line)
            if match:
                key = match.group(1).strip().lower().replace(' ', '_')
                value = match.group(2).strip()
                if len(key) > 2 and len(value) > 1:
                    triples.append((key, 'is', value))
    return triples

def extract_regular_triples(text):
    triples = []
    sentences = re.split(r"[.?!\n]", text)
    patterns = [
        r"\s+(is|are|was|were)\s+",
        r"\s+(has|have|had)\s+",
        r"\s+(uses|used|using)\s+",
        r"\s+(creates|created|creating)\s+",
        r"\s+(develops|developed|developing)\s+",
        r"\s+(leads|led|leading)\s+",
        r"\s+(affects|affected|affecting)\s+",
        r"\s+(contains|contained|containing)\s+",
        r"\s+(includes|included|including)\s+",
        r"\s+(requires|required|requiring)\s+",
        r"\s+(causes|caused|causing)\s+",
        r"\s+(results|resulted|resulting)\s+",
        r"\s+(enables|enabled|enabling)\s+",
        r"\s+(provides|provided|providing)\s+",
        r"\s+(supports|supported|supporting)\s+",
        r"\s+(located|situated|found)\s+",
        r"\s+(connects|links|relates)\s+",
        r"\s+(depends|relies|based)\s+",
        r"\s+(represents|symbolizes|stands)\s+",
        r"\s+(describes|explains|defines)\s+",
        r"\s+(refers|referring|referenced)\s+",
        r"\s+(concerns|concerning|concerned)\s+",
        r"\s+(relates|relating|related)\s+",
    ]
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        for pattern in patterns:
            parts = re.split(pattern, sentence, maxsplit=1)
            if len(parts) == 3:
                subj, pred, obj = parts
                subj = re.sub(r'^(the|a|an)\s+', '', subj.strip(), flags=re.IGNORECASE)
                obj = re.sub(r'^(the|a|an)\s+', '', obj.strip(), flags=re.IGNORECASE)
                if subj and pred and obj and len(subj) > 2 and len(obj) > 2:
                    triples.append((subj, pred.strip(), obj))
                    break
    return triples

def extract_triples(text):
    triples = []
    entities = extract_entities(text)
    for entity in entities:
        triples.append((entity, 'type', 'entity'))
    triples.extend(extract_structured_triples(text))
    triples.extend(extract_regular_triples_improved(text, entities))
    triples.extend(extract_regular_triples(text))
    unique_triples = []
    for s, p, o in triples:
        if s and p and o and len(s) > 2 and len(p) > 1 and len(o) > 2:
            s = s.strip()[:100]
            p = p.strip()[:50]
            o = o.strip()[:200]
            if (s, p, o) not in unique_triples:
                unique_triples.append((s, p, o))
    return unique_triples

def fact_exists(subject: str, predicate: str, object_val: str) -> bool:
    """
    Check if a fact (subject, predicate, object) already exists in the graph.
    Handles URI encoding/decoding to properly compare facts.
    Case-insensitive comparison to prevent duplicates with different cases.
    
    Args:
        subject: The subject of the fact
        predicate: The predicate of the fact
        object_val: The object of the fact
    
    Returns:
        True if the fact already exists (case-insensitive), False otherwise
    """
    global graph
    import rdflib
    from urllib.parse import quote, unquote
    
    # Normalize the input (case-insensitive)
    # Convert to lowercase and normalize spaces/underscores for consistent comparison
    subject_normalized = str(subject).strip().lower().replace('_', ' ')
    predicate_normalized = str(predicate).strip().lower().replace('_', ' ')
    object_normalized = str(object_val).strip().lower()
    
    # Check all facts in the graph for case-insensitive match
    for s, p, o in graph:
        # Extract and normalize subject from URI
        s_str = str(s)
        if 'urn:' in s_str:
            # Decode URI: unquote, replace underscores with spaces, lowercase
            s_decoded = unquote(s_str.replace('urn:', '')).replace('_', ' ').lower().strip()
        else:
            s_decoded = str(s).lower().strip().replace('_', ' ')
        
        # Extract and normalize predicate from URI
        p_str = str(p)
        if 'urn:' in p_str:
            p_decoded = unquote(p_str.replace('urn:', '')).replace('_', ' ').lower().strip()
        else:
            p_decoded = str(p).lower().strip().replace('_', ' ')
        
        # Normalize object (already a literal)
        o_decoded = str(o).lower().strip()
        
        # Compare case-insensitively (both normalized to lowercase with spaces)
        if (s_decoded == subject_normalized and 
            p_decoded == predicate_normalized and 
            o_decoded == object_normalized):
            return True
    
    return False

def add_to_graph(text):
    global graph
    import rdflib
    from urllib.parse import quote
    
    new_triples = extract_triples(text)
    added_count = 0
    skipped_count = 0
    
    for s, p, o in new_triples:
        # Check if fact already exists before adding
        if fact_exists(s, p, o):
            skipped_count += 1
            continue
        
        # Properly encode URIs like create_fact_endpoint does
        # Replace spaces with underscores and URL-encode to avoid RDFLib warnings
        subject_clean = str(s).strip().replace(' ', '_')
        predicate_clean = str(p).strip().replace(' ', '_')
        object_value = str(o).strip()
        
        # Create URIs (encode spaces to avoid RDFLib warnings)
        subject_uri = rdflib.URIRef(f"urn:{quote(subject_clean, safe='')}")
        predicate_uri = rdflib.URIRef(f"urn:{quote(predicate_clean, safe='')}")
        object_literal = rdflib.Literal(object_value)
        
        graph.add((subject_uri, predicate_uri, object_literal))
        added_count += 1
    
    save_knowledge_graph()
    
    if skipped_count > 0:
        return f" Added {added_count} new triples, skipped {skipped_count} duplicates. Total facts stored: {len(graph)}.\n Saved"
    return f" Added {added_count} new triples. Total facts stored: {len(graph)}.\n Saved"

def retrieve_context(question, limit=10):
    matches = []
    qwords = [w for w in question.lower().split() if w not in {
        'the','a','an','and','or','but','in','on','at','to','for','of','with','by','is','are','was','were','be','been','have','has','had','do','does','did','will','would','could','should','may','might','can','what','how','when','where','why','who'
    } and len(w) > 2]
    scored_matches = []
    for s, p, o in graph:
        subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
        predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
        object_val = str(o)
        fact_text = f"{subject} {predicate} {object_val}".lower()
        score = 0
        for word in qwords:
            if word in fact_text:
                score += 1
                if word == subject.lower() or word == predicate.lower():
                    score += 2
        if score > 0:
            scored_matches.append((score, f"{subject} {predicate} {object_val}"))
    scored_matches.sort(key=lambda x: x[0], reverse=True)
    matches = [m[1] for m in scored_matches[:limit]]
    if matches:
        result = "**Relevant Knowledge:**\n"
        for i, match in enumerate(matches, 1):
            result += f"{i}. {match}\n"
        return result
    return "**No directly relevant facts found.**\n\nTry asking about topics that might be in your knowledge base, or add more knowledge first!"

def show_graph_contents():
    if len(graph) == 0:
        return "**Knowledge Graph Status: EMPTY**\n\n**How to build your knowledge base:**\n1. **Add text directly** - Paste any text in the 'Add Knowledge from Text' box above\n2. **Upload documents** - Use the file upload to process PDF, DOCX, TXT, CSV files\n3. **Extract facts** - The system will automatically extract knowledge from your content\n4. **Build knowledge** - Add more text or files to expand your knowledge base\n5. **Save knowledge** - Use 'Save Knowledge' to persist your data\n\n**Start by adding some text or uploading a document!**"
    facts_by_subject = {}
    all_facts = []
    for s, p, o in graph:
        subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
        predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
        object_val = str(o)
        fact_text = f"{subject} {predicate} {object_val}"
        all_facts.append(fact_text)
        facts_by_subject.setdefault(subject, []).append(f"{predicate} {object_val}")
    result = f"**Knowledge Graph Overview**\n"
    result += f"**Total Facts:** {len(graph)}\n"
    result += f"**Unique Subjects:** {len(facts_by_subject)}\n\n"
    result += "## **Knowledge by Subject:**\n\n"
    for i, (subject, facts) in enumerate(facts_by_subject.items()):
        if i >= 10:
            remaining = len(facts_by_subject) - 10
            result += f"... and {remaining} more subjects\n"
            break
        result += f"**{subject}:**\n"
        for fact in facts:
            result += f"  • {fact}\n"
        result += "\n"
    result += "## **All Facts:**\n\n"
    for i, fact in enumerate(all_facts[:20]):
        result += f"{i+1}. {fact}\n"
    if len(all_facts) > 20:
        result += f"\n... and {len(all_facts) - 20} more facts"
    return result

def visualize_knowledge_graph():
    if len(graph) == 0:
        return "<p>No knowledge in graph. Add some text or upload a document first!</p>"
    try:
        G = nx.Graph()
        fact_data = {}
        for s, p, o in graph:
            subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
            predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
            object_val = str(o)
            subject_short = (subject[:30] + "...") if len(subject) > 30 else subject
            object_short = (object_val[:30] + "...") if len(object_val) > 30 else object_val
            if subject not in G:
                G.add_node(subject, display=subject_short, node_type='subject')
            if object_val not in G:
                G.add_node(object_val, display=object_short, node_type='object')
            G.add_edge(subject, object_val, label=predicate)
            fact_data[(subject, object_val)] = f"{subject} {predicate} {object_val}"
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        import numpy as np
        x_positions = [pos[n][0] for n in G.nodes()]
        y_positions = [pos[n][1] for n in G.nodes()]
        x_min, x_max = min(x_positions), max(x_positions)
        y_min, y_max = min(y_positions), max(y_positions)
        scale = min(500 / (x_max - x_min), 400 / (y_max - y_min)) if (x_max - x_min) > 0 and (y_max - y_min) > 0 else 50
        offset_x = 350
        offset_y = 300
        svg_elements = []
        for edge in G.edges():
            x1 = pos[edge[0]][0] * scale + offset_x
            y1 = pos[edge[0]][1] * scale + offset_y
            x2 = pos[edge[1]][0] * scale + offset_x
            y2 = pos[edge[1]][1] * scale + offset_y
            edge_data = G[edge[0]][edge[1]]
            label = edge_data.get('label', 'has')
            svg_elements.append(f"""
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" 
                  stroke="#999" stroke-width="2" opacity="0.5">
                <title>{label}</title>
            </line>
            """)
        node_info = []
        for i, node in enumerate(G.nodes()):
            x = pos[node][0] * scale + offset_x
            y = pos[node][1] * scale + offset_y
            display_name = G.nodes[node].get('display', node)
            node_type = G.nodes[node].get('node_type', 'unknown')
            color = '#4CAF50' if node_type == 'subject' else ('#2196F3' if node_type == 'object' else '#546E7A')
            neighbors = list(G.neighbors(node))
            neighbor_count = len(neighbors)
            node_info.append(f"""
            <circle cx="{x}" cy="{y}" r="{max(40, min(30, neighbor_count * 2 + 20))}" 
                    fill="{color}" stroke="#fff" stroke-width="2">
                <title>{display_name} ({neighbor_count} connections)</title>
            </circle>
            <text x="{x}" y="{y+6}" text-anchor="middle" font-size="15" font-weight="bold" fill="#000" 
                  pointer-events="none">{display_name[:15]}</text>
            """)
        svg_content = '\n'.join(svg_elements + node_info)
        html = f"""
        <div style="width: 100%; min-height: 700px; max-height: 800px; background: white; border: 2px solid #ddd; border-radius: 10px; padding: 20px; position: relative; overflow: auto;">
            <svg width="100%" height="550" style="border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9; display: block;">
                {svg_content}
            </svg>
        </div>
        """
        return html
    except Exception as e:
        return f"<p style='color: red; padding: 20px;'>Error creating visualization: {e}</p>"

def delete_all_knowledge():
    global graph
    count = len(graph)
    graph = rdflib.Graph()
    save_knowledge_graph()
    return f"🗑️ Deleted all {count} facts from the knowledge graph. Graph is now empty."

def delete_knowledge_by_keyword(keyword):
    global graph
    if not keyword or keyword.strip() == "":
        return "⚠️ Please enter a keyword to search for."
    keyword = keyword.strip().lower()
    deleted_count = 0
    facts_to_remove = []
    for s, p, o in graph:
        fact_text = f"{s} {p} {o}".lower()
        if keyword in fact_text:
            facts_to_remove.append((s, p, o))
    for fact in facts_to_remove:
        graph.remove(fact)
        deleted_count += 1
    if deleted_count > 0:
        save_knowledge_graph()
        return f"🗑️ Deleted {deleted_count} facts containing '{keyword}'"
    else:
        return f"ℹ️ No facts found containing '{keyword}'"

def delete_recent_knowledge(count=5):
    global graph
    if len(graph) == 0:
        return "ℹ️ Knowledge graph is already empty."
    facts = list(graph)
    facts_to_remove = facts[-count:] if count < len(facts) else facts
    for fact in facts_to_remove:
        graph.remove(fact)
    save_knowledge_graph()
    return f"🗑️ Deleted {len(facts_to_remove)} most recent facts"

def list_facts_for_editing():
    global fact_index
    fact_index = {}
    options = []
    for i, (s, p, o) in enumerate(list(graph), start=1):
        subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
        predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
        object_val = str(o)
        label = f"{i}. {subject} {predicate} {object_val}"
        options.append(label)
        fact_index[i] = (s, p, o)
    return options

def load_fact_by_label(fact_label):
    if not fact_label:
        return None
    try:
        fact_id = int(fact_label.split('.', 1)[0].strip())
        return fact_index.get(fact_id)
    except Exception:
        return None

def import_knowledge_from_json_file(file):
    try:
        if file is None:
            return "⚠️ No file selected."
        file_path = file.name if hasattr(file, 'name') else str(file)
        if not os.path.exists(file_path):
            return f"⚠️ File not found: {file_path}"
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'facts' in data:
            facts = data['facts']
        elif isinstance(data, list):
            facts = data
        else:
            return "❌ Unsupported JSON structure. Expect an object with 'facts' or a list of facts."
        added = 0
        skipped = 0
        for fact in facts:
            try:
                subject = fact.get('subject') or fact.get('full_subject')
                predicate = fact.get('predicate') or fact.get('full_predicate')
                obj = fact.get('object') or fact.get('full_object')
                if not subject or not predicate or obj is None:
                    skipped += 1
                    continue
                
                # Extract subject/predicate from URI if needed (handle both formats)
                # If it's already a URI like "urn:subject", extract the subject part
                subject_str = str(subject)
                if subject_str.startswith('urn:'):
                    # Extract the actual subject text from URI
                    from urllib.parse import unquote
                    subject_str = unquote(subject_str.replace('urn:', '')).replace('_', ' ')
                else:
                    subject_str = str(subject)
                
                predicate_str = str(predicate)
                if predicate_str.startswith('urn:'):
                    from urllib.parse import unquote
                    predicate_str = unquote(predicate_str.replace('urn:', '')).replace('_', ' ')
                else:
                    predicate_str = str(predicate)
                
                obj_str = str(obj)
                
                # Check if fact already exists using fact_exists function
                if fact_exists(subject_str, predicate_str, obj_str):
                    skipped += 1
                    continue
                
                # Create URIs using the same encoding as other functions
                from urllib.parse import quote
                subject_clean = subject_str.strip().replace(' ', '_')
                predicate_clean = predicate_str.strip().replace(' ', '_')
                s_ref = rdflib.URIRef(f"urn:{quote(subject_clean, safe='')}")
                p_ref = rdflib.URIRef(f"urn:{quote(predicate_clean, safe='')}")
                o_lit = rdflib.Literal(obj_str)
                
                graph.add((s_ref, p_ref, o_lit))
                added += 1
            except Exception as e:
                skipped += 1
                print(f"⚠️  Error importing fact: {e}")
        save_knowledge_graph()
        if skipped > 0:
            return f"✅ Imported {added} new facts, skipped {skipped} duplicates. Total facts: {len(graph)}."
        return f"✅ Imported {added} facts. Skipped {skipped}. Total facts: {len(graph)}."
    except Exception as e:
        return f"❌ Import failed: {e}"


