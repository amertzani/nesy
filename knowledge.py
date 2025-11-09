"""
Knowledge Graph Management Module
==================================

This module is the CORE of the knowledge extraction and management system.
It handles:
- RDF knowledge graph storage and retrieval
- Knowledge extraction from text (NLP-based with optional Triplex LLM)
- Fact management (add, delete, check duplicates)
- Graph visualization data generation
- Entity normalization and cleaning

CRITICAL FOR KNOWLEDGE EXTRACTION IMPROVEMENTS:
The main function to improve is: add_to_graph(text) - Line ~1100
This function extracts subject-predicate-object triples from raw text.

Triplex Integration (Optional LLM-based extraction):
- Enable by setting environment variable: USE_TRIPLEX=true
- Requires: transformers, torch, accelerate (see requirements.txt)
- Uses SciPhi/Triplex model (4B parameters) for high-quality extraction
- Falls back to regex-based extraction if Triplex is unavailable or disabled
- Model is loaded lazily on first use (may take time on first extraction)

Data Storage:
- knowledge_graph.pkl: Pickled RDFLib Graph object (binary)
- knowledge_backup.json: JSON backup of all facts
- entity_normalization.json: Entity normalization mappings

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
from collections import defaultdict

# Optional: Try to import transformers for Triplex model
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRIPLEX_AVAILABLE = True
    TRIPLEX_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TRIPLEX_AVAILABLE = False
    TRIPLEX_DEVICE = "cpu"
    print("⚠️  Transformers/Torch not available. Triplex extraction disabled. Using regex-based extraction.")

# spaCy is disabled due to Windows compilation issues
# Using improved regex-based extraction instead
SPACY_AVAILABLE = False
SPACY_NLP = None

# ============================================================================
# CONFIGURATION
# ============================================================================

# Storage file paths
KNOWLEDGE_FILE = "knowledge_graph.pkl"  # Main persistent storage (RDF graph)
BACKUP_FILE = "knowledge_backup.json"   # JSON backup for recovery
NORMALIZATION_FILE = "entity_normalization.json"  # Entity normalization mappings

# Triplex model configuration
USE_TRIPLEX = os.getenv("USE_TRIPLEX", "false").lower() == "true"  # Enable via environment variable
TRIPLEX_MODEL_NAME = "sciphi/triplex"
TRIPLEX_MODEL = None
TRIPLEX_TOKENIZER = None

# spaCy NER configuration (disabled - using improved regex instead)
USE_SPACY = False

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
    # Load normalization map on startup
    load_normalization_map()
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, 'rb') as f:
                loaded_graph = pickle.load(f)
            
            original_count = len(loaded_graph)
            
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
            
            if cleaned_count == 0 and original_count > 0:
                print(f"⚠️  CRITICAL ERROR: cleaned_graph is empty but original had {original_count} facts!")
                print("⚠️  Using original graph without cleaning")
                # Clear the global graph and add all facts from loaded_graph
                # RDFLib Graph doesn't have clear(), so remove all triples
                graph.remove((None, None, None))
                for triple in loaded_graph:
                    graph.add(triple)
            else:
                # Always use cleaned_graph (it has all facts, cleaned or not)
                # Clear the global graph and add all facts from cleaned_graph
                # This ensures we update the actual graph object, not replace the reference
                # RDFLib Graph doesn't have clear(), so remove all triples
                graph.remove((None, None, None))
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
            
            if final_count != original_count:
                print(f"⚠️  ERROR: Graph count changed from {original_count} to {final_count}!")
                # This shouldn't happen - restore original
                print("⚠️  Restoring original graph...")
                # Clear current graph and restore from loaded
                graph.remove((None, None, None))
                for triple in loaded_graph:
                    graph.add(triple)
                return f"📂 Loaded {len(graph)} facts from storage (restored original)"
            
            # Verify the graph actually has facts
            if len(graph) == 0 and original_count > 0:
                print(f"⚠️  CRITICAL ERROR: Graph is empty after loading {original_count} facts!")
                # Restore from file - use remove/add pattern instead of assignment
                graph.remove((None, None, None))
                for triple in loaded_graph:
                    graph.add(triple)
                print(f"✅ Restored {len(graph)} facts from loaded_graph")
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
        pattern = f'{keyword}([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)?)'
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

# ============================================================================
# ENTITY NORMALIZATION
# ============================================================================

# Entity normalization mappings (case-insensitive)
# Format: {variant: canonical_form}
entity_normalization_map = {}

def load_normalization_map():
    """Load entity normalization mappings from file"""
    global entity_normalization_map
    if os.path.exists(NORMALIZATION_FILE):
        try:
            with open(NORMALIZATION_FILE, 'r', encoding='utf-8') as f:
                entity_normalization_map = json.load(f)
        except:
            entity_normalization_map = {}
    else:
        # Initialize with common mappings
        entity_normalization_map = {
            # Abbreviations - Academic/Research
            'hw': 'homework',
            'homework': 'homework',
            'wp': 'work package',
            'wpl': 'work package',
            'work package': 'work package',
            'workplan': 'work plan',
            'work plan': 'work plan',
            'detailed work plan': 'work plan',
            'kom': 'kick-off meeting',
            'kom notes': 'kick-off meeting notes',
            'kg': 'knowledge graph',
            'knowledge graph': 'knowledge graph',
            
            # Geographic aliases
            'uk': 'united kingdom',
            'united kingdom': 'united kingdom',
            'great britain': 'united kingdom',
            'gb': 'united kingdom',
            'britain': 'united kingdom',
            'usa': 'united states',
            'us': 'united states',
            'united states': 'united states',
            'united states of america': 'united states',
            
            # Institutional
            'icl': 'imperial college london',
            'imperial college london': 'imperial college london',
            'imperial college': 'imperial college london',
            
            # Meeting variations
            'next meeting': 'meeting',
            'next meeting proposal': 'meeting',
            'meeting': 'meeting',
            'meeting proposal': 'meeting',
            
            # Day names
            'monday': 'monday',
            'mon': 'monday',
            'tuesday': 'tuesday',
            'tue': 'tuesday',
            'wednesday': 'wednesday',
            'wed': 'wednesday',
            'thursday': 'thursday',
            'thu': 'thursday',
            'friday': 'friday',
            'fri': 'friday',
            'saturday': 'saturday',
            'sat': 'saturday',
            'sunday': 'sunday',
            'sun': 'sunday',
        }
        save_normalization_map()

def save_normalization_map():
    """Save entity normalization mappings to file"""
    try:
        with open(NORMALIZATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(entity_normalization_map, f, indent=2, ensure_ascii=False)
    except:
        pass

def normalize_entity(entity):
    """
    Normalize an entity to its canonical form.
    Handles case-insensitive matching, abbreviations, and aliases.
    
    Examples:
    - "homework" -> "homework"
    - "HW" -> "homework"
    - "hw" -> "homework"
    - "WP" -> "work package"
    - "UK" -> "united kingdom"
    - "workplan" -> "work plan"
    """
    if not entity:
        return entity
    
    entity_lower = entity.lower().strip()
    
    # Check normalization map
    if entity_lower in entity_normalization_map:
        return entity_normalization_map[entity_lower]
    
    # Check if any variant matches (case-insensitive)
    for variant, canonical in entity_normalization_map.items():
        if entity_lower == variant.lower() or entity_lower.startswith(variant.lower() + ' ') or entity_lower.endswith(' ' + variant.lower()):
            return canonical
    
    # If no normalization found, return cleaned entity
    return entity

def add_normalization_mapping(variant, canonical):
    """Add a new normalization mapping"""
    global entity_normalization_map
    variant_lower = variant.lower().strip()
    canonical_lower = canonical.lower().strip()
    
    if variant_lower != canonical_lower:
        entity_normalization_map[variant_lower] = canonical_lower
        save_normalization_map()

def learn_normalizations_from_facts():
    """
    Learn normalization mappings from existing facts in the graph.
    Identifies entities that are similar (case-insensitive, abbreviations, etc.)
    and suggests normalizations.
    """
    global graph, entity_normalization_map
    
    # Group entities by normalized form (case-insensitive)
    entity_groups = defaultdict(list)
    
    for s, p, o in graph:
        # Extract and normalize entities
        s_str = str(s).split(':')[-1] if ':' in str(s) else str(s)
        o_str = str(o)
        
        # Normalize to lowercase for grouping
        s_normalized = s_str.lower().strip()
        o_normalized = o_str.lower().strip()
        
        # Group similar entities (exact match after normalization)
        entity_groups[s_normalized].append(s_str)
        entity_groups[o_normalized].append(o_str)
    
    # Find potential normalizations (entities that differ only by case or minor variations)
    for normalized, variants in entity_groups.items():
        unique_variants = list(set(variants))
        if len(unique_variants) > 1:
            # Use the most common variant as canonical
            canonical = max(set(variants), key=variants.count)
            canonical_lower = canonical.lower().strip()
            
            # Add mappings for all variants
            for variant in unique_variants:
                variant_lower = variant.lower().strip()
                if variant_lower != canonical_lower and variant_lower not in entity_normalization_map:
                    entity_normalization_map[variant_lower] = canonical_lower
    
    save_normalization_map()

def extract_core_entity(entity, context=""):
    """
    Extract the core entity from a longer phrase, moving all background/contextual
    information to the details field.
    
    Examples:
    - "3rd Monday of each month 10-11 CET" -> "monday" (with details: "3rd of each month 10-11 CET")
    - "Detailed Work Plan with Involved Partners till Month 18" -> "work plan" (with details: "Detailed with Involved Partners till Month 18")
    - "next meeting proposal" -> "meeting" (with details: "next proposal")
    - "Any discrepancies apart from the ones discovered in KoM notes" -> "discrepancies" (with details: "Any apart from the ones discovered in KoM notes")
    """
    if not entity:
        return entity, None
    
    entity = entity.strip()
    original = entity
    
    # Store all extracted background information here
    details_parts = []
    core_entity = entity
    
    # Extract time ranges like "10-11 CET" or "10-11"
    time_range_pattern = r'\b(\d{1,2}-\d{1,2}(?:\s+(?:CET|UTC|GMT|EST|PST|PDT|EDT|CDT|MDT))?)\b'
    time_matches = re.finditer(time_range_pattern, entity, re.IGNORECASE)
    for match in time_matches:
        details_parts.append(match.group(0))
        core_entity = core_entity.replace(match.group(0), '').strip()
    
    # Extract month references like "till Month 18" or "month 18"
    month_pattern = r'\b(till\s+month\s+\d+|month\s+\d+|each\s+month|of\s+each\s+month)\b'
    month_matches = re.finditer(month_pattern, entity, re.IGNORECASE)
    for match in month_matches:
        details_parts.append(match.group(0))
        core_entity = core_entity.replace(match.group(0), '').strip()
    
    # Extract ordinal day patterns like "3rd Monday" (but keep the day name for core entity)
    ordinal_day_pattern = r'\b(\d{1,2}(?:st|nd|rd|th)?)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b'
    ordinal_match = re.search(ordinal_day_pattern, entity, re.IGNORECASE)
    if ordinal_match:
        ordinal_part = ordinal_match.group(1)
        details_parts.append(ordinal_part)
        core_entity = core_entity.replace(ordinal_part, '').strip()
    
    # Extract core noun phrases
    # Remove descriptive adjectives and qualifiers
    descriptive_patterns = [
        r'^(detailed|comprehensive|extensive|complete|full|partial|brief|short|long)\s+',
        r'\s+(with|including|containing|having|featuring)\s+.+$',
        r'\s+(till|until|up to|through|by)\s+.+$',
        r'\s+(proposal|suggestion|recommendation|idea|concept|plan|draft|version)\s*$',
    ]
    
    for pattern in descriptive_patterns:
        matches = re.finditer(pattern, core_entity, re.IGNORECASE)
        for match in matches:
            details_parts.append(match.group(0).strip())
            core_entity = re.sub(pattern, '', core_entity, flags=re.IGNORECASE).strip()
    
    # Extract day names (e.g., "3rd Monday of each month 10-11 CET" -> "monday")
    # Check original entity first for day names
    day_pattern = r'\b(\d{1,2}(?:st|nd|rd|th)?\s+)?(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b'
    day_match = re.search(day_pattern, original, re.IGNORECASE)
    if day_match:
        day_name = day_match.group(2).lower() if day_match.group(2) else None
        if day_name:
            # Normalize day names
            day_map = {'mon': 'monday', 'tue': 'tuesday', 'wed': 'wednesday', 
                      'thu': 'thursday', 'fri': 'friday', 'sat': 'saturday', 'sun': 'sunday'}
            day_name = day_map.get(day_name, day_name)
            # If we found a day in a complex phrase, extract it as core entity
            # Example: "3rd Monday of each month 10-11 CET" -> "monday"
            if len(original.split()) > 3 or len(details_parts) > 0:
                core_entity = day_name
                if not details_parts:
                    details_parts = [original]
            elif day_name not in core_entity.lower():
                core_entity = day_name
                if not details_parts:
                    details_parts = [original]
    
    # Extract work plan / work package (e.g., "Detailed Work Plan with Involved Partners till Month 18" -> "work plan")
    if 'work plan' in core_entity.lower() or 'workplan' in core_entity.lower() or 'work package' in core_entity.lower():
        core_entity = 'work plan'
        if original.lower() != 'work plan' and not details_parts:
            details_parts = [original]
    
    # Extract meeting (e.g., "next meeting proposal" -> "meeting")
    if 'meeting' in core_entity.lower():
        core_entity = 'meeting'
        if original.lower() != 'meeting' and not details_parts:
            details_parts = [original]
    
    # Clean up core entity
    core_entity = clean_entity(core_entity)
    
    # Normalize the core entity
    core_entity = normalize_entity(core_entity)
    
    # Build details - include all background information
    details = None
    if details_parts:
        # Join all extracted details
        details = ' '.join(details_parts).strip()
    elif original.lower() != core_entity.lower():
        # If we modified the entity, store the original as details
        details = original
    
    # If we have context (full sentence), include it for better understanding
    # This helps preserve the full meaning even when we extract core entities
    if context and context.strip() and context != original:
        if details:
            # Append context if it's not already included
            if context[:100] not in details:
                details = f"{details} | Context: {context[:200]}"
        else:
            details = f"Context: {context[:200]}"
    
    return core_entity, details

def clean_entity(entity):
    """
    Clean an entity by removing articles, qualifiers, and common prefixes.
    Examples:
    - "Any discrepancies apart from the ones" -> "discrepancies"
    - "in KoM notes" -> "KoM notes"
    - "the Machine Learning" -> "Machine Learning"
    """
    if not entity:
        return entity
    
    entity = entity.strip()
    
    # Remove common prefixes and qualifiers
    prefixes_to_remove = [
        r'^(any|all|some|each|every|both|either|neither)\s+',
        r'^(the|a|an)\s+',
        r'^(this|that|these|those)\s+',
        r'^(other|another|same|different)\s+',
    ]
    
    for prefix in prefixes_to_remove:
        entity = re.sub(prefix, '', entity, flags=re.IGNORECASE)
    
    # Remove trailing qualifiers like "apart from the ones", "except for", etc.
    qualifiers_to_remove = [
        r'\s+apart\s+from\s+the\s+ones?.*$',
        r'\s+except\s+for.*$',
        r'\s+other\s+than.*$',
        r'\s+besides.*$',
    ]
    
    for qualifier in qualifiers_to_remove:
        entity = re.sub(qualifier, '', entity, flags=re.IGNORECASE)
    
    # Remove leading prepositions if they're not part of a proper noun
    # But keep "in" if it's part of a proper noun like "in KoM notes" -> "KoM notes"
    entity = re.sub(r'^(in|at|on|from|to|for|with|by)\s+', '', entity, flags=re.IGNORECASE)
    
    return entity.strip()

# ============================================================================
# TRIPLEX MODEL INTEGRATION
# ============================================================================

def load_triplex_model():
    """Load Triplex model for knowledge extraction (lazy loading)"""
    global TRIPLEX_MODEL, TRIPLEX_TOKENIZER
    
    if not TRIPLEX_AVAILABLE:
        return False
    
    if TRIPLEX_MODEL is not None and TRIPLEX_TOKENIZER is not None:
        return True
    
    try:
        print(f"🔄 Loading Triplex model ({TRIPLEX_MODEL_NAME})...")
        print(f"   Device: {TRIPLEX_DEVICE}")
        
        TRIPLEX_TOKENIZER = AutoTokenizer.from_pretrained(TRIPLEX_MODEL_NAME, trust_remote_code=True)
        TRIPLEX_MODEL = AutoModelForCausalLM.from_pretrained(
            TRIPLEX_MODEL_NAME, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if TRIPLEX_DEVICE == "cuda" else torch.float32,
            device_map="auto" if TRIPLEX_DEVICE == "cuda" else None
        )
        
        if TRIPLEX_DEVICE == "cpu":
            TRIPLEX_MODEL = TRIPLEX_MODEL.to(TRIPLEX_DEVICE)
        
        TRIPLEX_MODEL.eval()
        print(f"✅ Triplex model loaded successfully on {TRIPLEX_DEVICE}")
        return True
    except Exception as e:
        print(f"⚠️  Failed to load Triplex model: {e}")
        print("   Falling back to regex-based extraction")
        return False

def extract_with_triplex(text, entity_types=None, predicates=None):
    """
    Extract knowledge graph triplets using Triplex model.
    
    Args:
        text: Input text to extract from
        entity_types: Optional list of entity types to focus on
        predicates: Optional list of predicates to focus on
    
    Returns:
        List of tuples: (subject, predicate, object, details)
    """
    global TRIPLEX_MODEL, TRIPLEX_TOKENIZER
    
    if not TRIPLEX_AVAILABLE or not USE_TRIPLEX:
        return []
    
    # Lazy load model
    if not load_triplex_model():
        return []
    
    try:
        # Default entity types and predicates if not provided
        if entity_types is None:
            entity_types = [
                "PERSON", "ORGANIZATION", "LOCATION", "DATE", "CONCEPT", 
                "PROJECT", "DOCUMENT", "MEETING", "TASK", "DELIVERABLE"
            ]
        
        if predicates is None:
            predicates = [
                "is", "has", "uses", "creates", "requires", "enables", 
                "affects", "discovered", "studies", "proposes", "causes",
                "works with", "located in", "based in", "part of", "related to"
            ]
        
        # Format input according to Triplex format
        input_format = """Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.
      
        **Entity Types:**
        {entity_types}
        
        **Predicates:**
        {predicates}
        
        **Text:**
        {text}
        """
        
        message = input_format.format(
            entity_types=json.dumps({"entity_types": entity_types}),
            predicates=json.dumps({"predicates": predicates}),
            text=text[:2000]  # Limit text length for performance
        )
        
        messages = [{'role': 'user', 'content': message}]
        input_ids = TRIPLEX_TOKENIZER.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(TRIPLEX_DEVICE)
        
        # Generate with reasonable limits
        with torch.no_grad():
            # Fix for compatibility: disable cache to avoid DynamicCache issues
            try:
                output_ids = TRIPLEX_MODEL.generate(
                    input_ids,
                    max_new_tokens=512,  # Use max_new_tokens instead of max_length
                    do_sample=False,
                    pad_token_id=TRIPLEX_TOKENIZER.eos_token_id,
                    eos_token_id=TRIPLEX_TOKENIZER.eos_token_id,
                    use_cache=False  # Disable cache to avoid compatibility issues
                )
            except Exception as gen_error:
                print(f"⚠️  Generation error (first attempt): {gen_error}")
                # Fallback: try with attention_mask
                try:
                    attention_mask = torch.ones_like(input_ids)
                    output_ids = TRIPLEX_MODEL.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=TRIPLEX_TOKENIZER.eos_token_id,
                        eos_token_id=TRIPLEX_TOKENIZER.eos_token_id,
                        use_cache=False
                    )
                except Exception as gen_error2:
                    print(f"⚠️  Generation error (fallback): {gen_error2}")
                    raise gen_error2
        
        output = TRIPLEX_TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
        
        # Parse output to extract triplets
        # Triplex typically returns JSON or structured text
        triples = []
        
        # Try to parse as JSON first
        try:
            # Extract JSON from output if present
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict) and 'triplets' in parsed:
                    for triple in parsed['triplets']:
                        if 'subject' in triple and 'predicate' in triple and 'object' in triple:
                            s = triple['subject']
                            p = triple['predicate']
                            o = triple['object']
                            # Extract core entities and normalize
                            core_s, s_details = extract_core_entity(s, text)
                            core_o, o_details = extract_core_entity(o, text)
                            core_s = normalize_entity(core_s)
                            core_o = normalize_entity(core_o)
                            
                            # Combine details
                            details_parts = []
                            if s_details:
                                details_parts.append(f"Subject: {s_details}")
                            if o_details:
                                details_parts.append(f"Object: {o_details}")
                            if not details_parts:
                                details_parts.append(text[:200])  # Use original text as context
                            
                            details = ' | '.join(details_parts) if details_parts else None
                            triples.append((core_s, p, core_o, details))
        except:
            pass
        
        # Fallback: Try to extract triplets from text format
        # Pattern: (subject, predicate, object) or subject | predicate | object
        triplet_patterns = [
            r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)',
            r'([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)',
            r'Subject:\s*([^\n]+)\s+Predicate:\s*([^\n]+)\s+Object:\s*([^\n]+)',
        ]
        
        for pattern in triplet_patterns:
            matches = re.finditer(pattern, output)
            for match in matches:
                s = match.group(1).strip()
                p = match.group(2).strip()
                o = match.group(3).strip()
                
                # Extract core entities and normalize
                core_s, s_details = extract_core_entity(s, text)
                core_o, o_details = extract_core_entity(o, text)
                core_s = normalize_entity(core_s)
                core_o = normalize_entity(core_o)
                
                # Combine details
                details_parts = []
                if s_details:
                    details_parts.append(f"Subject: {s_details}")
                if o_details:
                    details_parts.append(f"Object: {o_details}")
                if not details_parts:
                    details_parts.append(text[:200])
                
                details = ' | '.join(details_parts) if details_parts else None
                triples.append((core_s, p, core_o, details))
        
        return triples
        
    except Exception as e:
        print(f"⚠️  Error in Triplex extraction: {e}")
        import traceback
        traceback.print_exc()
        return []

def extract_with_improved_patterns(text):
    """
    Extract knowledge graph triplets using improved regex patterns.
    This is an enhanced version that uses better linguistic patterns
    to identify subject-predicate-object relationships.
    
    Returns:
        List of tuples: (subject, predicate, object, details)
    """
    triples = []
    sentences = re.split(r'[.!?\n]+', text)
    
    # Common verb patterns for relationships
    verb_patterns = [
        # "X works at Y", "X is Y", "X has Y"
        (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(works?\s+at|is\s+at|located\s+at|based\s+at)\s+(.+)', 'works_at'),
        (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(is|are|was|were|becomes|represents|means|refers\s+to|denotes)\s+(.+)', 'is'),
        (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(has|have|had|contains|includes)\s+(.+)', 'has'),
        
        # "X started in Y", "X created Y", "X developed Y"
        (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(started|began|commenced)\s+(in|on|at)\s+(.+)', 'started_in'),
        (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(created|developed|designed|built|implemented)\s+(.+)', 'creates'),
        
        # "X uses Y", "X requires Y", "X needs Y"
        (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(uses?|employs?|utilizes?|applies?)\s+(.+)', 'uses'),
        (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(requires?|needs?|demands?)\s+(.+)', 'requires'),
        
        # "X enables Y", "X allows Y", "X permits Y"
        (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(enables?|allows?|permits?)\s+(.+)', 'enables'),
        
        # "X affects Y", "X impacts Y", "X influences Y"
        (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(affects?|impacts?|influences?)\s+(.+)', 'affects'),
        
        # "X discovered Y", "X found Y", "X identified Y"
        (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(discovered?|found|identified?|detected?)\s+(.+)', 'discovered'),
        
        # "X is on Y", "X happens on Y", "X occurs on Y"
        (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(is|are|happens?|occurs?|takes?\s+place)\s+(on|in|at)\s+(.+)', 'occurs_on'),
        
        # "X is to Y" (purpose/goal)
        (r'([A-Z][a-zA-Z\s]+(?:,\s+[A-Z][a-zA-Z\s]+)*)\s+(is|are)\s+to\s+(.+)', 'purpose_is'),
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        
        # Try each pattern
        for pattern, pred_type in verb_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    subject = groups[0].strip()
                    if len(groups) == 2:
                        obj = groups[1].strip()
                        predicate = pred_type
                    else:
                        # Pattern has preposition, skip it for predicate
                        obj = groups[-1].strip()
                        predicate = pred_type
                    
                    # Extract core entities and normalize
                    # Pass full sentence as context to preserve all background info
                    core_s, s_details = extract_core_entity(subject, sentence)
                    core_o, o_details = extract_core_entity(obj, sentence)
                    core_s = normalize_entity(core_s)
                    core_o = normalize_entity(core_o)
                    
                    # Only add if we have valid entities
                    if core_s and core_o and len(core_s) > 2 and len(core_o) > 2:
                        # Combine details - preserve all background information
                        details_parts = []
                        
                        # Add subject details if we extracted background info from subject
                        if s_details:
                            if "Context:" in s_details:
                                # Context already includes full sentence, use it
                                details_parts.append(s_details)
                            else:
                                details_parts.append(f"Subject context: {s_details}")
                        
                        # Add object details if we extracted background info from object
                        if o_details:
                            if "Context:" in o_details:
                                # Context already includes full sentence, use it
                                if not details_parts or "Context:" not in details_parts[0]:
                                    details_parts.append(o_details)
                            else:
                                details_parts.append(f"Object context: {o_details}")
                        
                        # If no details were extracted but original differs from core, store original sentence
                        if not details_parts:
                            if subject != core_s or obj != core_o:
                                details_parts.append(f"Original: {sentence.strip()}")
                            else:
                                # Store full sentence as context for reference
                                details_parts.append(f"Full context: {sentence.strip()}")
                        
                        details = ' | '.join(details_parts) if details_parts else None
                        triples.append((core_s, predicate, core_o, details))
                        break  # Found a match, move to next sentence
    
    return triples

def extract_triples_with_context(text):
    """
    Extract triples with cleaner entities and store original context in details.
    Returns list of tuples: (subject, predicate, object, details)
    """
    triples_with_context = []
    sentences = re.split(r'[.!?\n]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue
        
        # Try improved patterns first (case-insensitive to handle lowercase starts)
        improved_patterns = [
            (r'([A-Za-z][a-zA-Z\s]+(?:,\s+[A-Za-z][a-zA-Z\s]+)*)\s+(is|are|was|were|becomes|represents|means|refers to|denotes)\s+(.+)', 'relates to'),
            (r'([A-Za-z][a-zA-Z\s]+)\s+(uses|employs|utilizes|applies)\s+(.+)', 'uses'),
            (r'([A-Za-z][a-zA-Z\s]+)\s+(develops|created|designed|implemented)\s+(.+)', 'creates'),
            (r'([A-Za-z][a-zA-Z\s]+)\s+(requires|needs|demands)\s+(.+)', 'requires'),
            (r'([A-Za-z][a-zA-Z\s]+)\s+(enables|allows|permits)\s+(.+)', 'enables'),
            (r'([A-Za-z][a-zA-Z\s]+)\s+(affects|impacts|influences)\s+(.+)', 'affects'),
            (r'([A-Za-z][a-zA-Z\s]+)\s+(found|discovered|identified|observed|detected)\s+(.+)', 'discovered'),
            (r'([A-Za-z][a-zA-Z\s]+)\s+(studies|analyzes|examines|investigates)\s+(.+)', 'studies'),
            (r'([A-Za-z][a-zA-Z\s]+)\s+(proposes|suggests|recommends)\s+(.+)', 'proposes'),
            (r'([A-Za-z][a-zA-Z\s]+)\s+(results in|leads to|causes)\s+(.+)', 'causes'),
            (r'([A-Za-z][a-zA-Z\s]+)\s+(works with|collaborates with|partnered with)\s+(.+)', 'works with'),
            (r'([A-Za-z][a-zA-Z\s]+)\s+(located in|based in|situated in)\s+(.+)', 'located in'),
        ]
        
        for pattern, predicate in improved_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                groups = match.groups()
                original_subject = groups[0].strip() if len(groups) > 0 else ''
                original_object = groups[-1].strip() if len(groups) > 1 else ''
                
                # Extract core entities (this handles dates, times, descriptive text)
                core_subject, subject_details = extract_core_entity(original_subject, sentence)
                core_object, object_details = extract_core_entity(original_object, sentence)
                
                # Normalize entities (handles abbreviations, aliases, case-insensitive)
                core_subject = normalize_entity(core_subject)
                core_object = normalize_entity(core_object)
                
                # Only proceed if core entities are meaningful
                if core_subject and core_object and len(core_subject) > 2 and len(core_object) > 2:
                    # Combine details - preserve all background information
                    details_parts = []
                    
                    # Add subject details if we extracted background info from subject
                    if subject_details:
                        if "Context:" in subject_details:
                            # Context already includes full sentence, use it
                            details_parts.append(subject_details)
                        else:
                            details_parts.append(f"Subject context: {subject_details}")
                    
                    # Add object details if we extracted background info from object
                    if object_details:
                        if "Context:" in object_details:
                            # Context already includes full sentence, use it
                            if not details_parts or "Context:" not in details_parts[0]:
                                details_parts.append(object_details)
                        else:
                            details_parts.append(f"Object context: {object_details}")
                    
                    # If no details were extracted but original differs from core, store original sentence
                    if not details_parts:
                        if original_subject != core_subject or original_object != core_object:
                            details_parts.append(f"Original: {sentence.strip()}")
                        else:
                            # Store full sentence as context for reference
                            details_parts.append(f"Full context: {sentence.strip()}")
                    
                    details = ' | '.join(details_parts) if details_parts else None
                    
                    triples_with_context.append((core_subject, predicate, core_object, details))
                    break
        
        # Also try regular patterns
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
            r"\s+(discovered|discovering)\s+",
        ]
        
        for pattern in patterns:
            parts = re.split(pattern, sentence, maxsplit=1)
            if len(parts) == 3:
                original_subj, pred, original_obj = parts
                original_subj = original_subj.strip()
                original_obj = original_obj.strip()
                
                # Extract core entities (this handles dates, times, descriptive text)
                # Pass full sentence as context to preserve all background info
                core_subj, subject_details = extract_core_entity(original_subj, sentence)
                core_obj, object_details = extract_core_entity(original_obj, sentence)
                
                # Normalize entities (handles abbreviations, aliases, case-insensitive)
                core_subj = normalize_entity(core_subj)
                core_obj = normalize_entity(core_obj)
                
                if core_subj and core_obj and len(core_subj) > 2 and len(core_obj) > 2:
                    # Combine details - preserve all background information
                    details_parts = []
                    
                    # Add subject details if we extracted background info from subject
                    if subject_details:
                        if "Context:" in subject_details:
                            # Context already includes full sentence, use it
                            details_parts.append(subject_details)
                        else:
                            details_parts.append(f"Subject context: {subject_details}")
                    
                    # Add object details if we extracted background info from object
                    if object_details:
                        if "Context:" in object_details:
                            # Context already includes full sentence, use it
                            if not details_parts or "Context:" not in details_parts[0]:
                                details_parts.append(object_details)
                        else:
                            details_parts.append(f"Object context: {object_details}")
                    
                    # If no details were extracted but original differs from core, store original sentence
                    if not details_parts:
                        if original_subj != core_subj or original_obj != core_obj:
                            details_parts.append(f"Original: {sentence.strip()}")
                        else:
                            # Store full sentence as context for reference
                            details_parts.append(f"Full context: {sentence.strip()}")
                    
                    details = ' | '.join(details_parts) if details_parts else None
                    
                    triples_with_context.append((core_subj, pred.strip(), core_obj, details))
                    break
    
    return triples_with_context

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
    
    # Normalize the input (case-insensitive + entity normalization)
    # Convert to lowercase and normalize spaces/underscores for consistent comparison
    subject_normalized = normalize_entity(str(subject).strip().lower().replace('_', ' '))
    predicate_normalized = str(predicate).strip().lower().replace('_', ' ')
    object_normalized = normalize_entity(str(object_val).strip().lower())
    
    # OPTIMIZED: Skip metadata triples early to speed up duplicate checking
    # Check all facts in the graph for case-insensitive match with normalization
    for s, p, o in graph:
        # Skip metadata triples early (much faster than processing them)
        predicate_str = str(p)
        if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
            'fact_object' in predicate_str or 'has_details' in predicate_str or 
            'source_document' in predicate_str or 'uploaded_at' in predicate_str):
            continue
        
        # Extract and normalize subject from URI
        s_str = str(s)
        if 'urn:' in s_str:
            # Decode URI: unquote, replace underscores with spaces, lowercase
            s_decoded = unquote(s_str.replace('urn:', '')).replace('_', ' ').lower().strip()
        else:
            s_decoded = str(s).lower().strip().replace('_', ' ')
        
        # Normalize subject using entity normalization
        s_decoded = normalize_entity(s_decoded)
        
        # Extract and normalize predicate from URI
        p_str = str(p)
        if 'urn:' in p_str:
            p_decoded = unquote(p_str.replace('urn:', '')).replace('_', ' ').lower().strip()
        else:
            p_decoded = str(p).lower().strip().replace('_', ' ')
        
        # Normalize object (already a literal) using entity normalization
        o_decoded = normalize_entity(str(o).lower().strip())
        
        # Compare case-insensitively with normalization (handles abbreviations, aliases)
        if (s_decoded == subject_normalized and 
            p_decoded == predicate_normalized and 
            o_decoded == object_normalized):
            return True
    
    return False

def add_to_graph(text, source_document: str = "manual", uploaded_at: str = None):
    """
    Extract knowledge from text and add to graph.
    ALWAYS uses entity cleaning to produce cleaner subject/object pairs,
    and stores original context in details field.
    
    Uses Triplex model if available and enabled, otherwise falls back to regex-based extraction.
    
    Args:
        text: Text to extract knowledge from
        source_document: Name of the source document (default: "manual")
        uploaded_at: ISO format timestamp when the fact was added (default: current time)
    
    Returns:
        str: Status message with extraction method and counts
    """
    global graph
    import rdflib
    from urllib.parse import quote
    from datetime import datetime
    
    # Set default timestamp if not provided
    if uploaded_at is None:
        uploaded_at = datetime.now().isoformat()
    
    # Track which extraction method was used
    extraction_method = "regex"
    improved_count = 0
    triplex_count = 0
    regex_count = 0
    
    # OPTIMIZED: Combine fast extraction methods for comprehensive coverage
    # Strategy: Run both improved patterns AND context extraction (both fast, catch different patterns)
    # Only skip Triplex (slow) if we already have good results
    
    improved_triples = []
    triplex_triples = []
    new_triples_with_context = []
    new_triples = []
    
    # Always run improved pattern extraction (fast, catches capitalized entities)
    try:
        improved_triples = extract_with_improved_patterns(text)
        if improved_triples:
            improved_count = len(improved_triples)
            print(f"✅ Improved patterns extracted {improved_count} triples")
        else:
            improved_triples = []
    except Exception as e:
        print(f"⚠️  Improved pattern extraction failed: {e}")
        improved_triples = []
    
    # Always run context extraction (fast, catches lowercase/case-insensitive patterns)
    # This complements improved patterns by finding different types of relationships
    try:
        new_triples_with_context = extract_triples_with_context(text)
        if new_triples_with_context:
            print(f"✅ Context extraction found {len(new_triples_with_context)} triples")
        else:
            new_triples_with_context = []
    except Exception as e:
        print(f"⚠️  Context extraction failed: {e}")
        new_triples_with_context = []
    
    # Only try Triplex if we didn't find much from fast methods AND Triplex is enabled
    # Triplex is slow, so skip if we already have good results
    total_fast_triples = len(improved_triples) + len(new_triples_with_context)
    if total_fast_triples < 5 and USE_TRIPLEX and TRIPLEX_AVAILABLE:
        try:
            triplex_triples = extract_with_triplex(text)
            if triplex_triples and len(triplex_triples) > 0:
                triplex_count = len(triplex_triples)
                extraction_method = "triplex"
                print(f"✅ TRIPLEX: Extracted {triplex_count} triples using LLM")
            else:
                triplex_triples = []
        except Exception as e:
            print(f"⚠️  Triplex extraction failed: {e}")
            triplex_triples = []
    
    # Final fallback: regular triples only if nothing else worked
    if not improved_triples and not triplex_triples and not new_triples_with_context:
        try:
            new_triples = extract_triples(text)
            if new_triples:
                print(f"✅ Fallback extraction found {len(new_triples)} triples")
        except Exception as e:
            print(f"⚠️  Fallback extraction failed: {e}")
            new_triples = []
    
    # Determine extraction method for reporting
    if triplex_triples:
        extraction_method = "triplex"
    elif improved_triples and new_triples_with_context:
        extraction_method = "improved_patterns+context"
    elif improved_triples:
        extraction_method = "improved_patterns"
    elif new_triples_with_context:
        extraction_method = "context"
    else:
        extraction_method = "regex"
    
    added_count = 0
    skipped_count = 0
    
    # Track which facts we've added to avoid duplicates
    added_facts = set()
    
    # Process improved pattern triples first (if available) - enhanced regex
    for s, p, o, details in improved_triples:
        fact_key = (s.lower().strip(), p.lower().strip(), o.lower().strip())
        
        # Check if fact already exists before adding
        if fact_exists(s, p, o) or fact_key in added_facts:
            skipped_count += 1
            continue
        
        # Properly encode URIs like create_fact_endpoint does
        subject_clean = str(s).strip().replace(' ', '_')
        predicate_clean = str(p).strip().replace(' ', '_')
        object_value = str(o).strip()
        
        # Create URIs (encode spaces to avoid RDFLib warnings)
        subject_uri = rdflib.URIRef(f"urn:{quote(subject_clean, safe='')}")
        predicate_uri = rdflib.URIRef(f"urn:{quote(predicate_clean, safe='')}")
        object_literal = rdflib.Literal(object_value)
        
        graph.add((subject_uri, predicate_uri, object_literal))
        
        # Store details if provided (original context)
        if details and details.strip():
            add_fact_details(s, p, o, details)
        
        # Store source document and timestamp
        add_fact_source_document(s, p, o, source_document, uploaded_at)
        
        added_facts.add(fact_key)
        added_count += 1
    
    # Process Triplex triples (if available) - these are highest quality
    for s, p, o, details in triplex_triples:
        fact_key = (s.lower().strip(), p.lower().strip(), o.lower().strip())
        
        # Check if fact already exists before adding
        if fact_exists(s, p, o) or fact_key in added_facts:
            skipped_count += 1
            continue
        
        # Properly encode URIs like create_fact_endpoint does
        subject_clean = str(s).strip().replace(' ', '_')
        predicate_clean = str(p).strip().replace(' ', '_')
        object_value = str(o).strip()
        
        # Create URIs (encode spaces to avoid RDFLib warnings)
        subject_uri = rdflib.URIRef(f"urn:{quote(subject_clean, safe='')}")
        predicate_uri = rdflib.URIRef(f"urn:{quote(predicate_clean, safe='')}")
        object_literal = rdflib.Literal(object_value)
        
        graph.add((subject_uri, predicate_uri, object_literal))
        
        # Store details if provided (original context)
        if details and details.strip():
            add_fact_details(s, p, o, details)
        
        # Store source document and timestamp
        add_fact_source_document(s, p, o, source_document, uploaded_at)
        
        added_facts.add(fact_key)
        added_count += 1
    
    # Process triples with context (regex-based with entity cleaning)
    for s, p, o, details in new_triples_with_context:
        fact_key = (s.lower().strip(), p.lower().strip(), o.lower().strip())
        
        # Check if fact already exists before adding
        if fact_exists(s, p, o) or fact_key in added_facts:
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
        
        # Store details if provided (original context)
        if details and details.strip():
            add_fact_details(s, p, o, details)
        
        # Store source document and timestamp
        add_fact_source_document(s, p, o, source_document, uploaded_at)
        
        added_facts.add(fact_key)
        added_count += 1
    
    # Process regular triples as fallback, but clean entities and store context
    regex_triples_processed = 0
    for s, p, o in new_triples:
        fact_key = (s.lower().strip(), p.lower().strip(), o.lower().strip())
        
        # Skip if we already added this fact from context extraction
        if fact_key in added_facts:
            continue
        
        # Check if fact already exists before adding
        if fact_exists(s, p, o):
            skipped_count += 1
            continue
        
        # Extract core entities even for fallback triples (handles dates, times, descriptive text)
        original_subject = s
        original_object = o
        core_subject, subject_details = extract_core_entity(str(s), "")
        core_object, object_details = extract_core_entity(str(o), "")
        
        # Normalize entities (handles abbreviations, aliases, case-insensitive)
        core_subject = normalize_entity(core_subject)
        core_object = normalize_entity(core_object)
        
        # Use core entities if they're different and meaningful
        if core_subject and core_object and len(core_subject) > 2 and len(core_object) > 2:
            # Combine details from subject and object
            details_parts = []
            if subject_details:
                details_parts.append(f"Subject: {subject_details}")
            if object_details:
                details_parts.append(f"Object: {object_details}")
            if not details_parts and (original_subject != core_subject or original_object != core_object):
                details_parts.append(f"{original_subject} {p} {original_object}")
            
            details = ' | '.join(details_parts) if details_parts else None
            s, o = core_subject, core_object
        else:
            # If extraction didn't work well, normalize and use original
            s = normalize_entity(str(s))
            o = normalize_entity(str(o))
            details = None
        
        # Properly encode URIs like create_fact_endpoint does
        subject_clean = str(s).strip().replace(' ', '_')
        predicate_clean = str(p).strip().replace(' ', '_')
        object_value = str(o).strip()
        
        # Create URIs (encode spaces to avoid RDFLib warnings)
        subject_uri = rdflib.URIRef(f"urn:{quote(subject_clean, safe='')}")
        predicate_uri = rdflib.URIRef(f"urn:{quote(predicate_clean, safe='')}")
        object_literal = rdflib.Literal(object_value)
        
        graph.add((subject_uri, predicate_uri, object_literal))
        
        # Store details if we have original context
        if details and details.strip():
            add_fact_details(s, p, o, details)
        
        # Store source document and timestamp
        add_fact_source_document(s, p, o, source_document, uploaded_at)
        
        added_facts.add(fact_key)
        added_count += 1
        regex_triples_processed += 1
    
    # Count triples by method (only count what was actually added)
    triplex_added = sum(1 for s, p, o, _ in triplex_triples if (s.lower().strip(), p.lower().strip(), o.lower().strip()) in added_facts)
    improved_added = sum(1 for s, p, o, _ in improved_triples if (s.lower().strip(), p.lower().strip(), o.lower().strip()) in added_facts)
    context_added = sum(1 for s, p, o, _ in new_triples_with_context if (s.lower().strip(), p.lower().strip(), o.lower().strip()) in added_facts)
    regex_count = added_count - improved_added - triplex_added - context_added
    
    # Save graph to disk
    save_knowledge_graph()
    
    # Verify save worked
    import os
    if os.path.exists("knowledge_graph.pkl"):
        file_size = os.path.getsize("knowledge_graph.pkl")
        print(f"✅ Graph saved - file size: {file_size} bytes, facts in memory: {len(graph)}")
    
    # Build status message with extraction method info
    method_info = f"[{extraction_method.upper()}]"
    if extraction_method == "triplex":
        method_info = f"🤖 [TRIPLEX LLM] - {triplex_added} triples from LLM"
    elif extraction_method == "improved_patterns+context":
        method_info = f"✅ [IMPROVED PATTERNS + CONTEXT] - {improved_added} from patterns, {context_added} from context"
    elif extraction_method == "improved_patterns":
        method_info = f"✨ [IMPROVED PATTERNS] - {improved_added} triples from enhanced extraction"
    elif extraction_method == "context":
        method_info = f"✅ [CONTEXT EXTRACTION] - {context_added} triples"
    elif "improved" in extraction_method.lower() and "regex" in extraction_method.lower():
        method_info = f"✨ [IMPROVED/REGEX] - {improved_added} from patterns, {regex_count} from regex"
    elif "triplex" in extraction_method.lower():
        method_info = f"⚠️  [FALLBACK] - {extraction_method}"
    else:
        method_info = f"📝 [REGEX] - {regex_count} triples from regex patterns"
    
    if skipped_count > 0:
        return f"{method_info}\n Added {added_count} new triples, skipped {skipped_count} duplicates. Total facts stored: {len(graph)}.\n Saved"
    return f"{method_info}\n Added {added_count} new triples. Total facts stored: {len(graph)}.\n Saved"

def retrieve_context(question, limit=None):
    """
    Retrieve relevant context from knowledge graph for answering questions.
    Uses improved semantic matching and includes details for better context.
    """
    from urllib.parse import unquote
    from knowledge import get_fact_details
    
    # Extract meaningful keywords from question (remove stopwords)
    stopwords = {
        'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
        'is','are','was','were','be','been','have','has','had','do','does','did',
        'will','would','could','should','may','might','can','what','how','when',
        'where','why','who','which','this','that','these','those','tell','me','about',
        'show','give','explain','describe','list','from','there','here'
    }
    
    # Extract keywords - keep words longer than 2 chars and not stopwords
    qwords = [w.lower().strip() for w in question.split() 
              if w.lower().strip() not in stopwords and len(w.strip()) > 2]
    
    # Also extract potential entity names (capitalized words, acronyms)
    entities = [w for w in question.split() if (w[0].isupper() or w.isupper()) and len(w) > 1]
    qwords.extend([w.lower() for w in entities])
    
    # Remove duplicates
    qwords = list(set(qwords))
    
    if not qwords:
        # If no meaningful words, use the whole question (excluding very short words)
        qwords = [w.lower() for w in question.split() if len(w) > 2]
    
    scored_matches = []
    for s, p, o in graph:
        # Skip metadata triples
        predicate_str = str(p)
        if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
            'fact_object' in predicate_str or 'has_details' in predicate_str or 
            'source_document' in predicate_str or 'uploaded_at' in predicate_str):
            continue
        
        # Extract subject from URI
        subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
        subject = unquote(subject).replace('_', ' ')
        
        # Extract predicate from URI
        predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
        predicate = unquote(predicate).replace('_', ' ')
        
        # Object is already a literal
        object_val = str(o)
        
        # Build fact text for matching
        fact_text = f"{subject} {predicate} {object_val}".lower()
        
        # Calculate relevance score with improved matching
        score = 0
        
        # Exact word matches
        for word in qwords:
            word_lower = word.lower()
            
            # Check if word appears in any part of the fact
            if word_lower in fact_text:
                # Base score for any match
                score += 1
                
                # Higher scores for matches in important positions
                if word_lower in subject.lower():
                    score += 5  # Subject matches are most important
                if word_lower in predicate.lower():
                    score += 3  # Predicate matches are important
                if word_lower in object_val.lower():
                    score += 2  # Object matches are relevant
                
                # Bonus for exact word match (not substring)
                if f" {word_lower} " in f" {fact_text} ":
                    score += 2
        
        # Partial/substring matches (for abbreviations, partial words)
        for word in qwords:
            if len(word) > 3:  # Only for longer words
                if word in subject.lower():
                    score += 2
                if word in object_val.lower():
                    score += 1
        
        # Entity name matching (case-insensitive, handles acronyms)
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in subject.lower() or entity_lower in object_val.lower():
                score += 4  # Entity matches are very relevant
            # Also check if entity is part of a word (for acronyms)
            if len(entity) > 2:
                if entity_lower in fact_text:
                    score += 2
        
        # Only include facts with positive scores
        if score > 0:
            # Track where matches were found
            match_locations = []
            for word in qwords:
                word_lower = word.lower()
                if word_lower in subject.lower():
                    match_locations.append("subject")
                if word_lower in predicate.lower():
                    match_locations.append("predicate")
                if word_lower in object_val.lower():
                    match_locations.append("object")
            
            # Remove duplicates and create match location string
            unique_locations = list(set(match_locations))
            if unique_locations:
                match_str = f"[Match in: {', '.join(unique_locations)}]"
            else:
                match_str = "[Match found]"
            
            # Get details for richer context
            details = get_fact_details(subject, predicate, object_val)
            
            # Get source document
            source_doc, uploaded_at = get_fact_source_document(subject, predicate, object_val)
            
            # Build fact description with match location, details, and source
            fact_desc = f"{match_str} {subject} {predicate} {object_val}"
            if details:
                fact_desc += f" | Details: {details}"
            if source_doc:
                fact_desc += f" | Source: {source_doc}"
            
            scored_matches.append((score, fact_desc, subject, predicate, object_val, unique_locations))
    
    # Sort by score (highest first)
    scored_matches.sort(key=lambda x: x[0], reverse=True)
    
    # Remove duplicates (same fact with different scores) - but return ALL unique matches
    seen_facts = set()
    unique_matches = []
    for match_data in scored_matches:
        # Handle both old format (5 items) and new format (6 items)
        if len(match_data) == 6:
            score, fact_desc, subj, pred, obj, match_locs = match_data
        else:
            score, fact_desc, subj, pred, obj = match_data[:5]
            match_locs = []
        
        fact_key = (subj.lower(), pred.lower(), obj.lower())
        if fact_key not in seen_facts:
            seen_facts.add(fact_key)
            unique_matches.append(fact_desc)
            # No limit - return all relevant facts
    
    if unique_matches:
        result = f"**Relevant Knowledge from Your Documents ({len(unique_matches)} facts found):**\n\n"
        for i, match in enumerate(unique_matches, 1):
            result += f"{i}. {match}\n"
        return result
    
    # If no matches, try a broader search (lower threshold)
    if not unique_matches and scored_matches:
        # Return all even with lower scores
        result = f"**Partially Relevant Knowledge ({len(scored_matches)} facts found):**\n\n"
        for i, match_data in enumerate(scored_matches, 1):
            fact_desc = match_data[1] if len(match_data) > 1 else str(match_data)
            result += f"{i}. {fact_desc}\n"
        return result
    
    return "**No directly relevant facts found in the knowledge base.**\n\nTry asking about topics that might be in your knowledge base, or add more knowledge by uploading documents or adding text."

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
    # CRITICAL: Use remove() instead of creating a new graph object
    # This ensures all references to the graph (like kb_graph in api_server.py) stay in sync
    graph.remove((None, None, None))
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

def add_fact_details(subject: str, predicate: str, object_val: str, details: str):
    """
    Add details/comment to an existing fact by storing it as a separate RDF triple.
    Uses a special predicate 'has_details' to link details to the main fact.
    
    Args:
        subject: The subject of the main fact
        predicate: The predicate of the main fact
        object_val: The object of the main fact
        details: The details/comment text to store
    """
    global graph
    import rdflib
    from urllib.parse import quote
    
    if not details or not details.strip():
        return
    
    # Create a unique identifier for this fact to link details to it
    # We'll use a combination of subject, predicate, and object as the identifier
    fact_id = f"{subject}|{predicate}|{object_val}"
    fact_id_clean = fact_id.strip().replace(' ', '_')
    fact_id_uri = rdflib.URIRef(f"urn:fact:{quote(fact_id_clean, safe='')}")
    
    # Store details with special predicate
    details_predicate = rdflib.URIRef("urn:has_details")
    details_literal = rdflib.Literal(details.strip())
    
    # Remove any existing details for this fact first
    remove_fact_details(subject, predicate, object_val)
    
    # Add the details triple
    graph.add((fact_id_uri, details_predicate, details_literal))
    
    # Also link the fact_id to the actual fact components for easier retrieval
    subject_clean = subject.strip().replace(' ', '_')
    predicate_clean = predicate.strip().replace(' ', '_')
    subject_uri = rdflib.URIRef(f"urn:{quote(subject_clean, safe='')}")
    predicate_uri = rdflib.URIRef(f"urn:{quote(predicate_clean, safe='')}")
    
    # Link fact_id to subject, predicate, object for retrieval
    graph.add((fact_id_uri, rdflib.URIRef("urn:fact_subject"), subject_uri))
    graph.add((fact_id_uri, rdflib.URIRef("urn:fact_predicate"), predicate_uri))
    graph.add((fact_id_uri, rdflib.URIRef("urn:fact_object"), rdflib.Literal(object_val)))

def remove_fact_details(subject: str, predicate: str, object_val: str):
    """
    Remove details for a specific fact.
    
    Args:
        subject: The subject of the main fact
        predicate: The predicate of the main fact
        object_val: The object of the main fact
    """
    global graph
    import rdflib
    from urllib.parse import quote
    
    # Create the fact identifier
    fact_id = f"{subject}|{predicate}|{object_val}"
    fact_id_clean = fact_id.strip().replace(' ', '_')
    fact_id_uri = rdflib.URIRef(f"urn:fact:{quote(fact_id_clean, safe='')}")
    
    # Find and remove all triples related to this fact's details
    triples_to_remove = []
    for s, p, o in graph:
        if str(s) == str(fact_id_uri):
            triples_to_remove.append((s, p, o))
    
    for triple in triples_to_remove:
        graph.remove(triple)

def get_fact_details(subject: str, predicate: str, object_val: str) -> str:
    """
    Retrieve details/comment for a specific fact.
    
    Args:
        subject: The subject of the main fact
        predicate: The predicate of the main fact
        object_val: The object of the main fact
    
    Returns:
        The details text if found, empty string otherwise
    """
    global graph
    import rdflib
    from urllib.parse import quote, unquote
    
    # Create the fact identifier
    fact_id = f"{subject}|{predicate}|{object_val}"
    fact_id_clean = fact_id.strip().replace(' ', '_')
    fact_id_uri = rdflib.URIRef(f"urn:fact:{quote(fact_id_clean, safe='')}")
    
    # Look for details triple
    details_predicate = rdflib.URIRef("urn:has_details")
    
    for s, p, o in graph:
        if str(s) == str(fact_id_uri) and str(p) == str(details_predicate):
            return str(o)
    
    return ""

def add_fact_source_document(subject: str, predicate: str, object_val: str, source_document: str, uploaded_at: str):
    """
    Store source document and upload timestamp for a specific fact.
    
    Args:
        subject: The subject of the main fact
        predicate: The predicate of the main fact
        object_val: The object of the main fact
        source_document: Name of the source document (or "manual" for manually added)
        uploaded_at: ISO format timestamp when the fact was added
    """
    global graph
    import rdflib
    from urllib.parse import quote
    
    if not source_document or not uploaded_at:
        return
    
    # Create the fact identifier
    fact_id = f"{subject}|{predicate}|{object_val}"
    fact_id_clean = fact_id.strip().replace(' ', '_')
    fact_id_uri = rdflib.URIRef(f"urn:fact:{quote(fact_id_clean, safe='')}")
    
    # Store source document and timestamp with special predicates
    source_predicate = rdflib.URIRef("urn:source_document")
    timestamp_predicate = rdflib.URIRef("urn:uploaded_at")
    
    # Remove any existing source info for this fact first
    remove_fact_source_document(subject, predicate, object_val)
    
    # Add the source document and timestamp triples
    graph.add((fact_id_uri, source_predicate, rdflib.Literal(source_document.strip())))
    graph.add((fact_id_uri, timestamp_predicate, rdflib.Literal(uploaded_at.strip())))
    
    # Also link the fact_id to the actual fact components for easier retrieval
    subject_clean = subject.strip().replace(' ', '_')
    predicate_clean = predicate.strip().replace(' ', '_')
    subject_uri = rdflib.URIRef(f"urn:{quote(subject_clean, safe='')}")
    predicate_uri = rdflib.URIRef(f"urn:{quote(predicate_clean, safe='')}")
    
    # Link fact_id to subject, predicate, object for retrieval (if not already linked)
    graph.add((fact_id_uri, rdflib.URIRef("urn:fact_subject"), subject_uri))
    graph.add((fact_id_uri, rdflib.URIRef("urn:fact_predicate"), predicate_uri))
    graph.add((fact_id_uri, rdflib.URIRef("urn:fact_object"), rdflib.Literal(object_val)))

def remove_fact_source_document(subject: str, predicate: str, object_val: str):
    """
    Remove source document info for a specific fact.
    
    Args:
        subject: The subject of the main fact
        predicate: The predicate of the main fact
        object_val: The object of the main fact
    """
    global graph
    import rdflib
    from urllib.parse import quote
    
    # Create the fact identifier
    fact_id = f"{subject}|{predicate}|{object_val}"
    fact_id_clean = fact_id.strip().replace(' ', '_')
    fact_id_uri = rdflib.URIRef(f"urn:fact:{quote(fact_id_clean, safe='')}")
    
    # Find and remove source document and timestamp triples
    source_predicate = rdflib.URIRef("urn:source_document")
    timestamp_predicate = rdflib.URIRef("urn:uploaded_at")
    
    triples_to_remove = []
    for s, p, o in graph:
        if str(s) == str(fact_id_uri) and str(p) in [str(source_predicate), str(timestamp_predicate)]:
            triples_to_remove.append((s, p, o))
    
    for triple in triples_to_remove:
        graph.remove(triple)

def get_fact_source_document(subject: str, predicate: str, object_val: str) -> tuple[str, str]:
    """
    Retrieve source document and upload timestamp for a specific fact.
    
    Args:
        subject: The subject of the main fact
        predicate: The predicate of the main fact
        object_val: The object of the main fact
    
    Returns:
        Tuple of (source_document, uploaded_at) if found, ("", "") otherwise
    """
    global graph
    import rdflib
    from urllib.parse import quote
    
    # Create the fact identifier
    fact_id = f"{subject}|{predicate}|{object_val}"
    fact_id_clean = fact_id.strip().replace(' ', '_')
    fact_id_uri = rdflib.URIRef(f"urn:fact:{quote(fact_id_clean, safe='')}")
    
    # Look for source document and timestamp triples
    source_predicate = rdflib.URIRef("urn:source_document")
    timestamp_predicate = rdflib.URIRef("urn:uploaded_at")
    
    source_document = ""
    uploaded_at = ""
    
    for s, p, o in graph:
        if str(s) == str(fact_id_uri):
            if str(p) == str(source_predicate):
                source_document = str(o)
            elif str(p) == str(timestamp_predicate):
                uploaded_at = str(o)
    
    return source_document, uploaded_at

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
                
                # Import details if present
                details = fact.get('details')
                if details:
                    add_fact_details(subject_str, predicate_str, obj_str, details)
                
                # Import source document and timestamp if present
                source_document = fact.get('sourceDocument') or fact.get('source_document')
                uploaded_at = fact.get('uploadedAt') or fact.get('uploaded_at')
                if source_document and uploaded_at:
                    add_fact_source_document(subject_str, predicate_str, obj_str, source_document, uploaded_at)
                elif source_document:
                    # If only source document is provided, use current timestamp
                    from datetime import datetime
                    add_fact_source_document(subject_str, predicate_str, obj_str, source_document, datetime.now().isoformat())
                
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


