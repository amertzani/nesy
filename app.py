# app.py
import gradio as gr
import rdflib
import re
import os
import tempfile
from huggingface_hub import InferenceClient
import PyPDF2
from docx import Document
import pandas as pd

# =========================================================
#  🧠 1. Global Knowledge Graph with Persistent Storage
# =========================================================
import json
import pickle
from datetime import datetime

# Storage file paths
KNOWLEDGE_FILE = "knowledge_graph.pkl"
BACKUP_FILE = "knowledge_backup.json"

graph = rdflib.Graph()

def save_knowledge_graph():
    """Save the knowledge graph to persistent storage"""
    try:
        # Save as pickle for RDF graph
        with open(KNOWLEDGE_FILE, 'wb') as f:
            pickle.dump(graph, f)
        
        # Also save a human-readable backup
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
        
        print(f"💾 Saved {len(graph)} facts to persistent storage")
        return f"💾 Saved {len(graph)} facts to storage"
        
    except Exception as e:
        error_msg = f"❌ Error saving knowledge: {e}"
        print(error_msg)
        return error_msg

def load_knowledge_graph():
    """Load the knowledge graph from persistent storage"""
    global graph
    
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, 'rb') as f:
                graph = pickle.load(f)
            print(f"📂 Loaded {len(graph)} facts from storage")
            return f"📂 Loaded {len(graph)} facts from storage"
        else:
            print("📂 No existing knowledge file found, starting fresh")
            return "📂 No existing knowledge file found, starting fresh"
            
    except Exception as e:
        error_msg = f"❌ Error loading knowledge: {e}"
        print(error_msg)
        return error_msg

def get_knowledge_file():
    """Return the knowledge backup file for download"""
    try:
        # Create a comprehensive backup with all facts
        create_comprehensive_backup()
        return BACKUP_FILE
    except Exception as e:
        print(f"❌ Error getting knowledge file: {e}")
        return None

def create_comprehensive_backup():
    """Create a comprehensive backup file with all knowledge facts"""
    global graph
    
    try:
        # Create detailed backup data
        backup_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_facts": len(graph),
                "backup_type": "comprehensive_knowledge_base"
            },
            "facts": []
        }
        
        # Add all facts from the graph
        for i, (s, p, o) in enumerate(graph):
            # Clean up the subject, predicate, object for better readability
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
        
        # Save as JSON
        with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        # Also create a human-readable text version
        create_readable_backup()
        
        print(f"💾 Created comprehensive backup with {len(graph)} facts")
        
    except Exception as e:
        print(f"❌ Error creating comprehensive backup: {e}")

def create_readable_backup():
    """Create a human-readable text backup"""
    global graph
    
    try:
        # Create readable text file
        readable_text = f"# Knowledge Base Backup\n"
        readable_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        readable_text += f"Total Facts: {len(graph)}\n\n"
        
        if len(graph) == 0:
            readable_text += "No facts in knowledge base.\n"
        else:
            # Group facts by subject for better organization
            facts_by_subject = {}
            for s, p, o in graph:
                subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
                predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
                object_val = str(o)
                
                if subject not in facts_by_subject:
                    facts_by_subject[subject] = []
                facts_by_subject[subject].append(f"{predicate}: {object_val}")
            
            # Add organized facts
            for subject, facts in facts_by_subject.items():
                readable_text += f"## {subject}\n"
                for fact in facts:
                    readable_text += f"- {fact}\n"
                readable_text += "\n"
        
        # Save readable version
        with open("knowledge_readable.txt", 'w', encoding='utf-8') as f:
            f.write(readable_text)
        
        print(f"📄 Created readable backup: knowledge_readable.txt")
        
    except Exception as e:
        print(f"❌ Error creating readable backup: {e}")

def show_storage_info():
    """Show information about where files are stored"""
    info = f"📁 **Storage Information:**\n\n"
    
    # Check if files exist
    pkl_exists = os.path.exists(KNOWLEDGE_FILE)
    json_exists = os.path.exists(BACKUP_FILE)
    
    info += f"**Primary Storage:** `{KNOWLEDGE_FILE}` {'✅ Exists' if pkl_exists else '❌ Not found'}\n"
    info += f"**Backup Storage:** `{BACKUP_FILE}` {'✅ Exists' if json_exists else '❌ Not found'}\n"
    info += f"**Readable Backup:** `knowledge_readable.txt` {'✅ Exists' if os.path.exists('knowledge_readable.txt') else '❌ Not found'}\n\n"
    
    if pkl_exists:
        file_size = os.path.getsize(KNOWLEDGE_FILE)
        info += f"**File Size:** {file_size:,} bytes\n"
    
    info += f"**Total Facts:** {len(graph)}\n\n"
    
    info += "**How to Access:**\n"
    info += "• On Hugging Face Spaces: Files are in `/home/user/app/`\n"
    info += "• On Local Machine: Files are in your project folder\n"
    info += "• Use '📥 Download Knowledge' button to get the JSON backup\n"
    
    return info


def extract_triples(text):
    """
    Enhanced pattern-based extraction for general document processing.
    """
    triples = []
    print(f"🔍 Extracting triples from {len(text)} characters...")
    
    # Detect if this is structured data (key-value pairs, tables, etc.)
    structured_indicators = [
        'date', 'time', 'name', 'title', 'description', 'address', 'phone', 'email',
        'number', 'id', 'code', 'reference', 'amount', 'total', 'price', 'cost',
        'company', 'organization', 'department', 'location', 'status', 'type',
        'category', 'class', 'group', 'section', 'chapter', 'part', 'item'
    ]
    
    text_lower = text.lower()
    is_structured = any(indicator in text_lower for indicator in structured_indicators)
    
    if is_structured:
        print("🔍 Detected structured data (invoice/financial document)")
        triples.extend(extract_structured_triples(text))
    
    # Always try regular extraction too
    triples.extend(extract_regular_triples(text))
    
    print(f"🔍 Total extracted {len(triples)} triples")
    for i, (s, p, o) in enumerate(triples[:5]):  # Show first 5
        print(f"  {i+1}. {s} {p} {o}")
    
    return triples

def extract_structured_triples(text):
    """Extract triples from structured data (key-value pairs, tables, etc.)"""
    triples = []
    lines = text.split('\n')
    
    # General patterns for structured data extraction
    patterns = [
        # Date patterns
        (r'date\s*:?\s*([0-9\/\-\.]+)', 'date', 'is'),
        (r'time\s*:?\s*([0-9:]+)', 'time', 'is'),
        (r'created\s*:?\s*([0-9\/\-\.]+)', 'created_date', 'is'),
        (r'modified\s*:?\s*([0-9\/\-\.]+)', 'modified_date', 'is'),
        
        # ID and reference patterns
        (r'id\s*:?\s*([A-Z0-9\-]+)', 'id', 'is'),
        (r'number\s*:?\s*([A-Z0-9\-]+)', 'number', 'is'),
        (r'code\s*:?\s*([A-Z0-9\-]+)', 'code', 'is'),
        (r'reference\s*:?\s*([A-Z0-9\-]+)', 'reference', 'is'),
        
        # Name and title patterns
        (r'name\s*:?\s*([A-Za-z\s&.,]+)', 'name', 'is'),
        (r'title\s*:?\s*([A-Za-z\s&.,]+)', 'title', 'is'),
        (r'company\s*:?\s*([A-Za-z\s&.,]+)', 'company', 'is'),
        (r'organization\s*:?\s*([A-Za-z\s&.,]+)', 'organization', 'is'),
        
        # Contact patterns
        (r'email\s*:?\s*([A-Za-z0-9@\.\-]+)', 'email', 'is'),
        (r'phone\s*:?\s*([0-9\s\-\+\(\)]+)', 'phone', 'is'),
        (r'address\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'address', 'is'),
        
        # Description patterns
        (r'description\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'description', 'is'),
        (r'type\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'type', 'is'),
        (r'category\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'category', 'is'),
        (r'status\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'status', 'is'),
        
        # Location patterns
        (r'location\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'location', 'is'),
        (r'department\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'department', 'is'),
        (r'section\s*:?\s*([A-Za-z0-9\s\-\.,]+)', 'section', 'is'),
        
        # Amount patterns
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
                    break  # Only one match per line
    
    # General key-value pair extraction
    kv_patterns = [
        # Standard colon format
        r'([A-Za-z\s]+):\s*([A-Za-z0-9\s\$\-\.\/,]+)',
        # Equals format
        r'([A-Za-z\s]+)\s*=\s*([A-Za-z0-9\s\$\-\.\/,]+)',
        # Dash format
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
    
    # Extract any line that looks like "Label: Value" or "Label Value"
    for line in lines:
        line = line.strip()
        if ':' in line and len(line) > 10:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                if len(key) > 2 and len(value) > 1 and not key.isdigit():
                    # Clean the key
                    clean_key = re.sub(r'[^A-Za-z0-9\s]', '', key).strip().lower().replace(' ', '_')
                    if clean_key:
                        triples.append((clean_key, 'is', value))
    
    print(f"🔍 Structured extraction found {len(triples)} triples")
    return triples

def extract_regular_triples(text):
    """Extract triples using regular sentence patterns"""
    triples = []
    
    # Clean and split text into sentences
    sentences = re.split(r"[.?!\n]", text)
    print(f"🔍 Found {len(sentences)} sentences for regular extraction")
    
    # English extraction patterns
    patterns = [
        # Basic patterns
        r"\s+(is|are|was|were)\s+",
        r"\s+(has|have|had)\s+",
        r"\s+(uses|used|using)\s+",
        r"\s+(creates|created|creating)\s+",
        r"\s+(develops|developed|developing)\s+",
        r"\s+(leads|led|leading)\s+",
        r"\s+(affects|affected|affecting)\s+",
        r"\s+(contains|contained|containing)\s+",
        r"\s+(includes|included|including)\s+",
        r"\s+(involves|involved|involving)\s+",
        r"\s+(requires|required|requiring)\s+",
        r"\s+(produces|produced|producing)\s+",
        r"\s+(causes|caused|causing)\s+",
        r"\s+(results|resulted|resulting)\s+",
        r"\s+(enables|enabled|enabling)\s+",
        r"\s+(provides|provided|providing)\s+",
        r"\s+(supports|supported|supporting)\s+",
        r"\s+(allows|allowed|allowing)\s+",
        r"\s+(helps|helped|helping)\s+",
        r"\s+(improves|improved|improving)\s+",
        r"\s+(located|situated|found)\s+",
        r"\s+(consists|composed|made)\s+",
        r"\s+(operates|functions|works)\s+",
        r"\s+(generates|creates|produces)\s+",
        r"\s+(transforms|converts|changes)\s+",
        r"\s+(connects|links|relates)\s+",
        r"\s+(influences|impacts|affects)\s+",
        r"\s+(depends|relies|based)\s+",
        r"\s+(represents|symbolizes|stands)\s+",
        r"\s+(describes|explains|defines)\s+",
        r"\s+(refers|referring|referenced)\s+",
        r"\s+(concerns|concerning|concerned)\s+",
        r"\s+(relates|relating|related)\s+",
        r"\s+(analyzes|analyzing|analyzed)\s+",
        r"\s+(examines|examining|examined)\s+",
        r"\s+(studies|studying|studied)\s+",
        r"\s+(checks|checking|checked)\s+",
        r"\s+(manages|managing|managed)\s+",
        r"\s+(organizes|organizing|organized)\s+",
        r"\s+(coordinates|coordinating|coordinated)\s+",
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short sentences
            continue
            
        # Try each pattern
        for pattern in patterns:
            parts = re.split(pattern, sentence, maxsplit=1)
            if len(parts) == 3:
                subj, pred, obj = parts
                subj = subj.strip()
                pred = pred.strip()
                obj = obj.strip()
                
                # Clean up the parts
                if subj and pred and obj and len(subj) > 2 and len(obj) > 2:
                    # Remove common prefixes/suffixes
                    subj = re.sub(r'^(the|a|an)\s+', '', subj, flags=re.IGNORECASE)
                    obj = re.sub(r'^(the|a|an)\s+', '', obj, flags=re.IGNORECASE)
                    
                    triples.append((subj, pred, obj))
                    break  # Found a match, move to next sentence
    
    print(f"🔍 Regular extraction found {len(triples)} triples")
    return triples


def add_to_graph(text):
    """
    Parse text into triples and add them to the RDF graph.
    """
    new_triples = extract_triples(text)
    for s, p, o in new_triples:
        graph.add((rdflib.URIRef(f"urn:{s}"), rdflib.URIRef(f"urn:{p}"), rdflib.Literal(o)))
    
    # Automatically save after adding knowledge
    save_result = save_knowledge_graph()
    
    return f"✅ Added {len(new_triples)} new triples. Total facts stored: {len(graph)}.\n{save_result}"


def retrieve_context(question, limit=10):
    """
    Retrieve RDF facts related to keywords in the question with better matching.
    """
    matches = []
    qwords = question.lower().split()
    
    # Remove common words that don't add meaning
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why', 'who'}
    qwords = [w for w in qwords if w not in stop_words and len(w) > 2]
    
    print(f"🔍 Searching for: {qwords}")
    
    # Score matches by relevance
    scored_matches = []
    
    for s, p, o in graph:
        subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
        predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
        object_val = str(o)
        
        fact_text = f"{subject} {predicate} {object_val}".lower()
        
        # Calculate relevance score
        score = 0
        for word in qwords:
            if word in fact_text:
                score += 1
                # Bonus for exact matches
                if word == subject.lower() or word == predicate.lower():
                    score += 2
        
        if score > 0:
            scored_matches.append((score, f"{subject} {predicate} {object_val}"))
    
    # Sort by relevance score (highest first)
    scored_matches.sort(key=lambda x: x[0], reverse=True)
    
    # Take top matches
    matches = [match[1] for match in scored_matches[:limit]]
    
    print(f"🔍 Found {len(matches)} relevant facts")
    
    if matches:
        result = "**Relevant Knowledge:**\n"
        for i, match in enumerate(matches, 1):
            result += f"{i}. {match}\n"
        return result
    else:
        return "**No directly relevant facts found.**\n\nTry asking about topics that might be in your knowledge base, or add more knowledge first!"

def handle_add_knowledge(text):
    """Handle adding knowledge from text input"""
    if not text or text.strip() == "":
        return "⚠️ Please enter some text to add to the knowledge graph."
    
    print(f"📝 Adding knowledge from text input: {text[:1000]}...")
    result = add_to_graph(text)
    print(f"✅ Knowledge added: {result}")
    
    # Return enhanced status with current knowledge count
    total_facts = len(graph)
    return f"✅ **Text Knowledge Added Successfully!**\n\n{result}\n\n🧠 **Current Knowledge Base:** {total_facts} total facts"

def show_graph_contents():
    """
    Return all current triples as readable text with better formatting.
    """
    print(f"🔍 Showing graph contents. Total triples: {len(graph)}")
    
    if len(graph) == 0:
        return "📊 **Knowledge Graph Status: EMPTY**\n\n🚀 **How to build your knowledge base:**\n1. **Add text directly** - Paste any text in the 'Add Knowledge from Text' box above\n2. **Upload documents** - Use the file upload to process PDF, DOCX, TXT, CSV files\n3. **Extract facts** - The system will automatically extract knowledge from your content\n4. **Build knowledge** - Add more text or files to expand your knowledge base\n5. **Save knowledge** - Use 'Save Knowledge' to persist your data\n\n💡 **Start by adding some text or uploading a document!**"
    
    # Organize facts by subject for better readability
    facts_by_subject = {}
    all_facts = []
    
    for s, p, o in graph:
        subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
        predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
        object_val = str(o)
        
        fact_text = f"{subject} {predicate} {object_val}"
        all_facts.append(fact_text)
        
        if subject not in facts_by_subject:
            facts_by_subject[subject] = []
        facts_by_subject[subject].append(f"{predicate} {object_val}")
    
    # Create organized display
    result = f"📊 **Knowledge Graph Overview**\n"
    result += f"**Total Facts:** {len(graph)}\n"
    result += f"**Unique Subjects:** {len(facts_by_subject)}\n\n"
    
    # Show facts organized by subject
    result += "## 📚 **Knowledge by Subject:**\n\n"
    
    for i, (subject, facts) in enumerate(facts_by_subject.items()):
        if i >= 10:  # Limit to first 10 subjects for readability
            remaining = len(facts_by_subject) - 10
            result += f"... and {remaining} more subjects\n"
            break
            
        result += f"**{subject}:**\n"
        for fact in facts:
            result += f"  • {fact}\n"
        result += "\n"
    
    # Show all facts in a simple list
    result += "## 📋 **All Facts:**\n\n"
    for i, fact in enumerate(all_facts[:20]):  # Show first 20 facts
        result += f"{i+1}. {fact}\n"
    
    if len(all_facts) > 20:
        result += f"\n... and {len(all_facts) - 20} more facts"
    
    # Add search suggestions
    result += "\n\n## 🔍 **Search Suggestions:**\n"
    result += "Try asking the chatbot about any of these subjects or facts!\n"
    result += "Examples: 'What do you know about [subject]?' or 'Tell me about [fact]'"
    
    return result

# =========================================================
#  📁 File Processing Functions
# =========================================================

def extract_text_from_pdf(file_path):
    """Extract text from PDF file with better error handling"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            print(f"📄 PDF has {len(pdf_reader.pages)} pages")
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n"
                print(f"📄 Page {i+1}: {len(page_text)} characters")
            
            extracted_text = text.strip()
            print(f"📄 Total extracted: {len(extracted_text)} characters")
            print(f"📄 First 200 chars: {extracted_text[:200]}...")
            
            return extracted_text
    except Exception as e:
        error_msg = f"Error reading PDF: {e}"
        print(f"❌ {error_msg}")
        return error_msg

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading DOCX: {e}"

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        return f"Error reading TXT: {e}"

def extract_text_from_csv(file_path):
    """Extract text from CSV file"""
    try:
        df = pd.read_csv(file_path)
        # Convert DataFrame to readable text
        text = f"CSV Data with {len(df)} rows and {len(df.columns)} columns:\n\n"
        text += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Add first few rows as examples
        text += "Sample data:\n"
        for i, row in df.head(5).iterrows():
            text += f"Row {i+1}: {dict(row)}\n"
        
        return text.strip()
    except Exception as e:
        return f"Error reading CSV: {e}"

def process_uploaded_file(file):
    """Process uploaded file and extract text"""
    if file is None:
        return "No file uploaded."
    
    file_path = file.name
    file_extension = os.path.splitext(file_path)[1].lower()
    
    print(f"📁 Processing file: {file_path} (type: {file_extension})")
    
    # Extract text based on file type
    if file_extension == '.pdf':
        extracted_text = extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        extracted_text = extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        extracted_text = extract_text_from_txt(file_path)
    elif file_extension == '.csv':
        extracted_text = extract_text_from_csv(file_path)
    else:
        return f"❌ Unsupported file type: {file_extension}\n\nSupported formats: PDF, DOCX, TXT, CSV"
    
    if extracted_text.startswith("Error"):
        return f"❌ {extracted_text}"
    
    # Store extracted text for debugging
    update_extracted_text(extracted_text)
    
    # Show preview of extracted text
    preview = extracted_text[:300] + "..." if len(extracted_text) > 300 else extracted_text
    print(f"📄 Extracted text preview: {preview}")
    
    # Add extracted text to knowledge graph
    result = add_to_graph(extracted_text)
    
    # Return detailed summary
    file_size = len(extracted_text)
    return f"✅ Successfully processed {os.path.basename(file_path)}!\n\n📊 File stats:\n• Size: {file_size:,} characters\n• Type: {file_extension.upper()}\n\n📄 Text preview:\n{preview}\n\n{result}"

def handle_file_upload(files):
    """Handle multiple file uploads and processing"""
    global processed_files
    
    if not files or len(files) == 0:
        return "⚠️ Please select at least one file to upload."
    
    results = []
    new_processed = []
    
    for file in files:
        if file is None:
            continue
            
        try:
            # Handle both file objects and string paths
            if isinstance(file, str):
                file_path = file
                file_name = os.path.basename(file)
            else:
                file_path = file.name
                file_name = os.path.basename(file.name)
            
            # Check if file was already processed
            if any(f['name'] == file_name for f in processed_files):
                results.append(f"⏭️ {file_name} - Already processed, skipping")
                continue
            
            # Process the file
            result = process_uploaded_file(file_path)
            results.append(f"✅ {file_name} - {result}")
            
            # Add to processed files list
            new_processed.append({
                'name': file_name,
                'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'facts_added': len(graph) - sum(f.get('facts_count', 0) for f in processed_files)
            })
            
        except Exception as e:
            # Handle both file objects and string paths for error reporting
            if isinstance(file, str):
                file_name = os.path.basename(file)
            else:
                file_name = os.path.basename(file.name) if hasattr(file, 'name') else str(file)
            
            error_msg = f"❌ {file_name} - Error: {e}"
            print(error_msg)
            results.append(error_msg)
    
    # Update processed files list
    processed_files.extend(new_processed)
    
    # Create summary
    total_files = len(files)
    successful = len([r for r in results if r.startswith("✅")])
    skipped = len([r for r in results if r.startswith("⏭️")])
    failed = len([r for r in results if r.startswith("❌")])
    
    summary = f"📊 **Upload Summary:**\n"
    summary += f"• Total files: {total_files}\n"
    summary += f"• Successfully processed: {successful}\n"
    summary += f"• Already processed: {skipped}\n"
    summary += f"• Failed: {failed}\n"
    summary += f"• Total facts in knowledge base: {len(graph)}\n\n"
    
    # Add individual results
    summary += "📄 **File Results:**\n"
    for result in results:
        summary += f"{result}\n"
    
    return summary

def show_processed_files():
    """Show list of processed files"""
    global processed_files
    
    if not processed_files:
        return "📁 **No files processed yet.**\n\n🚀 **Start building your knowledge base:**\n1. Select one or more files (PDF, DOCX, TXT, CSV)\n2. Click 'Process Files' to extract knowledge\n3. View your processed files here\n4. Upload more files to expand your knowledge base!"
    
    result = f"📁 **Processed Files ({len(processed_files)}):**\n\n"
    
    for i, file_info in enumerate(processed_files, 1):
        result += f"**{i}. {file_info['name']}**\n"
        result += f"   • Size: {file_info['size']:,} bytes\n"
        result += f"   • Processed: {file_info['processed_at']}\n"
        result += f"   • Facts added: {file_info.get('facts_added', 'Unknown')}\n\n"
    
    result += f"🧠 **Total Knowledge Base:** {len(graph)} facts\n"
    result += f"📊 **Ready for more uploads!**"
    
    return result

def clear_processed_files():
    """Clear the processed files list"""
    global processed_files
    processed_files = []
    return "🗑️ Processed files list cleared. You can now re-upload previously processed files."


def simple_test():
    """Simple test function to verify event handlers work"""
    print("🔔 Simple test function called!")
    return "✅ Event handler is working! Button clicked successfully!"

# Global variable to store last extracted text
last_extracted_text = ""

# Global variable to track processed files
processed_files = []

def show_extracted_text():
    """Show the last extracted text from file processing"""
    global last_extracted_text
    
    if not last_extracted_text:
        return "📄 No file has been processed yet.\n\nUpload a file and process it to see the extracted text here."
    
    # Show first 1000 characters
    preview = last_extracted_text[:1000]
    if len(last_extracted_text) > 1000:
        preview += "\n\n... (truncated, showing first 1000 characters)"
    
    return f"📄 **Last Extracted Text:**\n\n{preview}"

def update_extracted_text(text):
    """Update the global variable with extracted text"""
    global last_extracted_text
    last_extracted_text = text

def delete_all_knowledge():
    """Delete all knowledge from the graph"""
    global graph
    count = len(graph)
    graph = rdflib.Graph()  # Create a new empty graph
    save_knowledge_graph()  # Save the empty graph
    return f"🗑️ Deleted all {count} facts from the knowledge graph. Graph is now empty."

def delete_knowledge_by_keyword(keyword):
    """Delete knowledge containing a specific keyword"""
    global graph
    if not keyword or keyword.strip() == "":
        return "⚠️ Please enter a keyword to search for."
    
    keyword = keyword.strip().lower()
    deleted_count = 0
    facts_to_remove = []
    
    # Find facts containing the keyword
    for s, p, o in graph:
        fact_text = f"{s} {p} {o}".lower()
        if keyword in fact_text:
            facts_to_remove.append((s, p, o))
    
    # Remove the facts
    for fact in facts_to_remove:
        graph.remove(fact)
        deleted_count += 1
    
    if deleted_count > 0:
        save_knowledge_graph()  # Save after deletion
        return f"🗑️ Deleted {deleted_count} facts containing '{keyword}'"
    else:
        return f"ℹ️ No facts found containing '{keyword}'"

def delete_recent_knowledge(count=5):
    """Delete the most recently added knowledge"""
    global graph
    if len(graph) == 0:
        return "ℹ️ Knowledge graph is already empty."
    
    # Convert graph to list to get order
    facts = list(graph)
    facts_to_remove = facts[-count:] if count < len(facts) else facts
    
    # Remove the facts
    for fact in facts_to_remove:
        graph.remove(fact)
    
    save_knowledge_graph()  # Save after deletion
    return f"🗑️ Deleted {len(facts_to_remove)} most recent facts"


# =========================================================
#  🤖 2. Intelligent Response Generation
# =========================================================

def generate_intelligent_response(message, context, system_message):
    """Generate intelligent responses based on available facts"""
    message_lower = message.lower()
    
    # Document understanding questions
    if any(phrase in message_lower for phrase in [
        'what is the document about', 'whats the document about', 'what is this about', 'whats this about', 
        'describe the document', 'summarize the document', 'what does this contain', 'what is this about'
    ]):
        return generate_document_summary(context)
    
    # General "what" questions
    elif message_lower.startswith('what'):
        return generate_what_response(message, context)
    
    # "Who" questions
    elif message_lower.startswith('who'):
        return generate_who_response(message, context)
    
    # "When" questions
    elif message_lower.startswith('when'):
        return generate_when_response(message, context)
    
    # "Where" questions
    elif message_lower.startswith('where'):
        return generate_where_response(message, context)
    
    # "How much" or amount questions
    elif any(phrase in message_lower for phrase in [
        'how much', 'amount', 'total', 'cost', 'price'
    ]):
        return generate_amount_response(message, context)
    
    # Default intelligent response
    else:
        return generate_general_response(message, context)

def generate_document_summary(context):
    """Generate a summary of what the document is about"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have enough information about this document to provide a summary. Please add more knowledge to the knowledge base first."
    
    # Extract key information from context
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            facts.append(line.strip())
    
    # Analyze the facts to understand document type
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
        elif 'company' in fact_lower or 'organization' in fact_lower or 'name' in fact_lower:
            key_info.append(fact)
        elif 'amount' in fact_lower or 'total' in fact_lower or 'cost' in fact_lower or 'price' in fact_lower:
            key_info.append(fact)
        elif 'date' in fact_lower or 'time' in fact_lower:
            key_info.append(fact)
        elif 'address' in fact_lower or 'location' in fact_lower:
            key_info.append(fact)
        elif 'description' in fact_lower or 'type' in fact_lower:
            key_info.append(fact)
        elif 'id' in fact_lower or 'number' in fact_lower or 'code' in fact_lower:
            key_info.append(fact)
    
    # Generate summary
    summary = f"Based on the information in my knowledge base, this appears to be a **{document_type}** document. "
    
    if key_info:
        summary += "Here are the key details I found:\n\n"
        for info in key_info[:5]:  # Limit to 5 most relevant facts
            summary += f"• {info}\n"
    else:
        summary += "However, I don't have enough specific details to provide a comprehensive summary."
    
    return summary

def generate_what_response(message, context):
    """Generate responses for 'what' questions"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have information about that topic in my knowledge base. Try asking about specific details that might be in the document."
    
    # Extract relevant facts
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            facts.append(line.strip())
    
    if not facts:
        return "I don't have specific information about that in my knowledge base."
    
    # Generate contextual response
    response = "Based on my knowledge base, here's what I can tell you:\n\n"
    for fact in facts[:3]:  # Show top 3 most relevant facts
        response += f"• {fact}\n"
    
    if len(facts) > 3:
        response += f"\nI have {len(facts)} total facts about this topic in my knowledge base."
    
    return response

def generate_who_response(message, context):
    """Generate responses for 'who' questions"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have information about people or entities in my knowledge base."
    
    # Look for person/company related facts
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            if any(keyword in line.lower() for keyword in ['company', 'name', 'person', 'επωνυμία', 'εταιρεία']):
                facts.append(line.strip())
    
    if not facts:
        return "I don't have specific information about people or companies in my knowledge base."
    
    response = "Here's what I know about people/entities:\n\n"
    for fact in facts:
        response += f"• {fact}\n"
    
    return response

def generate_when_response(message, context):
    """Generate responses for 'when' questions"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have date information in my knowledge base."
    
    # Look for date related facts
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            if any(keyword in line.lower() for keyword in ['date', 'ημερομηνία', 'due', 'προθεσμία']):
                facts.append(line.strip())
    
    if not facts:
        return "I don't have specific date information in my knowledge base."
    
    response = "Here's the date information I have:\n\n"
    for fact in facts:
        response += f"• {fact}\n"
    
    return response

def generate_where_response(message, context):
    """Generate responses for 'where' questions"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have location information in my knowledge base."
    
    # Look for address/location related facts
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            if any(keyword in line.lower() for keyword in ['address', 'διεύθυνση', 'location', 'place']):
                facts.append(line.strip())
    
    if not facts:
        return "I don't have specific location information in my knowledge base."
    
    response = "Here's the location information I have:\n\n"
    for fact in facts:
        response += f"• {fact}\n"
    
    return response

def generate_amount_response(message, context):
    """Generate responses for amount/money questions"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have financial information in my knowledge base."
    
    # Look for amount/money related facts
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            if any(keyword in line.lower() for keyword in ['amount', 'total', 'price', 'cost', 'σύνολο', 'φόρος', '€', '$']):
                facts.append(line.strip())
    
    if not facts:
        return "I don't have specific financial information in my knowledge base."
    
    response = "Here's the financial information I have:\n\n"
    for fact in facts:
        response += f"• {fact}\n"
    
    return response

def generate_general_response(message, context):
    """Generate general intelligent responses"""
    if not context or "No directly relevant facts found" in context:
        return "I don't have specific information about that topic in my knowledge base. Try asking about details that might be in the uploaded document, like company names, dates, amounts, or addresses."
    
    # Extract facts and provide intelligent response
    facts = []
    lines = context.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('**'):
            facts.append(line.strip())
    
    if not facts:
        return "I don't have relevant information about that in my knowledge base."
    
    response = "Based on my knowledge base, here's what I can tell you:\n\n"
    for fact in facts[:4]:  # Show top 4 most relevant facts
        response += f"• {fact}\n"
    
    if len(facts) > 4:
        response += f"\nI have {len(facts)} total relevant facts about this topic."
    
    return response

# =========================================================
#  🤖 3. Reasoning Function (LLM + Symbolic Context)
# =========================================================
def respond(message, history, system_message, max_tokens, temperature, top_p):
    # Step 1: retrieve context from symbolic KB
    context = retrieve_context(message)

    # Step 2: Try intelligent response generation first
    try:
        intelligent_response = generate_intelligent_response(message, context, system_message)
        print(f"🧠 Generated intelligent response for: {message[:50]}...")
        yield intelligent_response
        return
    except Exception as e:
        print(f"⚠️ Intelligent response failed: {e}")
        # Fall back to AI model approach

    # Step 3: Fallback to AI models if intelligent response fails
    # Enhanced prompt for better responses
    prompt = (
        f"{system_message}\n\n"
        f"Context from knowledge base:\n{context}\n\n"
        f"User Question: {message}\n\n"
        f"Instructions:\n"
        f"- Answer based ONLY on the facts provided above\n"
        f"- Be specific and factual\n"
        f"- If you don't have enough information, say so clearly\n"
        f"- Provide a helpful and informative response\n"
        f"- Keep your answer concise but complete\n\n"
        f"Answer:"
    )

    try:
        # Try to get HF token from environment variables
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        # Enhanced model list - more powerful free models, ordered by quality
        models_to_try = [
            # High-quality free models (no token required)
            ("microsoft/DialoGPT-medium", None),  # Conversational AI, good for Q&A
            ("facebook/blenderbot-400M-distill", None),  # Facebook's conversational model
            ("microsoft/DialoGPT-small", None),  # Smaller but reliable DialoGPT
            ("distilgpt2", None),  # Fast and reliable
            ("gpt2", None),  # Most reliable fallback
        ]
        
        # Add authenticated models if token is available (these are usually better)
        if hf_token:
            # Insert powerful authenticated models at the beginning
            authenticated_models = [
                ("HuggingFaceH4/zephyr-7b-beta", hf_token),  # High-quality instruction following
                ("microsoft/DialoGPT-large", hf_token),  # Large conversational model
                ("facebook/blenderbot-1B-distill", hf_token),  # Large Facebook model
                ("EleutherAI/gpt-neo-125M", hf_token),  # GPT-Neo model
            ]
            models_to_try = authenticated_models + models_to_try
        
        # Try each model
        for model, token in models_to_try:
            try:
                print(f"🔄 Attempting to use model: {model}")
                
                # Create client
                if token:
                    client = InferenceClient(model=model, token=token)
                else:
                    client = InferenceClient(model=model)
                
                # Try to generate response with optimized parameters
                result = client.text_generation(
                    prompt=prompt,
                    max_new_tokens=min(int(max_tokens), 150),  # Optimized for speed
                    temperature=min(float(temperature), 0.8),  # Cap temperature for consistency
                    top_p=min(float(top_p), 0.9),  # Cap top_p for better quality
                    repetition_penalty=1.1,  # Slightly higher to avoid repetition
                    do_sample=True,  # Enable sampling for better responses
                    stream=False,
                    return_full_text=False,
                )
                
                print(f"✅ Successfully generated response using: {model}")
                yield result.strip()
                return  # Success! Exit the function
                
            except Exception as model_error:
                print(f"❌ Model {model} failed: {model_error}")
                continue  # Try next model
        
        # If all models failed, provide intelligent fallback
        print("⚠️ All models failed, providing intelligent fallback")
        fallback_response = generate_intelligent_response(message, context, system_message)
        yield fallback_response
        
    except Exception as e:
        # Ultimate fallback - even if everything fails
        print(f"💥 Complete failure: {e}")
        yield f"🤖 I'm having trouble connecting to AI models right now, but I can still help!\n\nBased on your knowledge graph, I found these relevant facts:\n{context}\n\nFor your question '{message}', I'd suggest checking the facts above. Try adding more information to the knowledge graph or check back later when the AI models are working properly."

def generate_mock_response(message, context, system_message):
    """Generate a helpful response even when AI models fail"""
    
    # Simple keyword-based responses
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return f"👋 Hello! I'm your reasoning assistant. I found these facts in your knowledge base:\n\n{context}\n\nHow can I help you today?"
    
    elif any(word in message_lower for word in ['what', 'who', 'when', 'where', 'how', 'why']):
        return f"🤔 Great question! Based on your knowledge graph, here's what I found:\n\n{context}\n\nWhile I can't provide a full AI-generated answer right now, these facts from your knowledge base should help you understand the topic better."
    
    elif any(word in message_lower for word in ['help', 'assist', 'support']):
        return f"🆘 I'm here to help! Your knowledge graph contains:\n\n{context}\n\nYou can:\n• Add more information to the knowledge graph\n• Ask specific questions about the facts\n• Try again later when AI models are working"
    
    else:
        return f"💭 Interesting question! From your knowledge base, I found:\n\n{context}\n\nWhile I'm having technical difficulties with AI models, I can still help you explore the information you've added to the knowledge graph. Try asking more specific questions or adding more context!"

# =========================================================
#  💬 3. Gradio Interface Definition
# =========================================================

# =========================================================
#  🧩 4. Interface Layout
# =========================================================
with gr.Blocks(title="🧠 Reasoning Researcher Prototype") as demo:
    gr.Markdown("## 🧠 Reasoning Researcher\nA neurosymbolic assistant that combines a small knowledge graph with language reasoning.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Add Knowledge")
            
            # Text input
            upload_box = gr.Textbox(
                lines=6,
                placeholder="Paste any text here to extract knowledge and add to your knowledge base. Examples: articles, notes, descriptions, reports, etc.",
                label="Add Knowledge from Text",
            )
            add_button = gr.Button("Add to Knowledge Graph", variant="primary")
            
            gr.Markdown("### 📁 Upload Files")
            
            # File upload
            file_upload = gr.File(
                label="Upload Knowledge Files (Multiple files supported)",
                file_types=[".pdf", ".docx", ".txt", ".csv"],
                file_count="multiple"
            )
            upload_file_button = gr.Button("Process Files", variant="secondary")
            show_processed_button = gr.Button("📁 Show Processed Files", variant="secondary")
            clear_processed_button = gr.Button("🗑️ Clear File History", variant="secondary")
            file_status = gr.Textbox(label="File Processing Status", interactive=False)
            
            gr.Markdown("### 🧠 Knowledge Graph")
            
            # Graph management
            simple_test_button = gr.Button("🔔 Simple Test", variant="secondary")
            show_button = gr.Button("Show Knowledge Graph", variant="secondary")
            save_button = gr.Button("💾 Save Knowledge", variant="secondary")
            load_button = gr.Button("📂 Load Knowledge", variant="secondary")
            storage_info_button = gr.Button("📁 Storage Info", variant="secondary")
            extract_debug_button = gr.Button("🔍 Debug Extraction", variant="secondary")
            
            gr.Markdown("### 🗑️ Delete Knowledge")
            
            # Deletion options
            delete_keyword_input = gr.Textbox(
                placeholder="Enter keyword to delete facts containing it...",
                label="Delete by Keyword",
                lines=1
            )
            delete_keyword_button = gr.Button("🗑️ Delete by Keyword", variant="stop")
            delete_recent_button = gr.Button("🗑️ Delete Last 5 Facts", variant="stop")
            delete_all_button = gr.Button("⚠️ DELETE ALL KNOWLEDGE", variant="stop")
            
            download_button = gr.File(label="📥 Download Knowledge (JSON + Text)", value=get_knowledge_file)
            graph_info = gr.Textbox(label="Status / Graph Summary", interactive=False)
            graph_view = gr.Textbox(label="Knowledge Graph Contents", lines=15)
            
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat with AI")
            chatbot = gr.ChatInterface(
                respond,
                type="messages",
                additional_inputs=[
                    gr.Textbox(
                        value="You are an intelligent assistant that answers questions based on factual information from a knowledge base. You provide clear, accurate, and helpful responses. When you have relevant information, you share it directly. When you don't have enough information, you clearly state this limitation. You always stay grounded in the facts provided and never hallucinate information.",
                        label="System message",
                    ),
                    gr.Slider(minimum=64, maximum=1024, value=256, step=16, label="Max new tokens"),
                    gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
                    gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p"),
                ],
            )
            
    # Event handlers
    simple_test_button.click(fn=simple_test, inputs=None, outputs=graph_info)
    add_button.click(fn=handle_add_knowledge, inputs=upload_box, outputs=graph_info)
    upload_file_button.click(fn=handle_file_upload, inputs=file_upload, outputs=file_status)
    show_processed_button.click(fn=show_processed_files, inputs=None, outputs=file_status)
    clear_processed_button.click(fn=clear_processed_files, inputs=None, outputs=file_status)
    show_button.click(fn=show_graph_contents, inputs=None, outputs=graph_view)
    save_button.click(fn=save_knowledge_graph, inputs=None, outputs=graph_info)
    load_button.click(fn=load_knowledge_graph, inputs=None, outputs=graph_info)
    storage_info_button.click(fn=show_storage_info, inputs=None, outputs=graph_view)
    extract_debug_button.click(fn=show_extracted_text, inputs=None, outputs=graph_view)
    
    # Deletion event handlers
    delete_keyword_button.click(fn=delete_knowledge_by_keyword, inputs=delete_keyword_input, outputs=graph_info)
    delete_recent_button.click(fn=delete_recent_knowledge, inputs=None, outputs=graph_info)
    delete_all_button.click(fn=delete_all_knowledge, inputs=None, outputs=graph_info)

# =========================================================
#  🚀 5. Initialize Sample Data and Launch
# =========================================================


if __name__ == "__main__":
    # Initialize empty knowledge graph - no sample data
    # Knowledge will be built purely from uploaded documents
    graph = rdflib.Graph()
    print("📊 Knowledge graph initialized (empty) - ready for document uploads")
    demo.launch()
