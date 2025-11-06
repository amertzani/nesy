"""
FastAPI Backend Server - Research Brain
========================================

This is the MAIN BACKEND ENTRY POINT. It provides REST API endpoints
for the React frontend to interact with the knowledge management system.

Architecture:
- Receives HTTP requests from frontend (React app)
- Processes requests using knowledge.py, file_processing.py, documents_store.py
- Returns JSON responses

Key Endpoints:
- POST /api/knowledge/upload: Upload and process documents
- POST /api/knowledge/facts: Create a new fact
- GET /api/knowledge/facts: Get all facts
- DELETE /api/knowledge/facts/{id}: Delete a fact
- GET /api/documents: Get all uploaded documents
- GET /api/export: Export all knowledge as JSON
- POST /api/knowledge/import: Import knowledge from JSON

Connection:
- Frontend connects to: http://localhost:8001 (default)
- API docs available at: http://localhost:8001/docs

Author: Research Brain Team
Last Updated: 2025-01-15
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile

# Import your existing modules
from responses import respond as rqa_respond
from knowledge import (
    add_to_graph as kb_add_to_graph,
    show_graph_contents as kb_show_graph_contents,
    visualize_knowledge_graph as kb_visualize_knowledge_graph,
    save_knowledge_graph as kb_save_knowledge_graph,
    load_knowledge_graph as kb_load_knowledge_graph,
    delete_all_knowledge as kb_delete_all_knowledge,
    graph as kb_graph,
    import_knowledge_from_json_file as kb_import_json
)
from file_processing import handle_file_upload as fp_handle_file_upload
from documents_store import add_document, get_all_documents, delete_document as ds_delete_document, cleanup_documents_without_facts
from knowledge import create_comprehensive_backup as kb_create_comprehensive_backup

from contextlib import asynccontextmanager

# Load knowledge graph on startup using lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing knowledge graph...")
    try:
        load_result = kb_load_knowledge_graph()
        print(f"Startup: {load_result}")
        fact_count = len(kb_graph)
        print(f"Knowledge graph ready with {fact_count} facts")
        
        # Clean up documents without facts from previous sessions
        print("Cleaning up documents without facts...")
        cleaned_count = cleanup_documents_without_facts()
        if cleaned_count > 0:
            print(f"✅ Cleaned up {cleaned_count} documents without facts")
        else:
            print("✅ No documents needed cleanup")
        
        # Verify the count makes sense
        if 'Loaded' in load_result and fact_count == 0:
            print("⚠️  Warning: Graph loaded but shows 0 facts. This might indicate a cleanup issue.")
    except Exception as e:
        print(f"⚠️  Warning during knowledge graph load: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with empty graph...")
    yield
    # Shutdown (if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(title="NesyX API", description="Backend API for NesyX Knowledge Graph System", lifespan=lifespan)

# Configure CORS - Allow all origins (you can restrict this to specific domains later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Request/Response Models
# ==========================================================

class ChatMessage(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []

class AddKnowledgeRequest(BaseModel):
    text: str

class AddFactRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    source: Optional[str] = "manual"

class DeleteKnowledgeRequest(BaseModel):
    keyword: Optional[str] = None
    count: Optional[int] = None

# ==========================================================
# API Endpoints
# ==========================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "NesyX API",
        "facts_count": len(kb_graph)
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "facts": len(kb_graph)}

@app.post("/api/chat")
async def chat_endpoint(request: ChatMessage):
    """Chat endpoint - ask questions about the knowledge base"""
    try:
        response = rqa_respond(request.message, request.history)
        return {
            "response": response,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/api/knowledge/add")
async def add_knowledge_endpoint(request: AddKnowledgeRequest):
    """Add knowledge to the graph from text"""
    try:
        # Get current fact count before adding
        fact_count_before = len(kb_graph)
        
        # Add knowledge to graph
        result = kb_add_to_graph(request.text)
        kb_save_knowledge_graph()
        
        # Get newly added facts (those added after the operation)
        # Extract the last fact added (most recent)
        facts_list = []
        for i, (s, p, o) in enumerate(kb_graph):
            subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
            predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
            object_val = str(o)
            facts_list.append({
                "id": i + 1,
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "source": "manual"
            })
        
        # Return the last fact added (most recent)
        new_fact = facts_list[-1] if facts_list else None
        
        return {
            "message": result,
            "status": "success",
            "total_facts": len(kb_graph),
            "fact": new_fact  # Return the created fact for frontend
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding knowledge: {str(e)}")

@app.post("/api/knowledge/facts")
async def create_fact_endpoint(request: AddFactRequest):
    """Create a structured fact directly (subject, predicate, object)"""
    try:
        import rdflib
        from urllib.parse import quote
        from knowledge import fact_exists as kb_fact_exists
        
        # Check if fact already exists
        if kb_fact_exists(request.subject, request.predicate, str(request.object)):
            print(f"⚠️  POST /api/knowledge/facts: Duplicate fact detected - {request.subject} {request.predicate} {request.object}")
            return {
                "message": "Fact already exists in knowledge graph",
                "status": "duplicate",
                "fact": {
                    "subject": request.subject,
                    "predicate": request.predicate,
                    "object": str(request.object)
                },
                "total_facts": len(kb_graph)
            }
        
        # For structured facts, add directly to graph with proper URI encoding
        # Replace spaces with underscores in URIs to avoid RDFLib warnings
        subject_clean = request.subject.strip().replace(' ', '_')
        predicate_clean = request.predicate.strip().replace(' ', '_')
        object_value = str(request.object).strip()
        
        # Create URIs (encode spaces to avoid RDFLib warnings)
        subject_uri = rdflib.URIRef(f"urn:{quote(subject_clean, safe='')}")
        predicate_uri = rdflib.URIRef(f"urn:{quote(predicate_clean, safe='')}")
        object_literal = rdflib.Literal(object_value)
        
        # Add directly to graph
        kb_graph.add((subject_uri, predicate_uri, object_literal))
        
        # Save to disk
        save_result = kb_save_knowledge_graph()
        
        # Verify the fact was added
        fact_count = len(kb_graph)
        print(f"✅ POST /api/knowledge/facts: Added fact - {request.subject} {request.predicate} {request.object}")
        print(f"✅ Save result: {save_result}")
        print(f"✅ Total facts in graph: {fact_count}")
        
        # Verify file was written
        import os
        if os.path.exists("knowledge_graph.pkl"):
            file_size = os.path.getsize("knowledge_graph.pkl")
            print(f"✅ Knowledge file size: {file_size} bytes")
        
        # Create the fact object - use the actual index in the graph
        new_fact = {
            "id": fact_count,  # Use current count as ID
            "subject": request.subject,  # Return original subject (with spaces)
            "predicate": request.predicate,  # Return original predicate (with spaces)
            "object": object_value,  # Return original object
            "source": request.source
        }
        
        return {
            "message": f"✅ Added fact successfully. Total facts: {fact_count}",
            "status": "success",
            "total_facts": fact_count,
            "fact": new_fact
        }
    except Exception as e:
        print(f"❌ Error creating fact: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error creating fact: {str(e)}")

@app.post("/api/knowledge/upload")
async def upload_file_endpoint(files: List[UploadFile] = File(...)):
    """Upload and process files (PDF, DOCX, TXT, CSV)"""
    try:
        facts_before = len(kb_graph)
        tmp_paths = []
        file_info_list = []
        
        for file in files:
            # Save uploaded file temporarily
            suffix = os.path.splitext(file.filename)[1] if file.filename else ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_paths.append(tmp_file.name)
                file_info_list.append({
                    'name': file.filename or 'unknown',
                    'size': len(content),
                    'type': suffix.lstrip('.') or 'unknown'
                })
        
        try:
            # Process all files at once (handle_file_upload expects a list)
            result = fp_handle_file_upload(tmp_paths)
            
            # IMPORTANT: Reload graph from disk to ensure we have the latest facts
            # The graph might have been saved by add_to_graph, but we need to reload
            # to get the correct count and ensure consistency
            import time
            time.sleep(0.1)  # Small delay to ensure file write is complete
            kb_load_knowledge_graph()
            
            facts_after = len(kb_graph)
            facts_extracted = facts_after - facts_before
            
            # Parse result message to extract added/skipped counts
            import re
            added_match = re.search(r'Added (\d+) new triples', result)
            skipped_match = re.search(r'skipped (\d+) duplicates', result)
            added_count = int(added_match.group(1)) if added_match else facts_extracted
            skipped_count = int(skipped_match.group(1)) if skipped_match else 0
            
            print(f"✅ Upload processed {len(files)} file(s)")
            print(f"   Facts before: {facts_before}, after: {facts_after}, extracted: {facts_extracted}")
            print(f"   Added: {added_count}, Skipped duplicates: {skipped_count}")
            print(f"   Graph now has {len(kb_graph)} total facts")
            
            # Verify facts are actually in the graph
            if len(kb_graph) > 0:
                sample_fact = list(kb_graph)[0]
                print(f"   Sample fact from graph: {sample_fact}")
            else:
                print("   ⚠️  WARNING: Graph is empty after processing!")
            
            # Save document metadata - ONLY if facts were actually extracted
            # NEVER save documents with 0 facts
            processed_docs = []
            if facts_extracted > 0 and added_count > 0:
                # Only save/update documents if facts were actually extracted AND added
                for file_info in file_info_list:
                    doc = add_document(
                        name=file_info['name'],
                        size=file_info['size'],
                        file_type=file_info['type'],
                        facts_extracted=added_count  # Use actual added count (excluding duplicates)
                    )
                    processed_docs.append(doc)
                    print(f"✅ Saved document {file_info['name']} with {added_count} facts")
            else:
                # No facts extracted - PERMANENTLY REMOVE any existing documents with this name
                print(f"⚠️  No facts extracted from {len(files)} file(s) - REMOVING documents")
                for file_info in file_info_list:
                    from documents_store import load_documents, save_documents
                    docs = load_documents()
                    original_count = len(docs)
                    # PERMANENTLY REMOVE documents with this name
                    docs = [d for d in docs if d.get('name') != file_info['name']]
                    removed_count = original_count - len(docs)
                    if removed_count > 0:
                        save_documents(docs)
                        print(f"   🗑️  PERMANENTLY removed {file_info['name']} (no facts extracted)")
            
            print(f"✅ Upload processed {len(files)} file(s), added {added_count} new facts, skipped {skipped_count} duplicates")
            
            return {
                "message": result,
                "files_processed": len(files),
                "status": "success",
                "total_facts": len(kb_graph),
                "facts_extracted": added_count,  # Use actual added count
                "facts_skipped": skipped_count,   # Add skipped duplicates count
                "documents": processed_docs
            }
        finally:
            # Clean up temporary files
            for tmp_path in tmp_paths:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
    except Exception as e:
        print(f"❌ Error uploading files: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")

@app.get("/api/knowledge/graph")
async def get_graph_endpoint():
    """Get knowledge graph visualization"""
    try:
        graph_html = kb_visualize_knowledge_graph()
        return {
            "graph_html": graph_html,
            "total_facts": len(kb_graph),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting graph: {str(e)}")

@app.get("/api/knowledge/contents")
async def get_contents_endpoint():
    """Get all knowledge graph contents as text"""
    try:
        contents = kb_show_graph_contents()
        return {
            "contents": contents,
            "total_facts": len(kb_graph),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting contents: {str(e)}")

@app.get("/api/knowledge/facts")
async def get_facts_endpoint():
    """Get all knowledge graph facts as structured JSON array"""
    try:
        # ALWAYS reload from disk to ensure we have the latest persisted facts
        # This is critical after document processing
        load_result = kb_load_knowledge_graph()
        print(f"📥 GET /api/knowledge/facts: Reloaded graph - {load_result}")
        print(f"📥 Graph now has {len(kb_graph)} facts in memory")
        
        facts = []
        for i, (s, p, o) in enumerate(kb_graph):
            # Extract subject from URI (urn:subject -> subject)
            subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
            # Decode URL encoding and replace underscores back to spaces
            from urllib.parse import unquote
            subject = unquote(subject).replace('_', ' ')
            
            # Extract predicate from URI
            predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
            predicate = unquote(predicate).replace('_', ' ')
            
            # Object is already a literal, just get the string value
            object_val = str(o)
            
            facts.append({
                "id": i + 1,
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "source": "knowledge_graph"
            })
        
        print(f"✅ GET /api/knowledge/facts: Returning {len(facts)} facts")
        if len(facts) > 0:
            print(f"   Sample fact: {facts[0]}")
        else:
            print("   ⚠️  No facts in graph!")
            # Debug: Check if file exists
            import os
            if os.path.exists("knowledge_graph.pkl"):
                file_size = os.path.getsize("knowledge_graph.pkl")
                print(f"   📁 knowledge_graph.pkl exists ({file_size} bytes) but graph is empty!")
        
        response = {
            "facts": facts,
            "total_facts": len(kb_graph),
            "status": "success"
        }
        return response
    except Exception as e:
        print(f"❌ Error getting facts: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting facts: {str(e)}")

@app.get("/api/documents")
async def get_documents_endpoint(include_all: bool = False):
    """Get all uploaded documents
    
    Args:
        include_all: If True, return all documents. If False (default), only return documents that contributed facts.
    """
    try:
        # FIRST: ALWAYS clean up documents without facts before returning
        # This ensures documents with 0 facts are PERMANENTLY removed
        cleanup_documents_without_facts()
        
        all_documents = get_all_documents()
        
        # DOUBLE CHECK: Filter out any documents with facts_extracted = 0
        # This is a safety net in case cleanup didn't catch everything
        all_documents = [doc for doc in all_documents if doc.get('facts_extracted', 0) > 0]
        
        # Filter: ONLY return documents that have contributed facts (facts_extracted > 0)
        # This ensures we NEVER show documents without facts
        if not include_all:
            documents = all_documents  # Already filtered above
            print(f"✅ GET /api/documents: Returning {len(documents)} documents with facts")
        else:
            documents = all_documents
            print(f"✅ GET /api/documents: Returning {len(documents)} documents (all)")
        
        return {
            "documents": documents,
            "total_documents": len(documents),
            "total_all_documents": len(documents),  # Both are the same now (filtered)
            "status": "success"
        }
    except Exception as e:
        print(f"❌ Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")

@app.delete("/api/documents/{document_id}")
async def delete_document_endpoint(document_id: str):
    """Delete a document by ID"""
    try:
        success = ds_delete_document(document_id)
        if success:
            return {
                "message": "Document deleted successfully",
                "status": "success"
            }
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.delete("/api/knowledge/delete")
async def delete_knowledge_endpoint(request: DeleteKnowledgeRequest):
    """Delete knowledge from the graph"""
    try:
        if request.keyword:
            result = kb_delete_all_knowledge()  # You may want to implement keyword-based deletion
        elif request.count:
            from knowledge import delete_recent_knowledge
            result = delete_recent_knowledge(request.count)
        else:
            result = kb_delete_all_knowledge()
        
        kb_save_knowledge_graph()
        return {
            "message": result,
            "status": "success",
            "total_facts": len(kb_graph)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting knowledge: {str(e)}")

@app.delete("/api/knowledge/facts/{fact_id}")
async def delete_fact_endpoint(fact_id: str):
    """Delete a specific fact by ID (subject, predicate, object)"""
    try:
        import rdflib
        from urllib.parse import quote
        from knowledge import fact_exists as kb_fact_exists
        
        # Parse fact_id - it should be in format "subject|predicate|object"
        # Or we can accept it as a JSON string
        try:
            import json
            from urllib.parse import unquote
            # Decode URL encoding first
            decoded_id = unquote(fact_id)
            fact_data = json.loads(decoded_id)
            subject = fact_data.get('subject')
            predicate = fact_data.get('predicate')
            object_val = fact_data.get('object')
        except json.JSONDecodeError:
            # Try parsing as pipe-separated
            parts = fact_id.split('|')
            if len(parts) == 3:
                subject, predicate, object_val = parts
            else:
                # Try to find fact by searching all facts
                # This is a fallback - ideally fact_id should be structured
                raise HTTPException(status_code=400, detail="Invalid fact ID format. Expected JSON or 'subject|predicate|object'")
        
        if not subject or not predicate or object_val is None:
            raise HTTPException(status_code=400, detail="Missing subject, predicate, or object")
        
        # Check if fact exists
        if not kb_fact_exists(subject, predicate, str(object_val)):
            raise HTTPException(status_code=404, detail="Fact not found in knowledge graph")
        
        # Create URI-encoded triple to match how it's stored
        subject_clean = str(subject).strip().replace(' ', '_')
        predicate_clean = str(predicate).strip().replace(' ', '_')
        object_value = str(object_val).strip()
        
        subject_uri = rdflib.URIRef(f"urn:{quote(subject_clean, safe='')}")
        predicate_uri = rdflib.URIRef(f"urn:{quote(predicate_clean, safe='')}")
        object_literal = rdflib.Literal(object_value)
        
        # Remove from graph
        if (subject_uri, predicate_uri, object_literal) in kb_graph:
            kb_graph.remove((subject_uri, predicate_uri, object_literal))
            kb_save_knowledge_graph()
            
            print(f"✅ DELETE /api/knowledge/facts/{fact_id}: Deleted fact - {subject} {predicate} {object_val}")
            print(f"✅ Graph now has {len(kb_graph)} facts")
            
            return {
                "message": "Fact deleted successfully",
                "status": "success",
                "deleted_fact": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_val
                },
                "total_facts": len(kb_graph)
            }
        else:
            raise HTTPException(status_code=404, detail="Fact not found in knowledge graph")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting fact: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting fact: {str(e)}")

@app.get("/api/export")
async def export_knowledge_endpoint():
    """Export all knowledge graph facts as JSON"""
    try:
        from datetime import datetime
        from urllib.parse import unquote
        
        # Reload graph to ensure we have latest facts
        kb_load_knowledge_graph()
        
        # Extract facts from graph
        facts = []
        for i, (s, p, o) in enumerate(kb_graph):
            # Extract subject from URI (urn:subject -> subject)
            subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
            # Decode URL encoding and replace underscores back to spaces
            subject = unquote(subject).replace('_', ' ')
            
            # Extract predicate from URI
            predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
            predicate = unquote(predicate).replace('_', ' ')
            
            # Object is already a literal, just get the string value
            object_val = str(o)
            
            facts.append({
                "id": str(i + 1),
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "source": "knowledge_graph"
            })
        
        # Create export response with metadata
        export_data = {
            "facts": facts,
            "metadata": {
                "version": "1.2.3",
                "totalFacts": len(facts),
                "lastUpdated": datetime.now().isoformat()
            }
        }
        
        print(f"✅ GET /api/export: Exporting {len(facts)} facts")
        return export_data
    except Exception as e:
        print(f"❌ Error exporting knowledge: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error exporting knowledge: {str(e)}")

@app.post("/api/knowledge/import")
async def import_json_endpoint(file: UploadFile = File(...)):
    """Import knowledge from JSON file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            facts_before = len(kb_graph)
            result = kb_import_json(tmp_path)
            kb_save_knowledge_graph()
            facts_after = len(kb_graph)
            facts_added = facts_after - facts_before
            
            # Parse the result message to extract added/skipped counts
            import re
            added_match = re.search(r'Imported (\d+)', result)
            skipped_match = re.search(r'skipped (\d+)', result)
            added_count = int(added_match.group(1)) if added_match else facts_added
            skipped_count = int(skipped_match.group(1)) if skipped_match else 0
            
            print(f"✅ POST /api/knowledge/import: Added {added_count} facts, skipped {skipped_count} duplicates")
            
            return {
                "message": result,
                "status": "success",
                "total_facts": len(kb_graph),
                "facts_added": added_count,
                "facts_skipped": skipped_count
            }
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing JSON: {str(e)}")

@app.get("/api/knowledge/stats")
async def get_stats_endpoint():
    """Get knowledge graph statistics"""
    try:
        return {
            "total_facts": len(kb_graph),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/api/knowledge/save")
async def save_knowledge_endpoint():
    """Manually trigger knowledge graph save"""
    try:
        result = kb_save_knowledge_graph()
        return {
            "message": result,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving knowledge: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8001 (to avoid conflicts)
    port = int(os.getenv("API_PORT", 8001))
    host = os.getenv("API_HOST", "0.0.0.0")  # Bind to all interfaces for external access
    
    # Try to find an available port
    import socket
    for attempt_port in [port, 8000, 8001, 8002, 8003]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', attempt_port))
        sock.close()
        if result != 0:  # Port is available
            port = attempt_port
            break
    else:
        print("⚠️  Warning: Could not find available port, trying 8001 anyway")
        port = 8001
    
    print(f"Starting NesyX API server on http://{host}:{port}")
    print(f"API documentation available at http://localhost:{port}/docs")
    print(f"Frontend should connect to: http://localhost:{port}")
    
    uvicorn.run(app, host=host, port=port)

