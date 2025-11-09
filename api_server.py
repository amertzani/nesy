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
from documents_store import add_document, get_all_documents, delete_document as ds_delete_document, cleanup_documents_without_facts, delete_all_documents as ds_delete_all_documents
from knowledge import create_comprehensive_backup as kb_create_comprehensive_backup

from contextlib import asynccontextmanager

# Load knowledge graph on startup using lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing knowledge graph...")
    try:
        # Clear all facts on every restart
        print("🗑️  Clearing all facts on startup...")
        delete_result = kb_delete_all_knowledge()
        print(f"Startup: {delete_result}")
        
        # Also clear all documents
        print("🗑️  Clearing all documents...")
        deleted_docs = ds_delete_all_documents()
        if deleted_docs > 0:
            print(f"✅ Deleted {deleted_docs} documents")
        else:
            print("✅ No documents to delete")
        
        # Verify graph is empty after clearing
        fact_count = len(kb_graph)
        print(f"✅ Knowledge graph initialized with {fact_count} facts (fresh start)")
        
        # IMPORTANT: Verify the graph file is actually empty
        import os
        if os.path.exists("knowledge_graph.pkl"):
            file_size = os.path.getsize("knowledge_graph.pkl")
            print(f"✅ Graph file size after clear: {file_size} bytes")
            if file_size > 1000:  # If file is still large, something went wrong
                print(f"⚠️  WARNING: Graph file is {file_size} bytes but graph has {fact_count} facts!")
                print("⚠️  This might indicate the clear didn't work properly")
        
        # Pre-load LLM model in background to avoid timeout on first request
        print("🔄 Pre-loading LLM model for research assistant (this may take 1-2 minutes)...")
        import asyncio
        from responses import load_llm_model
        
        # Start pre-loading in background (don't block startup)
        def preload_llm_sync():
            try:
                result = load_llm_model()
                if result:
                    print("✅ LLM model pre-loaded successfully")
                else:
                    print("⚠️  LLM model not available, will use rule-based responses")
            except Exception as e:
                print(f"⚠️  Failed to pre-load LLM: {e}")
                print("   Will use rule-based responses")
        
        # Run in background thread (non-blocking)
        import threading
        preload_thread = threading.Thread(target=preload_llm_sync, daemon=True)
        preload_thread.start()
        print("   (Model loading in background, server is ready)")
        
    except Exception as e:
        print(f"⚠️  Warning during knowledge graph initialization: {e}")
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
    details: Optional[str] = None

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
    import asyncio
    try:
        # Run the response generation in a thread pool to avoid blocking
        # and set a timeout to prevent hanging
        loop = asyncio.get_event_loop()
        
        # Check if LLM is still loading and wait a bit if needed
        from responses import LLM_PIPELINE, load_llm_model, USE_LLM, LLM_AVAILABLE
        if USE_LLM and LLM_AVAILABLE and LLM_PIPELINE is None:
            # Model not loaded yet, try to load it (with timeout)
            print("⏳ LLM not loaded yet, loading now...")
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, load_llm_model),
                    timeout=90.0  # Give 90 seconds for model loading
                )
            except asyncio.TimeoutError:
                print("⚠️  LLM loading timed out, using rule-based responses")
        
        # Generate response with timeout
        response = await asyncio.wait_for(
            loop.run_in_executor(None, rqa_respond, request.message, request.history),
            timeout=45.0  # 45 second timeout for response generation (model should be loaded by now)
        )
        return {
            "response": response,
            "status": "success"
        }
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, 
            detail="Request timed out. The LLM is taking too long to respond. Try disabling LLM with USE_LLM=false or ask a simpler question."
        )
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
        # add_to_graph already saves, but ensure it's saved
        kb_save_knowledge_graph()
        
        # Verify save worked
        if os.path.exists("knowledge_graph.pkl"):
            file_size = os.path.getsize("knowledge_graph.pkl")
            print(f"✅ Knowledge saved - file size: {file_size} bytes, facts in graph: {len(kb_graph)}")
        
        # Extract extraction method from result message
        extraction_method = "regex"
        if "TRIPLEX" in result.upper():
            extraction_method = "triplex"
        elif "FALLBACK" in result.upper():
            extraction_method = "regex (triplex fallback)"
        
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
            "fact": new_fact,  # Return the created fact for frontend
            "extraction_method": extraction_method  # Indicate which method was used
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
        
        # Add details if provided
        if request.details and request.details.strip():
            from knowledge import add_fact_details as kb_add_fact_details
            kb_add_fact_details(request.subject, request.predicate, object_value, request.details)
        
        # Add source document and timestamp (manual for directly created facts)
        from datetime import datetime
        from knowledge import add_fact_source_document as kb_add_fact_source_document
        timestamp = datetime.now().isoformat()
        kb_add_fact_source_document(request.subject, request.predicate, object_value, "manual", timestamp)
        
        # Save to disk
        save_result = kb_save_knowledge_graph()
        
        # Verify the fact was added
        fact_count = len(kb_graph)
        print(f"✅ POST /api/knowledge/facts: Added fact - {request.subject} {request.predicate} {request.object}")
        if request.details:
            print(f"✅ Added details: {request.details[:50]}...")
        print(f"✅ Save result: {save_result}")
        print(f"✅ Total facts in graph: {fact_count}")
        
        # Verify file was written
        if os.path.exists("knowledge_graph.pkl"):
            file_size = os.path.getsize("knowledge_graph.pkl")
            print(f"✅ Knowledge file size: {file_size} bytes")
        
        # Get details for the response
        from knowledge import get_fact_details as kb_get_fact_details
        details = kb_get_fact_details(request.subject, request.predicate, object_value)
        
        # Create the fact object - use the actual index in the graph
        new_fact = {
            "id": str(fact_count),  # Use current count as ID (string format)
            "subject": request.subject,  # Return original subject (with spaces)
            "predicate": request.predicate,  # Return original predicate (with spaces)
            "object": object_value,  # Return original object
            "source": request.source,
            "details": details if details else None
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

@app.get("/api/knowledge/triplex-status")
async def triplex_status_endpoint():
    """Get Triplex model status and availability"""
    try:
        from knowledge import TRIPLEX_AVAILABLE, USE_TRIPLEX, TRIPLEX_MODEL, TRIPLEX_DEVICE
        
        status = {
            "available": TRIPLEX_AVAILABLE,
            "enabled": USE_TRIPLEX,
            "loaded": TRIPLEX_MODEL is not None,
            "device": TRIPLEX_DEVICE if TRIPLEX_AVAILABLE else "N/A"
        }
        
        if TRIPLEX_AVAILABLE and USE_TRIPLEX:
            status["message"] = "Triplex is available and enabled. LLM extraction will be used."
        elif TRIPLEX_AVAILABLE and not USE_TRIPLEX:
            status["message"] = "Triplex is available but disabled. Set USE_TRIPLEX=true to enable."
        else:
            status["message"] = "Triplex is not available. Using regex-based extraction."
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Triplex status: {str(e)}")

@app.post("/api/knowledge/upload")
async def upload_file_endpoint(files: List[UploadFile] = File(...)):
    """Upload and process files (PDF, DOCX, TXT, CSV)"""
    tmp_paths = []  # Initialize outside try block so finally can access it
    try:
        facts_before = len(kb_graph)
        file_info_list = []
        
        # Map temporary file paths to original filenames
        temp_to_original = {}
        
        for file in files:
            # Save uploaded file temporarily
            suffix = os.path.splitext(file.filename)[1] if file.filename else ""
            original_filename = file.filename or 'unknown'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
                tmp_paths.append(tmp_path)
                temp_to_original[tmp_path] = original_filename
                file_info_list.append({
                    'name': original_filename,
                    'size': len(content),
                    'type': suffix.lstrip('.') or 'unknown'
                })
        
        try:
            # Process all files at once (handle_file_upload expects a list)
            # Pass the mapping of temp paths to original filenames
            result = fp_handle_file_upload(tmp_paths, original_filenames=temp_to_original)
            
            # IMPORTANT: Ensure graph is saved to disk
            # add_to_graph already saves, but let's make sure it's persisted
            kb_save_knowledge_graph()
            
            # CRITICAL: Reload from disk to get the actual saved facts
            # The in-memory graph might be out of sync if there were multiple processes
            # or if the graph was cleared on startup
            kb_load_knowledge_graph()
            
            facts_after = len(kb_graph)
            facts_extracted = facts_after - facts_before
            
            
            # CRITICAL: If facts_extracted is 0 but we processed files, check the result message
            # The result message from add_to_graph contains the actual count
            if facts_extracted == 0 and result:
                # Try to extract the actual count from the result message
                import re
                total_match = re.search(r'Total facts stored: (\d+)', result)
                if total_match:
                    total_facts = int(total_match.group(1))
                    facts_extracted = max(0, total_facts - facts_before)
                    print(f"⚠️  Adjusted facts_extracted from result message: {facts_extracted}")
            
            # Parse result message to extract added/skipped counts and extraction method
            import re
            added_match = re.search(r'Added (\d+) new triples', result)
            skipped_match = re.search(r'skipped (\d+) duplicates', result)
            added_count = int(added_match.group(1)) if added_match else facts_extracted
            skipped_count = int(skipped_match.group(1)) if skipped_match else 0
            
            # Extract extraction method from result
            extraction_method = "regex"
            if "TRIPLEX" in result.upper():
                extraction_method = "triplex"
            elif "FALLBACK" in result.upper():
                extraction_method = "regex (triplex fallback)"
            
            print(f"✅ Upload processed {len(files)} file(s)")
            print(f"   Extraction method: {extraction_method}")
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
            # Save document if ANY facts were extracted (even if some were duplicates)
            processed_docs = []
            if facts_extracted > 0:
                # Save document if we extracted facts (use facts_extracted, not added_count)
                # This ensures documents are saved even if all facts were duplicates
                for file_info in file_info_list:
                    doc = add_document(
                        name=file_info['name'],
                        size=file_info['size'],
                        file_type=file_info['type'],
                        facts_extracted=facts_extracted  # Use total extracted, not just added
                    )
                    if doc:  # Only append if document was saved (has facts > 0)
                        processed_docs.append(doc)
                        print(f"✅ Saved document {file_info['name']} with {facts_extracted} facts extracted")
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
            
            # Final save to ensure everything is persisted to disk
            kb_save_knowledge_graph()
            
            # CRITICAL: Reload one more time to ensure in-memory graph matches disk
            # This is necessary because the graph might have been cleared on startup
            kb_load_knowledge_graph()
            
            # Verify final state
            final_fact_count = len(kb_graph)
            if os.path.exists("knowledge_graph.pkl"):
                file_size = os.path.getsize("knowledge_graph.pkl")
                print(f"✅ Final save - file size: {file_size} bytes, facts in graph: {final_fact_count}")
                
            # Update total_facts in response to reflect actual graph state
            if final_fact_count > 0:
                print(f"✅ Upload complete: Graph now has {final_fact_count} facts in memory")
            
            print(f"✅ Upload processed {len(files)} file(s), added {added_count} new facts, skipped {skipped_count} duplicates")
            
            # Get final fact count after all saves and reloads
            final_total = len(kb_graph)
            
            return {
                "message": result,
                "files_processed": len(files),
                "status": "success",
                "total_facts": final_total,  # Use final count after reload
                "facts_extracted": added_count,  # Use actual added count
                "facts_skipped": skipped_count,   # Add skipped duplicates count
                "extraction_method": extraction_method,  # Indicate which method was used
                "documents": processed_docs
            }
        finally:
            # Clean up temporary files
            if tmp_paths:  # Only clean up if tmp_paths was initialized
                for tmp_path in tmp_paths:
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except Exception as cleanup_error:
                            print(f"⚠️  Warning: Could not delete temp file {tmp_path}: {cleanup_error}")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error uploading files: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error uploading files: {error_msg}")

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
        # CRITICAL: Always reload from disk to get the latest saved facts
        # This ensures we have facts that were saved after upload, even if server was restarted
        # The in-memory graph might be empty if server was restarted (cleared on startup)
        load_result = kb_load_knowledge_graph()
        
        # Debug: If graph is empty but file exists, something is wrong
        import os
        if len(kb_graph) == 0 and os.path.exists("knowledge_graph.pkl"):
            file_size = os.path.getsize("knowledge_graph.pkl")
            if file_size > 1000:  # File has data but graph is empty
                print(f"⚠️  WARNING: Graph file is {file_size} bytes but graph is empty!")
                # Try reloading again
                kb_load_knowledge_graph()
        
        facts = []
        from urllib.parse import unquote, quote
        import rdflib
        
        # OPTIMIZED: Build lookup maps in a single pass instead of calling functions for each fact
        # This reduces O(n*m) complexity to O(n) where n = total triples, m = facts
        # Build fact_id_uri -> metadata map first
        metadata_map = {}  # fact_id_uri -> {details, source_document, uploaded_at}
        
        # Pass 1: Collect all metadata triples (O(n))
        for s, p, o in kb_graph:
            predicate_str = str(p)
            
            if 'has_details' in predicate_str:
                fact_id_uri = str(s)
                if fact_id_uri not in metadata_map:
                    metadata_map[fact_id_uri] = {}
                metadata_map[fact_id_uri]['details'] = str(o)
            elif 'source_document' in predicate_str:
                fact_id_uri = str(s)
                if fact_id_uri not in metadata_map:
                    metadata_map[fact_id_uri] = {}
                metadata_map[fact_id_uri]['source_document'] = str(o)
            elif 'uploaded_at' in predicate_str:
                fact_id_uri = str(s)
                if fact_id_uri not in metadata_map:
                    metadata_map[fact_id_uri] = {}
                metadata_map[fact_id_uri]['uploaded_at'] = str(o)
        
        # Pass 2: Collect facts and match with metadata using fact_id URI (O(n))
        fact_index = 0
        for s, p, o in kb_graph:
            # Skip metadata triples
            predicate_str = str(p)
            if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
                'fact_object' in predicate_str or 'has_details' in predicate_str or 
                'source_document' in predicate_str or 'uploaded_at' in predicate_str):
                continue
            
            fact_index += 1
            
            # Extract subject from URI
            subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
            subject = unquote(subject).replace('_', ' ')
            
            # Extract predicate from URI
            predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
            predicate = unquote(predicate).replace('_', ' ')
            
            # Object is already a literal
            object_val = str(o)
            
            # Build fact_id URI the same way get_fact_details does (for lookup)
            fact_id = f"{subject}|{predicate}|{object_val}"
            fact_id_clean = fact_id.strip().replace(' ', '_')
            fact_id_uri = f"urn:fact:{quote(fact_id_clean, safe='')}"
            
            # Get metadata from lookup map (O(1) lookup)
            metadata = metadata_map.get(fact_id_uri, {})
            
            facts.append({
                "id": str(fact_index),
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "source": "knowledge_graph",
                "details": metadata.get('details') if metadata.get('details') else None,
                "sourceDocument": metadata.get('source_document') if metadata.get('source_document') else None,
                "uploadedAt": metadata.get('uploaded_at') if metadata.get('uploaded_at') else None
            })
        
        print(f"✅ GET /api/knowledge/facts: Returning {len(facts)} facts")
        if len(facts) > 0:
            print(f"   Sample fact: {facts[0]}")
        else:
            print("   ⚠️  No facts in graph!")
            # Debug: Check if file exists
            if os.path.exists("knowledge_graph.pkl"):
                file_size = os.path.getsize("knowledge_graph.pkl")
                print(f"   📁 knowledge_graph.pkl exists ({file_size} bytes) but graph is empty!")
                # If file exists but no facts, try to see what's in the graph
                all_triples = list(kb_graph)
                print(f"   📊 Total triples in graph: {len(all_triples)}")
                if len(all_triples) > 0:
                    print(f"   📊 Sample triple: {all_triples[0]}")
        
        # CRITICAL: Return facts in the format the frontend expects
        # Frontend expects: { success: true, data: { facts: [...] } }
        # But FastAPI returns directly, so we need to ensure the response has the right structure
        response = {
            "facts": facts,
            "total_facts": len(facts),  # Use len(facts) not len(kb_graph) since we filter metadata
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
        
        # Remove details first (if any)
        from knowledge import remove_fact_details as kb_remove_fact_details
        kb_remove_fact_details(subject, predicate, object_value)
        
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
        
        # Import get_fact_details function
        from knowledge import get_fact_details as kb_get_fact_details
        
        # Extract facts from graph
        facts = []
        # Import get_fact_source_document function
        from knowledge import get_fact_source_document as kb_get_fact_source_document
        
        for i, (s, p, o) in enumerate(kb_graph):
            # Skip metadata triples (those with special predicates for details, source document, timestamp)
            predicate_str = str(p)
            if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
                'fact_object' in predicate_str or 'has_details' in predicate_str or 
                'source_document' in predicate_str or 'uploaded_at' in predicate_str):
                continue
            
            # Extract subject from URI (urn:subject -> subject)
            subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
            # Decode URL encoding and replace underscores back to spaces
            subject = unquote(subject).replace('_', ' ')
            
            # Extract predicate from URI
            predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
            predicate = unquote(predicate).replace('_', ' ')
            
            # Object is already a literal, just get the string value
            object_val = str(o)
            
            # Get details for this fact
            details = kb_get_fact_details(subject, predicate, object_val)
            
            # Get source document and timestamp
            source_document, uploaded_at = kb_get_fact_source_document(subject, predicate, object_val)
            
            facts.append({
                "id": str(i + 1),
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "source": "knowledge_graph",
                "details": details if details else None,
                "sourceDocument": source_document if source_document else None,
                "uploadedAt": uploaded_at if uploaded_at else None
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

