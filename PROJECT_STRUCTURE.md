# Research Brain - Project Structure Documentation

## 📁 Overview

Research Brain is a full-stack knowledge management system that extracts, stores, and visualizes knowledge from research documents. It consists of:

- **Frontend**: React/TypeScript application (`RandDKnowledgeGraph/client/`)
- **Backend**: FastAPI Python server (`api_server.py` and related modules)
- **Data Storage**: RDF knowledge graph (`knowledge_graph.pkl`) and document metadata (`documents_store.json`)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (User Interface)                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  React Frontend (RandDKnowledgeGraph/client/)        │  │
│  │  - Pages (upload, knowledge-base, graph, chat)       │  │
│  │  - Components (tables, dialogs, visualizations)     │  │
│  │  - State Management (knowledge-store.tsx)            │  │
│  │  - API Client (api-client.ts)                        │  │
│  └───────────────────┬──────────────────────────────────┘  │
└──────────────────────┼──────────────────────────────────────┘
                       │ HTTP REST API
                       │ (JSON requests/responses)
┌──────────────────────┼──────────────────────────────────────┐
│  ┌───────────────────▼──────────────────────────────────┐  │
│  │  FastAPI Backend (api_server.py)                     │  │
│  │  - API Endpoints (/api/knowledge/*, /api/documents)  │  │
│  │  - Request/Response handling                          │  │
│  └───────────┬───────────────────────┬──────────────────┘  │
│              │                       │                      │
│  ┌───────────▼──────────┐  ┌─────────▼──────────┐          │
│  │  knowledge.py        │  │  file_processing.py│          │
│  │  - RDF Graph         │  │  - Text extraction │          │
│  │  - Fact management   │  │  - PDF/DOCX/TXT    │          │
│  │  - Duplicate check   │  │  - CSV processing  │          │
│  └───────────┬──────────┘  └─────────┬──────────┘          │
│              │                       │                      │
│  ┌───────────▼───────────────────────▼──────────┐          │
│  │  documents_store.py                           │          │
│  │  - Document metadata                          │          │
│  │  - Facts count tracking                       │          │
│  └───────────┬───────────────────────────────────┘          │
└──────────────┼──────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │  Persistent Storage │
    │  - knowledge_graph.pkl (RDF graph)                    │
    │  - documents_store.json (metadata)                    │
    └─────────────────────┘
```

---

## 🎨 Frontend Structure (`RandDKnowledgeGraph/client/`)

### Directory Structure

```
client/
├── public/                    # Static assets
│   ├── favicon.png          # Browser tab icon (logo_G.jpeg)
│   ├── logo_G.jpeg          # Small logo (sidebar icon)
│   └── logo_GNOSES.jpeg     # Full logo (footer)
│
├── src/
│   ├── pages/                # Top-level page components
│   │   ├── home.tsx         # Landing page
│   │   ├── upload.tsx       # Document upload page
│   │   ├── knowledge-base.tsx  # Facts table view
│   │   ├── graph.tsx         # Knowledge graph visualization
│   │   ├── chat.tsx          # AI chat interface
│   │   ├── documents.tsx     # Document list
│   │   └── import-export.tsx # Import/export functionality
│   │
│   ├── components/            # Reusable UI components
│   │   ├── AppSidebar.tsx   # Main navigation sidebar
│   │   ├── KnowledgeBaseTable.tsx  # Facts table
│   │   ├── KnowledgeGraphVisualization.tsx  # Graph visualization
│   │   ├── FactEditDialog.tsx      # Edit fact dialog
│   │   ├── AddFactDialog.tsx       # Add fact dialog
│   │   └── ... (other UI components)
│   │
│   ├── lib/                  # Core logic and utilities
│   │   ├── api-client.ts    # ⭐ API communication layer
│   │   ├── knowledge-store.tsx  # ⭐ Global state management
│   │   ├── queryClient.ts   # React Query configuration
│   │   └── utils.ts         # Utility functions
│   │
│   ├── hooks/                # Custom React hooks
│   │   ├── use-toast.ts     # Toast notifications
│   │   └── use-mobile.tsx   # Mobile detection
│   │
│   ├── App.tsx              # Main app component (routing)
│   ├── main.tsx             # React entry point
│   └── index.css            # Global styles
│
├── index.html               # HTML template
├── package.json             # Dependencies and scripts
├── tsconfig.json           # TypeScript configuration
└── vite.config.ts          # Vite build configuration
```

### Key Frontend Files

#### 1. `src/lib/api-client.ts` - API Communication Layer
**Purpose**: Handles all HTTP communication with the backend.

**Key Functions**:
- `uploadDocuments(files)`: Uploads documents to `/api/knowledge/upload`
- `getFacts()`: Fetches all facts from `/api/knowledge/facts`
- `createFact(factData)`: Creates a fact via `/api/knowledge/facts`
- `updateFact(id, updates)`: Updates a fact (currently delete+create)
- `deleteFact(id, factData)`: Deletes a fact via `/api/knowledge/facts/{id}`
- `getDocuments()`: Fetches documents from `/api/documents`
- `exportKnowledgeBase()`: Exports all facts from `/api/export`
- `importKnowledgeBase(file)`: Imports facts via `/api/knowledge/import`

**Connection**: Uses `BASE_URL` (default: `http://localhost:8001`) to connect to FastAPI backend.

#### 2. `src/lib/knowledge-store.tsx` - Global State Management
**Purpose**: Manages application-wide state (facts, nodes, edges) and provides functions to manipulate them.

**Key Functions**:
- `refreshFacts()`: Fetches facts from backend and rebuilds graph
- `addFact(factData)`: Adds a fact (calls API, updates state)
- `updateFact(id, updates)`: Updates a fact (delete old + create new)
- `deleteFact(id)`: Deletes a fact (calls API, updates state)
- `addEdge(sourceId, targetId, label)`: Creates a connection (saves as fact)
- `deleteEdge(id)`: Deletes a connection (deletes corresponding fact)
- `updateEdge(id, updates)`: Updates a connection (updates fact)
- `rebuildGraphFromFacts(facts)`: Rebuilds graph visualization from facts

**State**:
- `facts`: Array of all facts
- `nodes`: Graph nodes (derived from facts)
- `edges`: Graph edges (derived from facts)

#### 3. `src/pages/upload.tsx` - Document Upload
**Flow**:
1. User selects files
2. Calls `hfApi.uploadDocuments(files)`
3. Backend processes files and extracts facts
4. Calls `refreshFacts()` to update UI

#### 4. `src/pages/knowledge-base.tsx` - Facts Management
**Flow**:
1. Displays facts in a table
2. User can edit/delete facts
3. Changes sync to backend and graph

#### 5. `src/pages/graph.tsx` - Graph Visualization
**Flow**:
1. Displays interactive 3D graph
2. User can create/delete/edit connections
3. Changes saved as facts in backend

---

## 🐍 Backend Structure (Python Files)

### Core Files

```
xNeSy2/
├── api_server.py          # ⭐ Main FastAPI server (API endpoints)
├── knowledge.py           # ⭐ Knowledge graph management (RDF)
├── file_processing.py     # ⭐ Document text extraction
├── documents_store.py     # ⭐ Document metadata management
├── responses.py           # Chat/AI response generation
├── requirements.txt       # Python dependencies
│
├── knowledge_graph.pkl    # Persistent RDF graph storage
├── documents_store.json   # Persistent document metadata
└── knowledge_backup.json  # Backup of knowledge graph
```

### 1. `api_server.py` - FastAPI Server (Main Backend Entry Point)

**Purpose**: Defines all HTTP API endpoints that the frontend calls.

**Key Endpoints**:

#### Document Upload & Processing
- `POST /api/knowledge/upload`
  - **Function**: `upload_file_endpoint(files)`
  - **Flow**:
    1. Receives uploaded files
    2. Calls `file_processing.handle_file_upload()` to extract text
    3. Calls `knowledge.add_to_graph()` to extract facts from text
    4. Saves document metadata via `documents_store.add_document()`
    5. Returns facts extracted count
  - **Location**: Lines ~220-320

#### Fact Management
- `POST /api/knowledge/facts`
  - **Function**: `create_fact_endpoint(request)`
  - **Flow**:
    1. Checks for duplicates via `knowledge.fact_exists()`
    2. Adds fact to RDF graph
    3. Saves graph to disk
  - **Location**: Lines ~163-220

- `DELETE /api/knowledge/facts/{fact_id}`
  - **Function**: `delete_fact_endpoint(fact_id)`
  - **Flow**:
    1. Parses fact_id (JSON with subject/predicate/object)
    2. Removes fact from RDF graph
    3. Saves graph to disk
  - **Location**: Lines ~500-568

- `GET /api/knowledge/facts`
  - **Function**: `get_facts_endpoint()`
  - **Flow**:
    1. Loads graph from disk
    2. Converts RDF triples to JSON facts
    3. Returns facts array
  - **Location**: Lines ~380-430

#### Document Management
- `GET /api/documents`
  - **Function**: `get_documents_endpoint()`
  - **Flow**:
    1. Cleans up documents without facts
    2. Returns document metadata
  - **Location**: Lines ~432-470

#### Import/Export
- `GET /api/export`
  - **Function**: `export_knowledge_endpoint()`
  - **Flow**:
    1. Loads graph from disk
    2. Converts to JSON format
    3. Returns facts with metadata
  - **Location**: Lines ~570-610

- `POST /api/knowledge/import`
  - **Function**: `import_json_endpoint(file)`
  - **Flow**:
    1. Reads JSON file
    2. Calls `knowledge.import_knowledge_from_json_file()`
    3. Returns import results
  - **Location**: Lines ~612-650

### 2. `knowledge.py` - Knowledge Graph Core (⭐ CRITICAL FOR KNOWLEDGE EXTRACTION)

**Purpose**: Manages the RDF knowledge graph using RDFLib. This is where knowledge extraction happens.

**Key Functions**:

#### Graph Management
- `load_knowledge_graph()`
  - **Purpose**: Loads graph from `knowledge_graph.pkl`
  - **Location**: Lines ~80-120
  - **Returns**: Status message

- `save_knowledge_graph()`
  - **Purpose**: Saves graph to `knowledge_graph.pkl`
  - **Location**: Lines ~122-140
  - **Returns**: Status message

#### ⭐ Knowledge Extraction (THIS IS WHERE YOU'LL IMPROVE)
- `add_to_graph(text)`
  - **Purpose**: **Extracts facts from text and adds them to the graph**
  - **Location**: Lines ~200-350
  - **Current Implementation**:
    1. Uses NLP to extract subject-predicate-object triples from text
    2. Creates RDF triples (subject, predicate, object)
    3. Checks for duplicates via `fact_exists()`
    4. Adds new facts to graph
    5. Saves graph to disk
  - **Returns**: Message with added/skipped counts
  - **⚠️ THIS IS WHERE YOU'LL IMPROVE THE EXTRACTION LOGIC**

- `fact_exists(subject, predicate, object_val)`
  - **Purpose**: Checks if a fact already exists (case-insensitive)
  - **Location**: Lines ~350-400
  - **Returns**: Boolean

#### Fact Management
- `get_all_facts()`
  - **Purpose**: Returns all facts as list of triples
  - **Location**: Lines ~400-450

- `delete_knowledge_by_keyword(keyword)`
  - **Purpose**: Deletes facts containing a keyword
  - **Location**: Lines ~599-617

- `import_knowledge_from_json_file(file)`
  - **Purpose**: Imports facts from JSON file
  - **Location**: Lines ~450-550
  - **Flow**:
    1. Reads JSON file
    2. For each fact, checks duplicates
    3. Adds to graph
    4. Saves graph

**Global Variable**:
- `graph`: RDFLib Graph object (in-memory representation)

### 3. `file_processing.py` - Document Text Extraction

**Purpose**: Extracts raw text from various file formats.

**Key Functions**:

- `process_uploaded_file(file)`
  - **Purpose**: Extracts text from a single file
  - **Location**: Lines ~50-150
  - **Supported Formats**:
    - PDF: Uses `PyPDF2` or `pdfplumber`
    - DOCX: Uses `python-docx`
    - TXT: Direct read
    - CSV: Uses `pandas`
    - PPTX: Uses `python-pptx`
  - **Returns**: Extracted text string

- `handle_file_upload(file_paths)`
  - **Purpose**: Processes multiple files
  - **Location**: Lines ~150-250
  - **Flow**:
    1. Processes each file
    2. Combines text
    3. Calls `knowledge.add_to_graph()` to extract facts
  - **Returns**: Status message

### 4. `documents_store.py` - Document Metadata

**Purpose**: Manages metadata about uploaded documents.

**Key Functions**:

- `add_document(name, size, file_type, facts_extracted)`
  - **Purpose**: Adds document metadata
  - **Location**: Lines ~50-100
  - **Storage**: `documents_store.json`
  - **Note**: Only saves documents with `facts_extracted > 0`

- `get_all_documents()`
  - **Purpose**: Returns all document metadata
  - **Location**: Lines ~100-150

- `cleanup_documents_without_facts()`
  - **Purpose**: Removes documents that didn't contribute facts
  - **Location**: Lines ~150-200

### 5. `responses.py` - AI Chat Responses

**Purpose**: Generates AI responses using the knowledge graph.

**Key Functions**:

- `respond(message, history)`
  - **Purpose**: Generates AI response based on knowledge graph
  - **Location**: Lines ~50-200
  - **Flow**:
    1. Queries knowledge graph for relevant facts
    2. Uses LLM to generate response
    3. Returns response

---

## 🔄 Data Flow: Complete Example (Document Upload)

### Step-by-Step Flow

1. **User Action**: User selects files and clicks "Process Documents"
   - **Location**: `client/src/pages/upload.tsx`

2. **Frontend**: Calls `hfApi.uploadDocuments(files)`
   - **Location**: `client/src/lib/api-client.ts`
   - **Action**: Sends HTTP POST to `http://localhost:8001/api/knowledge/upload`

3. **Backend Receives**: `api_server.py` → `upload_file_endpoint(files)`
   - **Location**: `api_server.py` lines ~220-320
   - **Action**: Saves files temporarily

4. **Text Extraction**: Calls `file_processing.handle_file_upload(tmp_paths)`
   - **Location**: `file_processing.py` lines ~150-250
   - **Action**: Extracts text from PDF/DOCX/TXT/CSV files
   - **Returns**: Combined text string

5. **⭐ Knowledge Extraction**: Calls `knowledge.add_to_graph(text)`
   - **Location**: `knowledge.py` lines ~200-350
   - **Action**: 
     - Uses NLP to extract subject-predicate-object triples
     - Checks for duplicates
     - Adds new facts to RDF graph
     - Saves graph to `knowledge_graph.pkl`
   - **Returns**: Message with counts

6. **Document Metadata**: Calls `documents_store.add_document(...)`
   - **Location**: `documents_store.py` lines ~50-100
   - **Action**: Saves document metadata to `documents_store.json`

7. **Response**: Backend returns JSON with facts extracted count
   - **Location**: `api_server.py` lines ~300-320

8. **Frontend Update**: Calls `refreshFacts()`
   - **Location**: `client/src/lib/knowledge-store.tsx`
   - **Action**: 
     - Fetches all facts from `/api/knowledge/facts`
     - Rebuilds graph visualization
     - Updates UI

---

## 🎯 Where to Improve Knowledge Extraction

### Primary Location: `knowledge.py` → `add_to_graph(text)`

**Current Implementation** (Lines ~200-350):
- Takes raw text as input
- Uses NLP to extract triples
- Adds to RDF graph

**To Improve**:
1. **Better NLP Models**: Replace/upgrade the extraction model
2. **Custom Rules**: Add domain-specific extraction rules
3. **Entity Recognition**: Improve entity detection
4. **Relationship Detection**: Better predicate extraction
5. **Context Awareness**: Use document context for better extraction

**Related Functions**:
- `fact_exists()`: Duplicate detection (Lines ~350-400)
- `import_knowledge_from_json_file()`: Import logic (Lines ~450-550)

### Secondary Location: `file_processing.py` → `process_uploaded_file()`

**To Improve**:
- Better text extraction from PDFs (preserve structure)
- Handle tables, figures, citations
- Extract metadata (title, authors, dates)

---

## 🔌 Frontend-Backend Connection Details

### API Base URL
- **Default**: `http://localhost:8001`
- **Location**: `client/src/lib/api-client.ts` line ~20
- **Configurable**: Can be changed via environment variable

### Communication Protocol
- **Method**: HTTP REST API
- **Format**: JSON (request and response)
- **CORS**: Enabled for all origins (configured in `api_server.py`)

### Request Flow
```
Frontend Component
    ↓
knowledge-store.tsx (state management)
    ↓
api-client.ts (HTTP request)
    ↓
FastAPI Endpoint (api_server.py)
    ↓
Business Logic (knowledge.py, file_processing.py, etc.)
    ↓
Persistent Storage (knowledge_graph.pkl, documents_store.json)
    ↓
Response (JSON)
    ↓
api-client.ts (parse response)
    ↓
knowledge-store.tsx (update state)
    ↓
React Component (re-render UI)
```

---

## 📝 Data Storage

### `knowledge_graph.pkl`
- **Format**: Pickled RDFLib Graph object
- **Content**: All facts as RDF triples (subject, predicate, object)
- **Location**: Root directory
- **Backup**: `knowledge_backup.json` (JSON export)

### `documents_store.json`
- **Format**: JSON array
- **Content**: Document metadata (name, size, type, facts_extracted)
- **Location**: Root directory
- **Structure**:
```json
[
  {
    "id": "doc_1",
    "name": "research_paper.pdf",
    "size": 1024000,
    "type": "pdf",
    "facts_extracted": 45,
    "uploaded_at": "2025-01-15T10:30:00"
  }
]
```

---

## 🚀 Running the Application

### Backend
```bash
python api_server.py
```
- Starts FastAPI server on `http://localhost:8001`
- API docs available at `http://localhost:8001/docs`

### Frontend
```bash
cd RandDKnowledgeGraph
npm install
npm run dev
```
- Starts Vite dev server (usually `http://localhost:5173`)
- Connects to backend at `http://localhost:8001`

---

## 📚 Key Dependencies

### Backend (Python)
- `fastapi`: Web framework
- `rdflib`: RDF graph management
- `PyPDF2` / `pdfplumber`: PDF processing
- `python-docx`: DOCX processing
- `pandas`: CSV processing

### Frontend (TypeScript/React)
- `react`: UI framework
- `wouter`: Routing
- `@tanstack/react-query`: Data fetching
- `lucide-react`: Icons
- `vite`: Build tool

---

## 🔍 Quick Reference: Finding Functions

| What You Want to Do | File | Function | Lines |
|---------------------|------|----------|-------|
| Extract facts from text | `knowledge.py` | `add_to_graph(text)` | ~200-350 |
| Check if fact exists | `knowledge.py` | `fact_exists()` | ~350-400 |
| Upload documents | `api_server.py` | `upload_file_endpoint()` | ~220-320 |
| Extract text from PDF | `file_processing.py` | `process_uploaded_file()` | ~50-150 |
| Add fact manually | `api_server.py` | `create_fact_endpoint()` | ~163-220 |
| Delete fact | `api_server.py` | `delete_fact_endpoint()` | ~500-568 |
| Get all facts | `api_server.py` | `get_facts_endpoint()` | ~380-430 |
| Frontend API calls | `client/src/lib/api-client.ts` | Various functions | Throughout |
| State management | `client/src/lib/knowledge-store.tsx` | Various functions | Throughout |

---

## 🎓 Next Steps for Improvement

1. **Knowledge Extraction**: Focus on `knowledge.py` → `add_to_graph()`
   - Improve NLP model
   - Add custom extraction rules
   - Better entity recognition

2. **Text Extraction**: Focus on `file_processing.py`
   - Better PDF structure preservation
   - Table extraction
   - Metadata extraction

3. **UI/UX**: Focus on `client/src/pages/` and `client/src/components/`
   - Better visualizations
   - Improved user workflows

4. **Performance**: Optimize graph operations in `knowledge.py`
   - Faster duplicate checking
   - Batch operations
   - Caching

---

**Last Updated**: 2025-01-15
**Version**: 1.0

