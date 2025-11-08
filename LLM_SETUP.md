# LLM Setup for Research Assistant

The Research Assistant now uses a lightweight LLM (Microsoft Phi-3-mini) for intelligent question answering based on your knowledge graph.

## Features

- **Lightweight LLM**: Uses Microsoft Phi-3-mini-4k-instruct (3.8B parameters)
- **RAG (Retrieval Augmented Generation)**: Retrieves relevant facts from your knowledge graph and uses them as context
- **Automatic Fallback**: Falls back to rule-based responses if LLM is unavailable
- **Lazy Loading**: Model loads only when first needed

## Configuration

### Enable/Disable LLM

By default, the LLM is enabled. To disable it:

**Windows (PowerShell):**
```powershell
$env:USE_LLM="false"
```

**Windows (CMD):**
```cmd
set USE_LLM=false
```

**Linux/Mac:**
```bash
export USE_LLM=false
```

### Model Selection

The default model is `microsoft/Phi-3-mini-4k-instruct`. This is a small, fast model that works well for RAG tasks.

## How It Works

1. **Question Processing**: When you ask a question, the system:
   - Extracts keywords from your question
   - Searches the knowledge graph for relevant facts
   - Retrieves up to 15 most relevant facts with their details

2. **Context Building**: The retrieved facts are formatted as context for the LLM

3. **Response Generation**: The LLM generates a natural language response based on:
   - The retrieved facts from your knowledge base
   - The question you asked
   - A system prompt that ensures factual, grounded responses

4. **Fallback**: If the LLM is unavailable or fails, the system uses rule-based pattern matching

## Performance

- **First Request**: May take 10-30 seconds to download and load the model (only once)
- **Subsequent Requests**: Typically 2-5 seconds per response
- **Memory Usage**: ~4-6 GB RAM (CPU) or ~2-3 GB VRAM (GPU)

## Troubleshooting

### Model Download Issues

If the model fails to download:
- Check your internet connection
- The model is ~7.6 GB, ensure you have enough disk space
- Model is cached in `~/.cache/huggingface/hub/`

### Memory Issues

If you run out of memory:
- Disable LLM: `set USE_LLM=false`
- Use CPU mode (slower but uses less memory)
- Consider using a smaller model

### Slow Responses

- First request is slow (model loading)
- CPU mode is slower than GPU mode
- Consider disabling LLM if speed is critical

## Frontend Integration

The frontend chat interface (`chat.tsx`) now:
- Calls the actual API endpoint `/api/chat`
- Displays real responses from the backend
- Shows loading states during processing
- Handles errors gracefully

## API Endpoint

The chat endpoint is available at:
```
POST /api/chat
Body: {
  "message": "Your question here",
  "history": []  // Optional conversation history
}
```

Response:
```json
{
  "response": "Generated answer...",
  "status": "success"
}
```

