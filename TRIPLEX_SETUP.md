# Triplex Integration Guide

## Overview

Triplex is an optional LLM-based knowledge extraction model that provides higher quality extraction compared to regex-based methods. It's a 4B parameter model fine-tuned specifically for knowledge graph construction.

## Installation

The required packages are already in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

This will install:
- `transformers>=4.30.0` - Hugging Face transformers library
- `torch>=2.0.0` - PyTorch for model execution
- `accelerate>=0.20.0` - For efficient model loading

## Enabling Triplex

### Option 1: Using Helper Scripts (Recommended)

**Windows (CMD):**
```cmd
enable_triplex.bat
python api_server.py
```

**Windows (PowerShell):**
```powershell
.\enable_triplex.ps1
python api_server.py
```

### Option 2: Manual Environment Variable

**Windows (CMD):**
```cmd
set USE_TRIPLEX=true
python api_server.py
```

**Windows (PowerShell):**
```powershell
$env:USE_TRIPLEX="true"
python api_server.py
```

**Linux/Mac:**
```bash
export USE_TRIPLEX=true
python api_server.py
```

### Option 3: Permanent Setup

Add to your system environment variables:
- Variable: `USE_TRIPLEX`
- Value: `true`

## How It Works

1. **First Use**: When Triplex is enabled and used for the first time, it will:
   - Download the model from Hugging Face (~4GB)
   - Load it into memory
   - This may take 5-10 minutes depending on your internet connection

2. **Subsequent Uses**: The model is cached locally, so subsequent uses are faster (but still slower than regex)

3. **Fallback**: If Triplex fails or is disabled, the system automatically falls back to regex-based extraction

## Performance Considerations

- **CPU Mode**: Works on CPU but is slower (may take 10-30 seconds per extraction)
- **GPU Mode**: Much faster if CUDA is available (2-5 seconds per extraction)
- **Memory**: Requires ~8GB RAM for the model
- **First Load**: Model loading takes 30-60 seconds on first use

## Disabling Triplex

Simply don't set the `USE_TRIPLEX` environment variable, or set it to `false`:

```cmd
set USE_TRIPLEX=false
```

## Troubleshooting

### Installation Errors

If you see errors during installation:

1. **Python Version**: Ensure you're using Python 3.8+ (3.14.0 is fine)
2. **pip Update**: Update pip first: `python -m pip install --upgrade pip`
3. **Separate Installation**: Try installing packages separately:
   ```bash
   pip install torch
   pip install transformers
   pip install accelerate
   ```

### Model Download Issues

If the model fails to download:

1. Check internet connection
2. Try manually downloading: The model is at `https://huggingface.co/sciphi/triplex`
3. Check disk space (needs ~4GB free)

### Out of Memory Errors

If you get memory errors:

1. The model requires ~8GB RAM
2. Close other applications
3. Consider using CPU mode (slower but uses less memory)
4. Disable Triplex and use regex-based extraction instead

### Slow Performance

If Triplex is too slow:

1. The system automatically falls back to regex if Triplex fails
2. You can disable Triplex: `set USE_TRIPLEX=false`
3. Regex-based extraction is much faster (milliseconds vs seconds)

## Verification

To check if Triplex is available and enabled:

```python
from knowledge import TRIPLEX_AVAILABLE, USE_TRIPLEX
print("Triplex available:", TRIPLEX_AVAILABLE)
print("Triplex enabled:", USE_TRIPLEX)
```

## Notes

- Triplex is **optional** - the system works perfectly fine without it
- Regex-based extraction is faster and uses less resources
- Triplex provides better quality extraction for complex texts
- The choice between speed and quality is yours!

