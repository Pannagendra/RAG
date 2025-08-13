# 🔧 Detailed Setup Guide

This guide provides step-by-step instructions for setting up the RAG Demo system on Windows.

## 📋 Prerequisites Check

### System Requirements
- **OS**: Windows 10/11 (64-bit)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB free space
- **CPU**: Intel i5/AMD Ryzen 5 or better

### Software Requirements
- **Python 3.11+**: [Download from python.org](https://python.org)
- **Git**: [Download from git-scm.com](https://git-scm.com)
- **Visual Studio Code** (Optional): [Download from code.visualstudio.com](https://code.visualstudio.com)

## 🚀 Installation Methods

### Method 1: Automated Setup (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/rag-demo.git
   cd rag-demo
   ```

2. **Run setup script as Administrator**:
   ```bash
   setup.bat
   ```

3. **Install Ollama**:
   - Download from https://ollama.ai/download/windows
   - Run installer
   - Restart your computer if prompted

4. **Download language model**:
   ```bash
   ollama pull llama3.1:8b
   ```

### Method 2: Manual Setup

1. **Clone repository**:
   ```bash
   git clone https://github.com/yourusername/rag-demo.git
   cd rag-demo
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv rag_demo
   rag_demo\Scripts\activate
   ```

3. **Upgrade pip**:
   ```bash
   python -m pip install --upgrade pip
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Fix potential conflicts** (if needed):
   ```bash
   fix_dependencies.bat
   ```

## 🔧 Ollama Setup

### Installation Steps

1. **Download Ollama**:
   - Visit: https://ollama.ai/download/windows
   - Download the Windows installer
   - Run as Administrator

2. **Verify Installation**:
   ```bash
   ollama --version
   ```

3. **Start Ollama Service**:
   ```bash
   ollama serve
   ```

### Model Selection

Choose based on your system capabilities:

#### For 16GB RAM Systems:
```bash
# Recommended: Good balance of speed and quality
ollama pull llama3.1:8b

# Alternative: Faster but lower quality
ollama pull phi3:mini
```

#### For 32GB+ RAM Systems:
```bash
# Best quality (requires more resources)
ollama pull llama3.1:70b
```

#### For Quick Testing:
```bash
# Smallest, fastest model
ollama pull tinyllama
```

### Model Testing

Test your installation:
```bash
ollama run llama3.1:8b "Hello, how are you today?"
```

## 🗂️ Project Structure Creation

Run this script to create the complete folder structure:

```bash
@echo off
mkdir src
mkdir src\components
mkdir src\utils
mkdir tests
mkdir docs
mkdir sample_data
mkdir chroma_db

echo. > src\__init__.py
echo. > src\components\__init__.py
echo. > src\utils\__init__.py
echo. > tests\__init__.py
echo. > chroma_db\.gitkeep

echo Project structure created successfully!
```

## 📁 File Placement

Place these files in your project directory:

### Root Directory:
- `README.md`
- `requirements.txt`
- `setup.bat`
- `fix_dependencies.bat`
- `.gitignore`
- `LICENSE`

### src/ Directory:
- `rag_demo.py` (main application)
- `components/` (modular components)
- `utils/` (utility functions)

### docs/ Directory:
- `SETUP_GUIDE.md`
- `API_REFERENCE.md`
- `TROUBLESHOOTING.md`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Ollama Configuration
OLLAMA_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=60

# Vector Database
VECTOR_DB_PATH=./chroma_db
COLLECTION_NAME=rag_documents

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Text Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_SOURCES=5

# Performance
BATCH_SIZE=100
EMBEDDING_BATCH_SIZE=32

# UI Configuration
PAGE_TITLE=RAG Demo - High Performance Q&A
PAGE_ICON=🚀
LAYOUT=wide

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/rag_demo.log
```

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[global]
developmentMode = false

[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 🧪 Testing Installation

### Basic Tests

1. **Test Python Environment**:
   ```bash
   python --version
   pip list | findstr streamlit
   ```

2. **Test Ollama Connection**:
   ```bash
   ollama list
   ollama ps
   ```

3. **Test Application Startup**:
   ```bash
   streamlit run src/rag_demo.py --server.headless true
   ```

### Component Tests

```bash
# Run all tests
pytest tests/ -v

# Test specific components
pytest tests/test_pdf_parser.py -v
pytest tests/test_vector_db.py -v
```

## 🔍 Performance Optimization

### System Optimizations

1. **Increase Virtual Memory**:
   - Control Panel → System → Advanced → Performance Settings
   - Set virtual memory to 2x your RAM size

2. **Disable Windows Defender for Project Folder**:
   - Add your project folder to exclusions
   - This prevents scanning delays during file operations

3. **Close Unnecessary Applications**:
   - Free up RAM for the demo
   - Disable startup programs

### Application Optimizations

1. **Adjust Chunk Size**:
   ```python
   # In .env file
   CHUNK_SIZE=800  # Smaller for faster processing
   CHUNK_OVERLAP=150
   ```

2. **Reduce Model Size**:
   ```bash
   # Switch to smaller model if needed
   ollama pull phi3:mini
   ```

3. **Limit Source Count**:
   ```python
   # In .env file
   MAX_SOURCES=3  # Fewer sources for faster responses
   ```

## 🚨 Common Issues & Solutions

### Python Issues

**Issue**: `python` command not found
```bash
# Solution: Add Python to PATH or use full path
C:\Users\[Username]\AppData\Local\Programs\Python\Python311\python.exe
```

**Issue**: Permission denied errors
```bash
# Solution: Run Command Prompt as Administrator
# Right-click Command Prompt → "Run as administrator"
```

### Package Installation Issues

**Issue**: `pip install` fails with network errors
```bash
# Solution: Use alternative index
pip install --index-url https://pypi.org/simple/ package-name
```

**Issue**: Conflicting package versions
```bash
# Solution: Use the fix script
fix_dependencies.bat
```

### Ollama Issues

**Issue**: "Ollama not found"
```bash
# Solution: Verify PATH and restart terminal
echo %PATH%
# Should contain Ollama installation directory
```

**Issue**: Model download fails
```bash
# Solution: Manual download with resume
ollama pull llama3.1:8b --insecure
```

## 📊 Monitoring & Logging

### Performance Monitoring

Add this to your startup routine:

```bash
# Monitor system resources
wmic cpu get loadpercentage /value
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value
```

### Application Logs

Check these locations for logs:
- `logs/rag_demo.log` (application logs)
- `chroma_db/chroma.log` (vector database logs)
- Windows Event Viewer (system issues)

## 🔄 Updates & Maintenance

### Regular Updates

```bash
# Update Python packages
pip list --outdated
pip install --upgrade package-name

# Update Ollama
ollama pull llama3.1:8b  # Re-download latest version
```

### Database Maintenance

```bash
# Clear vector database (if needed)
rmdir /s chroma_db
mkdir chroma_db
echo. > chroma_db\.gitkeep
```

## 🎯 Next Steps

After successful setup:

1. **Upload test documents** to verify functionality
2. **Run performance benchmarks** with your data
3. **Customize the interface** for your use case
4. **Set up monitoring** for production use
5. **Plan scaling strategy** for larger datasets

---

**Need help?** Check our [Troubleshooting Guide](TROUBLESHOOTING.md) or open an issue on GitHub.