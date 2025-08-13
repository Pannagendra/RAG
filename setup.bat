@echo off
echo ================================
echo    RAG Demo Setup Script
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo ✓ Python found
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv rag_demo
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call rag_demo\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install packages
echo Installing required packages...
echo.

echo Installing core ML packages...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install huggingface_hub==0.20.3
pip install sentence-transformers==2.7.0

echo Installing vector database...
pip install chromadb==0.4.22

echo Installing PDF processing...
pip install PyPDF2==3.0.1
pip install pymupdf==1.23.14

echo Installing LangChain components...
pip install langchain==0.1.5
pip install langchain-community==0.0.20
pip install langchain-chroma==0.1.0

echo Installing Ollama client...
pip install ollama==0.1.7

echo Installing web interface...
pip install streamlit==1.31.0

echo Installing utilities...
pip install numpy==1.24.3
pip install pandas==2.2.0
pip install python-dotenv==1.0.0

echo.
echo ================================
echo    Setup Complete!
echo ================================
echo.
echo Next steps:
echo 1. Install Ollama from: https://ollama.ai/download/windows
echo 2. Run: ollama pull llama3.1:8b
echo 3. Save the Python code as 'rag_demo.py'
echo 4. Run: streamlit run rag_demo.py
echo.
echo Your RAG demo will be available at: http://localhost:8501
echo.


@echo off
echo ================================
echo    Fixing RAG Demo Dependencies
echo ================================
echo.

REM Activate virtual environment
call rag_demo\Scripts\activate.bat

echo Uninstalling problematic packages...
pip uninstall sentence-transformers huggingface_hub -y

echo Installing compatible versions...
pip install huggingface_hub==0.20.3
pip install transformers==4.36.2
pip install sentence-transformers==2.7.0
pip install tokenizers==0.15.0

echo Reinstalling other packages...
pip install --upgrade chromadb==0.4.22
pip install --upgrade ollama==0.1.7

echo.
echo ================================
echo    Dependencies Fixed!
echo ================================
echo.
echo Now run: streamlit run rag_demo.py
pause