"""
PDF Parser Module for RAG Demo
Handles extraction of text from PDF documents with multiple parsing strategies.
"""

import streamlit as st
import PyPDF2
import fitz  # PyMuPDF
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFParser:
    """Enhanced PDF parser with multiple extraction methods"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF parser with configurable chunking.
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def extract_text_pypdf2(self, pdf_file) -> str:
        """
        Extract text using PyPDF2.
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            Extracted text as string
        """
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    st.warning(f"Failed to extract page {page_num + 1}: {e}")
                    continue
                    
            return text
        except Exception as e:
            st.error(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def extract_text_pymupdf(self, pdf_file) -> str:
        """
        Extract text using PyMuPDF (better for complex PDFs).
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            Extracted text as string
        """
        try:
            pdf_file.seek(0)  # Reset file pointer
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                        
                except Exception as e:
                    st.warning(f"Failed to extract page {page_num + 1}: {e}")
                    continue
                    
            doc.close()
            return text
        except Exception as e:
            st.error(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def extract_metadata(self, pdf_file) -> dict:
        """
        Extract metadata from PDF.
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {
            "title": "Unknown",
            "author": "Unknown", 
            "pages": 0,
            "file_size": 0
        }
        
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            if pdf_reader.metadata:
                metadata.update({
                    "title": pdf_reader.metadata.get("/Title", "Unknown"),
                    "author": pdf_reader.metadata.get("/Author", "Unknown"),
                    "subject": pdf_reader.metadata.get("/Subject", ""),
                    "creator": pdf_reader.metadata.get("/Creator", "")
                })
            
            metadata["pages"] = len(pdf_reader.pages)
            
            # Get file size
            pdf_file.seek(0, 2)  # Seek to end
            metadata["file_size"] = pdf_file.tell()
            pdf_file.seek(0)  # Reset
            
        except Exception as e:
            st.warning(f"Could not extract metadata: {e}")
        
        return metadata
    
    def validate_pdf(self, pdf_file) -> bool:
        """
        Validate if the file is a proper PDF.
        
        Args:
            pdf_file: Uploaded file object
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            pdf_file.seek(0)
            header = pdf_file.read(4)
            pdf_file.seek(0)
            
            return header == b'%PDF'
        except Exception:
            return False
    
    def parse_pdf(self, pdf_file, filename: str = "unknown.pdf") -> List[str]:
        """
        Parse PDF and return text chunks with fallback strategy.
        
        Args:
            pdf_file: Uploaded PDF file object
            filename: Name of the file for logging
            
        Returns:
            List of text chunks
        """
        if not self.validate_pdf(pdf_file):
            raise ValueError(f"Invalid PDF file: {filename}")
        
        # Extract metadata for logging
        metadata = self.extract_metadata(pdf_file)
        st.info(f"📄 Processing: {filename} ({metadata['pages']} pages, "
                f"{metadata['file_size'] / 1024:.1f} KB)")
        
        # Try PyMuPDF first (better quality), fallback to PyPDF2
        text = self.extract_text_pymupdf(pdf_file)
        
        if not text.strip():
            st.warning("PyMuPDF extraction failed, trying PyPDF2...")
            pdf_file.seek(0)  # Reset for second attempt
            text = self.extract_text_pypdf2(pdf_file)
        
        if not text.strip():
            raise ValueError(f"Could not extract text from {filename}")
        
        # Clean and preprocess text
        text = self._clean_text(text)
        
        # Split text into chunks
        chunks = self.text_splitter.split