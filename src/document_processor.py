"""
Document Processor for Mini RAG Project
Handles reading PDF and DOCX files and extracting text
"""
import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
from docx import Document as DocxDocument

class DocumentProcessor:
    """Process documents and extract text content"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.supported_extensions = ['.pdf', '.docx', '.txt', '.md']
    
    def read_pdf(self, filepath: Path) -> str:
        """Extract text from PDF file"""
        text = []
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text.append(f"[Page {page_num + 1}]\n{page_text}")
        except Exception as e:
            print(f"Error reading PDF {filepath}: {e}")
            return ""
        
        return "\n\n".join(text)
    
    def read_docx(self, filepath: Path) -> str:
        """Extract text from DOCX file"""
        text = []
        try:
            doc = DocxDocument(filepath)
            for para in doc.paragraphs:
                if para.text.strip():
                    text.append(para.text)
        except Exception as e:
            print(f"Error reading DOCX {filepath}: {e}")
            return ""
        
        return "\n".join(text)
    
    def read_txt(self, filepath: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT {filepath}: {e}")
            return ""
    
    def read_md(self, filepath: Path) -> str:
        """Extract text from Markdown file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading MD {filepath}: {e}")
            return ""
    
    def process_file(self, filepath: Path) -> Dict[str, str]:
        """Process a single file and return document info"""
        extension = filepath.suffix.lower()
        
        if extension == '.pdf':
            text = self.read_pdf(filepath)
        elif extension == '.docx':
            text = self.read_docx(filepath)
        elif extension == '.txt':
            text = self.read_txt(filepath)
        elif extension == '.md':
            text = self.read_md(filepath)
        else:
            print(f"Unsupported file type: {extension}")
            return None
        
        return {
            'filename': filepath.name,
            'filepath': str(filepath),
            'text': text,
            'length': len(text),
            'extension': extension
        }
    
    def process_all_documents(self) -> List[Dict[str, str]]:
        """Process all documents in the data directory"""
        documents = []
        
        if not self.data_dir.exists():
            print(f"Error: Directory {self.data_dir} does not exist!")
            return documents
        
        print(f" Scanning directory: {self.data_dir.absolute()}\n")
        
        # Find all supported files
        files = []
        for ext in self.supported_extensions:
            files.extend(self.data_dir.glob(f"*{ext}"))
        
        if not files:
            print(f"  No documents found in {self.data_dir}")
            print(f"Supported formats: {', '.join(self.supported_extensions)}")
            return documents
        
        print(f"Found {len(files)} document(s):\n")
        
        # Process each file
        for filepath in sorted(files):
            print(f" Processing: {filepath.name}")
            doc_info = self.process_file(filepath)
            
            if doc_info and doc_info['text']:
                documents.append(doc_info)
                print(f"    Extracted {doc_info['length']:,} characters")
            else:
                print(f"    Failed to extract text")
        
        print(f"\n{'='*50}")
        print(f" Processing Summary:")
        print(f"{'='*50}")
        print(f"Total documents processed: {len(documents)}")
        total_chars = sum(doc['length'] for doc in documents)
        print(f"Total characters extracted: {total_chars:,}")
        
        return documents

def main():
    """Test the document processor"""
    processor = DocumentProcessor()
    documents = processor.process_all_documents()
    
    if documents:
        print("\n Document processing complete!")
        print(f"Ready to chunk and embed {len(documents)} document(s)")
    else:
        print("\n No documents were processed")
        print("Please add PDF, DOCX, or TXT files to the 'data' folder")

if __name__ == "__main__":
    main()