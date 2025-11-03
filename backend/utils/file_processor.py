import pandas as pd
import io
from typing import List, Tuple
import base64

# PDF processing
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    PyPDF2 = None

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    pdfplumber = None

# Image processing
try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    Image = None
    pytesseract = None

def process_excel(file_content: bytes, filename: str) -> List[Tuple[str, dict]]:
    """Process Excel file and extract text content"""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        documents = []
        
        # Convert DataFrame to text
        for idx, row in df.iterrows():
            text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            metadata = {
                "row_index": idx,
                "columns": list(df.columns),
                "filename": filename
            }
            documents.append((text, metadata))
        
        return documents
    except Exception as e:
        raise Exception(f"Error processing Excel file: {str(e)}")

def process_csv(file_content: bytes, filename: str) -> List[Tuple[str, dict]]:
    """Process CSV file and extract text content"""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        documents = []
        
        # Convert DataFrame to text
        for idx, row in df.iterrows():
            text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            metadata = {
                "row_index": idx,
                "columns": list(df.columns),
                "filename": filename
            }
            documents.append((text, metadata))
        
        return documents
    except Exception as e:
        raise Exception(f"Error processing CSV file: {str(e)}")

def process_txt(file_content: bytes, filename: str) -> List[Tuple[str, dict]]:
    """Process TXT file and extract text content"""
    try:
        text = file_content.decode('utf-8')
        # Split by paragraphs or sentences
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        documents = []
        for idx, paragraph in enumerate(paragraphs):
            metadata = {
                "paragraph_index": idx,
                "filename": filename
            }
            documents.append((paragraph, metadata))
        
        return documents
    except Exception as e:
        raise Exception(f"Error processing TXT file: {str(e)}")

def process_pdf(file_content: bytes, filename: str) -> List[Tuple[str, dict]]:
    """Process PDF file and extract text content"""
    try:
        documents = []
        
        # Try pdfplumber first (better text extraction)
        if HAS_PDFPLUMBER:
            pdf = pdfplumber.open(io.BytesIO(file_content))
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    metadata = {
                        "page_number": page_num + 1,
                        "total_pages": len(pdf.pages),
                        "filename": filename,
                        "method": "pdfplumber"
                    }
                    documents.append((text.strip(), metadata))
            pdf.close()
            return documents
        
        # Fallback to PyPDF2
        elif HAS_PYPDF2:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    metadata = {
                        "page_number": page_num + 1,
                        "total_pages": len(pdf_reader.pages),
                        "filename": filename,
                        "method": "PyPDF2"
                    }
                    documents.append((text.strip(), metadata))
            return documents
        
        else:
            raise Exception("No PDF processing library available. Install pdfplumber or PyPDF2")
    
    except Exception as e:
        raise Exception(f"Error processing PDF file: {str(e)}")

def process_image(file_content: bytes, filename: str) -> List[Tuple[str, dict]]:
    """Process image file and extract text using OCR"""
    try:
        if not HAS_OCR:
            raise Exception("OCR libraries not available. Install Pillow and pytesseract")
        
        # Load image
        image = Image.open(io.BytesIO(file_content))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using Tesseract OCR
        try:
            text = pytesseract.image_to_string(image)
        except Exception as ocr_error:
            # If pytesseract executable not found, provide helpful error
            if "tesseract" in str(ocr_error).lower():
                raise Exception("Tesseract OCR not installed. Install tesseract-ocr package on your system")
            raise
        
        if not text or not text.strip():
            # If no text found, store image metadata instead
            metadata = {
                "filename": filename,
                "image_width": image.width,
                "image_height": image.height,
                "image_mode": image.mode,
                "has_text": False,
                "ocr_method": "tesseract"
            }
            # Store base64 encoded image in content
            image_base64 = base64.b64encode(file_content).decode('utf-8')
            return [(f"Image file: {filename} (No text detected)", metadata)]
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        documents = []
        for idx, paragraph in enumerate(paragraphs):
            metadata = {
                "paragraph_index": idx,
                "filename": filename,
                "image_width": image.width,
                "image_height": image.height,
                "image_mode": image.mode,
                "has_text": True,
                "ocr_method": "tesseract"
            }
            documents.append((paragraph, metadata))
        
        return documents
    
    except Exception as e:
        raise Exception(f"Error processing image file: {str(e)}")

def process_plain_text(text: str, metadata: dict = None) -> List[Tuple[str, dict]]:
    """Process plain text"""
    if metadata is None:
        metadata = {}
    
    # Split into sentences or chunks
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    documents = []
    for idx, sentence in enumerate(sentences):
        doc_metadata = {**metadata, "sentence_index": idx}
        documents.append((sentence, doc_metadata))
    
    return documents

def get_file_type(filename: str) -> str:
    """Determine file type from filename"""
    ext = filename.lower().split('.')[-1]
    
    if ext in ['xlsx', 'xls']:
        return 'excel'
    elif ext == 'csv':
        return 'csv'
    elif ext == 'txt':
        return 'txt'
    elif ext == 'pdf':
        return 'pdf'
    elif ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp']:
        return 'image'
    else:
        return 'plain_text'

def process_file(file_content: bytes, filename: str) -> List[Tuple[str, dict]]:
    """Universal file processor"""
    file_type = get_file_type(filename)
    
    if file_type == 'excel':
        return process_excel(file_content, filename)
    elif file_type == 'csv':
        return process_csv(file_content, filename)
    elif file_type == 'txt':
        return process_txt(file_content, filename)
    elif file_type == 'pdf':
        return process_pdf(file_content, filename)
    elif file_type == 'image':
        return process_image(file_content, filename)
    else:
        raise Exception(f"Unsupported file type: {file_type}")

