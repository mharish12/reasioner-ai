# Feature Update: PDF & Image Support

## Overview
Added comprehensive support for PDF and image file processing in the AI Model Training Platform.

## New Features

### PDF File Support
- **Extraction Methods**: Dual support for pdfplumber and PyPDF2
- **Processing**: Page-by-page text extraction
- **Metadata**: Tracks page numbers, total pages, filename
- **Auto-fallback**: Uses pdfplumber if available, falls back to PyPDF2

### Image File Support with OCR
- **Formats**: JPG, JPEG, PNG, GIF, BMP, TIFF, TIF, WebP
- **OCR Engine**: Tesseract OCR via pytesseract
- **Text Extraction**: Paragraph-based chunking
- **Metadata**: Image dimensions, mode, processing method
- **Fallback**: Handles images with no text gracefully

## Technical Implementation

### Backend Changes

#### New Dependencies (`backend/requirements.txt`)
```
PyPDF2>=3.0.0          # PDF text extraction
pdfplumber>=0.9.0      # Better PDF text extraction
Pillow>=10.0.0         # Image processing
pytesseract>=0.3.10    # OCR wrapper
```

#### New Functions (`backend/utils/file_processor.py`)

**process_pdf(file_content: bytes, filename: str)**
- Extracts text from PDF documents
- Handles multi-page documents
- Returns list of (text, metadata) tuples

**process_image(file_content: bytes, filename: str)**
- Performs OCR on images
- Handles various image formats
- Returns extracted text with image metadata

**Updated Functions**
- `get_file_type()`: Now recognizes PDF and image extensions
- `process_file()`: Routes to appropriate processor based on file type

### Frontend Changes

#### Updated Component (`frontend/src/components/ModelTraining.jsx`)
- Expanded file upload `accept` attribute to include PDF and image formats
- Updated UI text to reflect new supported formats
- Maintains existing file preview functionality

## Usage

### PDF Processing
1. Upload a PDF file through the training interface
2. System extracts text from each page
3. Text is chunked and stored with metadata
4. Ready for model training

### Image OCR Processing
1. Upload an image file (JPG, PNG, etc.)
2. System performs OCR to extract text
3. Text is chunked into paragraphs
4. Image metadata is preserved
5. Ready for model training

### Code Example

```python
# Backend usage
from utils.file_processor import process_file

# Process any supported file
with open('document.pdf', 'rb') as f:
    content = f.read()
    documents = process_file(content, 'document.pdf')

# Process image
with open('image.png', 'rb') as f:
    content = f.read()
    documents = process_file(content, 'image.png')
```

## System Requirements

### For PDF Processing
- Python packages: PyPDF2 or pdfplumber
- No additional system dependencies

### For Image OCR
- Python packages: Pillow, pytesseract
- **System requirement**: Tesseract OCR binary

**Installation:**
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- Windows: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)

## Metadata Structure

### PDF Metadata
```python
{
    "page_number": 1,
    "total_pages": 5,
    "filename": "document.pdf",
    "method": "pdfplumber"  # or "PyPDF2"
}
```

### Image Metadata (with text)
```python
{
    "paragraph_index": 0,
    "filename": "image.png",
    "image_width": 1920,
    "image_height": 1080,
    "image_mode": "RGB",
    "has_text": True,
    "ocr_method": "tesseract"
}
```

### Image Metadata (no text)
```python
{
    "filename": "image.png",
    "image_width": 1920,
    "image_height": 1080,
    "image_mode": "RGB",
    "has_text": False,
    "ocr_method": "tesseract"
}
```

## Error Handling

### Graceful Degradation
- If pdfplumber unavailable, uses PyPDF2
- If no PDF libraries available, returns clear error message
- If Tesseract not installed, returns helpful installation guide
- Images without text are stored with metadata

### Error Messages
- "No PDF processing library available. Install pdfplumber or PyPDF2"
- "OCR libraries not available. Install Pillow and pytesseract"
- "Tesseract OCR not installed. Install tesseract-ocr package on your system"

## Performance Considerations

- PDFs: Processing time depends on number of pages
- Images: OCR processing depends on image resolution and text content
- Large files are handled efficiently with streaming
- Automatic chunking prevents memory issues

## Testing Recommendations

### Test Cases
1. Upload PDF with text
2. Upload multi-page PDF
3. Upload scanned PDF (OCR may be needed)
4. Upload clear image with text
5. Upload image without text
6. Upload very large image
7. Upload corrupted file

### Expected Behavior
- All supported formats process successfully
- Text extraction works accurately
- Metadata is correctly preserved
- Training proceeds without errors
- Query results include text from files

## Documentation Updates

### Files Updated
- `README.md`: Added file format support section
- `QUICKSTART.md`: Added PDF and image examples
- `backend/requirements.txt`: Added new dependencies

### Sections Added
- File Processing Details
- PDF/Image Processing Issues troubleshooting
- System prerequisites for OCR
- Usage examples for new formats

## Future Enhancements

### Potential Improvements
- Table extraction from PDFs
- Advanced OCR preprocessing (deskew, denoise)
- Support for Word documents (.docx)
- Image preprocessing optimizations
- Parallel processing for large files
- Progress indicators for long-running operations

## Backward Compatibility

- Existing functionality unchanged
- All previous file formats still supported
- No breaking changes to API
- Database schema compatible
- Frontend UI remains familiar

## Migration Notes

### For Existing Installations
1. Install new dependencies:
   ```bash
   pip install PyPDF2 pdfplumber Pillow pytesseract
   ```
2. Install Tesseract OCR on system
3. Restart backend server
4. New file formats automatically available

### No Database Migration Needed
- Uses existing `training_data` table
- No schema changes required
- Metadata stored as JSON as before

## Summary

The AI Model Training Platform now supports comprehensive document and image processing, making it suitable for a wider range of use cases including:
- Document knowledge bases
- Scanned document processing
- Image-based text recognition
- Multi-format training data

All implementations follow best practices for error handling, metadata preservation, and user experience.

---

**Date**: 2025-01-11  
**Version**: 1.1.0  
**Status**: ✅ Complete

