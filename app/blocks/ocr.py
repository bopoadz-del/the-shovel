"""OCR Block - Extract text from images with Typed Schema"""

import os
import io
import tempfile
from typing import Any, Dict
from app.core.typed_block import TypedBlock, Schema, ContentType
import logging

logger = logging.getLogger(__name__)


def _ocr_lang() -> str:
    """Tesseract language code(s). Defaults to 'eng'; set RAG_OCR_LANG to
    'ara+eng' on the production image where tesseract-ocr-ara is installed
    (FOLLOW-UP #93) so Arabic BOQ pages stop producing CMAP-less mojibake."""
    return os.getenv("RAG_OCR_LANG", "eng")


class OCRBlock(TypedBlock):
    """Optical Character Recognition from images with typed I/O"""
    
    auto_validate = False
    name = "ocr"
    version = "2.0.0"
    description = "Extract text from images using OCR with preprocessing"
    layer = 3
    tags = ["domain", "vision", "ocr", "documents", "typed"]
    requires = []
    
    default_config = {
        "languages": ["en"],
        "preprocess": True,
        "deskew": True,
        "contrast_factor": 1.5
    }
    
    # Type schemas for chain validation
    input_schema = Schema(
        content_type=ContentType.IMAGE,
        required_fields=["file_path"],
        optional_fields=["path", "url"],
        format_hints={"accept": [".jpg", ".jpeg", ".png", ".webp"]}
    )
    
    output_schema = Schema(
        content_type=ContentType.TEXT,
        required_fields=["text"],
        optional_fields=["confidence", "quality", "has_markup", "markup", "word_count", "engine", "preprocessed", "status"],
        format_hints={}
    )
    
    ui_schema = {
        "input": {
            "type": "image",
            "accept": [".jpg", ".jpeg", ".png", ".webp"],
            "placeholder": "Upload image to extract text...",
            "multiline": False
        },
        "output": {
            "type": "text",
            "fields": [
                {"name": "text", "type": "text", "label": "Extracted Text"},
                {"name": "confidence", "type": "percentage", "label": "Confidence"}
            ]
        },
        "quick_actions": [
            {"icon": "🔍", "label": "Extract Text", "prompt": "Extract all text from this image"},
            {"icon": "🔢", "label": "Extract Numbers", "prompt": "Extract all numbers and measurements from this image"},
            {"icon": "📋", "label": "Full OCR", "prompt": "Perform full OCR and return structured content"}
        ]
    }
    
    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Extract text from image (or PDF) with preprocessing"""
        params = params or {}

        # Download from URL if needed (handles bare URL strings and InputAdapter {"text": "url"} wrapping)
        url = None
        if isinstance(input_data, str) and input_data.startswith("http"):
            url = input_data
        elif isinstance(input_data, dict):
            url = input_data.get("url")
            if not url:
                raw = input_data.get("text") or input_data.get("input") or ""
                if raw.startswith("http"):
                    url = raw

        if url:
            import httpx
            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    response = await client.get(url, timeout=30)
                    response.raise_for_status()
                    ext = ".jpg"
                    for candidate in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
                        if candidate in url.lower():
                            ext = candidate
                            break
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                        f.write(response.content)
                        input_data = f.name
            except Exception as e:
                return {"status": "error", "text": "", "confidence": 0, "error": f"Download failed: {str(e)}"}

        image_path = self._get_image_path(input_data)
        if not image_path:
            return {"status": "error", "text": "", "confidence": 0, "error": "No image provided"}

        if not os.path.exists(image_path):
            return {"status": "error", "text": "", "confidence": 0, "error": f"File not found: {image_path}"}

        # Decrypt-to-temp if the stored file is encrypted at rest. PIL /
        # Tesseract / PyMuPDF all need a real file on disk, so every read below
        # goes through this plaintext path. No-op for plaintext / legacy files.
        from app.core.file_crypto import open_plaintext
        with open_plaintext(image_path) as image_path:
            return await self._process_image(image_path, params)

    async def _process_image(self, image_path: str, params: Dict) -> Dict:
        """Run OCR on a plaintext image/PDF path (post-decryption)."""
        preprocess = params.get("preprocess", self.config.get("preprocess", True))
        languages = params.get("languages", self.config.get("languages", ["en"]))

        # Detect coloured markup / redlines BEFORE preprocessing greys the image
        # out (Roadmap V2 · Epic 5). Annotated regions are flagged, not mangled
        # into the extracted text.
        markup = self._detect_markup(image_path)

        # Detect if input is a PDF and convert pages to images
        page_images = self._prepare_images(image_path, preprocess)
        if not page_images:
            return {"status": "error", "text": "", "confidence": 0, "error": "Could not process input file"}
        
        # Try pytesseract first, then fall back to PyMuPDF text-layer extraction.
        all_texts = []
        all_confs = []
        engine_used = None
        tesseract_available = True

        try:
            import pytesseract
            pytesseract.get_tesseract_version()  # raises if not installed
        except Exception:
            tesseract_available = False

        word_confidences = []
        if tesseract_available:
            try:
                import pytesseract
                from PIL import Image
                engine_used = "pytesseract"

                for page_img_path in page_images:
                    img = Image.open(page_img_path)
                    text = pytesseract.image_to_string(img, lang=_ocr_lang())
                    if text.strip():
                        all_texts.append(text.strip())
                    # Capture REAL per-word confidence (Roadmap V2 · Epic 5 / 1)
                    # instead of the old hardcoded 0.85.
                    try:
                        data = pytesseract.image_to_data(
                            img, lang=_ocr_lang(), output_type=pytesseract.Output.DICT
                        )
                        for conf in data.get("conf", []):
                            try:
                                c = float(conf)
                            except (TypeError, ValueError):
                                continue
                            if c >= 0:
                                word_confidences.append(c / 100.0)
                    except Exception:
                        logger.debug(
                            "swallowed %s in _process_image() — continuing",
                            "Exception", exc_info=True,
                        )
            except Exception as e:
                tesseract_available = False

        if not tesseract_available:
            # Local-only fallback: PyMuPDF text extraction (text-layer PDFs only).
            # No cloud vision dependency — the OCR block stays fully on-prem.
            pdf_text = self._extract_pdf_text(image_path)
            if pdf_text:
                return {
                    "status": "success",
                    "text": pdf_text,
                    "confidence": 0.95,
                    "has_markup": markup["has_markup"],
                    "markup": markup,
                    "word_count": len(pdf_text.split()),
                    "engine": "pymupdf_fallback",
                    "preprocessed": False,
                    "pages": 1,
                    "note": "Tesseract not installed; used PyMuPDF text extraction",
                }
            return {
                "status": "error",
                "text": "",
                "confidence": 0,
                "error": "Tesseract not installed and no PDF text layer found. Install tesseract-ocr to enable local OCR.",
            }

        from app.core.image_quality import summarize_ocr_quality
        quality = summarize_ocr_quality(word_confidences)

        if not all_texts:
            return {
                "status": "success", "text": "", "confidence": 0,
                "quality": quality, "message": "No text detected",
                "has_markup": markup["has_markup"],
                "markup": markup,
            }

        full_text = "\n".join(all_texts)

        return {
            "status": "success",
            "text": full_text,
            "confidence": quality["ocr_confidence"],
            "quality": quality,
            "has_markup": markup["has_markup"],
            "markup": markup,
            "word_count": len(full_text.split()),
            "engine": engine_used or "unknown",
            "preprocessed": preprocess,
            "pages": len(page_images)
        }
    
    def _get_image_path(self, input_data: Any) -> str:
        """Extract image path from input"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            return input_data.get("file_path") or input_data.get("path") or input_data.get("url")
        return None
    
    def _detect_markup(self, file_path: str) -> Dict:
        """Detect coloured markup / redlines on the input (Roadmap V2 · Epic 5).

        Run on the ORIGINAL colour image, before preprocessing converts it to
        greyscale. For PDFs the first page is rendered and checked. Returns the
        `summarize_markup` verdict (`has_markup`, `coverage`, `region_count`,
        `regions`, `caveat`) — annotated regions are flagged for the user, not
        merged into the extracted text. Failures degrade gracefully to "clean".
        """
        from app.core.redline import detect_redlines, summarize_markup

        clean = {
            "has_markup": False, "coverage": 0.0, "region_count": 0,
            "regions": [], "caveat": None,
        }
        try:
            from PIL import Image

            if file_path.lower().endswith(".pdf"):
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                if len(doc) == 0:
                    doc.close()
                    return clean
                pix = doc.load_page(0).get_pixmap(dpi=150)
                fd, tmp = tempfile.mkstemp(suffix="_markup.png")
                os.close(fd)
                pix.save(tmp)
                doc.close()
                img = Image.open(tmp)
            else:
                img = Image.open(file_path)

            result = detect_redlines(img)
            summary = summarize_markup(result)
            summary["regions"] = result["regions"]
            return summary
        except Exception:
            return clean

    def _prepare_images(self, file_path: str, preprocess: bool = True) -> list:
        """Convert input to a list of image file paths (handles PDFs and images)."""
        from PIL import Image
        
        is_pdf = file_path.lower().endswith(".pdf")
        if not is_pdf:
            # Try to open as image; if it fails, try as PDF via PyMuPDF
            try:
                Image.open(file_path)
                # It's a valid image
                if preprocess:
                    return [self._preprocess_image(file_path)]
                return [file_path]
            except Exception:
                is_pdf = True
        
        if is_pdf:
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                images = []
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=200)
                    fd, img_path = tempfile.mkstemp(suffix=f"_page{page_num + 1}.png")
                    os.close(fd)
                    pix.save(img_path)
                    if preprocess:
                        img_path = self._preprocess_image(img_path)
                    images.append(img_path)
                doc.close()
                return images
            except ImportError:
                return []
            except Exception:
                return []
        
        return []
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from a PDF using PyMuPDF (fallback when OCR is unavailable)."""
        try:
            import fitz
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception:
            return ""
    
    def _preprocess_image(self, image_path: str) -> str:
        """Enhance image for better OCR quality"""
        from PIL import Image, ImageEnhance, ImageFilter
        
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Resize if too small (improve effective DPI)
        min_dim = min(img.size)
        if min_dim < 1000:
            scale = 1200 / min_dim
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        gray = img.convert('L')

        # Deskew (Roadmap V2 · Epic 5) — straighten rotated/tilted scans
        if self.config.get("deskew", True):
            try:
                from app.core.image_quality import deskew as _deskew
                gray, _angle = _deskew(gray)
            except Exception:
                logger.debug(
                    "swallowed %s in _preprocess_image() — continuing",
                    "Exception", exc_info=True,
                )

        # Enhance contrast
        contrast_factor = self.config.get("contrast_factor", 1.5)
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(contrast_factor)
        
        # Sharpen
        gray = gray.filter(ImageFilter.SHARPEN)
        
        # Apply mild denoise
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        
        # Adaptive thresholding: if image is low-contrast, apply point operation
        # to increase separation between text and background
        stat = gray.getextrema()
        if stat and (stat[1] - stat[0]) < 100:
            # Low dynamic range — apply threshold
            gray = gray.point(lambda x: 0 if x < 128 else 255, '1').convert('L')
        
        # Save to temp file
        fd, temp_path = tempfile.mkstemp(suffix="_ocr.png")
        os.close(fd)
        gray.save(temp_path, "PNG")
        return temp_path
