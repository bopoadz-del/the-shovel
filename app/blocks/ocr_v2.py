"""OCR Block v2 - Extract text from images using TypedBlock

This is the v2 implementation using the TypedBlock system with schema validation.
"""

import os
import io
import tempfile
from typing import Any, Dict

from app.core.typed_block import TypedBlock
from app.core.schema_registry import TextContent


class OCRBlockV2(TypedBlock):
    """Optical Character Recognition from images - v2 with schema support"""
    
    name = "ocr_v2"
    version = "2.0"
    description = "Extract text from images using OCR with typed output"
    layer = 3
    tags = ["domain", "vision", "ocr", "documents", "v2"]
    requires = []
    
    default_config = {
        "languages": ["en"],
        "preprocess": True,
        "deskew": False,
        "contrast_factor": 1.5
    }
    
    # Input: image path or dict with image info
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "path": {"type": "string"},
            "url": {"type": "string"},
            "image_path": {"type": "string"}
        },
        "anyOf": [
            {"required": ["file_path"]},
            {"required": ["path"]},
            {"required": ["url"]},
            {"required": ["image_path"]}
        ]
    }
    
    # Output: TextContent schema
    output_schema = TextContent
    
    # Type declarations for orchestrator
    accepted_input_types = ["ImageContent", "FileContent"]
    produced_output_types = ["TextContent"]
    
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
        }
    }
    
    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Extract text from image and return TextContent format"""
        params = params or {}
        
        image_path = self._get_image_path(input_data)
        if not image_path:
            return self._error_response("No image provided")

        if image_path.startswith("http://") or image_path.startswith("https://"):
            import httpx, tempfile
            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    resp = await client.get(image_path, timeout=30)
                    resp.raise_for_status()
                    ext = ".jpg"
                    for candidate in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
                        if candidate in image_path.lower():
                            ext = candidate
                            break
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                        f.write(resp.content)
                        image_path = f.name
            except Exception as e:
                return self._error_response(f"Download failed: {str(e)}")

        if not os.path.exists(image_path):
            return self._error_response(f"File not found: {image_path}")

        # Decrypt-to-temp if the stored file is encrypted at rest. EasyOCR / PIL
        # need a real file on disk. No-op for plaintext / legacy files.
        from app.core.file_crypto import open_plaintext
        with open_plaintext(image_path) as image_path:
            return await self._process_image(image_path, params)

    async def _process_image(self, image_path: str, params: Dict) -> Dict:
        """Run OCR on a plaintext image path (post-decryption)."""
        preprocess = params.get("preprocess", self.config.get("preprocess", True))
        languages = params.get("languages", self.config.get("languages", ["en"]))

        # Detect coloured markup / redlines on the ORIGINAL image, before
        # preprocessing greys it out (Roadmap V2 · Epic 5). Annotated regions
        # are flagged in the result, not mangled into the extracted text.
        markup = self._detect_markup(image_path)

        # Preprocess image if enabled
        if preprocess:
            image_path = self._preprocess_image(image_path)
        
        # Try EasyOCR (pure Python, no system deps)
        try:
            import easyocr
            reader = easyocr.Reader(languages, gpu=False)
            results = reader.readtext(image_path)
            
            if not results:
                return {
                    "text": "",
                    "source": "ocr",
                    "confidence": 0,
                    "has_markup": markup["has_markup"],
                    "markup": markup,
                    "metadata": {
                        "message": "No text detected",
                        "word_count": 0,
                        "engine": "easyocr",
                        "preprocessed": preprocess,
                        "has_markup": markup["has_markup"],
                        "markup": markup
                    }
                }

            texts = [r[1] for r in results if r[1].strip()]
            confs = [r[2] for r in results]

            full_text = "\n".join(texts)
            avg_conf = sum(confs) / len(confs) if confs else 0

            # Return TextContent format
            return {
                "text": full_text,
                "source": "ocr",
                "confidence": round(avg_conf, 2),
                "has_markup": markup["has_markup"],
                "markup": markup,
                "metadata": {
                    "word_count": len(full_text.split()),
                    "engine": "easyocr",
                    "preprocessed": preprocess,
                    "has_markup": markup["has_markup"],
                    "markup": markup,
                    "extracted_at": self._timestamp()
                }
            }
            
        except ImportError:
            return self._error_response("EasyOCR not installed")
        except Exception as e:
            return self._error_response(f"OCR failed: {str(e)}")
    
    def _get_image_path(self, input_data: Any) -> str:
        """Extract image path from input"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            return (input_data.get("file_path") or 
                    input_data.get("path") or 
                    input_data.get("url") or 
                    input_data.get("image_path"))
        return None
    
    def _detect_markup(self, file_path: str) -> Dict:
        """Detect coloured markup / redlines on the input (Roadmap V2 · Epic 5).

        Run on the ORIGINAL colour image, before preprocessing converts it to
        greyscale. For PDFs the first page is rendered and checked. Returns the
        `summarize_markup` verdict (`has_markup`, `coverage`, `region_count`,
        `regions`, `caveat`). Failures degrade gracefully to "clean".
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
        
        # Enhance contrast
        contrast_factor = self.config.get("contrast_factor", 1.5)
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(contrast_factor)
        
        # Sharpen
        gray = gray.filter(ImageFilter.SHARPEN)
        
        # Apply mild denoise
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        
        # Adaptive thresholding: if image is low-contrast, apply point operation
        stat = gray.getextrema()
        if stat and (stat[1] - stat[0]) < 100:
            # Low dynamic range — apply threshold
            gray = gray.point(lambda x: 0 if x < 128 else 255, '1').convert('L')
        
        # Save to temp file
        fd, temp_path = tempfile.mkstemp(suffix="_ocr.png")
        os.close(fd)
        gray.save(temp_path, "PNG")
        return temp_path
    
    def _error_response(self, message: str) -> Dict:
        """Return standardized error response"""
        return {
            "text": "",
            "source": "ocr",
            "confidence": 0,
            "metadata": {
                "error": message,
                "extracted_at": self._timestamp()
            }
        }
    
    def _timestamp(self) -> str:
        """Get current ISO timestamp"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
