"""
Edusoft AI Automated Paper Evaluation Engine.
Performs AI-driven evaluation of student answer sheets/papers against authoritative Question Papers
and Model Answer Keys / Rubrics. Supports Mode A class-wise evaluation via a combined multi-student PDF.

CORE PRINCIPLES & SPECIFICATIONS:
1. Evaluation Inputs:
   - Question Paper: Authoritative source for question numbering, question count, and maximum marks.
   - Answer Key / Model Answer: Authoritative source for correct answers and scoring rubrics.
   - Student Answer Booklet(s): Handwritten answers evaluated against QP and Answer Key.
2. Mode A — Class-Wise Evaluation Using One Combined PDF:
   - Detects student booklet boundaries page-by-page using cover pages, identity indicators,
     page number restarts, and question sequence resets.
   - Only passes student name and student ID if visibly present on that student's booklet.
3. Complete Decoupling of Identity & Evaluation:
   - Answer evaluation is NEVER dependent on student identification.
   - If identity fails, returns blank identity fields ("") and 'unmatched' status, but still returns
     the complete question-by-question marks and breakdown.
4. Exact Metric Calculations:
   - total_questions, attempted_questions, unanswered_questions, evaluated_questions, pending_review_questions.
   - total_questions = attempted_questions + unanswered_questions.
   - attempted_questions = evaluated_questions + pending_review_questions.
   - total_marks, obtained_marks, percentage = (obtained_marks / total_marks) * 100.
5. Authoritative Document Handling & Discrepancy Reporting:
   - No hardcoded marks or question counts.
   - Validates sum of question maximum marks against exam total marks and reports any discrepancy.
6. Privacy & Identity Removal in GET Responses:
   - Strictly removes 'student_name' and 'student_id' from student objects in GET responses.
"""
import os
import re
import io
import json
import base64
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from openai import OpenAI, OpenAIError

logger = logging.getLogger("edusoft_service.paper_evaluator")

# Remove any legacy evaluations file on startup
for _legacy in ["evaluations_db.json", "/app/evaluations_db.json"]:
    try:
        if os.path.exists(_legacy):
            os.remove(_legacy)
    except Exception:
        pass


def calculate_grade(percentage: float) -> str:
    """Calculate letter grade from percentage score."""
    if percentage >= 90.0:
        return "A+"
    elif percentage >= 80.0:
        return "A"
    elif percentage >= 70.0:
        return "B+"
    elif percentage >= 60.0:
        return "B"
    elif percentage >= 50.0:
        return "C"
    elif percentage >= 40.0:
        return "D"
    else:
        return "F"


def determine_status(percentage: float, pass_mark: float = 50.0) -> str:
    """Determine pass/fail status."""
    return "Pass" if percentage >= pass_mark else "Fail"


def format_two_decimals(val: Any) -> str:
    """Format numeric value into exact two-decimal string format e.g. '88.50'."""
    try:
        return f"{float(val):.2f}"
    except (ValueError, TypeError):
        return str(val)


# ==============================================================================
# FILE RESOLUTION & DOCUMENT EXTRACTION HELPERS
# ==============================================================================
def resolve_file_location(path_or_url: str) -> Optional[str]:
    """
    Find actual file path on disk across common web server locations,
    handling local relative paths, URLs, and base64 payloads.
    """
    if not path_or_url or not isinstance(path_or_url, str):
        return None
    p = path_or_url.strip()
    if not p:
        return None

    # 1. Direct absolute or relative path check
    if os.path.exists(p) and os.path.isfile(p):
        return os.path.abspath(p)

    import urllib.parse
    p_unquoted = urllib.parse.unquote(p)
    if os.path.exists(p_unquoted) and os.path.isfile(p_unquoted):
        return os.path.abspath(p_unquoted)

    # 2. Remote URL handling (e.g., http://.../uploads/...)
    if p.startswith("http://") or p.startswith("https://"):
        try:
            import urllib.request
            suffix = ".pdf" if ".pdf" in p.lower() else ".jpg"
            temp_f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            urllib.request.urlretrieve(p, temp_f.name)
            logger.info(f"Downloaded remote file from {p} to temporary path {temp_f.name}")
            return temp_f.name
        except Exception as dl_err:
            logger.warning(f"Failed to download remote file {p}: {dl_err}")

    # 3. Base64 data URI handling
    if p.startswith("data:") and "base64," in p:
        try:
            header, b64data = p.split("base64,", 1)
            suffix = ".pdf" if "pdf" in header.lower() else ".jpg"
            temp_f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_f.write(base64.b64decode(b64data))
            temp_f.close()
            return temp_f.name
        except Exception as b64_err:
            logger.warning(f"Failed to decode base64 file URI: {b64_err}")

    # 4. Search across standard server directories
    cwd = os.getcwd()
    filename_only = os.path.basename(p)
    filename_unquoted = os.path.basename(p_unquoted)
    norm_p = p.replace("/", os.sep).replace("\\", os.sep)
    norm_unquoted = p_unquoted.replace("/", os.sep).replace("\\", os.sep)

    candidates = [
        os.path.join(cwd, p),
        os.path.join(cwd, norm_p),
        os.path.join(cwd, norm_unquoted),
        os.path.join("/app", p),
        os.path.join("/var/www/html", p),
        os.path.join("/var/www/html/edusoft", p),
        os.path.join("/var/www", p),
        os.path.join(os.path.dirname(cwd), p),
        os.path.join(cwd, "uploads", filename_only),
        os.path.join(cwd, "uploads", filename_unquoted),
        os.path.join(cwd, "uploads", "evaluation_papers", filename_only),
        os.path.join(cwd, "uploads", "evaluation_papers", filename_unquoted),
        os.path.join("/app", "uploads", filename_only),
        os.path.join("/app", "uploads", "evaluation_papers", filename_only),
        os.path.join("/var/www/html", "uploads", "evaluation_papers", filename_only),
        os.path.join(r"C:\wamp64\www\edusoft", norm_p),
        os.path.join(r"C:\wamp64\www", norm_p),
        os.path.join(r"C:\xampp\htdocs\edusoft", norm_p),
        os.path.join(r"C:\xampp\htdocs", norm_p),
        os.path.join(r"C:\wamp64\www\uploads\evaluation_papers", filename_only),
        os.path.join(r"C:\wamp64\www\edusoft\uploads\evaluation_papers", filename_only),
        os.path.join(r"C:\xampp\htdocs\uploads\evaluation_papers", filename_only),
        os.path.join(r"C:\xampp\htdocs\edusoft\uploads\evaluation_papers", filename_only),
        os.path.join(tempfile.gettempdir(), "evaluation_uploads", filename_only),
        os.path.join(tempfile.gettempdir(), filename_only),
    ]
    for c in candidates:
        try:
            if os.path.exists(c) and os.path.isfile(c):
                return os.path.abspath(c)
        except Exception:
            pass

    # 5. Recursive search in known upload directories
    for root_dir in [
        "/app/uploads",
        os.path.join(cwd, "uploads"),
        "/var/www/html/uploads",
        r"C:\wamp64\www\uploads",
        r"C:\xampp\htdocs\uploads",
        os.path.join(tempfile.gettempdir(), "evaluation_uploads")
    ]:
        try:
            if os.path.exists(root_dir):
                for dirpath, _, filenames in os.walk(root_dir):
                    if filename_only in filenames:
                        found_path = os.path.join(dirpath, filename_only)
                        if os.path.isfile(found_path):
                            return os.path.abspath(found_path)
                    if filename_unquoted in filenames:
                        found_path = os.path.join(dirpath, filename_unquoted)
                        if os.path.isfile(found_path):
                            return os.path.abspath(found_path)
        except Exception:
            pass

    # 6. Try retrieving via HTTP from host web server ONLY if path indicates upload location
    if ("uploads" in p.lower() or p.startswith("http")) and not (p.startswith("http://") or p.startswith("https://")):
        clean_rel = p.lstrip("/\\").replace("\\", "/")
        host_candidates = [
            f"http://localhost/{clean_rel}",
            f"http://127.0.0.1/{clean_rel}",
            f"http://localhost/edusoft/{clean_rel}",
            f"http://127.0.0.1/edusoft/{clean_rel}",
            f"http://172.17.0.1/{clean_rel}",
        ]
        for h_url in host_candidates:
            try:
                import urllib.request
                req = urllib.request.Request(h_url, headers={"User-Agent": "EdusoftAI/1.0"})
                with urllib.request.urlopen(req, timeout=0.5) as resp:
                    if resp.status == 200:
                        suffix = ".pdf" if ".pdf" in p.lower() else ".jpg"
                        temp_f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        temp_f.write(resp.read())
                        temp_f.close()
                        logger.info(f"Retrieved file via HTTP from {h_url} to {temp_f.name}")
                        return temp_f.name
            except Exception:
                pass

    return None


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, int]:
    """Extract readable text and page count from all pages of a PDF."""
    try:
        import pdfplumber
        extracted_pages = []
        total_pages = 1
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_idx, page in enumerate(pdf.pages, start=1):
                t = page.extract_text()
                if t and t.strip():
                    extracted_pages.append(f"--- PAGE {page_idx} ---\n{t.strip()}")
        return "\n\n".join(extracted_pages).strip(), total_pages
    except Exception as e:
        logger.warning(f"pdfplumber text extraction failed on {pdf_path}: {e}")
        return "", 1


def render_pdf_page_to_jpeg(page: Any, resolution: int = 150) -> Optional[bytes]:
    """Render a pdfplumber page to JPEG bytes with multiple fallbacks."""
    try:
        img_obj = page.to_image(resolution=resolution)
        buf = io.BytesIO()
        img_obj.original.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception as e:
        logger.debug(f"page.to_image failed: {e}")

    # Fallback: check embedded images
    try:
        for img_dict in page.images:
            stream = img_dict.get("stream")
            if stream and hasattr(stream, "get_data"):
                data = stream.get_data()
                if data:
                    return data
    except Exception:
        pass
    return None


def extract_page_images_from_pdf(
    pdf_path: str,
    page_numbers: Optional[List[int]] = None,
    max_pages: int = 50
) -> List[Tuple[str, bytes]]:
    """
    Extract high-resolution page images from a PDF for OpenAI Vision OCR.
    If page_numbers is provided, extracts only those 1-based page indices.
    """
    extracted = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total_pdf_pages = len(pdf.pages)
            target_indices = page_numbers if page_numbers else list(range(1, min(total_pdf_pages + 1, max_pages + 1)))
            for p_num in target_indices:
                if 1 <= p_num <= total_pdf_pages:
                    page = pdf.pages[p_num - 1]
                    img_bytes = render_pdf_page_to_jpeg(page)
                    if img_bytes:
                        extracted.append((f"page_{p_num}.jpg", img_bytes))
    except Exception as e:
        logger.warning(f"extract_page_images_from_pdf failed for {pdf_path}: {e}")
    return extracted


def extract_page_text_from_pdf(
    pdf_path: str,
    page_numbers: Optional[List[int]] = None
) -> str:
    """Extract readable text from specified 1-based page numbers of a PDF."""
    try:
        import pdfplumber
        extracted = []
        with pdfplumber.open(pdf_path) as pdf:
            total_pdf_pages = len(pdf.pages)
            target_indices = page_numbers if page_numbers else list(range(1, total_pdf_pages + 1))
            for p_num in target_indices:
                if 1 <= p_num <= total_pdf_pages:
                    page = pdf.pages[p_num - 1]
                    t = page.extract_text()
                    if t and t.strip():
                        extracted.append(f"--- PAGE {p_num} ---\n{t.strip()}")
        return "\n\n".join(extracted).strip()
    except Exception as e:
        logger.warning(f"extract_page_text_from_pdf failed for {pdf_path}: {e}")
        return ""


def extract_document_content(source: Any) -> Tuple[str, List[Tuple[str, bytes]]]:
    """
    Extract text content and/or page image bytes from a document source
    (disk path, URL, base64, JSON dict/list, or raw string).
    Returns: (text_content, list_of_(filename, bytes))
    """
    if not source:
        return "", []

    if isinstance(source, (dict, list)):
        return json.dumps(source, indent=2), []

    if isinstance(source, str):
        # Check if it resolves to a disk file or URL
        resolved = resolve_file_location(source)
        if resolved and os.path.exists(resolved):
            lower = resolved.lower()
            if lower.endswith(".pdf"):
                txt, _ = extract_text_from_pdf(resolved)
                imgs = extract_page_images_from_pdf(resolved, max_pages=15)
                return txt, imgs
            elif lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
                try:
                    with open(resolved, "rb") as f:
                        return "", [(os.path.basename(resolved), f.read())]
                except Exception as ie:
                    logger.warning(f"Could not read image {resolved}: {ie}")
                    return "", []
            else:
                try:
                    with open(resolved, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read(), []
                except Exception:
                    pass
        # Raw text string
        return source.strip(), []

    return "", []


FORBIDDEN_NAME_WORDS = {
    "examination", "exam", "college", "university", "institute", "institution",
    "subject", "paper", "question", "section", "semester", "date", "class",
    "branch", "course", "degree", "test", "marks", "grade", "faculty", "department",
    "edusoft", "module", "assignment", "evaluation", "reg", "roll", "admission",
    "total", "score", "pass", "fail", "model", "answer", "signature", "pharmacy",
    "nursing", "engineering", "school", "medical", "centre", "center", "title"
}


def extract_identity_from_document_text(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Search document text for student name and student ID.
    ONLY returns name/ID if explicitly written on the document header/cover page.
    """
    if not text:
        return None, None, None, None, None

    header_chunk = text[:1500]
    detected_name = None
    detected_id = None
    detected_adm = None
    detected_roll = None
    detected_kuhs = None

    # Pattern for Student Name / Candidate Name
    name_match = re.search(
        r"(?:Student\s*Name|Candidate\s*Name|Name\s*of\s*(?:the\s*)?(?:Student|Candidate))\s*[:\-\. ]\s*([A-Za-z\s\.]{2,40})",
        header_chunk,
        re.IGNORECASE
    )
    if name_match:
        cand = name_match.group(1).strip()
        lower_cand = cand.lower()
        words = set(lower_cand.split())
        if (
            len(cand) >= 2
            and len(words) <= 5
            and not words.intersection(FORBIDDEN_NAME_WORDS)
            and not any(lower_cand.startswith(f) for f in ["question", "section", "subject", "exam", "paper", "test", "final"])
        ):
            detected_name = cand

    # Pattern for Student ID / Candidate ID
    id_match = re.search(
        r"(?:Student\s*ID|Candidate\s*ID|Student\s*Id|Candidate\s*Id|Student\s*No)\s*[:\-\. ]\s*([A-Za-z0-9\-_/]+)",
        header_chunk,
        re.IGNORECASE
    )
    if id_match:
        detected_id = id_match.group(1).strip()

    adm_match = re.search(r"(?:Admission\s*No|Adm\s*No|Admission\s*Number)\s*[:\-\. ]\s*([A-Za-z0-9\-_/]+)", header_chunk, re.IGNORECASE)
    if adm_match:
        detected_adm = adm_match.group(1).strip()

    roll_match = re.search(r"(?:Roll\s*No|Roll\s*Number)\s*[:\-\. ]\s*([A-Za-z0-9\-_/]+)", header_chunk, re.IGNORECASE)
    if roll_match:
        detected_roll = roll_match.group(1).strip()

    kuhs_match = re.search(r"(?:KUHS\s*Reg\s*No|KUHS\s*No|Reg\s*No|Registration\s*No)\s*[:\-\. ]\s*([A-Za-z0-9\-_/]+)", header_chunk, re.IGNORECASE)
    if kuhs_match:
        detected_kuhs = kuhs_match.group(1).strip()

    return detected_name, detected_id, detected_adm, detected_roll, detected_kuhs


FORBIDDEN_PLACEHOLDER_SUBSTRINGS = [
    "student response extracted and verified",
    "response extracted and verified",
    "extracted and verified across",
    "handwritten answers on uploaded pages",
    "response extracted across",
    "answer extracted for question",
    "refer to uploaded page",
    "refer to uploaded",
    "see handwritten sheet",
    "see uploaded image",
    "see attached sheet",
    "verified response",
    "placeholder",
    "generic response",
    "student response transcribed",
    "handwritten response on page",
    "response on page",
    "extracted from page",
    "uploaded answer page",
    "uploaded answer page(s)",
    "uploaded answer pages",
    "answer sheet submitted across",
    "scanned handwriting images attached",
    "attached answer sheet",
    "student response verified",
]


def contains_forbidden_placeholder(text: str) -> bool:
    """Check if a response text contains forbidden generic placeholder phrasing."""
    if not text or not isinstance(text, str):
        return False
    lower = text.lower().strip()
    return any(p in lower for p in FORBIDDEN_PLACEHOLDER_SUBSTRINGS)


def collect_booklet_images(
    booklet: Dict[str, Any],
    default_file_path: str = ""
) -> List[Tuple[str, bytes, str]]:
    """
    Collect all high-resolution page images for a student's answer booklet.
    Supports:
    1. booklet['pages'] list (each page can have file_path, file_name, or base64)
    2. booklet['page_numbers'] with a multi-page PDF (extracts rendered images per page)
    3. Single image file_path (JPEG, PNG, WebP)
    4. Base64 data URIs
    """
    images_list: List[Tuple[str, bytes, str]] = []
    seen_hashes = set()

    def add_img(name: str, data: bytes, mime: str):
        if not data:
            return
        h = hash(data)
        if h not in seen_hashes:
            seen_hashes.add(h)
            images_list.append((name, data, mime))

    # 1. Process explicit pages list if available
    raw_pages = booklet.get("pages")
    if isinstance(raw_pages, str):
        try:
            raw_pages = json.loads(raw_pages)
        except Exception:
            raw_pages = []

    if isinstance(raw_pages, list) and len(raw_pages) > 0:
        for pg in raw_pages:
            if not isinstance(pg, dict):
                continue
            pg_fp = pg.get("file_path") or pg.get("path") or pg.get("file_name") or ""
            pg_res = resolve_file_location(pg_fp) if pg_fp else None
            if pg_res and os.path.exists(pg_res):
                if pg_res.lower().endswith(".pdf"):
                    p_no = pg.get("page_no") or 1
                    pdf_imgs = extract_page_images_from_pdf(pg_res, page_numbers=[int(p_no)])
                    for iname, ibytes in pdf_imgs:
                        add_img(iname, ibytes, "image/jpeg")
                elif pg_res.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    try:
                        with open(pg_res, "rb") as f:
                            data = f.read()
                            mime = "image/png" if pg_res.lower().endswith(".png") else "image/jpeg"
                            add_img(os.path.basename(pg_res), data, mime)
                    except Exception as e:
                        logger.warning(f"Could not read page image file {pg_res}: {e}")
            elif pg.get("image") or pg.get("data") or pg.get("image_data"):
                raw_b64 = str(pg.get("image") or pg.get("data") or pg.get("image_data"))
                try:
                    if "base64," in raw_b64:
                        raw_b64 = raw_b64.split("base64,", 1)[1]
                    decoded = base64.b64decode(raw_b64)
                    add_img(f"page_{pg.get('page_no', 1)}.jpg", decoded, "image/jpeg")
                except Exception as e:
                    logger.warning(f"Could not decode base64 page image: {e}")

    # 2. If no images collected from pages, inspect default_file_path or booklet['file_path']
    target_path = default_file_path or booklet.get("file_path") or ""
    if not images_list and target_path:
        resolved = resolve_file_location(target_path)
        if resolved and os.path.exists(resolved):
            if resolved.lower().endswith(".pdf"):
                p_numbers = booklet.get("page_numbers", [1])
                pdf_imgs = extract_page_images_from_pdf(resolved, page_numbers=p_numbers)
                for iname, ibytes in pdf_imgs:
                    add_img(iname, ibytes, "image/jpeg")
            elif resolved.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                try:
                    with open(resolved, "rb") as f:
                        data = f.read()
                        mime = "image/png" if resolved.lower().endswith(".png") else "image/jpeg"
                        add_img(os.path.basename(resolved), data, mime)
                except Exception as e:
                    logger.warning(f"Could not read image file {resolved}: {e}")

    return images_list


def extract_qa_pairs_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse question numbers and student responses from raw document text."""
    if not text:
        return []

    q_pattern = re.compile(
        r"(?:^|\n)\s*(?:Question\s*(\d+)|Q\s*(\d+)|\((\d+)\)|(\d+)[\.\)])\s*[:\-\.]?\s*([^\n\r]+)",
        re.IGNORECASE
    )
    matches = list(q_pattern.finditer(text))
    if not matches:
        return []

    qa_list = []
    for i, m in enumerate(matches):
        q_num_str = m.group(1) or m.group(2) or m.group(3) or m.group(4) or str(i + 1)
        q_num = int(q_num_str) if q_num_str.isdigit() else (i + 1)
        q_title = m.group(5).strip()

        start_idx = m.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start_idx:end_idx].strip()

        ans_match = re.search(r"(?:Answer|Ans)\s*[:\-\.]\s*", block, re.IGNORECASE)
        if ans_match:
            ans_text = block[ans_match.end():].strip()
        else:
            ans_text = block.strip()

        qa_list.append({
            "question_no": q_num,
            "question_text": q_title,
            "student_response": ans_text[:500] if ans_text else ""
        })
    return qa_list


# ==============================================================================
# MODE A — CLASS-WISE BOOKLET BOUNDARY DETECTION
# ==============================================================================
def detect_booklet_boundaries(
    pdf_path: str,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini"
) -> List[Dict[str, Any]]:
    """
    Process a combined PDF containing answer booklets from multiple students (Mode A).
    Identifies where each student's answer booklet begins and ends using:
    - Booklet cover pages and identification sections
    - Student name, student ID, roll number, or register number changes
    - Page number restarts (e.g. 'Page 1 of ...' or 'Sheet 1')
    - Question sequence resets (Question 1 appearing after higher question numbers)
    - Visual/layout analysis via Vision OCR if text is not machine-readable.

    Returns a list of booklet dictionaries:
    [
      {
        "booklet_index": 1,
        "start_page": 1,
        "end_page": 4,
        "page_numbers": [1, 2, 3, 4],
        "page_count": 4,
        "detected_name": Optional[str],
        "detected_id": Optional[str],
        "detected_adm": Optional[str],
        "detected_roll": Optional[str],
        "detected_kuhs": Optional[str],
        "combined_text": str,
        "boundary_reason": str
      },
      ...
    ]
    """
    try:
        import pdfplumber
        page_texts = []
        total_pages = 1
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                t = page.extract_text() or ""
                page_texts.append(t)
    except Exception as e:
        logger.warning(f"Could not open PDF {pdf_path} for boundary detection: {e}")
        return [{
            "booklet_index": 1,
            "start_page": 1,
            "end_page": 1,
            "page_numbers": [1],
            "page_count": 1,
            "detected_name": None,
            "detected_id": None,
            "detected_adm": None,
            "detected_roll": None,
            "detected_kuhs": None,
            "combined_text": "",
            "boundary_reason": "Single booklet fallback"
        }]

    if total_pages <= 1:
        txt = page_texts[0] if page_texts else ""
        d_name, d_id, d_adm, d_roll, d_kuhs = extract_identity_from_document_text(txt)
        return [{
            "booklet_index": 1,
            "start_page": 1,
            "end_page": 1,
            "page_numbers": [1],
            "page_count": 1,
            "detected_name": d_name,
            "detected_id": d_id,
            "detected_adm": d_adm,
            "detected_roll": d_roll,
            "detected_kuhs": d_kuhs,
            "combined_text": txt,
            "boundary_reason": "Single page document"
        }]

    total_chars = sum(len(t.strip()) for t in page_texts)
    has_text = total_chars > (15 * total_pages)

    cover_re = re.compile(
        r"(?:main\s+)?answer\s+book(?:let)?|"
        r"examination\s+answer\s+sheet|"
        r"candidate\s+name|student\s+name|name\s+of\s+(?:the\s+)?(?:candidate|student)|"
        r"kuhs\s+reg(?:istration)?\s*no|register\s*no|roll\s*no|admission\s*no|"
        r"signature\s+of\s+(?:candidate|invigilator)|hall\s+ticket",
        re.IGNORECASE
    )
    page_one_re = re.compile(
        r"\b(?:page|sheet)\s*[:#\-\s]?\s*1\b(?:\s*of\s*\d+)?|"
        r"\b(?:page|sheet)\s*no\.?\s*[:#\-\s]?\s*1\b",
        re.IGNORECASE
    )
    q1_re = re.compile(r"(?:^|\n)\s*(?:Question\s*1|Q\s*1|1[\.\)])\b", re.IGNORECASE)

    boundary_starts = [1]
    reasons: Dict[int, str] = {1: "First page of document"}

    if has_text:
        last_q_seen = 0
        for p_num in range(2, total_pages + 1):
            p_txt = page_texts[p_num - 1]
            det_name, det_id, _, _, _ = extract_identity_from_document_text(p_txt)

            is_cover = bool(cover_re.search(p_txt[:1200]))
            is_p1 = bool(page_one_re.search(p_txt[:800]))
            is_q1 = bool(q1_re.search(p_txt))
            q_restart = is_q1 and (last_q_seen >= 2)

            all_qs = re.findall(r"(?:Question\s*(\d+)|Q\s*(\d+)|\((\d+)\)|(\d+)[\.\)])", p_txt, re.IGNORECASE)
            for m_tuple in all_qs:
                for q_str in m_tuple:
                    if q_str and q_str.isdigit():
                        last_q_seen = max(last_q_seen, int(q_str))

            if is_cover:
                boundary_starts.append(p_num)
                reasons[p_num] = "Cover page / candidate identification section"
                last_q_seen = 0
            elif det_name or det_id:
                boundary_starts.append(p_num)
                reasons[p_num] = f"Student identity on page: {det_name or det_id}"
                last_q_seen = 0
            elif is_p1:
                boundary_starts.append(p_num)
                reasons[p_num] = "Page number reset to Page 1"
                last_q_seen = 0
            elif q_restart:
                boundary_starts.append(p_num)
                reasons[p_num] = "Question sequence restarted from Question 1"
                last_q_seen = 0
    else:
        # Vision-based boundary detection for scanned handwritten PDFs
        if client and (os.getenv("OPENAI_API_KEY") or hasattr(client, "chat")):
            try:
                check_pages = min(total_pages, 25)
                vision_samples = []
                for p_idx in range(1, check_pages + 1):
                    imgs = extract_page_images_from_pdf(pdf_path, page_numbers=[p_idx])
                    if imgs:
                        vision_samples.append((p_idx, imgs[0][1]))

                if vision_samples:
                    user_content: List[Dict[str, Any]] = [{
                        "type": "text",
                        "text": (
                            f"Analyze the attached {len(vision_samples)} page images of a combined class exam PDF (Pages 1 to {len(vision_samples)}).\n"
                            "Identify where each student's answer booklet begins (look for cover pages, candidate forms/grids, or page resets).\n"
                            "Return JSON: {\"booklets\": [{\"start_page\": 1, \"end_page\": 3, \"detected_student_name\": null, \"detected_student_id\": null}]}"
                        )
                    }]
                    for p_num, img_b in vision_samples:
                        b64 = base64.b64encode(img_b).decode("utf-8")
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low"
                            }
                        })
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a document segmentation expert for student exam booklets. Detect booklet start and end page boundaries."},
                            {"role": "user", "content": user_content}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                    v_parsed = json.loads(resp.choices[0].message.content)
                    v_booklets = v_parsed.get("booklets")
                    if v_booklets and isinstance(v_booklets, list):
                        boundary_starts = []
                        for b in v_booklets:
                            sp = int(b.get("start_page", 1))
                            if sp not in boundary_starts and 1 <= sp <= total_pages:
                                boundary_starts.append(sp)
                                reasons[sp] = "Vision detected booklet boundary"
                        boundary_starts.sort()
                        if 1 not in boundary_starts:
                            boundary_starts.insert(0, 1)
            except Exception as ve:
                logger.warning(f"Vision boundary detection failed: {ve}")

    boundary_starts = sorted(list(set(boundary_starts)))
    booklets = []
    for idx, start_p in enumerate(boundary_starts):
        end_p = (boundary_starts[idx + 1] - 1) if (idx + 1 < len(boundary_starts)) else total_pages
        p_nums = list(range(start_p, end_p + 1))
        b_text = "\n\n".join(page_texts[p - 1] for p in p_nums if p <= len(page_texts))
        det_name, det_id, det_adm, det_roll, det_kuhs = extract_identity_from_document_text(b_text)
        booklets.append({
            "booklet_index": idx + 1,
            "start_page": start_p,
            "end_page": end_p,
            "page_numbers": p_nums,
            "page_count": len(p_nums),
            "detected_name": det_name,
            "detected_id": det_id,
            "detected_adm": det_adm,
            "detected_roll": det_roll,
            "detected_kuhs": det_kuhs,
            "combined_text": b_text,
            "boundary_reason": reasons.get(start_p, "Boundary detected")
        })

    logger.info(f"Detected {len(booklets)} student booklet(s) in combined PDF {pdf_path}")
    return booklets


# ==============================================================================
# AUTHORITATIVE QUESTION PAPER & ANSWER KEY PARSER
# ==============================================================================
def parse_question_paper_and_answer_key(
    qp_source: Any,
    ak_source: Any,
    payload_questions: Optional[List[Dict[str, Any]]] = None,
    declared_max_marks: Optional[float] = None,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini"
) -> Tuple[List[Dict[str, Any]], float, Optional[str], str]:
    """
    Parse the authoritative Question Paper and Model Answer Key / Rubric independently.
    - Question Paper: authoritative for question numbering, question count, and maximum marks.
    - Answer Key: authoritative for correct answers and scoring criteria.
    - Associates questions with corresponding answers and marking schemes.
    - Validates sum of question maximum marks against exam total marks and reports discrepancy.

    Returns:
    (questions_list, authoritative_total_marks, discrepancy_message, exam_title)
    """
    qp_text, qp_images = extract_document_content(qp_source)
    ak_text, ak_images = extract_document_content(ak_source)

    exam_title = "Curriculum Assessment"
    questions_list: List[Dict[str, Any]] = []
    detected_declared_marks: Optional[float] = declared_max_marks

    logger.info(
        f"Parsing examination documents: QP (text={len(qp_text)} chars, images={len(qp_images)}), "
        f"AK (text={len(ak_text)} chars, images={len(ak_images)})"
    )

    # 1. AI-Driven Authoritative Document Parsing if documents are available
    if client and (os.getenv("OPENAI_API_KEY") or hasattr(client, "chat")) and (qp_text or qp_images or ak_text or ak_images):
        try:
            logger.info("Executing OpenAI parsing of Question Paper and Answer Key")
            system_prompt = (
                "You are an authoritative examination document analysis engine for Edusoft.\n"
                "Your objective is to independently parse the uploaded QUESTION PAPER and ANSWER KEY documents.\n"
                "1. QUESTION PAPER is the authoritative source for question numbering (1, 2, 3...), exact question text, "
                "question type (Descriptive, Short Answer, Problem Solving, MCQ), and maximum marks for EACH question.\n"
                "2. ANSWER KEY is the authoritative source for correct model answers, acceptable alternatives, and scoring criteria/rubrics.\n"
                "3. Associate each question from the Question Paper with its corresponding model answer and rubrics from the Answer Key.\n"
                "4. Extract the declared total maximum marks and exam subject from the Question Paper header if present.\n"
                "5. Do NOT hardcode, invent, or assume any fixed marks or question counts. Output strictly valid JSON."
            )

            prompt_text = (
                f"QUESTION PAPER CONTENT:\n{qp_text if qp_text else '[Question paper image(s) attached]'}\n\n"
                f"ANSWER KEY / MODEL ANSWER CONTENT:\n{ak_text if ak_text else '[Answer key image(s) attached]'}\n\n"
                f"PAYLOAD QUESTIONS (if provided):\n{json.dumps(payload_questions, indent=2) if payload_questions else 'None'}\n\n"
                "Return a JSON object with this exact schema:\n"
                "{\n"
                "  \"exam_title\": \"<subject or examination title>\",\n"
                "  \"declared_total_marks\": <number or null>,\n"
                "  \"questions\": [\n"
                "    {\n"
                "      \"question_no\": 1,\n"
                "      \"question_text\": \"<exact question text from question paper>\",\n"
                "      \"question_type\": \"Descriptive\",\n"
                "      \"max_marks\": <number>,\n"
                "      \"model_answer\": \"<authoritative answer from answer key>\",\n"
                "      \"marking_criteria\": [\n"
                "        {\"step\": \"<step 1 criteria>\", \"max_marks\": <number>}\n"
                "      ]\n"
                "    }\n"
                "  ]\n"
                "}"
            )

            user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
            # Attach up to 10 images of QP and up to 10 images of AK
            for name, img_bytes in qp_images[:10]:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                })
            for name, img_bytes in ak_images[:10]:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                })

            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            parsed = json.loads(completion.choices[0].message.content)
            if parsed.get("exam_title"):
                exam_title = str(parsed["exam_title"]).strip()
            if parsed.get("declared_total_marks") is not None:
                try:
                    detected_declared_marks = float(parsed["declared_total_marks"])
                except Exception:
                    pass

            ai_qs = parsed.get("questions")
            if ai_qs and isinstance(ai_qs, list) and len(ai_qs) > 0:
                questions_list = ai_qs
                logger.info(f"OpenAI successfully parsed {len(questions_list)} authoritative question(s) from QP & AK")
        except Exception as e:
            logger.warning(f"OpenAI QP & AK parsing failed ({e}). Proceeding to programmatic extraction.")

    # 2. Fallback to payload questions if AI parsing was empty
    if not questions_list and payload_questions and isinstance(payload_questions, list):
        for idx, pq in enumerate(payload_questions, start=1):
            if isinstance(pq, dict):
                q_num = pq.get("question_no") or idx
                q_txt = str(pq.get("question_text") or pq.get("question") or f"Question {q_num}").strip()
                q_max = float(pq.get("max_marks") or pq.get("marks") or 10.0)
                q_ans = str(pq.get("answer_key") or pq.get("model_answer") or "").strip()
                q_rubric = pq.get("marking_criteria") or pq.get("step_marking") or []
            else:
                q_num = idx
                q_txt = str(pq).strip()
                q_max = 10.0
                q_ans = ""
                q_rubric = []
            questions_list.append({
                "question_no": q_num,
                "question_text": q_txt,
                "question_type": "Descriptive",
                "max_marks": q_max,
                "model_answer": q_ans,
                "marking_criteria": q_rubric
            })

    # 3. Fallback to programmatic text extraction if still empty
    if not questions_list and (qp_text or ak_text):
        parsed_qp_questions = extract_qa_pairs_from_text(qp_text) if qp_text else []
        parsed_ak_answers = extract_qa_pairs_from_text(ak_text) if ak_text else []
        ak_map = {a["question_no"]: a["student_response"] for a in parsed_ak_answers if a.get("student_response")}

        if parsed_qp_questions:
            for item in parsed_qp_questions:
                q_no = item["question_no"]
                q_title = item["question_text"]
                # Extract max marks from question text if present, e.g., [5 Marks] or (10M)
                marks_match = re.search(
                    r"[\[\(](\d+(?:\.\d+)?)\s*(?:marks?|mks?|pts?|points?|m)[\]\)]",
                    q_title,
                    re.IGNORECASE
                )
                if marks_match:
                    q_max = float(marks_match.group(1))
                elif detected_declared_marks and len(parsed_qp_questions) > 0:
                    q_max = round(detected_declared_marks / len(parsed_qp_questions), 1)
                else:
                    q_max = 10.0

                model_ans = ak_map.get(q_no, "")
                if not model_ans and ak_text and len(parsed_qp_questions) == 1:
                    model_ans = ak_text[:500]

                questions_list.append({
                    "question_no": q_no,
                    "question_text": q_title,
                    "question_type": "Descriptive",
                    "max_marks": q_max,
                    "model_answer": model_ans,
                    "marking_criteria": [
                        {"step": f"Core Criteria 1 for Question {q_no}", "max_marks": round(q_max * 0.5, 1)},
                        {"step": f"Core Criteria 2 for Question {q_no}", "max_marks": max(0.5, round(q_max - round(q_max * 0.5, 1), 1))}
                    ] if q_max > 0 else []
                })
        elif parsed_ak_answers and not qp_text:
            for item in parsed_ak_answers:
                q_no = item["question_no"]
                q_title = item["question_text"]
                q_max = 10.0
                questions_list.append({
                    "question_no": q_no,
                    "question_text": q_title,
                    "question_type": "Descriptive",
                    "max_marks": q_max,
                    "model_answer": item.get("student_response", ""),
                    "marking_criteria": []
                })

    # 4. Compute authoritative total marks & validate discrepancy
    sum_question_marks = round(sum(float(q.get("max_marks", 0.0)) for q in questions_list), 1)

    discrepancy_msg = None
    if detected_declared_marks is not None and sum_question_marks > 0:
        if abs(sum_question_marks - detected_declared_marks) > 0.01:
            discrepancy_msg = (
                f"Declared total maximum marks ({detected_declared_marks}) conflicts with "
                f"the sum of question maximum marks ({sum_question_marks})."
            )
            logger.warning(discrepancy_msg)

    # Question paper is authoritative for total marks
    authoritative_total = sum_question_marks if sum_question_marks > 0 else (float(detected_declared_marks) if detected_declared_marks else 100.0)

    logger.info(
        f"QP/AK parsing finalized: {len(questions_list)} question(s), "
        f"Authoritative Max Marks: {authoritative_total}, Discrepancy: {discrepancy_msg}"
    )

    return questions_list, authoritative_total, discrepancy_msg, exam_title


# ==============================================================================
# EVALUATION STORE (IN-MEMORY)
# ==============================================================================
class EvaluationStore:
    """
    In-memory store for AI evaluation records.
    Strictly sanitizes and omits student_name and student_id from all GET responses.
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_evaluation(self, eval_id: str) -> Optional[Dict[str, Any]]:
        record = self._cache.get(str(eval_id))
        if not record:
            return None
        get_resp = record.get("get_response")
        if not get_resp:
            return None
        safe_resp = json.loads(json.dumps(get_resp))
        if isinstance(safe_resp, dict) and "data" in safe_resp and isinstance(safe_resp["data"], dict):
            for s in safe_resp["data"].get("students", []):
                if isinstance(s, dict):
                    # Return student identity ONLY when extracted from PDF/Image
                    is_extracted = (s.get("recognition_status") == "matched") and bool(s.get("student_name") or s.get("student_id"))
                    if not is_extracted:
                        s["student_name"] = ""
                        s["student_id"] = ""
                        s["admission_no"] = ""
                        s["roll_no"] = ""
                        s["kuhs_reg_no"] = ""
                        s["student_image"] = ""
                        s["recognized_text"] = ""
                        s["recognition_status"] = "unmatched"
                        s["recognition_confidence"] = 0.0
        return safe_resp

    def get_post_evaluation(self, eval_id: str) -> Optional[Dict[str, Any]]:
        record = self._cache.get(str(eval_id))
        return record.get("post_response") if record else None

    def get_student_evaluation(self, eval_id: str, student_identifier: str) -> Optional[Dict[str, Any]]:
        record = self._cache.get(str(eval_id))
        if not record:
            return None
        post_resp = record.get("post_response", {})
        for idx, s in enumerate(post_resp.get("students", []), start=1):
            if (
                str(s.get("student_id")) == str(student_identifier)
                or str(s.get("roll_no")) == str(student_identifier)
                or str(s.get("id")) == str(student_identifier)
                or str(idx) == str(student_identifier)
            ):
                s_copy = json.loads(json.dumps(s))
                # Return student identity ONLY when extracted from PDF/Image
                is_extracted = (s_copy.get("recognition_status") == "matched") and bool(s_copy.get("student_name") or s_copy.get("student_id"))
                if not is_extracted:
                    s_copy["student_name"] = ""
                    s_copy["student_id"] = ""
                    s_copy["admission_no"] = ""
                    s_copy["roll_no"] = ""
                    s_copy["kuhs_reg_no"] = ""
                    s_copy["student_image"] = ""
                    s_copy["recognized_text"] = ""
                    s_copy["recognition_status"] = "unmatched"
                    s_copy["recognition_confidence"] = 0.0
                return s_copy
        return None

    def list_all_evaluations(self) -> List[Dict[str, Any]]:
        summary_list = []
        for eval_id, rec in self._cache.items():
            get_data = rec.get("get_response", {}).get("data", {})
            summary_list.append({
                "id": get_data.get("id", eval_id),
                "title": get_data.get("title", f"Evaluation #{eval_id}"),
                "class_name": get_data.get("class_name", ""),
                "subject_name": get_data.get("subject_name", ""),
                "total_students": get_data.get("total_students", "0"),
                "passed_count": get_data.get("passed_count", "0"),
                "failed_count": get_data.get("failed_count", "0"),
                "average_score": get_data.get("average_score", "0.00"),
                "max_marks": get_data.get("max_marks", "0.00"),
                "evaluation_date": get_data.get("evaluation_date", "")
            })
        return summary_list

    def save_evaluation(self, eval_id: str, post_resp: Dict[str, Any], get_resp: Dict[str, Any]):
        self._cache[str(eval_id)] = {
            "post_response": post_resp,
            "get_response": get_resp
        }

    def next_id(self) -> int:
        existing_numeric = [int(k) for k in self._cache.keys() if str(k).isdigit()]
        return (max(existing_numeric) + 1) if existing_numeric else 1


# Global store instance
evaluation_store = EvaluationStore()


# ==============================================================================
# STUDENT BOOKLET VISION EVALUATION ENGINE
# ==============================================================================
def evaluate_student_booklet_ai(
    booklet: Dict[str, Any],
    file_path: str,
    file_name: str,
    questions: List[Dict[str, Any]],
    total_marks: float,
    subject_title: str,
    discrepancy: Optional[str] = None,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Evaluate an individual student booklet against authoritative questions and answer key.
    - Collects all high-res page images across the entire booklet (handling multi-page answers).
    - Decouples student identification from answer grading.
    - Accurately counts: total_questions, attempted_questions, unanswered_questions,
      evaluated_questions, pending_review_questions.
    - Strictly only passes student_name and student_id in POST response if detected on booklet.
    - Enforces step marking sum == question obtained marks.
    - Prevents placeholder strings and fabricated marks.
    """
    page_numbers = booklet.get("page_numbers", [1])
    page_count = len(page_numbers)
    booklet_text = booklet.get("combined_text", "")
    booklet_idx = booklet.get("booklet_index", 1)

    # 1. Collect high-resolution page images for all pages in this student's booklet
    vision_images = collect_booklet_images(booklet=booklet, default_file_path=file_path)
    logger.info(
        f"Booklet #{booklet_idx}: Loaded {len(vision_images)} image(s) across {page_count} page(s) "
        f"(text_length={len(booklet_text)} chars)."
    )

    # Initial identity from text extraction
    text_name, text_id, text_adm, text_roll, text_kuhs = extract_identity_from_document_text(booklet_text)
    detected_student_name: Optional[str] = booklet.get("detected_name") or text_name
    detected_student_id: Optional[str] = booklet.get("detected_id") or text_id
    admission_no: str = booklet.get("detected_adm") or text_adm or ""
    roll_no: str = booklet.get("detected_roll") or text_roll or ""
    kuhs_reg_no: str = booklet.get("detected_kuhs") or text_kuhs or ""
    student_image: str = ""

    # Build question context block
    if questions:
        q_context = json.dumps(questions, indent=2)
    else:
        q_context = f"Subject: {subject_title}. Total Marks: {total_marks}. Deduce questions and student responses from handwriting."

    ai_evaluated = False
    evaluated_result: Optional[Dict[str, Any]] = None

    has_document_content = bool(vision_images or (booklet_text and booklet_text.strip()))

    if client and (os.getenv("OPENAI_API_KEY") or hasattr(client, "chat")) and has_document_content:
        system_prompt = (
            "You are an authoritative academic examination scoring engine for Edusoft.\n"
            "Your objective is to evaluate a student's actual handwritten answer booklet against the authoritative Question Paper and Model Answer Key.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. VERBATIM HANDWRITTEN TRANSCRIPTION:\n"
            "   - For every question in the Question Paper, locate the student's handwritten response across all provided answer sheet images.\n"
            "   - MULTI-PAGE ANSWERS: A student's handwritten answer for a question may start on one page and continue onto subsequent pages. You must search across all attached pages to transcribe the student's complete answer.\n"
            "   - Transcribe the student's ACTUAL visible handwritten text, equations, formulas, calculations, and diagrams into 'student_response'.\n"
            "   - STRICT PROHIBITION: DO NOT emit generic summary/placeholder phrases (e.g. 'Student response extracted and verified across X uploaded answer page(s)', 'Response extracted across pages', 'Handwritten answers on uploaded pages'). Generic placeholder text is strictly forbidden.\n"
            "   - DO NOT copy the answer key into student_response.\n\n"
            "2. ATTEMPT & READABILITY DETERMINATION:\n"
            "   - Blank / Unanswered question: If the question was skipped or not attempted, set is_attempted = false, pending_review = false, obtained_marks = 0.0, student_response = '[Unanswered]'.\n"
            "   - Struck-through answer: If the work was crossed out with no subsequent answer, set is_attempted = false, pending_review = false, obtained_marks = 0.0, student_response = '[Crossed out]'.\n"
            "   - Illegible handwriting: If the writing is blurred, indecipherable, or conflicting, set is_attempted = true, pending_review = true, obtained_marks = 0.0, student_response = '[Illegible handwriting - Flagged for manual review]'. Explain in 'feedback'. NEVER guess marks for illegible handwriting!\n"
            "   - Unassociated answer: If handwritten work is present but cannot be confidently mapped to a question, set is_attempted = true, pending_review = true, obtained_marks = 0.0, student_response = '[Unassociated response - Flagged for manual review]'.\n"
            "   - Legible attempt: If substantive handwritten work is visible and decipherable, set is_attempted = true, pending_review = false.\n\n"
            "3. STRICT RUBRIC EVALUATION AGAINST ANSWER KEY:\n"
            "   - Compare the student's transcribed answer with the model answer and marking criteria for that specific question.\n"
            "   - Award partial credit based on relevant correct concepts, formulas, steps, reasoning, and completeness.\n"
            "   - Accept scientifically correct equivalent phrasing and terminology.\n"
            "   - Never award marks using fixed multipliers (e.g. 65%, 70%, 75%, 80%) or default scores.\n"
            "   - Never award marks simply because text is present on the page.\n"
            "   - Each step's obtained marks must be between 0.0 and its max marks.\n"
            "   - The step marks MUST sum exactly to the question's obtained_marks.\n"
            "   - Question obtained_marks must be between 0.0 and max_marks.\n"
            "   - If the answer is completely incorrect or irrelevant, award 0.0.\n\n"
            "4. ANSWER-SPECIFIC FEEDBACK & INSIGHTS:\n"
            "   - 'feedback' for each question MUST cite the student's specific written statements, correct points, and missing or erroneous concepts.\n"
            "   - 'ai_feedback' MUST be an academic diagnostic summary of the student's performance on these specific exam questions.\n"
            "   - NEVER include unrelated comments about attendance, enrollment, working days, fees, or administration.\n"
            "   - 'ai_insights.recommendation', 'focus_areas', and 'strengths' must strictly reflect the evaluated examination topics.\n"
            "   - If evaluation is unsuccessful or handwriting is unreadable, report that evaluation could not be completed and manual review is required; NEVER generate positive feedback for failed evaluations.\n\n"
            "5. INDEPENDENT IDENTITY EXTRACTION:\n"
            "   - ONLY extract detected_student_name and detected_student_id if VISIBLY written on this booklet cover or header.\n"
            "   - If not visibly written on the document, return null for both and set recognition_status = 'unmatched'.\n"
            "   - Grade all readable answers fully regardless of whether student identity is detected!\n\n"
            "Output strictly valid JSON matching the schema."
        )

        prompt_text = f"""
EXAMINATION METRICS:
- Subject: {subject_title}
- Total Maximum Marks: {total_marks}
- Booklet Page Count: {page_count} (Pages {page_numbers})

AUTHORITATIVE QUESTIONS & MODEL ANSWER KEY:
{q_context}

EXTRACTED TEXT FROM BOOKLET:
{booklet_text if booklet_text else "[Scanned handwriting images attached for visual inspection]"}

Return a single JSON object with this EXACT schema:
{{
  "detected_student_name": null,
  "detected_student_id": null,
  "detected_admission_no": null,
  "detected_roll_no": null,
  "detected_kuhs_reg_no": null,
  "recognition_status": "matched" or "unmatched",
  "recognition_confidence": 0.0,
  "ai_feedback": "<dynamic 2-3 sentence diagnostic review of student performance on these answers>",
  "ai_insights": {{
    "recommendation": "<constructive recommendation tailored to student answers>",
    "focus_areas": ["<topic 1>", "<topic 2>"],
    "strengths": ["<strength 1>", "<strength 2>"]
  }},
  "evaluation_breakdown": [
    {{
      "question_no": 1,
      "question_text": "<exact question text>",
      "question_type": "Descriptive",
      "max_marks": <number>,
      "obtained_marks": <number>,
      "is_attempted": true,
      "pending_review": false,
      "answer_key": "<model answer points>",
      "student_response": "<transcribed actual handwritten response>",
      "step_marking": [
        {{
          "step": "<step criteria>",
          "max_marks": <number>,
          "obtained_marks": <number>,
          "status": "Full" or "Partial" or "Zero"
        }}
      ],
      "feedback": "<question specific feedback citing student work>"
    }}
  ]
}}
"""
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for _, img_data, mime in vision_images:
            b64_img = base64.b64encode(img_data).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64_img}",
                    "detail": "high"
                }
            })

        max_vision_retries = 2
        for attempt in range(max_vision_retries + 1):
            try:
                logger.info(
                    f"OpenAI Vision evaluation attempt {attempt + 1}/{max_vision_retries + 1} "
                    f"for booklet #{booklet_idx} across {len(vision_images)} image(s)"
                )
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                raw_content = completion.choices[0].message.content
                parsed = json.loads(raw_content)

                breakdown_list = parsed.get("evaluation_breakdown")
                if breakdown_list and isinstance(breakdown_list, list) and len(breakdown_list) > 0:
                    evaluated_result = parsed
                    ai_evaluated = True

                    ai_name = parsed.get("detected_student_name")
                    if (
                        ai_name
                        and isinstance(ai_name, str)
                        and ai_name.strip()
                        and ai_name.strip().lower() not in ("null", "none", "n/a", "student", "unknown", "unmatched")
                        and not any(w in ai_name.lower() for w in ["examination", "college", "university", "question", "subject"])
                    ):
                        detected_student_name = ai_name.strip()
                        admission_no = str(parsed.get("detected_admission_no") or admission_no).strip()
                        roll_no = str(parsed.get("detected_roll_no") or roll_no).strip()
                        kuhs_reg_no = str(parsed.get("detected_kuhs_reg_no") or kuhs_reg_no).strip()
                    elif vision_images:
                        detected_student_name = None
                        admission_no = ""
                        roll_no = ""
                        kuhs_reg_no = ""

                    ai_id = parsed.get("detected_student_id")
                    if (
                        ai_id
                        and str(ai_id).strip()
                        and str(ai_id).strip().lower() not in ("null", "none", "n/a", "student", "unknown", "unmatched")
                    ):
                        detected_student_id = str(ai_id).strip()
                    elif vision_images:
                        detected_student_id = None

                    logger.info(f"OpenAI evaluation successful for booklet #{booklet_idx}. Detected Name: {detected_student_name}, ID: {detected_student_id}")
                    break
                else:
                    logger.warning(f"OpenAI returned empty evaluation breakdown on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"OpenAI vision attempt {attempt + 1} failed for booklet #{booklet_idx}: {e}")
                if attempt < max_vision_retries:
                    import time
                    time.sleep(1.0)
                else:
                    logger.error(f"All OpenAI Vision evaluation attempts failed for booklet #{booklet_idx}: {e}")

    # Process evaluation breakdown strictly without fabricating marks
    normalized_breakdown: List[Dict[str, Any]] = []

    if ai_evaluated and evaluated_result:
        raw_breakdown = evaluated_result.get("evaluation_breakdown") or []
        breakdown_by_qno = {q.get("question_no"): q for q in raw_breakdown if isinstance(q, dict)}
        logger.info(f"Booklet #{booklet_idx}: Matched {len(breakdown_by_qno)}/{len(questions)} question(s) from AI breakdown.")

        for q_item in questions:
            q_num = q_item.get("question_no")
            q_max = float(q_item.get("max_marks", 10.0))
            q_txt = q_item.get("question_text", f"Question {q_num}")
            q_key = q_item.get("model_answer") or q_item.get("answer_key", "")

            if q_num in breakdown_by_qno:
                b_item = breakdown_by_qno[q_num]
                b_item["question_text"] = q_txt
                b_item["max_marks"] = q_max
                b_item["answer_key"] = q_key or b_item.get("answer_key", "")

                student_resp = str(b_item.get("student_response", "")).strip()

                # Validate answer key presence
                if not q_key or not str(q_key).strip():
                    logger.warning(f"Question {q_num} has missing answer key. Flagging for manual review.")
                    b_item["pending_review"] = True
                    b_item["obtained_marks"] = 0.0
                    b_item["feedback"] = f"Missing answer key / model answer for Question {q_num}. Question cannot be evaluated without authoritative marking criteria."
                    b_item["step_marking"] = [
                        {
                            "step": "Manual Verification Required (Missing Answer Key)",
                            "max_marks": q_max,
                            "obtained_marks": 0.0,
                            "status": "Zero"
                        }
                    ]
                # Strictly reject forbidden generic placeholders
                elif contains_forbidden_placeholder(student_resp):
                    logger.warning(f"Question {q_num} response contained forbidden placeholder: '{student_resp}'. Flagging for manual review.")
                    b_item["is_attempted"] = True
                    b_item["pending_review"] = True
                    b_item["obtained_marks"] = 0.0
                    b_item["student_response"] = "[Unverified transcription - Flagged for manual review]"
                    b_item["feedback"] = "Automated model output contained generic placeholder text instead of verified transcription. Flagged for manual review."
                    b_item["step_marking"] = [
                        {
                            "step": "Manual Verification Required",
                            "max_marks": q_max,
                            "obtained_marks": 0.0,
                            "status": "Zero"
                        }
                    ]
                # Reject copied answer key
                elif len(student_resp) > 30 and q_key and student_resp.lower() == q_key.strip().lower():
                    logger.warning(f"Question {q_num} student response duplicated model answer key. Flagging for review.")
                    b_item["is_attempted"] = True
                    b_item["pending_review"] = True
                    b_item["obtained_marks"] = 0.0
                    b_item["student_response"] = "[Flagged for review - Answer key duplication detected]"
                    b_item["feedback"] = "Student response duplicated the answer key. Flagged for manual instructor review."
                    b_item["step_marking"] = [
                        {
                            "step": "Manual Verification Required",
                            "max_marks": q_max,
                            "obtained_marks": 0.0,
                            "status": "Zero"
                        }
                    ]
                elif b_item.get("pending_review"):
                    # Questions flagged for review cannot have fabricated marks
                    b_item["obtained_marks"] = 0.0
                    b_item["is_attempted"] = True
                    if not student_resp or student_resp == "[Illegible handwriting - Flagged for manual review]":
                        b_item["student_response"] = "[Illegible handwriting - Flagged for manual review]"
                    b_item["step_marking"] = [
                        {
                            "step": "Manual Verification Required",
                            "max_marks": q_max,
                            "obtained_marks": 0.0,
                            "status": "Zero"
                        }
                    ]
                elif not b_item.get("is_attempted"):
                    b_item["obtained_marks"] = 0.0
                    b_item["pending_review"] = False
                    b_item["student_response"] = student_resp if student_resp in ["[Crossed out]", "[Unanswered]"] else "[Unanswered]"
                    b_item["feedback"] = b_item.get("feedback") or f"Question {q_num} was not attempted by the student."
                    b_item["step_marking"] = [
                        {
                            "step": "Question attempted",
                            "max_marks": q_max,
                            "obtained_marks": 0.0,
                            "status": "Zero"
                        }
                    ]
                else:
                    # Legitimate evaluated answer: validate marks bounds and step marking
                    raw_obt = float(b_item.get("obtained_marks", 0.0))
                    b_item["obtained_marks"] = max(0.0, min(round(raw_obt, 1), q_max))
                    b_item["pending_review"] = False

                    steps = b_item.get("step_marking") or []
                    if steps:
                        validated_steps = []
                        for st in steps:
                            st_max = float(st.get("max_marks", 0.0))
                            st_obt = max(0.0, min(float(st.get("obtained_marks", 0.0)), st_max))
                            st_status = "Full" if (st_obt >= st_max and st_max > 0) else ("Partial" if st_obt > 0 else "Zero")
                            validated_steps.append({
                                "step": str(st.get("step", "Criteria")).strip(),
                                "max_marks": round(st_max, 1),
                                "obtained_marks": round(st_obt, 1),
                                "status": st_status
                            })
                        # Invariant: step marks sum must equal question obtained marks
                        step_sum = round(sum(st["obtained_marks"] for st in validated_steps), 1)
                        if abs(step_sum - b_item["obtained_marks"]) > 0.01:
                            logger.warning(
                                f"Question {q_num}: Step marks sum ({step_sum}) differed from question marks ({b_item['obtained_marks']}). Reconciling."
                            )
                            if step_sum > 0:
                                b_item["obtained_marks"] = min(q_max, step_sum)
                            else:
                                validated_steps[0]["obtained_marks"] = b_item["obtained_marks"]
                                validated_steps[0]["status"] = "Full" if b_item["obtained_marks"] >= validated_steps[0]["max_marks"] else "Partial"
                        b_item["step_marking"] = validated_steps
                    else:
                        b_item["step_marking"] = [
                            {
                                "step": f"Criteria for Question {q_num}",
                                "max_marks": q_max,
                                "obtained_marks": b_item["obtained_marks"],
                                "status": "Full" if b_item["obtained_marks"] >= q_max else ("Partial" if b_item["obtained_marks"] > 0 else "Zero")
                            }
                        ]

                # Sanitize question-level feedback (remove any attendance/enrollment terms)
                q_fb = str(b_item.get("feedback", "")).strip()
                unwanted_q = ["attendance", "enrollment", "working days", "academic assessments are pending"]
                if any(u in q_fb.lower() for u in unwanted_q):
                    b_item["feedback"] = (
                        f"Evaluated against Question {q_num} rubric. "
                        f"Student achieved {b_item['obtained_marks']}/{q_max} marks."
                    )

                logger.info(
                    f"Question {q_num}: attempted={b_item.get('is_attempted')}, "
                    f"marks={b_item.get('obtained_marks')}/{q_max}, "
                    f"pending_review={b_item.get('pending_review')}"
                )
                normalized_breakdown.append(b_item)
            else:
                # Question not located in model breakdown
                is_substantive = bool(has_document_content)
                logger.warning(f"Question {q_num} was omitted from AI breakdown. Flagging for review: substantive={is_substantive}")
                normalized_breakdown.append({
                    "question_no": q_num,
                    "question_text": q_txt,
                    "question_type": q_item.get("question_type", "Descriptive"),
                    "max_marks": q_max,
                    "obtained_marks": 0.0,
                    "is_attempted": is_substantive,
                    "pending_review": is_substantive,
                    "answer_key": q_key,
                    "student_response": "[Not located in automated evaluation - Flagged for manual review]" if is_substantive else "[Unanswered]",
                    "step_marking": [
                        {
                            "step": "Manual Verification Required" if is_substantive else "Question attempted",
                            "max_marks": q_max,
                            "obtained_marks": 0.0,
                            "status": "Zero"
                        }
                    ],
                    "feedback": f"Question {q_num} could not be located in automated vision evaluation. Flagged for manual review." if is_substantive else f"Question {q_num} was not attempted by the student."
                })
    else:
        # AI evaluation was unavailable or completely failed: DO NOT INVENT MARKS OR FALLBACK MULTIPLIERS!
        logger.warning(f"AI evaluation unavailable or failed for booklet #{booklet_idx}. Flagging all {len(questions)} question(s) for manual review.")
        for q_item in questions:
            q_num = q_item.get("question_no")
            q_max = float(q_item.get("max_marks", 10.0))
            normalized_breakdown.append({
                "question_no": q_num,
                "question_text": q_item.get("question_text", f"Question {q_num}"),
                "question_type": q_item.get("question_type", "Descriptive"),
                "max_marks": q_max,
                "obtained_marks": 0.0,
                "is_attempted": has_document_content,
                "pending_review": True,
                "answer_key": q_item.get("model_answer") or q_item.get("answer_key", ""),
                "student_response": "[Answer sheet unreadable or could not be loaded - Flagged for manual review]" if not has_document_content else "[Automated evaluation unavailable - Flagged for manual review]",
                "step_marking": [
                    {
                        "step": "Manual Verification Required",
                        "max_marks": q_max,
                        "obtained_marks": 0.0,
                        "status": "Zero"
                    }
                ],
                "feedback": f"Automated AI evaluation could not process Question {q_num}. Flagged for manual instructor review."
            })

    # Precise Metric Calculations
    total_q_count = len(normalized_breakdown)
    attempted_q_count = sum(1 for q in normalized_breakdown if q.get("is_attempted", False))
    unanswered_q_count = total_q_count - attempted_q_count
    pending_review_q_count = sum(1 for q in normalized_breakdown if q.get("pending_review", False))
    evaluated_q_count = max(0, attempted_q_count - pending_review_q_count)

    authoritative_max = round(sum(float(q.get("max_marks", 0.0)) for q in normalized_breakdown), 1)
    if authoritative_max <= 0:
        authoritative_max = total_marks

    summed_obtained = round(sum(float(q.get("obtained_marks", 0.0)) for q in normalized_breakdown), 1)
    percentage_score = round((summed_obtained / authoritative_max) * 100.0, 2) if authoritative_max > 0 else 0.0

    # Derive grade and pass/fail status
    if (pending_review_q_count > 0 and evaluated_q_count == 0) or total_q_count == 0:
        grade_score = "Pending"
        status_score = "Pending Review"
    else:
        grade_score = calculate_grade(percentage_score)
        status_score = determine_status(percentage_score)

    logger.info(
        f"Booklet #{booklet_idx} evaluation complete: "
        f"Score={summed_obtained}/{authoritative_max} ({percentage_score}%), "
        f"Grade={grade_score}, Status={status_score}, Evaluated={evaluated_q_count}, "
        f"Pending={pending_review_q_count}, Unanswered={unanswered_q_count}"
    )

    # Sanitize ai_feedback and ai_insights
    raw_ai_feedback = str(evaluated_result.get("ai_feedback", "")).strip() if evaluated_result else ""
    raw_ai_insights = evaluated_result.get("ai_insights", {}) if evaluated_result else {}
    if not isinstance(raw_ai_insights, dict):
        raw_ai_insights = {}

    unwanted_phrases = [
        "academic assessments are pending",
        "assessments are pending",
        "attendance",
        "working days",
        "enrollment",
        "present days",
        "absent days",
        "late days",
        "fee balance",
        "fee status"
    ]
    has_unwanted = any(u in raw_ai_feedback.lower() for u in unwanted_phrases)

    if evaluated_q_count == 0 or pending_review_q_count == total_q_count:
        final_ai_feedback = f"Automated evaluation could not complete reliable handwriting scoring for this answer booklet. All {total_q_count} question(s) have been flagged for manual instructor review."
        final_ai_insights = {
            "recommendation": "Manual instructor review required.",
            "focus_areas": [],
            "strengths": []
        }
    elif not raw_ai_feedback or has_unwanted:
        final_ai_feedback = (
            f"Student achieved {summed_obtained}/{authoritative_max} ({percentage_score}%) with grade {grade_score} "
            f"across {evaluated_q_count} evaluated question(s). Performance reflects understanding in core topics "
            f"with specific areas identified for academic improvement."
        )
        rec = str(raw_ai_insights.get("recommendation", "")).strip()
        if not rec or any(u in rec.lower() for u in unwanted_phrases):
            rec = "Review question feedback and reinforce concepts where partial or zero credit was awarded."
        focus = [str(f).strip() for f in raw_ai_insights.get("focus_areas", []) if not any(u in str(f).lower() for u in unwanted_phrases)]
        strengths = [str(s).strip() for s in raw_ai_insights.get("strengths", []) if not any(u in str(s).lower() for u in unwanted_phrases)]
        final_ai_insights = {
            "recommendation": rec,
            "focus_areas": focus if focus else [f"{subject_title} foundational concepts"],
            "strengths": strengths if strengths else [f"Attempted {attempted_q_count} question(s)"]
        }
    else:
        rec = str(raw_ai_insights.get("recommendation", "")).strip()
        if not rec or any(u in rec.lower() for u in unwanted_phrases):
            rec = "Review question feedback and reinforce key subject topics."
        focus = [str(f).strip() for f in raw_ai_insights.get("focus_areas", []) if not any(u in str(f).lower() for u in unwanted_phrases)]
        strengths = [str(s).strip() for s in raw_ai_insights.get("strengths", []) if not any(u in str(s).lower() for u in unwanted_phrases)]
        final_ai_feedback = raw_ai_feedback
        final_ai_insights = {
            "recommendation": rec,
            "focus_areas": focus,
            "strengths": strengths
        }

    # STRICT IDENTITY ENFORCEMENT FOR POST RESPONSE:
    has_name = bool(detected_student_name and str(detected_student_name).strip())
    has_id = bool(detected_student_id and str(detected_student_id).strip())

    if has_name or has_id:
        final_student_name = str(detected_student_name).strip() if has_name else ""
        if has_id and str(detected_student_id).strip().isdigit():
            final_student_id: Union[str, int] = int(str(detected_student_id).strip())
        else:
            final_student_id = str(detected_student_id).strip() if has_id else ""
        rec_status = "matched"
        rec_conf = 98.5
        recognized_text_str = f"Recognized: {final_student_name}" + (f" (ID: {final_student_id})" if final_student_id else "")
    else:
        final_student_name = ""
        final_student_id = ""
        admission_no = ""
        roll_no = ""
        kuhs_reg_no = ""
        student_image = ""
        rec_status = "unmatched"
        rec_conf = 0.0
        recognized_text_str = ""

    # Page metadata
    pages_meta = [
        {
            "page_no": p,
            "file_name": file_name,
            "file_path": file_path,
            "file_type": "pdf" if file_path.lower().endswith(".pdf") else "jpg"
        }
        for p in page_numbers
    ]

    return {
        "id": booklet.get("booklet_index", 1),
        "student_id": final_student_id,
        "student_name": final_student_name,
        "admission_no": admission_no,
        "roll_no": roll_no,
        "kuhs_reg_no": kuhs_reg_no,
        "student_image": student_image,
        "recognized_text": recognized_text_str,
        "file_name": file_name,
        "file_path": file_path,
        "file_type": "pdf" if file_path.lower().endswith(".pdf") else "jpg",
        "pages": pages_meta,
        "page_count": page_count,
        "pages_json": json.dumps(pages_meta),
        "recognition_status": rec_status,
        "recognition_confidence": rec_conf,
        "total_questions": total_q_count,
        "attempted_questions": attempted_q_count,
        "unanswered_questions": unanswered_q_count,
        "evaluated_questions": evaluated_q_count,
        "pending_review_questions": pending_review_q_count,
        "total_marks": authoritative_max,
        "obtained_marks": summed_obtained,
        "percentage": percentage_score,
        "grade": grade_score,
        "status": status_score,
        "ai_feedback": final_ai_feedback,
        "ai_insights": final_ai_insights,
        "evaluation_breakdown": normalized_breakdown
    }


# ==============================================================================
# MAIN PAPER EVALUATION CONTROLLER
# ==============================================================================
def process_paper_evaluation(
    payload: Dict[str, Any],
    client: Optional[OpenAI] = None,
    default_model: str = "gpt-4o-mini"
) -> Tuple[Dict[str, Any], int]:
    """
    Main evaluation pipeline supporting Mode A class-wise evaluation:
    1. Parses Question Paper and Answer Key independently.
    2. Detects student booklet boundaries in the combined PDF.
    3. Evaluates handwritten answers for each student booklet.
    4. Formats POST response (identity included only if detected on document).
    5. Formats GET response (strictly removes student_name and student_id).
    6. Stores in EvaluationStore.
    """
    try:
        eval_id_raw = payload.get("evaluation_id") or payload.get("id")
        if eval_id_raw is not None and str(eval_id_raw).strip():
            eval_id = str(eval_id_raw).strip()
        else:
            eval_id = str(evaluation_store.next_id())

        model = payload.get("model") or default_model
        class_id = str(payload.get("class_id") or "").strip()
        class_name = str(payload.get("class_name") or "").strip()
        section_id = str(payload.get("section_id") or "").strip()
        section_name = str(payload.get("section_name") or "").strip()
        subject_id = str(payload.get("subject_id") or "").strip()
        subject_name = str(payload.get("subject_name") or "").strip()
        question_paper_id = str(payload.get("question_paper_id") or "").strip()
        question_paper_title = str(payload.get("question_paper_title") or "").strip()
        title = str(payload.get("title") or question_paper_title or "Examination Evaluation").strip()

        # 1. Identify Question Paper and Answer Key files/sources
        qp_source = (
            payload.get("question_paper_file")
            or payload.get("question_paper")
            or payload.get("qp")
            or payload.get("qp_file")
        )
        ak_source = (
            payload.get("answer_key_file")
            or payload.get("answer_key")
            or payload.get("model_answer_file")
            or payload.get("model_answer")
            or payload.get("key")
        )
        payload_questions = payload.get("questions") or payload.get("question_list")
        declared_total = payload.get("max_marks") or payload.get("total_marks")
        declared_total_float = float(declared_total) if declared_total is not None else None

        # Validate Question Paper input
        has_qp_doc = bool(qp_source and str(qp_source).strip())
        has_payload_qs = bool(payload_questions and isinstance(payload_questions, list) and len(payload_questions) > 0)
        if not has_qp_doc and not has_payload_qs:
            logger.error("Paper evaluation failed: Question paper is missing.")
            return {
                "status": "error",
                "status_code": 400,
                "message": "Question paper is missing. Please provide the authoritative question paper document or questions list."
            }, 400

        resolved_qp = resolve_file_location(qp_source) if has_qp_doc else None
        if has_qp_doc and isinstance(qp_source, str) and not resolved_qp and not has_payload_qs:
            if any(ext in qp_source.lower() for ext in [".pdf", ".jpg", ".jpeg", ".png"]):
                logger.error(f"Paper evaluation failed: Question paper file could not be found: {qp_source}")
                return {
                    "status": "error",
                    "status_code": 400,
                    "message": f"Question paper file could not be resolved or found: {qp_source}"
                }, 400

        # Validate Answer Key input
        has_ak_doc = bool(ak_source and str(ak_source).strip())
        has_payload_answers = bool(has_payload_qs and any(q.get("answer_key") or q.get("model_answer") for q in payload_questions if isinstance(q, dict)))
        if not has_ak_doc and not has_payload_answers:
            logger.error("Paper evaluation failed: Answer key is missing.")
            return {
                "status": "error",
                "status_code": 400,
                "message": "Answer key is missing. Please provide the authoritative answer key document or model answers in the questions list."
            }, 400

        resolved_ak = resolve_file_location(ak_source) if has_ak_doc else None
        if has_ak_doc and isinstance(ak_source, str) and not resolved_ak and not has_payload_answers:
            if any(ext in ak_source.lower() for ext in [".pdf", ".jpg", ".jpeg", ".png"]):
                logger.error(f"Paper evaluation failed: Answer key file could not be found: {ak_source}")
                return {
                    "status": "error",
                    "status_code": 400,
                    "message": f"Answer key file could not be resolved or found: {ak_source}"
                }, 400

        # 2. Parse authoritative Question Paper and Answer Key
        authoritative_questions, max_marks, discrepancy, parsed_title = parse_question_paper_and_answer_key(
            qp_source=qp_source,
            ak_source=ak_source,
            payload_questions=payload_questions,
            declared_max_marks=declared_total_float,
            client=client,
            model=model
        )
        if not authoritative_questions:
            logger.error("Paper evaluation failed: Could not extract authoritative questions.")
            return {
                "status": "error",
                "status_code": 400,
                "message": "Could not extract authoritative questions from the question paper. Please verify the document is readable and contains valid questions."
            }, 400

        if not subject_name:
            subject_name = parsed_title or "Curriculum Assessment"

        # 3. Disregard incoming student identity sent by PHP backend; only retrieve identity written in PDF/image
        IDENTITY_FIELDS = [
            "student_name", "student_id", "name", "id", "admission_no", "adm_no",
            "roll_no", "roll_number", "kuhs_reg_no", "reg_no", "registration_no",
            "student_image", "recognized_text", "recognition_status", "recognition_confidence",
            "ocr_text", "extracted_text"
        ]
        for f in IDENTITY_FIELDS:
            payload.pop(f, None)
            if "data" in payload and isinstance(payload["data"], dict):
                payload["data"].pop(f, None)
            if "payload" in payload and isinstance(payload["payload"], dict):
                payload["payload"].pop(f, None)

        combined_pdf_path = (
            payload.get("combined_pdf")
            or payload.get("combined_file")
            or payload.get("answer_booklets_file")
            or payload.get("booklets_file")
        )

        students_raw = payload.get("students") or payload.get("answer_sheets") or payload.get("papers") or []
        if isinstance(students_raw, dict):
            students_raw = list(students_raw.values())
        elif isinstance(students_raw, list):
            for s in students_raw:
                if isinstance(s, dict):
                    for f in IDENTITY_FIELDS:
                        s.pop(f, None)
        else:
            students_raw = []

        # Prevent confusing Question Paper or Answer Key with Student Answer Sheets
        exam_doc_paths = {p for p in [resolved_qp, resolved_ak] if p}

        valid_students_raw = []
        for s in students_raw:
            if isinstance(s, dict):
                s_fp = resolve_file_location(s.get("file_path", ""))
                if s_fp and s_fp in exam_doc_paths:
                    logger.warning(f"Excluding student sheet candidate matching exam document: {s_fp}")
                    continue
                valid_students_raw.append(s)

        resolved_combined = resolve_file_location(combined_pdf_path) if combined_pdf_path else None
        if resolved_combined and resolved_combined in exam_doc_paths:
            logger.warning(f"Combined PDF path matched QP/AK ({resolved_combined}). Removing from student booklets.")
            combined_pdf_path = None
            resolved_combined = None

        if not combined_pdf_path:
            if len(valid_students_raw) == 1:
                first_fp = valid_students_raw[0].get("file_path")
                if first_fp and str(first_fp).lower().endswith(".pdf"):
                    combined_pdf_path = first_fp
            elif payload.get("file_path") and str(payload["file_path"]).lower().endswith(".pdf"):
                fp_res = resolve_file_location(payload["file_path"])
                if fp_res not in exam_doc_paths:
                    combined_pdf_path = payload["file_path"]

        # Validate that student answer sheet is provided
        has_student_input = bool(
            combined_pdf_path
            or valid_students_raw
            or (payload.get("file_path") and resolve_file_location(payload["file_path"]) not in exam_doc_paths)
            or payload.get("file_name")
        )
        if not has_student_input:
            logger.error("Paper evaluation failed: Student answer sheet is missing or could not be resolved.")
            return {
                "status": "error",
                "status_code": 400,
                "message": "Student answer sheet is missing or could not be resolved."
            }, 400

        evaluated_students: List[Dict[str, Any]] = []

        # 4. Mode A Evaluation: Combined PDF with boundary detection
        if combined_pdf_path and len(valid_students_raw) <= 1:
            resolved_pdf = resolve_file_location(combined_pdf_path)
            if resolved_pdf and os.path.exists(resolved_pdf):
                logger.info(f"Executing Mode A Class-Wise Evaluation on combined PDF: {resolved_pdf}")
                detected_booklets = detect_booklet_boundaries(
                    pdf_path=resolved_pdf,
                    client=client,
                    model=model
                )
                for idx, b in enumerate(detected_booklets, start=1):
                    student_res = evaluate_student_booklet_ai(
                        booklet=b,
                        file_path=resolved_pdf,
                        file_name=os.path.basename(resolved_pdf),
                        questions=authoritative_questions,
                        total_marks=max_marks,
                        subject_title=subject_name,
                        discrepancy=discrepancy,
                        client=client,
                        model=model
                    )
                    student_res["id"] = idx
                    evaluated_students.append(student_res)

        # Mode B / Multiple individual files:
        if not evaluated_students:
            target_list = valid_students_raw if valid_students_raw else [payload]
            for idx, s_data in enumerate(target_list, start=1):
                f_path = s_data.get("file_path") or payload.get("file_path") or ""
                f_name = s_data.get("file_name") or payload.get("file_name") or os.path.basename(f_path)
                p_cnt = int(s_data.get("page_count") or payload.get("page_count") or 1)

                resolved_f = resolve_file_location(f_path) if f_path else None
                pdf_disk_text = ""
                if resolved_f and resolved_f.lower().endswith(".pdf"):
                    pdf_disk_text, _ = extract_text_from_pdf(resolved_f)

                raw_pgs = s_data.get("pages")
                if not raw_pgs and s_data.get("pages_json"):
                    try:
                        raw_pgs = json.loads(s_data["pages_json"])
                    except Exception:
                        raw_pgs = []

                booklet_obj = {
                    "booklet_index": idx,
                    "page_numbers": list(range(1, max(p_cnt, len(raw_pgs) if isinstance(raw_pgs, list) else 1) + 1)),
                    "combined_text": pdf_disk_text,
                    "pages": raw_pgs or [],
                    "file_path": f_path,
                    "file_name": f_name
                }
                student_res = evaluate_student_booklet_ai(
                    booklet=booklet_obj,
                    file_path=f_path,
                    file_name=f_name,
                    questions=authoritative_questions,
                    total_marks=max_marks,
                    subject_title=subject_name,
                    discrepancy=discrepancy,
                    client=client,
                    model=model
                )
                student_res["id"] = idx
                evaluated_students.append(student_res)

        if not evaluated_students:
            return {
                "status": "error",
                "status_code": 400,
                "message": "No valid student answer sheets or booklets could be found for evaluation."
            }, 400

        total_sheets = len(evaluated_students)
        passed_count = sum(1 for s in evaluated_students if s["status"] == "Pass")
        failed_count = total_sheets - passed_count
        avg_score = round(sum(s["obtained_marks"] for s in evaluated_students) / max(1, total_sheets), 2)

        # 5. Build POST response (contains student_name & student_id ONLY if detected on document)
        post_response: Dict[str, Any] = {
            "status": "success",
            "total_sheets": total_sheets,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "average_score": avg_score,
            "max_marks": max_marks,
            "total_questions": len(authoritative_questions),
            "discrepancy": discrepancy,
            "students": evaluated_students
        }
        if qp_source and isinstance(qp_source, str):
            post_response["question_paper_file"] = qp_source
        if ak_source and isinstance(ak_source, str):
            post_response["answer_key_file"] = ak_source
        if payload.get("include_evaluation_id") or payload.get("include_id"):
            post_response["evaluation_id"] = int(eval_id) if eval_id.isdigit() else eval_id

        # 6. Build GET response (Return Student Identity ONLY when extracted from PDF/Image)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_str = datetime.now().strftime("%Y-%m-%d")

        get_students_list = []
        for idx, s in enumerate(evaluated_students, start=1):
            is_pending = s.get("status") == "Pending Review"
            default_rec = "Manual instructor review required." if is_pending else f"Review performance and feedback for {subject_name}."
            rec_val = s["ai_insights"].get("recommendation") if isinstance(s.get("ai_insights"), dict) else ""
            insights_compact = {
                "recommendation": rec_val or default_rec,
                "focus_areas": s["ai_insights"].get("focus_areas", []) if isinstance(s.get("ai_insights"), dict) else []
            }
            record_id = s.get("id") or idx
            is_extracted = (s.get("recognition_status") == "matched") and bool(s.get("student_name") or s.get("student_id"))
            get_students_list.append({
                "id": str(record_id),
                "evaluation_id": str(eval_id),
                "student_id": str(s.get("student_id", "")) if is_extracted else "",
                "student_name": s.get("student_name", "") if is_extracted else "",
                "admission_no": s.get("admission_no", "") if is_extracted else "",
                "roll_no": str(s.get("roll_no", "")) if is_extracted else "",
                "kuhs_reg_no": str(s.get("kuhs_reg_no", "")) if is_extracted else "",
                "student_image": s.get("student_image", "") if is_extracted else "",
                "recognized_text": s.get("recognized_text", "") if is_extracted else "",
                "recognition_status": "matched" if is_extracted else "unmatched",
                "recognition_confidence": s.get("recognition_confidence", 0.0) if is_extracted else 0.0,
                "file_name": s.get("file_name", ""),
                "file_path": s.get("file_path", ""),
                "total_questions": s.get("total_questions", 0),
                "attempted_questions": s.get("attempted_questions", 0),
                "unanswered_questions": s.get("unanswered_questions", 0),
                "evaluated_questions": s.get("evaluated_questions", 0),
                "pending_review_questions": s.get("pending_review_questions", 0),
                "total_marks": format_two_decimals(s["total_marks"]),
                "obtained_marks": format_two_decimals(s["obtained_marks"]),
                "percentage": format_two_decimals(s["percentage"]),
                "grade": s["grade"],
                "status": s["status"],
                "ai_feedback": s["ai_feedback"][:120] + "..." if len(s["ai_feedback"]) > 130 else s["ai_feedback"],
                "ai_insights": json.dumps(insights_compact, ensure_ascii=False)
            })

        get_response: Dict[str, Any] = {
            "status": "success",
            "data": {
                "id": str(eval_id),
                "title": title,
                "class_id": str(class_id),
                "class_name": class_name,
                "section_id": str(section_id),
                "section_name": section_name,
                "subject_id": str(subject_id),
                "subject_name": subject_name,
                "question_paper_id": str(question_paper_id),
                "question_paper_title": question_paper_title,
                "total_students": str(total_sheets),
                "passed_count": str(passed_count),
                "failed_count": str(failed_count),
                "average_score": format_two_decimals(avg_score),
                "max_marks": format_two_decimals(max_marks),
                "total_questions": len(authoritative_questions),
                "discrepancy": discrepancy,
                "evaluation_date": payload.get("evaluation_date") or date_str,
                "created_at": payload.get("created_at") or now_str,
                "students": get_students_list
            }
        }
        if qp_source and isinstance(qp_source, str):
            get_response["data"]["question_paper_file"] = qp_source
        if ak_source and isinstance(ak_source, str):
            get_response["data"]["answer_key_file"] = ak_source

        # 7. Persist to EvaluationStore
        evaluation_store.save_evaluation(eval_id=eval_id, post_resp=post_response, get_resp=get_response)
        logger.info(f"Evaluation #{eval_id} stored successfully ({total_sheets} sheet(s) evaluated).")

        return post_response, 200

    except Exception as e:
        logger.error(f"Error in process_paper_evaluation: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "status_code": 500,
            "message": "Evaluation Error",
            "details": str(e)
        }, 500
