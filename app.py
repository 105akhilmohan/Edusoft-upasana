import os
import io
import json
import base64
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("edusoft_service")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OPENAI_API_KEY is not set in environment variables.")

client = OpenAI(api_key=openai_api_key)
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@app.after_request
def log_response_status(response):
    """Log incoming request method, path, and HTTP response status code."""
    logger.info(f"{request.remote_addr} - \"{request.method} {request.path}\" {response.status_code}")
    return response


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Edusoft AI Analytics & Question Generator API",
        "status": "online",
        "status_code": 200,
        "version": "1.3.0",
        "endpoints": {
            "health": "GET /health",
            "generate_insights": "POST /api/generate-insights",
            "generate_questions": "POST /api/generate-questions",
            "extract_syllabus": "POST /api/extract-syllabus"
        }
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "status_code": 200,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }), 200


# ==============================================================================
# 1. STUDENT ANALYSIS & INSIGHTS GENERATION ENDPOINT
# ==============================================================================
@app.route("/api/generate-insights", methods=["POST"])
@app.route("/generate-insights", methods=["POST"])
@app.route("/api/student-analysis", methods=["POST"])
@app.route("/analyze-student", methods=["POST"])
def generate_insights():
    """
    Generate student analytical insights (summary, strengths, focus_areas, recommendation)
    from student profile, overall scores, attendance, and examination records.
    """
    try:
        raw_body = request.get_json()
        if not raw_body:
            return jsonify({
                "status": "error",
                "message": "Request body must be a valid JSON object."
            }), 400

        # Handle payload whether wrapped inside 'data' or sent directly at top-level
        payload = raw_body.get("data") if ("data" in raw_body and isinstance(raw_body["data"], dict)) else raw_body

        # Check if flat payload format (e.g. PHP $ai_payload)
        if "student_name" in payload or "student_id" in payload or "academic_percentage" in payload:
            student = {
                "id": payload.get("student_id") or payload.get("id"),
                "name": payload.get("student_name") or payload.get("name", "The student"),
                "admission_no": payload.get("admission_no", ""),
                "roll_no": payload.get("roll_no", ""),
                "class": payload.get("class", ""),
                "section": payload.get("section", ""),
                "gender": payload.get("gender", "")
            }
            overall_score = {
                "composite_score": payload.get("composite_score"),
                "performance_label": payload.get("performance_label", ""),
                "academic_percentage": payload.get("academic_percentage"),
                "academic_grade": payload.get("academic_grade", ""),
                "attendance_percentage": payload.get("attendance_percentage"),
                "attendance_health": payload.get("attendance_health", "")
            }
            attendance = {
                "total_working_days": payload.get("total_working_days"),
                "present": payload.get("present_days") if payload.get("present_days") is not None else payload.get("present"),
                "absent": payload.get("absent_days") if payload.get("absent_days") is not None else payload.get("absent"),
                "late": payload.get("late_days") if payload.get("late_days") is not None else payload.get("late"),
                "half_day": payload.get("half_days") if payload.get("half_days") is not None else payload.get("half_day"),
                "percentage": payload.get("attendance_percentage"),
                "health_status": payload.get("attendance_health", "")
            }
            examinations = {
                "exams_count": payload.get("exams_count"),
                "overall_percentage": payload.get("academic_percentage"),
                "overall_grade": payload.get("academic_grade", ""),
                "subject_summary": payload.get("subject_summary", []),
                "top_subjects": payload.get("top_subjects", []),
                "lowest_subjects": payload.get("lowest_subjects", [])
            }
            extra_info = {}
            if payload.get("fee_status") is not None or payload.get("fee_balance") is not None:
                extra_info["fee_info"] = {
                    "fee_status": payload.get("fee_status"),
                    "fee_balance": payload.get("fee_balance")
                }
            if payload.get("behavior_points") is not None or payload.get("incident_count") is not None:
                extra_info["behavior_and_discipline"] = {
                    "behavior_points": payload.get("behavior_points"),
                    "incident_count": payload.get("incident_count")
                }
        else:
            student = payload.get("student", {})
            overall_score = payload.get("overall_score", {})
            attendance = payload.get("attendance", {})
            examinations = payload.get("examinations", {})
            extra_info = {}
            if "fee_info" in payload:
                extra_info["fee_info"] = payload.get("fee_info")
            elif "fee_balance" in payload or "fee_status" in payload:
                extra_info["fee_info"] = {
                    "fee_status": payload.get("fee_status"),
                    "fee_balance": payload.get("fee_balance")
                }
            if "discipline" in payload:
                extra_info["behavior_and_discipline"] = payload.get("discipline")
            elif "behavior_points" in payload or "incident_count" in payload:
                extra_info["behavior_and_discipline"] = {
                    "behavior_points": payload.get("behavior_points"),
                    "incident_count": payload.get("incident_count")
                }

        model = payload.get("model", DEFAULT_MODEL)
        student_name = student.get("name") or payload.get("student_name", "The student")

        # ----------------------------------------------------------------------
        # PROMPT 1: Academic Summary & Key Strengths Diagnostic
        # ----------------------------------------------------------------------
        system_prompt_1 = (
            "You are a senior academic diagnostic specialist for Edusoft. "
            "Your objective is to review student performance, grades, and attendance "
            "to produce an executive summary and highlight key academic strengths with precise metrics. "
            "Return valid JSON only."
        )

        extra_str = f"\nADDITIONAL METRICS:\n{json.dumps(extra_info, indent=2)}" if extra_info else ""

        user_prompt_1 = f"""
Analyze the student's performance data below:

STUDENT PROFILE:
{json.dumps(student, indent=2)}

OVERALL SCORES:
{json.dumps(overall_score, indent=2)}

ATTENDANCE:
{json.dumps(attendance, indent=2)}

EXAMINATION & SUBJECT DATA:
{json.dumps(examinations, indent=2)}
{extra_str}

SPECIAL RULES FOR NEW/INITIAL STUDENTS (0 working days or 0 exams):
- If total_working_days is 0 or no attendance is recorded, state that attendance logging has just begun or is pending. Do NOT treat 0 working days as attendance failure or 0% critical absent.
- If exams_count is 0 or no exam data exists, state that academic assessments are pending for the term. Do NOT treat 0 exams as Grade F failure.

Generate a JSON object with:
{{
  "summary": "1-2 sentence overview mentioning {student_name}'s academic status, grade, and attendance standing (or pending assessment status if newly enrolled).",
  "strengths": [
    "Highlight strong attendance or consistency with exact % (or positive behavioral/enrollment status if newly enrolled)",
    "Highlight top subject mastery with subject name, percentage, and grade (or general positive orientation if assessments are pending)"
  ]
}}
"""

        # ----------------------------------------------------------------------
        # PROMPT 2: Diagnostic Gap Analysis, Focus Areas & Actionable Recommendations
        # ----------------------------------------------------------------------
        system_prompt_2 = (
            "You are an educational psychologist and student guidance counselor for Edusoft. "
            "Your objective is to identify academic vulnerabilities, lowest performing subjects, attendance risks, "
            "and create forward-looking recommendations. "
            "Return valid JSON only."
        )

        user_prompt_2 = f"""
Analyze the student's areas of growth based ONLY on the supplied data.

STUDENT PROFILE:
{json.dumps(student, indent=2)}

OVERALL SCORES:
{json.dumps(overall_score, indent=2)}

ATTENDANCE:
{json.dumps(attendance, indent=2)}

EXAMINATIONS & SUBJECT BREAKDOWN:
{json.dumps(examinations, indent=2)}
{extra_str}

IMPORTANT DATA RULES:
- Use ONLY information present above.
- Do NOT invent any information.
- Do NOT mention fee dues unless fee information is explicitly provided with an unpaid balance > 0.
- Do NOT create subjects that are not present.
- Do NOT create marks or percentages that are not present.
- Do NOT infer a student's financial status.
- If total_working_days is 0 or exams_count is 0, focus on establishing baseline attendance and preparing for initial assessments rather than penalizing for missing data.
- Every percentage mentioned must match the supplied data.
- Focus areas must be based on actual weaknesses in the supplied academic or attendance data.
- Recommendations must be constructive and tailored to the actual student performance or enrollment stage.

Generate:

{{
  "focus_areas": [
    "Specific subject or metric requiring improvement with exact percentage (or baseline focus for initial term)",
    "Another actual academic or attendance focus area if supported by the data"
  ],
  "recommendation": "A clear, actionable, encouraging recommendation based only on the supplied data."
}}

Return valid JSON only.
"""

        logger.info(f"Executing Prompt 1 (Summary & Strengths) for: {student_name}")
        resp1 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt_1},
                {"role": "user", "content": user_prompt_1}
            ],
            response_format={"type": "json_object"},
            temperature=0.5
        )

        logger.info(f"Executing Prompt 2 (Focus Areas & Recommendations) for: {student_name}")
        resp2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt_2},
                {"role": "user", "content": user_prompt_2}
            ],
            response_format={"type": "json_object"},
            temperature=0.6
        )

        # Parse both responses
        res1_json = json.loads(resp1.choices[0].message.content)
        res2_json = json.loads(resp2.choices[0].message.content)

        # Merge insights
        insights = {
            "summary": res1_json.get("summary", ""),
            "strengths": res1_json.get("strengths", []),
            "focus_areas": res2_json.get("focus_areas", []),
            "recommendation": res2_json.get("recommendation", "")
        }

        # Build full data payload preserving student, overall_score, attendance, examinations
        response_data = dict(payload)
        response_data["insights"] = insights

        logger.info(f"Successfully generated two-stage insights for student: {student_name}")

        return jsonify({
            "status": "success",
            "status_code": 200,
            "data": response_data,
            "insights": insights
        }), 200

    except OpenAIError as oe:
        logger.error(f"OpenAI API Error: {str(oe)}", exc_info=True)
        return jsonify({
            "status": "error",
            "status_code": 502,
            "message": "OpenAI API Error",
            "details": str(oe)
        }), 502
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "status_code": 500,
            "message": "Internal Server Error",
            "details": str(e)
        }), 500


from textbook_repo import (
    get_textbook_content,
    validate_question,
    normalize_text_for_comparison,
    normalize_class,
    normalize_subject
)

# ==============================================================================
# 2. QUESTION GENERATION ENDPOINT (Textbook-Grounded & Strict Validation)
# ==============================================================================
@app.route("/api/generate-questions", methods=["POST"])
@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    """
    Generate educational questions strictly grounded in textbook content with post-generation validation.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body must be a valid JSON object."
            }), 400

        # Extract Fields with fallbacks
        class_name = str(data.get("class_name", "")).strip()
        subject_name = str(data.get("subject_name") or data.get("topic", "")).strip()
        
        # Chapters
        chapters_raw = data.get("chapters", [])
        if isinstance(chapters_raw, list):
            chapters = [str(c).strip() for c in chapters_raw if str(c).strip()]
        elif isinstance(chapters_raw, str) and chapters_raw.strip():
            chapters = [chapters_raw.strip()]
        else:
            chapters = []

        if not subject_name and not chapters:
            return jsonify({
                "status": "error",
                "message": "At least 'subject_name' (or 'topic') or 'chapters' must be provided."
            }), 400

        primary_chapter = chapters[0] if chapters else (subject_name or "General")

        # Question Count
        count_val = data.get("question_count") if data.get("question_count") is not None else data.get("count")
        if count_val is None:
            question_count = 5
        else:
            try:
                question_count = int(count_val)
                if question_count <= 0 or question_count > 50:
                    return jsonify({
                        "status": "error",
                        "message": "Field 'question_count' must be an integer between 1 and 50."
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    "status": "error",
                    "message": "Field 'question_count' must be a valid integer."
                }), 400

        # Difficulty & Question Types
        difficulty = data.get("difficulty", "Medium")
        q_types_raw = data.get("question_types") or data.get("question_type") or ["MCQ", "Short", "Long"]
        if isinstance(q_types_raw, list):
            question_types = [str(t).strip() for t in q_types_raw if str(t).strip()]
        elif isinstance(q_types_raw, str) and q_types_raw.strip():
            question_types = [q_types_raw.strip()]
        else:
            question_types = ["MCQ", "Short", "Long"]

        suggestions = data.get("suggestions") or data.get("description", "")
        payload_content = data.get("textbook_content") or data.get("content") or data.get("chapter_content")
        model = data.get("model", DEFAULT_MODEL)

        # ----------------------------------------------------------------------
        # STEP 1: RETRIEVE TEXTBOOK CONTENT (Source of Truth)
        # ----------------------------------------------------------------------
        textbook_content, content_id = get_textbook_content(
            class_name=class_name,
            subject_name=subject_name,
            chapter_name=primary_chapter,
            payload_content=payload_content
        )

        if not textbook_content:
            logger.warning(f"Textbook content not found for Class='{class_name}', Subject='{subject_name}', Chapter='{primary_chapter}'")
            return jsonify({
                "status": "error",
                "message": "Textbook content not found for the selected class, subject, and chapter."
            }), 404

        # Logging source content metadata as required
        logger.info(f"Generating questions: Class = {class_name}, Subject = {subject_name}, Chapter = {primary_chapter}")
        logger.info(f"Retrieved textbook content: {content_id}, Content length = {len(textbook_content)}")

        # ----------------------------------------------------------------------
        # STEP 2: BUILD PROMPTS WITH TEXTBOOK AS ONLY SOURCE OF TRUTH
        # ----------------------------------------------------------------------
        import uuid
        import time
        generation_nonce = f"{uuid.uuid4().hex[:8]}-{int(time.time() * 1000)}"

        system_prompt = f"""
You are an expert school curriculum designer and examination question generator for Edusoft.

MANDATORY RULES:
1. The supplied textbook content is the ONLY source of truth.
2. Every question, option, correct answer, and step marking MUST be directly and strictly derived from the provided textbook content below.
3. Target Class Level: Class {class_name}.
4. Target Subject: {subject_name}.
5. Chapter: {primary_chapter}.
6. Do NOT use outside knowledge or introduce any concepts from other subjects (e.g. absolutely no Physics/Science when generating Maths questions).
7. For Class 1: Only simple single-digit numbers (1-20), counting, basic addition/subtraction, and shapes. NEVER use high-school physics or mechanics.
8. Do NOT include any 'explanation' field.
9. Return strictly valid JSON only.
"""

        user_prompt = f"""
Generate exactly {question_count} distinct educational questions based STRICTLY on the supplied textbook content below:

============================================================
OFFICIAL TEXTBOOK CONTENT (ONLY SOURCE OF TRUTH)
============================================================
{textbook_content}

============================================================
SPECIFICATIONS
============================================================
Target Class: {class_name}
Target Subject: {subject_name}
Target Chapter: {primary_chapter}
Difficulty: {difficulty}
Requested Types: {', '.join(question_types)}
{f"Suggestions: {suggestions}" if suggestions else ""}
Random Variation Seed: {generation_nonce}

============================================================
STRICT GENERATION RULES
============================================================
- Generate exactly {question_count} questions answerable directly from the textbook content above.
- Every question MUST be pure {subject_name} suitable for Class {class_name}.
- For MCQs: 4 plausible options within the textbook scope.
- For Numerical: Simple age-appropriate calculations.
- For Short/Long: Model answer and step-by-step marking.
- Do NOT include any 'explanation' field.

Return a valid JSON object matching this schema:
{{
  "class_name": "{class_name}",
  "subject_name": "{subject_name}",
  "chapters": {json.dumps(chapters)},
  "total_questions": {question_count},
  "difficulty": "{difficulty}",
  "questions": [
    {{
      "id": 1,
      "chapter": "{primary_chapter}",
      "type": "MCQ | Short | Long | Numerical",
      "marks": 2,
      "question": "Question text here",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "Correct answer or model answer",
      "step_marking": [
        {{
          "step": "Step description",
          "marks": 1
        }}
      ],
      "difficulty": "Easy | Medium | Hard"
    }}
  ]
}}
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            top_p=0.9
        )

        raw_content = response.choices[0].message.content
        try:
            parsed_result = json.loads(raw_content)
        except json.JSONDecodeError:
            parsed_result = {"questions": []}

        generated_raw_questions = parsed_result.get("questions", [])

        # ----------------------------------------------------------------------
        # STEP 3: STRICT POST-GENERATION VALIDATION & DUPLICATE DETECTION
        # ----------------------------------------------------------------------
        valid_questions = []
        seen_normalized_texts = set()

        for idx, q_obj in enumerate(generated_raw_questions, start=1):
            q_text = q_obj.get("question", "")
            norm_text = normalize_text_for_comparison(q_text)

            # Duplicate Check
            if norm_text in seen_normalized_texts:
                logger.warning(f"Question {idx}: INVALID - Duplicate question detected ('{q_text[:50]}...')")
                continue

            # Validation against Class, Subject, Chapter & Content
            is_valid, reason = validate_question(
                q_obj=q_obj,
                class_name=class_name,
                subject_name=subject_name,
                chapter_name=primary_chapter,
                textbook_content=textbook_content
            )

            if is_valid:
                logger.info(f"Question {idx}: VALID - '{q_text[:60]}...'")
                # Enforce clean fields
                clean_q = dict(q_obj)
                clean_q["id"] = len(valid_questions) + 1
                clean_q["chapter"] = primary_chapter  # Hard constraint: exact requested chapter name
                clean_q.pop("explanation", None)       # Enforce: no explanation
                valid_questions.append(clean_q)
                seen_normalized_texts.add(norm_text)
            else:
                logger.warning(f"Question {idx}: INVALID - {reason} - '{q_text[:60]}...'")

        # ----------------------------------------------------------------------
        # STEP 4: REGENERATE REPLACEMENTS IF INVALID QUESTIONS WERE DROPPED
        # ----------------------------------------------------------------------
        retry_count = 0
        max_retries = 2
        while len(valid_questions) < question_count and retry_count < max_retries:
            needed = question_count - len(valid_questions)
            retry_count += 1
            logger.info(f"Regenerating {needed} replacement question(s) (Attempt {retry_count}/{max_retries})...")

            replacement_prompt = f"""
Generate exactly {needed} fresh, distinct replacement questions based ONLY on the supplied textbook content:
{textbook_content}

Exclude already generated questions:
{list(seen_normalized_texts)[:10]}

Specifications:
Class: {class_name}
Subject: {subject_name}
Chapter: {primary_chapter}
Difficulty: {difficulty}
Types: {', '.join(question_types)}

Return JSON with "questions" array. Do NOT include explanations.
"""
            rep_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": replacement_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            try:
                rep_json = json.loads(rep_response.choices[0].message.content)
                rep_questions = rep_json.get("questions", [])
            except Exception:
                rep_questions = []

            for q_obj in rep_questions:
                if len(valid_questions) >= question_count:
                    break
                q_text = q_obj.get("question", "")
                norm_text = normalize_text_for_comparison(q_text)
                if norm_text in seen_normalized_texts:
                    continue
                is_valid, reason = validate_question(
                    q_obj=q_obj,
                    class_name=class_name,
                    subject_name=subject_name,
                    chapter_name=primary_chapter,
                    textbook_content=textbook_content
                )
                if is_valid:
                    clean_q = dict(q_obj)
                    clean_q["id"] = len(valid_questions) + 1
                    clean_q["chapter"] = primary_chapter
                    clean_q.pop("explanation", None)
                    valid_questions.append(clean_q)
                    seen_normalized_texts.add(norm_text)
                    logger.info(f"Replacement Question: VALID - '{q_text[:60]}...'")
                else:
                    logger.warning(f"Replacement Question: INVALID - {reason}")

        # Final response formatting
        response_payload = {
            "class_name": class_name,
            "subject_name": subject_name,
            "chapters": chapters,
            "total_questions": len(valid_questions),
            "difficulty": difficulty,
            "questions": valid_questions
        }

        return jsonify({
            "status": "success",
            "status_code": 200,
            "data": response_payload
        }), 200

    except OpenAIError as oe:
        logger.error(f"OpenAI API Error: {str(oe)}", exc_info=True)
        return jsonify({
            "status": "error",
            "status_code": 502,
            "message": "OpenAI API Error",
            "details": str(oe)
        }), 502
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "status_code": 500,
            "message": "Internal Server Error",
            "details": str(e)
        }), 500


# ==============================================================================
# 3. SYLLABUS PDF EXTRACTION & PARSING ENDPOINT (pdfplumber + OpenAI)
def chunk_pdf_pages(page_records, max_chunk_chars=40000):
    """
    Split extracted PDF pages into safe chunks adhering strictly to page boundaries.
    Never split in the middle of a page unless an individual page exceeds max_chunk_chars.
    """
    chunks = []
    current_chunk_pages = []
    current_chunk_len = 0

    for page_num, text in page_records:
        formatted_page = f"--- PAGE {page_num} ---\n{text}\n\n"
        page_len = len(formatted_page)

        # If single page exceeds max_chunk_chars, split that single page
        if page_len > max_chunk_chars:
            # Flush accumulated pages first
            if current_chunk_pages:
                chunks.append("".join(current_chunk_pages).strip())
                current_chunk_pages = []
                current_chunk_len = 0

            # Slice large page
            start_idx = 0
            part_num = 1
            while start_idx < len(text):
                sub_text = text[start_idx:start_idx + (max_chunk_chars - 200)]
                chunks.append(f"--- PAGE {page_num} (Part {part_num}) ---\n{sub_text}".strip())
                start_idx += (max_chunk_chars - 200)
                part_num += 1
            continue

        # Check if adding this page exceeds chunk size
        if current_chunk_len + page_len > max_chunk_chars and current_chunk_pages:
            chunks.append("".join(current_chunk_pages).strip())
            current_chunk_pages = [formatted_page]
            current_chunk_len = page_len
        else:
            current_chunk_pages.append(formatted_page)
            current_chunk_len += page_len

    if current_chunk_pages:
        chunks.append("".join(current_chunk_pages).strip())

    return chunks


# ==============================================================================
# 3. SYLLABUS PDF EXTRACTION & PARSING ENDPOINT (pdfplumber + OpenAI Chunked)
# ==============================================================================
@app.route("/api/extract-syllabus", methods=["POST"])
@app.route("/extract-syllabus", methods=["POST"])
@app.route("/api/parse-syllabus-pdf", methods=["POST"])
def extract_syllabus():
    """
    Extract text from uploaded PDF using pdfplumber and generate structured
    syllabus overview, chapter list with descriptions, and learning outcomes.
    Supports both normal syllabus PDFs and large books (e.g. 600+ pages) via
    page-boundary chunking, TOC-aware extraction, and sequential aggregation.
    """
    try:
        subject_name = ""
        subject_code = ""
        pdf_stream = None
        extracted_text = ""
        page_records = []
        total_pages = 0
        pages_with_text = 0

        # Case 1: multipart/form-data upload
        if request.files:
            file_obj = request.files.get("file") or request.files.get("pdf") or request.files.get("document")
            if file_obj:
                pdf_stream = io.BytesIO(file_obj.read())
            subject_name = request.form.get("subject_name", "").strip()
            subject_code = request.form.get("subject_code", "").strip()
            model = request.form.get("model") or DEFAULT_MODEL or "gpt-4o-mini"

        # Case 2: JSON payload (base64, direct text, or URL)
        elif request.is_json:
            data = request.get_json() or {}
            subject_name = str(data.get("subject_name", "")).strip()
            subject_code = str(data.get("subject_code", "")).strip()
            model = data.get("model") or DEFAULT_MODEL or "gpt-4o-mini"

            if "pdf_base64" in data or "file_base64" in data or "pdf" in data:
                b64_str = data.get("pdf_base64") or data.get("file_base64") or data.get("pdf")
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                try:
                    pdf_bytes = base64.b64decode(b64_str)
                    pdf_stream = io.BytesIO(pdf_bytes)
                except Exception as b64_err:
                    return jsonify({
                        "status": "error",
                        "message": f"Failed to decode base64 PDF: {str(b64_err)}"
                    }), 400
            elif "pdf_text" in data or "text" in data or "content" in data:
                extracted_text = str(data.get("pdf_text") or data.get("text") or data.get("content")).strip()
        else:
            model = DEFAULT_MODEL or "gpt-4o-mini"

        # Extract text page-by-page preserving boundaries with pdfplumber
        if pdf_stream:
            try:
                import pdfplumber
                with pdfplumber.open(pdf_stream) as pdf:
                    total_pages = len(pdf.pages)
                    for page_num, page in enumerate(pdf.pages, start=1):
                        try:
                            txt = page.extract_text()
                        except Exception as page_err:
                            logger.warning(f"Error extracting text from page {page_num}: {str(page_err)}")
                            txt = ""

                        if txt and txt.strip():
                            page_records.append((page_num, txt.strip()))
                            pages_with_text += 1

            except Exception as pdf_err:
                logger.error(f"Error parsing PDF with pdfplumber: {str(pdf_err)}", exc_info=True)
                return jsonify({
                    "status": "error",
                    "message": f"Error parsing PDF file: {str(pdf_err)}"
                }), 400

        elif extracted_text:
            # Reconstruct page records if simulated markers exist or treat as single block
            raw_lines = extracted_text.splitlines()
            current_page_num = 1
            current_page_lines = []
            for line in raw_lines:
                if line.strip().startswith("--- PAGE ") or line.strip().startswith("PAGE "):
                    if current_page_lines:
                        page_records.append((current_page_num, "\n".join(current_page_lines).strip()))
                        current_page_lines = []
                        current_page_num += 1
                current_page_lines.append(line)
            if current_page_lines:
                page_records.append((current_page_num, "\n".join(current_page_lines).strip()))

            if not page_records and extracted_text.strip():
                page_records.append((1, extracted_text.strip()))

            total_pages = len(page_records)
            pages_with_text = len([p for p in page_records if p[1]])

        # Handle scanned/image-only PDFs gracefully
        if not page_records or pages_with_text == 0:
            return jsonify({
                "status": "error",
                "message": (
                    f"No readable text could be extracted from the provided PDF ({total_pages} total pages checked). "
                    "The document appears to contain scanned or image-only pages without an embedded text layer. "
                    "Please ensure a valid text-based or OCR-processed PDF is uploaded."
                ),
                "total_pages": total_pages,
                "pages_with_text": 0
            }), 400

        total_chars = sum(len(text) for _, text in page_records)
        logger.info(
            f"PDF Extracted: {pages_with_text}/{total_pages} pages have text ({total_chars} total characters) "
            f"for Subject: '{subject_name}' ({subject_code})"
        )

        # Chunk pages safely at page boundaries
        chunks = chunk_pdf_pages(page_records, max_chunk_chars=40000)
        logger.info(f"Formed {len(chunks)} chunk(s) for extraction.")

        # ======================================================================
        # PATH A: SINGLE CHUNK (Small/Medium PDF)
        # ======================================================================
        if len(chunks) == 1:
            full_text = chunks[0]
            system_prompt = (
                "You are an expert curriculum and syllabus extraction specialist. "
                "Your objective is to analyze the provided extracted PDF document text for the specified subject and subject code, "
                "and extract a clean, complete, structured syllabus curriculum.\n"
                "MANDATORY RULES:\n"
                "1. The provided PDF text is the ONLY source of truth. Do NOT invent chapters, units, or outside topics.\n"
                "2. If a Table of Contents (TOC) or syllabus outline is present, use it to accurately identify syllabus chapters and units.\n"
                "3. Do NOT assume every heading or section is a syllabus chapter. Extract only legitimate curriculum units/chapters.\n"
                "4. 'syllabus_content': A concise 2-3 sentence executive overview of the subject curriculum and its core scope.\n"
                "5. 'chapters': An ordered list of all chapters/units with 'chapter_no', 'chapter_name', and a comprehensive 'description' of topics covered.\n"
                "6. 'learning_outcomes': An array of key academic and practical learning outcomes.\n"
                "Output strictly valid JSON matching the exact schema."
            )

            user_prompt = f"""
Analyze the following extracted PDF text and extract the structured syllabus for:
Subject Name: {subject_name}
Subject Code: {subject_code}

============================================================
EXTRACTED PDF DOCUMENT TEXT:
============================================================
{full_text}

============================================================
REQUIRED JSON OUTPUT SCHEMA:
============================================================
{{
  "status": "success",
  "subject_name": "{subject_name}",
  "subject_code": "{subject_code}",
  "syllabus_content": "Detailed overview of human anatomical systems, osteology, arthrology, myology, and systemic organ relations for clinical practice.",
  "chapters": [
    {{
      "chapter_no": "1",
      "chapter_name": "Introduction to Anatomical Terms & Organization",
      "description": "Anatomical planes, positions, cavities, cell structure, tissues, and membranes."
    }},
    {{
      "chapter_no": "2",
      "chapter_name": "The Skeletal & Muscular System",
      "description": "Axial and appendicular skeleton, joints, muscle classification, and biomechanics."
    }}
  ],
  "learning_outcomes": [
    "Identify anatomical landmarks on human models and radiographic images",
    "Correlate anatomical structures with nursing procedures and clinical interventions"
  ]
}}
"""
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            raw_output = response.choices[0].message.content
            parsed_result = json.loads(raw_output)

        # ======================================================================
        # PATH B: MULTI-CHUNK SEQUENTIAL EXTRACTION & AGGREGATION (Large PDF / Book)
        # ======================================================================
        else:
            raw_chapters = []
            raw_learning_outcomes = []

            chunk_system_prompt = (
                "You are an expert curriculum and syllabus extraction specialist. "
                "Analyze the provided text excerpt from a multi-page document or textbook.\n"
                "MANDATORY RULES:\n"
                "1. Extract all syllabus units/chapters, topics covered, and learning outcomes mentioned in this specific excerpt.\n"
                "2. The provided text is the ONLY source of truth. Do NOT invent chapters or use outside knowledge.\n"
                "3. If this excerpt contains a Table of Contents (TOC), course outline, or chapter headings, use them to capture accurate chapter names and numbers.\n"
                "4. Do NOT treat casual body headings, sub-sections, preface remarks, figure captions, or index listings as syllabus chapters.\n"
                "5. Extract only legitimate curriculum units/chapters with 'chapter_no', 'chapter_name', and a comprehensive 'description'.\n"
                "6. Extract any explicit or implied learning outcomes/objectives present in this excerpt.\n"
                "7. If this excerpt does not contain syllabus units or chapters, return empty lists: {\"chapters\": [], \"learning_outcomes\": []}.\n"
                "Return valid JSON only."
            )

            for idx, chunk_text in enumerate(chunks, start=1):
                logger.info(f"Extracting syllabus from Chunk {idx}/{len(chunks)} ({len(chunk_text)} chars)...")
                chunk_user_prompt = f"""
Analyze this document excerpt (Chunk {idx} of {len(chunks)}) for:
Subject Name: {subject_name}
Subject Code: {subject_code}

============================================================
EXTRACTED DOCUMENT EXCERPT (Chunk {idx}/{len(chunks)}):
============================================================
{chunk_text}

============================================================
JSON OUTPUT SCHEMA:
============================================================
{{
  "chapters": [
    {{
      "chapter_no": "1",
      "chapter_name": "Chapter or Unit Title",
      "description": "Comprehensive description of topics and concepts covered in this chapter."
    }}
  ],
  "learning_outcomes": [
    "Specific learning competency or outcome"
  ]
}}
"""
                try:
                    chunk_resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": chunk_system_prompt},
                            {"role": "user", "content": chunk_user_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.2
                    )
                    chunk_parsed = json.loads(chunk_resp.choices[0].message.content)
                    extracted_ch = chunk_parsed.get("chapters", [])
                    extracted_lo = chunk_parsed.get("learning_outcomes", [])

                    if isinstance(extracted_ch, list):
                        for ch in extracted_ch:
                            if isinstance(ch, dict) and ch.get("chapter_name"):
                                raw_chapters.append(ch)

                    if isinstance(extracted_lo, list):
                        for lo in extracted_lo:
                            if isinstance(lo, str) and lo.strip():
                                raw_learning_outcomes.append(lo.strip())

                except Exception as chunk_err:
                    logger.warning(f"Error processing chunk {idx}/{len(chunks)}: {str(chunk_err)}")

            logger.info(
                f"Completed multi-chunk pass. Collected {len(raw_chapters)} raw chapter entries "
                f"and {len(raw_learning_outcomes)} raw learning outcomes. Aggregating final syllabus..."
            )

            # Aggregation Step
            agg_system_prompt = (
                "You are a master educational curriculum aggregation specialist. "
                "Your task is to consolidate, deduplicate, and organize the extracted chapters and learning outcomes "
                "from all chunks of a large textbook/syllabus document into a single, clean, cohesive, ordered syllabus curriculum.\n"
                "MANDATORY RULES:\n"
                "1. Merge duplicate chapters (e.g., chapters appearing both in the Table of Contents and in individual chapter body chunks) into single entries with rich, consolidated descriptions.\n"
                "2. Maintain strict chronological / sequential order of chapters (e.g., Chapter 1, Chapter 2... or Unit I, Unit II...).\n"
                "3. Deduplicate learning outcomes while preserving specific and actionable competencies.\n"
                "4. Generate a concise 2-3 sentence 'syllabus_content' executive overview of the subject curriculum and its core scope based ONLY on the extracted content.\n"
                "5. Do NOT invent new chapters or use outside knowledge. Rely strictly on the provided aggregated data.\n"
                "6. Return strictly valid JSON matching the exact schema."
            )

            agg_user_prompt = f"""
Consolidate the extracted syllabus data below into the final structured curriculum:
Subject Name: {subject_name}
Subject Code: {subject_code}

============================================================
COLLECTED EXTRACTED CHAPTERS ACROSS ALL CHUNKS:
============================================================
{json.dumps(raw_chapters, indent=2)}

============================================================
COLLECTED LEARNING OUTCOMES ACROSS ALL CHUNKS:
============================================================
{json.dumps(raw_learning_outcomes, indent=2)}

============================================================
REQUIRED FINAL JSON OUTPUT SCHEMA:
============================================================
{{
  "status": "success",
  "subject_name": "{subject_name}",
  "subject_code": "{subject_code}",
  "syllabus_content": "Executive overview of the subject curriculum and core scope...",
  "chapters": [
    {{
      "chapter_no": "1",
      "chapter_name": "Chapter Name",
      "description": "Comprehensive description of topics covered."
    }}
  ],
  "learning_outcomes": [
    "Learning outcome competency"
  ]
}}
"""
            agg_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": agg_system_prompt},
                    {"role": "user", "content": agg_user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            parsed_result = json.loads(agg_response.choices[0].message.content)

        # Guarantee status, subject_name, subject_code, chapters, and learning_outcomes
        parsed_result["status"] = "success"
        if subject_name and not parsed_result.get("subject_name"):
            parsed_result["subject_name"] = subject_name
        if subject_code and not parsed_result.get("subject_code"):
            parsed_result["subject_code"] = subject_code

        if "chapters" not in parsed_result or not isinstance(parsed_result["chapters"], list):
            parsed_result["chapters"] = []
        if "learning_outcomes" not in parsed_result or not isinstance(parsed_result["learning_outcomes"], list):
            parsed_result["learning_outcomes"] = []
        if "syllabus_content" not in parsed_result or not parsed_result["syllabus_content"]:
            parsed_result["syllabus_content"] = f"Curriculum overview for {subject_name or 'the subject'}."

        return jsonify(parsed_result), 200

    except OpenAIError as oe:
        logger.error(f"OpenAI API Error: {str(oe)}", exc_info=True)
        return jsonify({
            "status": "error",
            "status_code": 502,
            "message": "OpenAI API Error",
            "details": str(oe)
        }), 502
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "status_code": 500,
            "message": "Internal Server Error",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production").lower() == "development"
    logger.info(f"Starting Edusoft AI server on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
