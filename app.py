import os
import io
import re
import json
import base64
import logging
import tempfile
import gc
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

        # Difficulty & Question Types (Descriptive & Application-level, No MCQs)
        difficulty = data.get("difficulty", "Medium")
        q_types_raw = data.get("question_types") or data.get("question_type") or ["Descriptive", "Short Answer", "Long Answer", "Application", "Case-Based"]
        if isinstance(q_types_raw, list):
            question_types = [str(t).strip() for t in q_types_raw if str(t).strip() and str(t).strip().upper() != "MCQ"]
        elif isinstance(q_types_raw, str) and q_types_raw.strip():
            question_types = [t.strip() for t in q_types_raw.split(",") if t.strip() and t.strip().upper() != "MCQ"]
        else:
            question_types = ["Descriptive", "Short Answer", "Long Answer", "Application", "Case-Based"]

        if not question_types:
            question_types = ["Descriptive", "Short Answer", "Long Answer", "Application", "Case-Based"]

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
                "status_code": 404,
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
You are an expert curriculum designer and examination question generator for Edusoft.

MANDATORY RULES:
1. The supplied textbook content is the ONLY source of truth.
2. Every question, model answer, and step marking MUST be directly and strictly derived from the provided textbook content below.
3. Target Class Level: Class {class_name}.
4. Target Subject: {subject_name}.
5. Chapter: {primary_chapter}.
6. QUESTION FORMATS: Generate ONLY Descriptive (Short Answer, Long Answer, Analytical) and Application-level / Case-Based questions. Do NOT generate Multiple Choice Questions (MCQs).
7. Application-level questions must require practical problem-solving, clinical/functional correlations, or scenario analysis based strictly on the textbook content.
8. Include thorough, step-by-step marking schemes and complete model answers.
9. Do NOT include any 'explanation' field.
10. Return strictly valid JSON only.
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
- Focus on Descriptive questions (Short Answer, Long Answer, Conceptual Breakdowns) and Application-Level questions (Clinical/Practical Scenarios, Problem-Solving).
- Absolutely NO Multiple Choice Questions (MCQs).
- Provide detailed model answers (`correct_answer`) and granular step-by-step marking schemes (`step_marking`).
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
      "type": "Descriptive | Short Answer | Long Answer | Application | Case-Based | Numerical",
      "marks": 5,
      "question": "Detailed descriptive or application-level question text here",
      "correct_answer": "Complete, comprehensive model answer based strictly on textbook content",
      "step_marking": [
        {{
          "step": "Specific concept, step, or key point",
          "marks": 2
        }},
        {{
          "step": "Detailed clinical/practical explanation or conclusion",
          "marks": 3
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
                clean_q.pop("options", None)           # Remove options since no MCQs
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
# 3. FULL TEXTBOOK & SYLLABUS PDF EXTRACTION PIPELINE (Reading Order + TOC + Boundaries)
# ==============================================================================

def extract_page_lines_reading_order(page, page_num):
    """
    Extract words from a pdfplumber page using bounding-box coordinates to reconstruct
    natural reading order, handle two-column vs single-column layouts, and repair line-break hyphens.
    Optimized for low-memory footprint on large textbooks.
    """
    try:
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=False,
            extra_attrs=[]
        )
    except Exception as e:
        logger.warning(f"extract_words failed on page {page_num}: {e}. Falling back to extract_text.")
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
        return [l.strip() for l in raw_text.splitlines() if l.strip()]

    if not words:
        return []

    page_width = float(page.width or 612.0)
    page_height = float(page.height or 792.0)
    mid_x = page_width / 2.0

    # Determine if page is two-column
    left_words = [w for w in words if w["x1"] <= (mid_x - 8)]
    right_words = [w for w in words if w["x0"] >= (mid_x + 8)]
    spanning_words = [w for w in words if w["x0"] < mid_x and w["x1"] > mid_x]

    is_two_column = (
        len(left_words) > 20 and
        len(right_words) > 20 and
        len(spanning_words) < (0.15 * len(words))
    )

    def group_words_into_lines(word_list):
        if not word_list:
            return []
        # Quantize vertical position to group words on the same line
        sorted_words = sorted(word_list, key=lambda w: (round(w["top"] / 3.5) * 3.5, w["x0"]))
        lines = []
        current_line_words = []
        current_top = None

        for w in sorted_words:
            w_top = w["top"]
            if current_top is None:
                current_top = w_top
                current_line_words.append(w)
            elif abs(w_top - current_top) <= 3.5:
                current_line_words.append(w)
            else:
                sorted_line = sorted(current_line_words, key=lambda x: x["x0"])
                line_str = " ".join(x["text"] for x in sorted_line if x.get("text"))
                if line_str.strip():
                    lines.append(line_str.strip())
                current_line_words = [w]
                current_top = w_top

        if current_line_words:
            sorted_line = sorted(current_line_words, key=lambda x: x["x0"])
            line_str = " ".join(x["text"] for x in sorted_line if x.get("text"))
            if line_str.strip():
                lines.append(line_str.strip())

        return lines

    if is_two_column:
        header_words = [w for w in spanning_words if w["top"] < (page_height * 0.18)]
        header_lines = group_words_into_lines(header_words) if header_words else []

        left_lines = group_words_into_lines(left_words)
        right_lines = group_words_into_lines(right_words)

        footer_words = [w for w in spanning_words if w["top"] > (page_height * 0.85)]
        footer_lines = group_words_into_lines(footer_words) if footer_words else []

        raw_lines = header_lines + left_lines + right_lines + footer_lines
    else:
        raw_lines = group_words_into_lines(words)

    # Hyphenation repair: join split words across line breaks (e.g. "show-" + "ing" -> "showing")
    fixed_lines = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if i + 1 < len(raw_lines) and line.endswith("-") and not line.endswith("--"):
            next_line = raw_lines[i + 1]
            first_word_match = re.match(r"^([A-Za-z]+)(.*)$", next_line)
            last_word_match = re.search(r"([A-Za-z]+)-$", line)
            if first_word_match and last_word_match:
                prefix = last_word_match.group(1)
                suffix = first_word_match.group(1)
                rest_of_next_line = first_word_match.group(2).strip()

                merged_word = prefix + suffix
                line_without_hyphen_word = line[:last_word_match.start(1)]
                new_first_line = (line_without_hyphen_word + merged_word).strip()
                fixed_lines.append(new_first_line)

                if rest_of_next_line:
                    raw_lines[i + 1] = rest_of_next_line
                    i += 1
                else:
                    i += 2
                continue
        fixed_lines.append(line)
        i += 1

    return fixed_lines


def clean_page_headers_and_footers(page_records):
    """
    Conservatively detect and remove repeated running headers, footers,
    standalone page numbers, and digitization stamps across pages,
    while strictly preserving legitimate chapter titles, subsection headings,
    figure captions, and body text.
    """
    if not page_records:
        return []

    top_line_freq = {}
    bottom_line_freq = {}
    total_pages = len(page_records)

    for rec in page_records:
        lines = rec.get("lines", [])
        if lines:
            top_line = lines[0].strip().upper()
            top_line_freq[top_line] = top_line_freq.get(top_line, 0) + 1
        if len(lines) > 1:
            bottom_line = lines[-1].strip().upper()
            bottom_line_freq[bottom_line] = bottom_line_freq.get(bottom_line, 0) + 1

    # Detect repeated headers/footers appearing on multiple pages (>= 4% of book)
    threshold = max(3, int(total_pages * 0.04)) if total_pages >= 15 else 2
    repeated_top_headers = {k for k, v in top_line_freq.items() if v >= threshold and len(k) > 2}
    repeated_bottom_footers = {k for k, v in bottom_line_freq.items() if v >= threshold and len(k) > 2}

    archive_markers = ["INTERNET ARCHIVE", "DIGITIZED BY", "MICROSOFT CORP", "GOOGLE BOOK", "LIBRARY OF", "UNIVERSITY OF"]

    cleaned_records = []
    for rec in page_records:
        lines = list(rec.get("lines", []))
        p_num = rec["page_number"]
        if not lines:
            continue

        # Check top line 1
        if lines:
            top = lines[0].strip()
            top_upper = top.upper()
            if re.match(r"^\d{1,4}$", top):
                lines.pop(0)
            elif top_upper in repeated_top_headers:
                lines.pop(0)
            elif any(stamp in top_upper for stamp in archive_markers):
                lines.pop(0)

        # Check top line 2
        if lines:
            top = lines[0].strip()
            top_upper = top.upper()
            if re.match(r"^\d{1,4}$", top):
                lines.pop(0)
            elif top_upper in repeated_top_headers:
                lines.pop(0)

        # Check bottom line 1
        if lines:
            bot = lines[-1].strip()
            bot_upper = bot.upper()
            if re.match(r"^\d{1,4}$", bot):
                lines.pop()
            elif bot_upper in repeated_bottom_footers:
                lines.pop()
            elif any(stamp in bot_upper for stamp in archive_markers):
                lines.pop()

        # Check bottom line 2
        if lines:
            bot = lines[-1].strip()
            bot_upper = bot.upper()
            if re.match(r"^\d{1,4}$", bot):
                lines.pop()
            elif bot_upper in repeated_bottom_footers:
                lines.pop()

        cleaned_text = "\n".join(lines).strip()
        if cleaned_text:
            cleaned_records.append({
                "page_number": p_num,
                "text": cleaned_text,
                "lines": lines
            })

    return cleaned_records


def detect_table_of_contents_and_chapters(cleaned_pages, client, model, subject_name=""):
    """
    Dynamically scans the opening portion of the book for Table of Contents,
    and extracts authentic chapter names directly from the book structure.
    Returns list of dicts: [{"chapter_no": "1", "chapter_name": "...", "printed_page": 16}, ...]
    """
    toc_text_pages = []
    toc_detected = False
    toc_start_page = None

    # Dynamically search opening pages (up to first 60 pages) for TOC
    max_search_pages = min(len(cleaned_pages), 60)
    for p in cleaned_pages[:max_search_pages]:
        txt_upper = p["text"].upper()
        if any(h in txt_upper for h in ["CONTENTS", "TABLE OF CONTENTS", "INDEX OF CHAPTERS", "LIST OF CHAPTERS"]):
            toc_detected = True
            if toc_start_page is None:
                toc_start_page = p["page_number"]
            toc_text_pages.append(f"--- PAGE {p['page_number']} ---\n{p['text']}")
        elif toc_detected:
            # Continue collecting consecutive TOC pages
            if re.search(r"(\.{3,}|\b(?:chapter|unit|section|part)\b|\b\d{1,4}\b)", p["text"], re.I):
                toc_text_pages.append(f"--- PAGE {p['page_number']} ---\n{p['text']}")
            else:
                break

    toc_combined = "\n\n".join(toc_text_pages)

    if toc_combined and len(toc_combined) > 40:
        logger.info(f"TOC pages detected: {len(toc_text_pages)} (Starting at page {toc_start_page})")
        toc_prompt = f"""
You are an expert textbook curriculum analyzer.
Extract the exact major chapters/units and their printed start pages from the Table of Contents text below.

MANDATORY RULES:
1. ONLY extract authentic chapter/section titles explicitly listed in this Table of Contents.
2. Do NOT invent, summarize, or alter chapter names. Preserve the exact textbook titles (e.g. 'THE SCALP', 'THE SKULL', 'THE MENINGES', 'THE BRAIN').
3. Extract 'chapter_no', 'chapter_name', and 'printed_page' (integer if visible, else null).
4. Return valid JSON only.

============================================================
TABLE OF CONTENTS TEXT:
============================================================
{toc_combined}

JSON SCHEMA:
{{
  "chapters": [
    {{
      "chapter_no": "1",
      "chapter_name": "THE SCALP",
      "printed_page": 16
    }}
  ]
}}
"""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You extract exact chapter titles from Table of Contents text. Return valid JSON only."},
                    {"role": "user", "content": toc_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            parsed = json.loads(resp.choices[0].message.content)
            extracted = parsed.get("chapters", [])
            if isinstance(extracted, list) and len(extracted) > 0:
                clean_extracted = []
                for idx, c in enumerate(extracted, start=1):
                    c_name = str(c.get("chapter_name", "")).strip()
                    # Filter out short invalid artifacts
                    if c_name and len(c_name) >= 3 and not c_name.lower().startswith("ch "):
                        clean_extracted.append({
                            "chapter_no": str(c.get("chapter_no") or idx),
                            "chapter_name": c_name,
                            "printed_page": c.get("printed_page")
                        })
                if clean_extracted:
                    logger.info(f"Chapters detected from TOC: {len(clean_extracted)}")
                    return clean_extracted
        except Exception as toc_err:
            logger.warning(f"Error extracting TOC via OpenAI: {toc_err}")

    # Fallback: Structural scan for explicit "CHAPTER I: ...", "CHAPTER 1 - ...", "UNIT 1: ..."
    logger.info("Scanning document for explicit chapter/unit headings via structural patterns...")
    discovered_chapters = []
    seen_names = set()

    for p in cleaned_pages:
        p_num = p["page_number"]
        for line in p["lines"][:8]:
            line_str = line.strip()
            # Match strictly explicit chapter titles (e.g. "CHAPTER 1. THE SCALP", "CHAPTER I — THE SKULL")
            m = re.match(r"^(?:CHAPTER|UNIT|SECTION|PART)\s+([0-9IVXLCDM]+)[\s:\.\-—]+([A-Za-z0-9\s,\-\(\)]{3,80})$", line_str, re.IGNORECASE)
            if m:
                ch_num = m.group(1).strip()
                ch_name = m.group(2).strip()
                norm_n = ch_name.upper()
                if norm_n not in seen_names and len(ch_name) >= 3:
                    seen_names.add(norm_n)
                    discovered_chapters.append({
                        "chapter_no": str(ch_num),
                        "chapter_name": ch_name,
                        "start_page": p_num
                    })

    if discovered_chapters:
        logger.info(f"Discovered {len(discovered_chapters)} chapters from structural heading scan.")
        return discovered_chapters

    # Ultimate fallback: single chapter representing the whole textbook
    return [{
        "chapter_no": "1",
        "chapter_name": subject_name or "Textbook Curriculum Content",
        "start_page": cleaned_pages[0]["page_number"] if cleaned_pages else 1
    }]


def build_full_chapters_with_boundaries(cleaned_pages, chapter_map, subject_name=""):
    """
    Locates exact start_page and end_page for every chapter using standalone heading matching
    and proximity search, preventing paragraph substrings from falsely matching.
    Extracts the 100% COMPLETE readable textbook text belonging to each chapter.
    """
    if not cleaned_pages:
        return []

    all_page_numbers = [p["page_number"] for p in cleaned_pages]
    page_lookup = {p["page_number"]: p for p in cleaned_pages}

    positioned_chapters = []

    for c in chapter_map:
        c_name = c["chapter_name"].strip()
        c_no = str(c.get("chapter_no", ""))
        c_printed = c.get("printed_page")
        c_name_norm = re.sub(r"[^\w\s]", "", c_name.upper()).strip()

        matched_page = None

        # Strategy 1: Proximity search around printed_page (±45 pages)
        candidate_pages = []
        if c_printed and isinstance(c_printed, int):
            candidate_pages = [p for p in cleaned_pages if abs(p["page_number"] - c_printed) <= 45]
        
        # Check candidate pages first for standalone line match
        for p in candidate_pages:
            p_lines = p.get("lines", [])
            for line in p_lines[:12]:  # Headings appear in top 12 lines of chapter start page
                line_clean = re.sub(r"[^\w\s]", "", line.upper()).strip()
                # Exact line match or explicit chapter title line
                if line_clean == c_name_norm or (line_clean.startswith("CHAPTER") and c_name_norm in line_clean):
                    matched_page = p["page_number"]
                    break
            if matched_page is not None:
                break

        # Strategy 2: Search all pages for standalone heading line
        if matched_page is None:
            for p in cleaned_pages:
                p_lines = p.get("lines", [])
                for line in p_lines[:12]:
                    line_clean = re.sub(r"[^\w\s]", "", line.upper()).strip()
                    if line_clean == c_name_norm or (line_clean.startswith("CHAPTER") and c_name_norm in line_clean):
                        matched_page = p["page_number"]
                        break
                    elif c_name_norm in line_clean and len(line_clean) <= len(c_name_norm) + 15:
                        # Near exact line match
                        matched_page = p["page_number"]
                        break
                if matched_page is not None:
                    break

        # Log match or rejection
        if matched_page is not None:
            logger.info(f"Chapter heading matched: '{c_name}' -> PDF page {matched_page}")
        else:
            # Fallback to existing start_page or printed_page if within page bounds
            if c.get("start_page"):
                matched_page = c["start_page"]
            elif c_printed and c_printed in page_lookup:
                matched_page = c_printed
            else:
                matched_page = all_page_numbers[0]
            logger.warning(f"WARNING: rejected possible chapter heading: '{c_name}' (No standalone heading line found, assigned to page {matched_page})")

        positioned_chapters.append({
            "chapter_no": c_no or str(len(positioned_chapters) + 1),
            "chapter_name": c_name,
            "start_page": matched_page
        })

    # Sort chapters strictly by start_page
    positioned_chapters.sort(key=lambda x: x["start_page"])

    # Resolve end_page and assemble 100% COMPLETE original cleaned content per chapter
    final_chapters = []
    total_assigned_pages = 0

    for i, ch in enumerate(positioned_chapters):
        start_p = ch["start_page"]
        if i + 1 < len(positioned_chapters):
            next_start_p = positioned_chapters[i + 1]["start_page"]
            end_p = max(start_p, next_start_p - 1)
        else:
            end_p = all_page_numbers[-1]

        logger.info(f"Chapter boundary: '{ch['chapter_name']}' pages {start_p}-{end_p}")

        # Gather ALL pages from start_p to end_p
        chapter_pages_text = []
        for p in cleaned_pages:
            if start_p <= p["page_number"] <= end_p:
                chapter_pages_text.append(f"--- PAGE {p['page_number']} ---\n{p['text']}")
                total_assigned_pages += 1

        full_content = "\n\n".join(chapter_pages_text).strip()
        if not full_content and cleaned_pages:
            for p in cleaned_pages:
                if p["page_number"] == start_p:
                    full_content = f"--- PAGE {p['page_number']} ---\n{p['text']}"
                    break

        # Generate concise description separately from first few content paragraphs
        sample_snippet = full_content[:3000] if full_content else ch["chapter_name"]
        desc_lines = [l.strip() for l in sample_snippet.splitlines() if len(l.strip()) > 30 and not l.strip().startswith("---")]
        if desc_lines:
            description = " ".join(desc_lines[:3])[:300].strip()
            if not description.endswith("."):
                description += "."
        else:
            description = f"Comprehensive curriculum and textbook coverage of {ch['chapter_name']}."

        final_chapters.append({
            "chapter_no": ch["chapter_no"],
            "chapter_name": ch["chapter_name"],
            "start_page": start_p,
            "end_page": end_p,
            "description": description,
            "content": full_content  # COMPLETE READABLE ORIGINAL TEXTBOOK CONTENT PRESERVED!
        })

    logger.info(f"Total pages assigned: {total_assigned_pages}")
    return final_chapters, total_assigned_pages


@app.route("/api/extract-syllabus", methods=["POST"])
@app.route("/extract-syllabus", methods=["POST"])
@app.route("/api/parse-syllabus-pdf", methods=["POST"])
def extract_syllabus():
    """
    Extract text from uploaded full textbook or syllabus PDF using pdfplumber,
    reconstruct coordinate reading order, clean headers/footers, detect chapters
    via Table of Contents, map exact chapter boundaries, and preserve 100% full chapter content.
    Uses disk-backed streaming and aggressive memory reclamation to prevent OOM on 600+ page books.
    """
    temp_pdf_path = None
    try:
        subject_name = ""
        subject_code = ""
        extracted_text = ""
        raw_page_records = []
        total_pages = 0
        pages_with_text = 0

        # Case 1: multipart/form-data upload
        if request.files:
            file_obj = request.files.get("file") or request.files.get("pdf") or request.files.get("document")
            if file_obj:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    temp_pdf_path = tmp.name
                    file_obj.save(tmp.name)
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
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        temp_pdf_path = tmp.name
                        tmp.write(pdf_bytes)
                    del pdf_bytes
                    gc.collect()
                except Exception as b64_err:
                    return jsonify({
                        "status": "error",
                        "status_code": 400,
                        "message": f"Failed to decode base64 PDF: {str(b64_err)}"
                    }), 400
            elif "pdf_text" in data or "text" in data or "content" in data:
                extracted_text = str(data.get("pdf_text") or data.get("text") or data.get("content")).strip()
        else:
            model = DEFAULT_MODEL or "gpt-4o-mini"

        # Step 1: Extract all pages with bounding-box coordinate reading order
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                import pdfplumber
                with pdfplumber.open(temp_pdf_path) as pdf:
                    total_pages = len(pdf.pages)
                    logger.info(f"Total PDF pages: {total_pages}")
                    for page_num, page in enumerate(pdf.pages, start=1):
                        lines = extract_page_lines_reading_order(page, page_num)
                        try:
                            page.flush_cache()
                        except Exception:
                            pass
                        if lines:
                            clean_t = "\n".join(lines).strip()
                            raw_page_records.append({
                                "page_number": page_num,
                                "raw_text": clean_t,
                                "lines": lines
                            })
                            pages_with_text += 1

                        # Periodic garbage collection for large textbooks
                        if page_num % 25 == 0:
                            gc.collect()

            except Exception as pdf_err:
                logger.error(f"Error parsing PDF with pdfplumber: {str(pdf_err)}", exc_info=True)
                return jsonify({
                    "status": "error",
                    "status_code": 400,
                    "message": f"Error parsing PDF file: {str(pdf_err)}"
                }), 400

        elif extracted_text:
            raw_lines = extracted_text.splitlines()
            current_page_num = 1
            current_page_lines = []
            for line in raw_lines:
                if line.strip().startswith("--- PAGE ") or line.strip().startswith("PAGE "):
                    if current_page_lines:
                        clean_t = "\n".join(current_page_lines).strip()
                        raw_page_records.append({
                            "page_number": current_page_num,
                            "raw_text": clean_t,
                            "lines": [l.strip() for l in clean_t.splitlines() if l.strip()]
                        })
                        current_page_lines = []
                        current_page_num += 1
                current_page_lines.append(line)
            if current_page_lines:
                clean_t = "\n".join(current_page_lines).strip()
                raw_page_records.append({
                    "page_number": current_page_num,
                    "raw_text": clean_t,
                    "lines": [l.strip() for l in clean_t.splitlines() if l.strip()]
                })

            if not raw_page_records and extracted_text.strip():
                clean_t = extracted_text.strip()
                raw_page_records.append({
                    "page_number": 1,
                    "raw_text": clean_t,
                    "lines": [l.strip() for l in clean_t.splitlines() if l.strip()]
                })

            total_pages = len(raw_page_records)
            pages_with_text = len(raw_page_records)

        # Graceful handling of scanned/image-only PDFs
        if not raw_page_records or pages_with_text == 0:
            return jsonify({
                "status": "error",
                "status_code": 400,
                "message": (
                    f"No readable text could be extracted from the provided PDF ({total_pages} total pages checked). "
                    "The document appears to contain scanned or image-only pages without an embedded text layer. "
                    "Please ensure a valid text-based or OCR-processed PDF is uploaded."
                ),
                "total_pages": total_pages,
                "pages_with_text": 0
            }), 400

        logger.info(f"Pages with text: {pages_with_text}")

        # Step 2: Clean repeated running headers & footers conservatively
        cleaned_pages = clean_page_headers_and_footers(raw_page_records)

        # Step 3: Dynamically detect Table of Contents & authentic chapter list
        chapter_map = detect_table_of_contents_and_chapters(
            cleaned_pages=cleaned_pages,
            client=client,
            model=model,
            subject_name=subject_name
        )

        # Step 4: Map chapter boundaries & assemble 100% COMPLETE textbook content per chapter
        final_chapters, total_pages_assigned = build_full_chapters_with_boundaries(
            cleaned_pages=cleaned_pages,
            chapter_map=chapter_map,
            subject_name=subject_name
        )

        # Build final response payload matching exact requested schema
        response_payload = {
            "status": "success",
            "subject_name": subject_name,
            "subject_code": subject_code,
            "total_pages": total_pages,
            "chapters": final_chapters,
            "learning_outcomes": [],
            "extraction_metadata": {
                "total_pdf_pages": total_pages,
                "pages_with_text": pages_with_text,
                "chapters_detected": len(final_chapters),
                "pages_assigned_to_chapters": total_pages_assigned
            }
        }

        return jsonify(response_payload), 200

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
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception:
                pass
        gc.collect()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production").lower() == "development"
    logger.info(f"Starting Edusoft AI server on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
