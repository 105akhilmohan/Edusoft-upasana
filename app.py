import os
import io
import re
import sys
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

# Configure logging with immediate stdout streaming
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)
EXTRACTION_DEBUG = os.getenv("EXTRACTION_DEBUG", "false").lower() in ("true", "1")

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
    get_textbook_chapter_record,
    get_textbook_content,
    register_extracted_textbook,
    validate_question,
    normalize_text_for_comparison,
    normalize_class,
    normalize_subject,
    normalize_chapter_key
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
        chapter_record = get_textbook_chapter_record(
            class_name=class_name,
            subject_name=subject_name,
            chapter_identifier=primary_chapter,
            payload_content=payload_content
        )
        textbook_content = chapter_record.get("content", "")
        real_chapter_name = chapter_record.get("chapter_name", primary_chapter)
        chapter_id = chapter_record.get("chapter_id", "")

        if not textbook_content:
            logger.warning(f"Textbook content not found for Class='{class_name}', Subject='{subject_name}', Chapter='{primary_chapter}'")
            return jsonify({
                "status": "error",
                "status_code": 404,
                "message": "Textbook content not found for the selected class, subject, and chapter."
            }), 404

        logger.info(
            "[QUESTION GENERATION]\nChapter=%r",
            real_chapter_name
        )

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
5. Chapter: {real_chapter_name}.
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
Target Chapter: {real_chapter_name}
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
  "chapters": {json.dumps([real_chapter_name] if chapters else [real_chapter_name])},
  "total_questions": {question_count},
  "difficulty": "{difficulty}",
  "questions": [
    {{
      "id": 1,
      "chapter": "{real_chapter_name}",
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
                chapter_name=real_chapter_name,
                textbook_content=textbook_content
            )

            if is_valid:
                logger.info(f"Question {idx}: VALID - '{q_text[:60]}...'")
                # Enforce clean fields
                clean_q = dict(q_obj)
                clean_q["id"] = len(valid_questions) + 1
                clean_q["chapter"] = real_chapter_name  # Exact authentic chapter name from textbook
                clean_q.pop("explanation", None)        # Enforce: no explanation
                clean_q.pop("options", None)            # Remove options since no MCQs
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
Chapter: {real_chapter_name}
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
                    chapter_name=real_chapter_name,
                    textbook_content=textbook_content
                )
                if is_valid:
                    clean_q = dict(q_obj)
                    clean_q["id"] = len(valid_questions) + 1
                    clean_q["chapter"] = real_chapter_name
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
            "chapters": [real_chapter_name] if chapters else [real_chapter_name],
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

def extract_page_lines_reading_order(page, page_num, debug_mode=False):
    """
    Coordinate-based word extractor for textbooks:
    1. Extracts words with font metadata and coordinates.
    2. Groups words into lines using vertical overlap / height tolerance.
    3. Dynamically detects columns from x-coordinate distribution (no fixed 50% split).
    4. Separates full-width headings from column blocks.
    5. Reconstructs top-to-bottom natural reading order across multi-column pages.
    6. Repairs line-break hyphenation while preserving genuine compound words.
    7. Detects and logs suspicious fragments without silently dropping or mangling words.
    """
    try:
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=False,
            extra_attrs=["fontname", "size"]
        )
    except Exception as e:
        logger.warning("[EXTRACT FAILED] page=%s error=%s", page_num, str(e))
        return [], 1, []

    if not words:
        return [], 1, []

    # Step A: Group words into lines using dynamic vertical overlap
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines = []
    current_line = []

    for w in sorted_words:
        if not current_line:
            current_line.append(w)
            continue

        line_top = min(item["top"] for item in current_line)
        line_bottom = max(item["bottom"] for item in current_line)
        line_height = max(1.0, line_bottom - line_top)
        w_height = max(1.0, w["bottom"] - w["top"])

        overlap = max(0.0, min(line_bottom, w["bottom"]) - max(line_top, w["top"]))
        min_h = min(line_height, w_height)

        if overlap >= 0.35 * min_h or abs(w["top"] - line_top) <= 3.5:
            current_line.append(w)
        else:
            sorted_l = sorted(current_line, key=lambda x: x["x0"])
            lines.append({
                "words": sorted_l,
                "text": " ".join(x.get("text", "") for x in sorted_l if x.get("text")).strip(),
                "x0": min(x["x0"] for x in sorted_l),
                "x1": max(x["x1"] for x in sorted_l),
                "top": min(x["top"] for x in sorted_l),
                "bottom": max(x["bottom"] for x in sorted_l),
                "max_size": max(float(x.get("size", 10.0) or 10.0) for x in sorted_l)
            })
            current_line = [w]

    if current_line:
        sorted_l = sorted(current_line, key=lambda x: x["x0"])
        lines.append({
            "words": sorted_l,
            "text": " ".join(x.get("text", "") for x in sorted_l if x.get("text")).strip(),
            "x0": min(x["x0"] for x in sorted_l),
            "x1": max(x["x1"] for x in sorted_l),
            "top": min(x["top"] for x in sorted_l),
            "bottom": max(x["bottom"] for x in sorted_l),
            "max_size": max(float(x.get("size", 10.0) or 10.0) for x in sorted_l)
        })

    # Step B: Dynamic Column Detection
    min_x = min(l["x0"] for l in lines)
    max_x = max(l["x1"] for l in lines)
    content_width = max(1.0, max_x - min_x)

    column_count = 1
    ordered_lines = []

    if len(lines) >= 8 and content_width >= 200:
        gutter_min_search = min_x + (content_width * 0.30)
        gutter_max_search = min_x + (content_width * 0.70)

        slice_candidates = []
        step = 5.0
        curr_x = gutter_min_search
        while curr_x <= gutter_max_search:
            crossing_count = sum(1 for l in lines if l["x0"] < (curr_x - 6) and l["x1"] > (curr_x + 6))
            left_count = sum(1 for l in lines if l["x1"] <= (curr_x + 6))
            right_count = sum(1 for l in lines if l["x0"] >= (curr_x - 6))
            slice_candidates.append((crossing_count, left_count, right_count, curr_x))
            curr_x += step

        if slice_candidates:
            slice_candidates.sort(key=lambda s: (s[0], -min(s[1], s[2])))
            best_crossing, left_c, right_c, best_gutter_x = slice_candidates[0]

            if left_c >= 4 and right_c >= 4 and best_crossing <= (0.22 * len(lines)):
                column_count = 2
                gutter_x = best_gutter_x

                current_left = []
                current_right = []

                def flush_columns():
                    col_lines = []
                    if current_left:
                        current_left.sort(key=lambda l: l["top"])
                        col_lines.extend(current_left)
                        current_left.clear()
                    if current_right:
                        current_right.sort(key=lambda l: l["top"])
                        col_lines.extend(current_right)
                        current_right.clear()
                    return col_lines

                vertical_sorted_lines = sorted(lines, key=lambda l: l["top"])

                for l in vertical_sorted_lines:
                    is_spanning = (l["x0"] < (gutter_x - 15) and l["x1"] > (gutter_x + 15)) or ((l["x1"] - l["x0"]) >= 0.70 * content_width)
                    if is_spanning:
                        col_lines = flush_columns()
                        if col_lines:
                            ordered_lines.extend(col_lines)
                        ordered_lines.append(l)
                    elif l["x1"] <= (gutter_x + 8):
                        current_left.append(l)
                    elif l["x0"] >= (gutter_x - 8):
                        current_right.append(l)
                    else:
                        if (gutter_x - l["x0"]) > (l["x1"] - gutter_x):
                            current_left.append(l)
                        else:
                            current_right.append(l)

                col_lines = flush_columns()
                if col_lines:
                    ordered_lines.extend(col_lines)

    if column_count == 1:
        ordered_lines = sorted(lines, key=lambda l: l["top"])

    raw_text_lines = [l["text"] for l in ordered_lines if l.get("text")]

    # Step C: Hyphenation repair: join split words across line breaks (e.g. "show-" + "ing" -> "showing")
    fixed_lines = []
    suspicious_fragments = []
    i = 0
    while i < len(raw_text_lines):
        line = raw_text_lines[i]

        if re.search(r"\bCh\s+(?:ing|is|made|care|devoted|down|in|it|so|passing)\b", line, re.IGNORECASE):
            suspicious_fragments.append(line)
            logger.warning("[SUSPICIOUS TEXT] page=%s line=%r", page_num, line)

        if i + 1 < len(raw_text_lines) and line.endswith("-") and not line.endswith("--"):
            next_line = raw_text_lines[i + 1]
            first_word_match = re.match(r"^([A-Za-z]+)(.*)$", next_line)
            last_word_match = re.search(r"([A-Za-z]+)-$", line)
            if first_word_match and last_word_match:
                prefix = last_word_match.group(1)
                suffix = first_word_match.group(1)
                rest_of_next_line = first_word_match.group(2).strip()

                if suffix.islower() or (prefix.isupper() and suffix.isupper()):
                    merged_word = prefix + suffix
                    line_without_hyphen = line[:last_word_match.start(1)]
                    new_first_line = (line_without_hyphen + merged_word).strip()
                    fixed_lines.append(new_first_line)

                    if rest_of_next_line:
                        raw_text_lines[i + 1] = rest_of_next_line
                        i += 1
                    else:
                        i += 2
                    continue

        fixed_lines.append(line)
        i += 1

    if debug_mode and (suspicious_fragments or page_num <= 5):
        logger.info(
            "[DEBUG] Page %s coords: cols=%s | lines=%s | content_w=%.1f | sample=%r",
            page_num, column_count, len(fixed_lines), content_width, fixed_lines[:3]
        )

    return fixed_lines, column_count, suspicious_fragments


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
    Dynamically scans the opening portion of the book for Table of Contents across multiple pages,
    and extracts all authentic chapter names directly from the book structure.
    Returns: (list_of_chapters, list_of_toc_page_numbers)
    """
    toc_text_pages = []
    toc_page_numbers = []

    max_search_pages = min(len(cleaned_pages), 80)
    toc_start_idx = None

    for idx, p in enumerate(cleaned_pages[:max_search_pages]):
        txt_upper = p["text"].upper()
        if any(h in txt_upper for h in ["CONTENTS", "TABLE OF CONTENTS", "INDEX OF CHAPTERS", "LIST OF CHAPTERS"]):
            toc_start_idx = idx
            break

    if toc_start_idx is not None:
        for idx in range(toc_start_idx, min(toc_start_idx + 12, len(cleaned_pages))):
            p = cleaned_pages[idx]
            if idx == toc_start_idx or re.search(r"(\.{2,}|\b(?:chapter|unit|section|part|ch)\b|\b\d{1,4}\b|[A-Z\s]{4,}\s+\d+)", p["text"], re.I):
                toc_page_numbers.append(p["page_number"])
                toc_text_pages.append(f"--- PAGE {p['page_number']} ---\n{p['text']}")
            else:
                break

    toc_combined = "\n\n".join(toc_text_pages)

    if toc_combined and len(toc_combined) > 40:
        logger.info("[TOC] Detected TOC pages: %s", toc_page_numbers)
        toc_prompt = f"""
You are an expert textbook curriculum analyzer.
Extract the COMPLETE list of ALL major chapters/units and their printed start pages from the Table of Contents text below.

MANDATORY RULES:
1. Extract EVERY SINGLE authentic chapter/section listed in this Table of Contents from the first chapter to the very last chapter. Do NOT stop early, omit, or summarize chapters.
2. Preserve the EXACT chapter titles as written in the Table of Contents.
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
      "chapter_name": "EXACT TITLE FROM TOC",
      "printed_page": 1
    }}
  ]
}}
"""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You extract all exact chapter titles from Table of Contents text. Return valid JSON only."},
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
                    if c_name and len(c_name) >= 3 and not c_name.lower().startswith("ch "):
                        clean_extracted.append({
                            "chapter_no": str(c.get("chapter_no") or idx),
                            "chapter_name": c_name,
                            "printed_page": c.get("printed_page")
                        })
                if clean_extracted:
                    logger.info("[TOC] Detected %s textbook chapters", len(clean_extracted))
                    return clean_extracted, toc_page_numbers
        except Exception as toc_err:
            logger.warning("[TOC ERROR] Error extracting TOC via OpenAI: %s", str(toc_err))

    # Fallback: Structural scan for explicit "CHAPTER I: ...", "CHAPTER 1 - ...", "CHAPTER ONE – ...", "UNIT 1: ..."
    logger.info("Scanning document for explicit chapter/unit headings via structural patterns...")
    discovered_chapters = []
    seen_names = set()

    word_to_num = {
        "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5",
        "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9", "TEN": "10",
        "ELEVEN": "11", "TWELVE": "12", "THIRTEEN": "13", "FOURTEEN": "14", "FIFTEEN": "15"
    }

    for p in cleaned_pages:
        p_num = p["page_number"]
        for line in p["lines"][:8]:
            line_str = line.strip()
            m = re.match(
                r"^(?:CHAPTER|UNIT|SECTION|PART)\s+([0-9IVXLCDM]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN)[\s:\.\-—–]+([A-Za-z0-9\s,\-\(\)]{3,80})$",
                line_str,
                re.IGNORECASE
            )
            if m:
                raw_num = m.group(1).strip().upper()
                ch_num = word_to_num.get(raw_num, raw_num)
                ch_name = m.group(2).strip()
                norm_n = normalize_chapter_key(ch_name)
                if norm_n not in seen_names and len(ch_name) >= 3:
                    seen_names.add(norm_n)
                    discovered_chapters.append({
                        "chapter_no": str(ch_num),
                        "chapter_name": ch_name,
                        "start_page": p_num
                    })

    if discovered_chapters:
        logger.info("[TOC] Detected %s textbook chapters", len(discovered_chapters))
        return discovered_chapters, toc_page_numbers

    logger.info("[TOC] Detected 1 textbook chapter (fallback)")
    return [{
        "chapter_no": "1",
        "chapter_name": subject_name or "Textbook Curriculum Content",
        "start_page": cleaned_pages[0]["page_number"] if cleaned_pages else 1
    }], toc_page_numbers


def check_page_for_heading(page_dict, c_name_norm, c_no=""):
    """
    Look for candidate chapter heading in the top portion (first 18 lines) of a page.
    Supports:
    - one-line exact heading
    - multiple consecutive lines forming the heading (wrapped headings)
    - explicit CHAPTER/UNIT heading followed by title or title on subsequent lines
    - whitespace and punctuation normalized only for comparison
    """
    lines = page_dict.get("lines", [])[:18]
    if not lines or not c_name_norm:
        return False

    def clean_str(s):
        s = s.replace("&", " AND ")
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.upper())).strip()

    target_clean = clean_str(c_name_norm)
    if not target_clean:
        return False

    clean_lines = [clean_str(l) for l in lines if clean_str(l)]
    if not clean_lines:
        return False

    # 1. Single line checks
    for l in clean_lines:
        if l == target_clean:
            return True
        if l.startswith("CHAPTER") or l.startswith("UNIT") or l.startswith("SECTION") or l.startswith("PART"):
            if target_clean in l:
                return True
        elif target_clean in l and len(l) <= len(target_clean) + 15:
            return True

    # 2. Consecutive 2-line checks
    for idx in range(len(clean_lines) - 1):
        l1 = clean_lines[idx]
        l2 = clean_lines[idx + 1]
        comb = f"{l1} {l2}".strip()
        if comb == target_clean:
            return True
        if (comb.startswith("CHAPTER") or comb.startswith("UNIT") or comb.startswith("SECTION") or comb.startswith("PART")) and target_clean in comb:
            return True
        if target_clean in comb and len(comb) <= len(target_clean) + 20:
            return True
        if re.match(r"^(?:CHAPTER|UNIT|SECTION|PART)\s+[0-9IVXLCDM]+$", l1):
            if l2 == target_clean or (target_clean in l2 and len(l2) <= len(target_clean) + 15):
                return True

    # 3. Consecutive 3-line checks
    for idx in range(len(clean_lines) - 2):
        l1 = clean_lines[idx]
        l2 = clean_lines[idx + 1]
        l3 = clean_lines[idx + 2]
        comb = f"{l1} {l2} {l3}".strip()
        if comb == target_clean:
            return True
        if (comb.startswith("CHAPTER") or comb.startswith("UNIT") or comb.startswith("SECTION") or comb.startswith("PART")) and target_clean in comb:
            return True
        if target_clean in comb and len(comb) <= len(target_clean) + 25:
            return True
        if re.match(r"^(?:CHAPTER|UNIT|SECTION|PART)\s+[0-9IVXLCDM]+$", l1):
            comb_23 = f"{l2} {l3}".strip()
            if comb_23 == target_clean or (target_clean in comb_23 and len(comb_23) <= len(target_clean) + 20):
                return True

    return False


def build_full_chapters_with_boundaries(cleaned_pages, chapter_map, subject_name=""):
    """
    Locates exact start_page and end_page for every chapter using multi-line wrapped heading matching
    and dynamic offset calculation, preventing paragraph substrings from falsely matching.
    Extracts the 100% COMPLETE readable textbook text belonging to each chapter.
    """
    if not cleaned_pages:
        return [], 0

    all_page_numbers = [p["page_number"] for p in cleaned_pages]
    page_lookup = {p["page_number"]: p for p in cleaned_pages}

    # Step 1: Calculate dynamic offset consensus (PDF page - Printed page)
    offset_samples = []
    for c in chapter_map:
        c_name = c["chapter_name"].strip()
        c_printed = c.get("printed_page")
        c_no = str(c.get("chapter_no", ""))

        if c_printed and isinstance(c_printed, int):
            for p in cleaned_pages:
                if check_page_for_heading(p, c_name, c_no):
                    offset_samples.append(p["page_number"] - c_printed)
                    break

    median_offset = 0
    if offset_samples:
        offset_samples.sort()
        median_offset = offset_samples[len(offset_samples) // 2]
        logger.info("[OFFSET] Dynamic consensus offset calculated: %s (samples=%s)", median_offset, offset_samples)

    # Step 2: Match exact standalone heading lines for every chapter with multi-stage bounded windows
    positioned_chapters = []

    for c in chapter_map:
        c_name = c["chapter_name"].strip()
        c_no = str(c.get("chapter_no", ""))
        c_printed = c.get("printed_page")

        mapped_page = None
        if c_printed and isinstance(c_printed, int):
            mapped_page = c_printed + median_offset
        elif c.get("start_page"):
            mapped_page = c["start_page"]

        actual_heading_page = None
        is_verified = False

        # STEP 3 & 4: Search reasonable window around mapped_page
        if mapped_page is not None:
            # Window 1: mapped_page - 5 through mapped_page + 8
            w1_pages = [p for p in cleaned_pages if (mapped_page - 5) <= p["page_number"] <= (mapped_page + 8)]
            for p in w1_pages:
                if check_page_for_heading(p, c_name, c_no):
                    actual_heading_page = p["page_number"]
                    is_verified = True
                    break

            # Window 2: wider bounded window mapped_page - 15 through mapped_page + 25
            if actual_heading_page is None:
                w2_pages = [p for p in cleaned_pages if (mapped_page - 15) <= p["page_number"] <= (mapped_page + 25)]
                for p in w2_pages:
                    if check_page_for_heading(p, c_name, c_no):
                        actual_heading_page = p["page_number"]
                        is_verified = True
                        break

        # Window 3: Document-wide scan across all pages
        if actual_heading_page is None:
            for p in cleaned_pages:
                if check_page_for_heading(p, c_name, c_no):
                    actual_heading_page = p["page_number"]
                    is_verified = True
                    break

        if is_verified:
            logger.info(
                "[CHAPTER VERIFY]\ncandidate=%r\nmapped_page=%s\nactual_heading_page=%s\nverified=true",
                c_name, mapped_page, actual_heading_page
            )
            final_start_page = actual_heading_page
        else:
            logger.warning(
                "[CHAPTER UNVERIFIED] name=%r toc_page=%s mapped_page=%s",
                c_name, c_printed, mapped_page
            )
            final_start_page = mapped_page if (mapped_page and mapped_page in page_lookup) else (c_printed if c_printed and c_printed in page_lookup else None)

        positioned_chapters.append({
            "chapter_no": c_no,
            "chapter_name": c_name,
            "start_page": final_start_page,
            "verified": is_verified
        })

    # Filter out chapters without any start_page and sort strictly by chronological start_page
    valid_positioned = [ch for ch in positioned_chapters if ch["start_page"] is not None]
    if not valid_positioned and cleaned_pages:
        valid_positioned = [{
            "chapter_no": "1",
            "chapter_name": subject_name or "Textbook Curriculum Content",
            "start_page": cleaned_pages[0]["page_number"],
            "verified": False
        }]

    valid_positioned.sort(key=lambda x: x["start_page"])

    # Step 3: Resolve end_page and assemble 100% COMPLETE readable textbook content
    final_chapters = []
    total_assigned_pages = 0

    for i, ch in enumerate(valid_positioned):
        start_p = ch["start_page"]
        if i + 1 < len(valid_positioned):
            next_start_p = valid_positioned[i + 1]["start_page"]
            end_p = max(start_p, next_start_p - 1)
        else:
            end_p = all_page_numbers[-1]

        ch_num = ch["chapter_no"] or str(i + 1)

        logger.info(
            "[TEXTBOOK CHAPTER]\n#%s name=%r\nstart=%s\nend=%s",
            ch_num, ch["chapter_name"], start_p, end_p
        )
        if ch.get("verified"):
            logger.info(
                "[CHAPTER VERIFIED]\nname=%r\nsource=textbook_pdf",
                ch["chapter_name"]
            )
        else:
            logger.warning(
                "[CHAPTER UNVERIFIED] name=%r",
                ch["chapter_name"]
            )

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

        logger.info(
            "[CHAPTER CONTENT]\nname=%r\ncharacters=%s",
            ch["chapter_name"], len(full_content)
        )

        if len(full_content.strip()) < 1000:
            logger.warning(
                "[SHORT CHAPTER CONTENT] name=%r characters=%s",
                ch["chapter_name"], len(full_content)
            )

        sample_snippet = full_content[:3000] if full_content else ch["chapter_name"]
        desc_lines = [l.strip() for l in sample_snippet.splitlines() if len(l.strip()) > 30 and not l.strip().startswith("---")]
        if desc_lines:
            description = " ".join(desc_lines[:3])[:300].strip()
            if not description.endswith("."):
                description += "."
        else:
            description = f"Comprehensive curriculum and textbook coverage of {ch['chapter_name']}."

        ch_id = normalize_chapter_key(ch["chapter_name"])

        final_chapters.append({
            "chapter_id": ch_id,
            "chapter_no": ch_num,
            "chapter_name": ch["chapter_name"],
            "chapter_name_source": "textbook_pdf",
            "chapter_name_verified": ch.get("verified", True),
            "source_pdf_page": start_p,
            "start_page": start_p,
            "end_page": end_p,
            "description": description,
            "content": full_content
        })

    logger.info("[EXTRACTION COMPLETE]\npages=%s\nchapters=%s", len(cleaned_pages), len(final_chapters))
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
        detected_column_counts = []
        all_suspicious_pages = set()
        all_suspicious_fragments = []

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

        logger.info("========== TEXTBOOK EXTRACTION START ==========")

        # Step 1: Extract all pages with bounding-box coordinate reading order
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                import pdfplumber
                with pdfplumber.open(temp_pdf_path) as pdf:
                    total_pages = len(pdf.pages)
                    logger.info("[PDF] Total pages=%s", total_pages)

                    for page_num, page in enumerate(pdf.pages, start=1):
                        lines, col_count, suspicious = extract_page_lines_reading_order(
                            page=page,
                            page_num=page_num,
                            debug_mode=EXTRACTION_DEBUG
                        )
                        detected_column_counts.append(col_count)
                        if suspicious:
                            all_suspicious_pages.add(page_num)
                            all_suspicious_fragments.extend(suspicious)

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

                        logger.info(
                            "[PDF PAGE] %s/%s extracted",
                            page_num,
                            total_pages
                        )

                        if page_num % 10 == 0 or page_num == total_pages:
                            logger.info(
                                "[PROGRESS] %s/%s pages",
                                page_num,
                                total_pages
                            )

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
            detected_column_counts = [1] * total_pages

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

        # Step 2: Clean repeated running headers & footers conservatively
        cleaned_pages = clean_page_headers_and_footers(raw_page_records)

        # Step 3: Dynamically detect Table of Contents & authentic chapter list
        chapter_map, toc_page_numbers = detect_table_of_contents_and_chapters(
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

        logger.info("[EXTRACT RESULT] chapters=%s", len(final_chapters))
        for chapter in final_chapters:
            logger.info(
                "[EXTRACTED CHAPTER] no=%s name=%r start=%s end=%s chars=%s",
                chapter.get("chapter_no"),
                chapter.get("chapter_name"),
                chapter.get("start_page"),
                chapter.get("end_page"),
                len(chapter.get("content", ""))
            )

        # Register authentic chapters in textbook registry for subsequent question generation
        register_extracted_textbook(
            subject_name=subject_name,
            subject_code=subject_code,
            chapters=final_chapters
        )

        avg_columns = (sum(detected_column_counts) / len(detected_column_counts)) if detected_column_counts else 1

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
                "toc_pages_detected": len(toc_page_numbers),
                "chapters_detected": len(final_chapters),
                "pages_assigned_to_chapters": total_pages_assigned,
                "columns_detected": round(avg_columns),
                "suspicious_pages": len(all_suspicious_pages),
                "suspicious_fragments": len(all_suspicious_fragments)
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

