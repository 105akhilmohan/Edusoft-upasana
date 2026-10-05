import os
import json
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
        "version": "1.2.0",
        "endpoints": {
            "health": "GET /health",
            "generate_insights": "POST /api/generate-insights",
            "generate_questions": "POST /api/generate-questions"
        }
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
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
            "data": response_data,
            "insights": insights
        }), 200

    except OpenAIError as oe:
        logger.error(f"OpenAI API Error: {str(oe)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "OpenAI API Error",
            "details": str(oe)
        }), 502
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Internal Server Error",
            "details": str(e)
        }), 500


# ==============================================================================
# 2. QUESTION GENERATION ENDPOINT (Explanation Removed)
# ==============================================================================
@app.route("/api/generate-questions", methods=["POST"])
@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    """
    Generate educational questions for Edusoft without explanation.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body must be a valid JSON object."
            }), 400

        # Extract Fields with fallbacks
        class_name = data.get("class_name", "")
        subject_name = data.get("subject_name") or data.get("topic", "")
        
        # Chapters
        chapters_raw = data.get("chapters", [])
        if isinstance(chapters_raw, list):
            chapters = [str(c).strip() for c in chapters_raw if str(c).strip()]
        elif isinstance(chapters_raw, str) and chapters_raw.strip():
            chapters = [chapters_raw.strip()]
        else:
            chapters = []

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

        # Difficulty
        difficulty = data.get("difficulty", "Medium")

        # Question Types
        q_types_raw = data.get("question_types") or data.get("question_type") or ["MCQ", "Short", "Long"]
        if isinstance(q_types_raw, list):
            question_types = [str(t).strip() for t in q_types_raw if str(t).strip()]
        elif isinstance(q_types_raw, str) and q_types_raw.strip():
            question_types = [q_types_raw.strip()]
        else:
            question_types = ["MCQ", "Short", "Long"]

        # Suggestions / Description
        suggestions = data.get("suggestions") or data.get("description", "")
        exclude_questions = data.get("exclude_questions", [])
        model = data.get("model", DEFAULT_MODEL)

        if not subject_name and not chapters:
            return jsonify({
                "status": "error",
                "message": "At least 'subject_name' (or 'topic') or 'chapters' must be provided."
            }), 400

        # Generate a unique generation nonce to prevent deterministic repetition
        import uuid
        import time
        generation_nonce = f"{uuid.uuid4().hex[:8]}-{int(time.time() * 1000)}"

        context_parts = []
        if class_name:
            context_parts.append(f"Target Class / Grade: {class_name}")
        if subject_name:
            context_parts.append(f"Subject / Topic: {subject_name}")
        chapters_formatted = ("\n  - " + "\n  - ".join(chapters)) if chapters else f"All topics under {subject_name}"
        if chapters:
            context_parts.append(f"Chapters Covered:{chapters_formatted}")
        context_parts.append(f"Difficulty Level: {difficulty}")
        context_parts.append(f"Requested Question Types: {', '.join(question_types)}")
        if suggestions:
            context_parts.append(f"Specific Suggestions & Requirements: {suggestions}")
        if exclude_questions and isinstance(exclude_questions, list):
            context_parts.append(f"Exclude These Previously Generated Questions (Do not repeat):\n  - " + "\n  - ".join([str(q) for q in exclude_questions[:10]]))
        
        # Dynamically build disallowed subjects list excluding the current subject
        all_common_subjects = [
            "Mathematics", "Physics", "Chemistry", "Biology", "General Science",
            "History", "Geography", "Civics", "Social Science", "Economics",
            "English Language", "Malayalam", "Hindi", "Computer Science", "General Knowledge"
        ]
        disallowed_subjects = [s for s in all_common_subjects if s.lower() not in subject_name.lower() and subject_name.lower() not in s.lower()]
        disallowed_formatted = "\n- ".join(disallowed_subjects)

        # Grade-level cognitive syllabus guidance
        class_str_clean = str(class_name).lower().replace("class", "").replace("grade", "").strip()
        if class_str_clean in ["1", "i", "first", "one"]:
            grade_guidance = (
                "CRITICAL CLASS LEVEL CONSTRAINT (CLASS 1 - PRIMARY SCHOOL / AGE 5-6):\n"
                "- Questions MUST be extremely simple elementary school level suitable for a 6-year-old child.\n"
                "- For Mathematics: Single-digit addition and subtraction (e.g., 4 + 3 = ?, 7 - 2 = ?), counting objects (1 to 20), simple patterns, recognizing basic numbers and shapes (circle, square, triangle).\n"
                "- STRICTLY FORBIDDEN: Physics, Optics, Lens formulas, Newton's Laws, Kinetic Energy, Vectors, Friction, Chemistry, Biology, high-school algebra, and advanced formulas. Any question containing secondary school terminology will be rejected immediately.\n"
            )
        elif class_str_clean in ["2", "ii", "second", "two"]:
            grade_guidance = (
                "CRITICAL CLASS LEVEL CONSTRAINT (CLASS 2 - PRIMARY SCHOOL / AGE 6-7):\n"
                "- Questions MUST be simple elementary level: addition/subtraction up to 2-digit numbers (within 100), basic skip counting, simple word problems with toys/fruits/candies.\n"
                "- STRICTLY FORBIDDEN: Secondary/high school physics, chemistry, biology, or advanced mechanics.\n"
            )
        elif class_str_clean in ["3", "iii", "third", "three"]:
            grade_guidance = (
                "CRITICAL CLASS LEVEL CONSTRAINT (CLASS 3 - PRIMARY SCHOOL / AGE 7-8):\n"
                "- Questions MUST be elementary school level: basic multiplication tables, simple division, elementary place value, basic word problems.\n"
            )
        elif class_str_clean in ["4", "iv", "fourth", "5", "v", "fifth"]:
            grade_guidance = (
                f"CRITICAL CLASS LEVEL CONSTRAINT (CLASS {class_name} - PRIMARY SCHOOL):\n"
                "- Questions MUST strictly follow elementary primary school syllabus for this grade.\n"
            )
        else:
            grade_guidance = f"Keep the questions strictly aligned with the standard school curriculum for Class {class_name}."

        system_prompt = f"""
You are an expert school curriculum designer and examination question generator for Edusoft.

PRIMARY RULE: ABSOLUTE SUBJECT, CHAPTER, AND CLASS/GRADE LEVEL COMPLIANCE.

You MUST generate questions strictly matching:
- SUBJECT: {subject_name}
- CHAPTER(S): {chapters_formatted}
- CLASS LEVEL: Class {class_name}

STRICT SCOPE RULES:
1. Every question MUST belong 100% to the subject "{subject_name}".
2. Every question MUST belong 100% to the specified chapter(s): {chapters_formatted}.
3. Under NO circumstances should you generate questions from any other subject (such as Science/Physics when Mathematics is requested).
4. Do NOT attempt to camouflage questions from other subjects by merely appending the chapter name into the question text.
5. All concepts, vocabulary, and numerical values MUST be strictly appropriate for Class {class_name}.
6. For MCQs, all 4 options must belong to {subject_name} and the specified chapter.
7. For Short/Long/Numerical questions, provide clear model answers and step-by-step marking schemes appropriate for Class {class_name}.
8. Do NOT include any 'explanation' field.
9. Do NOT repeat questions.
10. Return strictly valid JSON only.
"""

        user_prompt = f"""
Generate exactly {question_count} fresh and distinct educational questions based ONLY on the following specifications:

============================================================
TARGET SPECIFICATIONS
============================================================
Class / Grade: {class_name}
Subject: {subject_name}
Chapter(s):
{chapters_formatted}
Difficulty: {difficulty}
Question Types: {', '.join(question_types)}
{f"Suggestions: {suggestions}" if suggestions else ""}

{grade_guidance}

============================================================
ABSOLUTE SUBJECT & CHAPTER RESTRICTION
============================================================
Subject: "{subject_name}"
Chapter(s): {chapters_formatted}

Every generated question MUST be a pure {subject_name} question from the requested chapter(s).

Do NOT generate content from:
- {disallowed_formatted}

============================================================
QUESTION GENERATION RULES
============================================================
- Generate exactly {question_count} questions.
- Distribute across the requested types: {', '.join(question_types)}.
- For MCQs: 4 plausible options strictly within {subject_name}.
- For Numerical: Realistic calculations strictly appropriate for Class {class_name}.
- For Short/Long: Complete model answers and step marking.
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
      "chapter": "Chapter name",
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

        logger.info(f"Generating {question_count} distinct questions for '{subject_name}' ({class_name}) [Nonce: {generation_nonce}]")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.35,
            top_p=0.9
        )

        raw_content = response.choices[0].message.content
        try:
            parsed_result = json.loads(raw_content)
        except json.JSONDecodeError:
            parsed_result = {"raw_output": raw_content}

        return jsonify({
            "status": "success",
            "data": parsed_result
        }), 200

    except OpenAIError as oe:
        logger.error(f"OpenAI API Error: {str(oe)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "OpenAI API Error",
            "details": str(oe)
        }), 502
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Internal Server Error",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production").lower() == "development"
    logger.info(f"Starting Edusoft AI server on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
