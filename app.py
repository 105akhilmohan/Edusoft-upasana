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

        student = payload.get("student", {})
        overall_score = payload.get("overall_score", {})
        attendance = payload.get("attendance", {})
        examinations = payload.get("examinations", {})
        model = payload.get("model", DEFAULT_MODEL)

        student_name = student.get("name", "The student")

        # ----------------------------------------------------------------------
        # PROMPT 1: Academic Summary & Key Strengths Diagnostic
        # ----------------------------------------------------------------------
        system_prompt_1 = (
            "You are a senior academic diagnostic specialist for Edusoft. "
            "Your objective is to review student performance, grades, and attendance "
            "to produce an executive summary and highlight key academic strengths with precise metrics. "
            "Return valid JSON only."
        )

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

Generate a JSON object with:
{{
  "summary": "1-2 sentence overview mentioning {student_name}'s academic percentage, grade, and attendance status.",
  "strengths": [
    "Highlight strong attendance or consistency with exact %",
    "Highlight top subject mastery with subject name, percentage, and grade"
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
Analyze the student's areas of growth based on the data below:

STUDENT PROFILE:
{json.dumps(student, indent=2)}

OVERALL SCORES & ATTENDANCE:
{json.dumps(overall_score, indent=2)}
{json.dumps(attendance, indent=2)}

EXAMINATIONS & SUBJECT BREAKDOWN:
{json.dumps(examinations, indent=2)}

Generate a JSON object with:
{{
  "focus_areas": [
    "Specific subject or metric requiring improvement with exact percentages",
    "Another critical focus area (e.g., fee dues, attendance gaps, or subject score drops)"
  ],
  "recommendation": "A clear, actionable, and encouraging recommendation for {student_name} ahead of upcoming examinations."
}}
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

        logger.info(f"Successfully generated two-stage insights for student: {student_name}")

        return jsonify({
            "status": "success",
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
        model = data.get("model", DEFAULT_MODEL)

        if not subject_name and not chapters:
            return jsonify({
                "status": "error",
                "message": "At least 'subject_name' (or 'topic') or 'chapters' must be provided."
            }), 400

        context_parts = []
        if class_name:
            context_parts.append(f"Target Class / Grade: {class_name}")
        if subject_name:
            context_parts.append(f"Subject / Topic: {subject_name}")
        if chapters:
            chapters_formatted = "\n  - " + "\n  - ".join(chapters)
            context_parts.append(f"Chapters Covered:{chapters_formatted}")
        context_parts.append(f"Difficulty Level: {difficulty}")
        context_parts.append(f"Requested Question Types: {', '.join(question_types)}")
        if suggestions:
            context_parts.append(f"Specific Suggestions & Requirements: {suggestions}")
        
        context_str = "\n".join(context_parts)

        system_prompt = (
            "You are an expert curriculum designer for Edusoft. "
            "Your role is to create high-standard academic questions aligned with school syllabi. "
            "Do NOT include explanations. Output strictly valid JSON."
        )

        user_prompt = f"""
Generate exactly {question_count} educational question(s) based on the following specifications:

{context_str}

Ensure:
1. Questions are distributed across the requested chapters and question types ({', '.join(question_types)}).
2. For MCQ questions: Provide 4 clear options and indicate the correct answer.
3. For Short and Long answer questions: Provide complete model answers and step-by-step marking schemes where applicable.
4. Do NOT include any 'explanation' field.

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
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."], // empty array [] if not MCQ
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

        logger.info(f"Generating {question_count} questions for '{subject_name}' ({class_name})")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
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
