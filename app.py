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

        system_prompt = (
            "You are an expert AI academic analyst and educational advisor for Edusoft. "
            "Your task is to analyze the provided student's academic performance, grades, attendance records, "
            "subject summaries, and trends. Generate concise, actionable, and encouraging insights strictly in valid JSON format."
        )

        user_prompt = f"""
Analyze the following student data and generate analytical insights:

STUDENT PROFILE:
{json.dumps(student, indent=2)}

OVERALL SCORES & PERFORMANCE:
{json.dumps(overall_score, indent=2)}

ATTENDANCE DETAILS:
{json.dumps(attendance, indent=2)}

EXAMINATIONS & SUBJECT SUMMARY:
{json.dumps(examinations, indent=2)}

Generate a response adhering strictly to this JSON format:
{{
  "insights": {{
    "summary": "Brief 1-2 sentence overview of academic average, grade, and attendance health for {student_name}.",
    "strengths": [
      "Key positive highlight regarding attendance, top subject scores, or consistency.",
      "Another specific academic or behavioral strength with metrics."
    ],
    "focus_areas": [
      "Specific subject or area requiring improvement (e.g. lowest scoring subjects, attendance drops, or pending balances).",
      "Actionable focus point."
    ],
    "recommendation": "A clear, motivational recommendation for the student/teacher/parent ahead of the upcoming term."
  }}
}}
"""

        logger.info(f"Generating insights for student: {student_name}")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.6
        )

        raw_content = response.choices[0].message.content
        logger.info("Successfully generated student insights from OpenAI")

        try:
            parsed_result = json.loads(raw_content)
            # Ensure insights key exists
            insights = parsed_result.get("insights", parsed_result)
        except json.JSONDecodeError:
            insights = {"summary": raw_content, "strengths": [], "focus_areas": [], "recommendation": ""}

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
