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
logger = logging.getLogger("edusoft_question_generator")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OPENAI_API_KEY is not set in environment variables.")

client = OpenAI(api_key=openai_api_key)
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Edusoft Question Generator API",
        "status": "online",
        "version": "1.1.0",
        "endpoints": {
            "health": "GET /health",
            "generate_questions": "POST /api/generate-questions"
        }
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }), 200


@app.route("/api/generate-questions", methods=["POST"])
@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    """
    Generate educational questions for Edusoft.
    
    Supported JSON Payload formats:
    
    Format 1 (Edusoft Class & Chapter Schema):
    {
      "class_name": "Class 10",
      "subject_name": "Physics",
      "chapters": [
        "Chapter 1: Force, Work, Energy and Power",
        "Chapter 2: Light and Refraction"
      ],
      "question_count": 5,
      "difficulty": "Medium",
      "question_types": ["MCQ", "Short", "Long"],
      "suggestions": "Include numerical problems with step markings"
    }
    
    Format 2 (Generic Topic Schema):
    {
      "topic": "Python Programming",
      "description": "Object-oriented programming concepts",
      "count": 5,
      "difficulty": "medium",
      "question_type": "multiple_choice"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Request body must be a valid JSON object."
            }), 400

        # Extract Fields with fallbacks
        class_name = data.get("class_name", "")
        subject_name = data.get("subject_name") or data.get("topic", "")
        
        # Chapters (can be list or string or empty)
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
                        "success": False,
                        "error": "Field 'question_count' must be an integer between 1 and 50."
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    "success": False,
                    "error": "Field 'question_count' must be a valid integer."
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

        # Validation: must have subject/topic or chapters
        if not subject_name and not chapters:
            return jsonify({
                "success": False,
                "error": "At least 'subject_name' (or 'topic') or 'chapters' must be provided."
            }), 400

        # Build context details for the prompt
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

        # Construct System Prompt
        system_prompt = (
            "You are an expert curriculum designer and senior assessment author for Edusoft. "
            "Your role is to create high-standard academic questions aligned with school syllabi. "
            "Follow all requested question types (e.g. MCQ, Short Answer, Long Answer, Numerical), "
            "difficulty levels, and specific user suggestions (such as step markings and answer keys). "
            "Output strictly valid JSON matching the requested schema."
        )

        user_prompt = f"""
Generate exactly {question_count} high-quality educational question(s) based on the following specifications:

{context_str}

Ensure:
1. Questions are distributed across the requested chapters and question types ({', '.join(question_types)}).
2. For MCQ questions: Provide 4 clear options and indicate the correct answer.
3. For Short and Long answer questions: Provide complete model answers and step-by-step marking schemes where applicable (especially for numerical problems).
4. Include clear explanations and marks for each question.

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
      "chapter": "Chapter name (or topic)",
      "type": "MCQ | Short | Long | Numerical",
      "marks": 2,
      "question": "Question text with clear problem statement",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."], // empty array [] if not MCQ
      "correct_answer": "Correct answer or full model answer",
      "step_marking": [
        {{
          "step": "Description of step or formula used",
          "marks": 1
        }}
      ],
      "explanation": "Detailed explanation of concept, working, and logic",
      "difficulty": "Easy | Medium | Hard"
    }}
  ]
}}
"""

        logger.info(f"Generating {question_count} questions for '{subject_name}' ({class_name}) using model '{model}'")

        # Call OpenAI API
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
        logger.info("Successfully received response from OpenAI")

        try:
            parsed_result = json.loads(raw_content)
        except json.JSONDecodeError:
            parsed_result = {"raw_output": raw_content}

        return jsonify({
            "success": True,
            "data": parsed_result
        }), 200

    except OpenAIError as oe:
        logger.error(f"OpenAI API Error: {str(oe)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "OpenAI API Error",
            "details": str(oe)
        }), 502
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production").lower() == "development"
    logger.info(f"Starting Edusoft Question Generator server on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
