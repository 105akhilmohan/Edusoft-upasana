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
        "version": "1.0.0",
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
    Generate questions based on topic, description, and question count.
    
    Expected JSON Body:
    {
        "topic": "Python Programming",
        "description": "Object-oriented programming concepts including inheritance and polymorphism",
        "count": 5,
        "difficulty": "medium",          # Optional: easy | medium | hard | mixed
        "question_type": "multiple_choice" # Optional: multiple_choice | true_false | short_answer | mixed
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Request body must be a valid JSON object."
            }), 400

        # Extract fields
        topic = data.get("topic")
        description = data.get("description", "")
        count = data.get("count")
        difficulty = data.get("difficulty", "medium")
        question_type = data.get("question_type", "multiple_choice")
        model = data.get("model", DEFAULT_MODEL)

        # Validation
        if not topic or not isinstance(topic, str) or not topic.strip():
            return jsonify({
                "success": False,
                "error": "Field 'topic' is required and must be a non-empty string."
            }), 400

        if count is None:
            count = 5  # default count
        else:
            try:
                count = int(count)
                if count <= 0 or count > 50:
                    return jsonify({
                        "success": False,
                        "error": "Field 'count' must be an integer between 1 and 50."
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    "success": False,
                    "error": "Field 'count' must be a valid integer."
                }), 400

        # Construct OpenAI Prompt
        system_prompt = (
            "You are an expert educational content creator and curriculum assessment specialist for Edusoft. "
            "Your task is to generate high-quality educational questions based strictly on the provided topic, "
            "description, and constraints. Output your response strictly as valid JSON matching the requested schema."
        )

        user_prompt = f"""
Generate {count} educational question(s) according to these specifications:
- Topic: {topic.strip()}
- Description / Context: {description.strip() if description else "N/A"}
- Target Difficulty: {difficulty}
- Question Type: {question_type}

Return a valid JSON object with the following schema:
{{
    "topic": "{topic.strip()}",
    "total_questions": {count},
    "difficulty": "{difficulty}",
    "questions": [
        {{
            "id": 1,
            "type": "multiple_choice | true_false | short_answer",
            "question": "Question text here",
            "options": ["Option A", "Option B", "Option C", "Option D"], // If multiple choice or true/false, omit or empty if short answer
            "correct_answer": "Correct answer here",
            "explanation": "Brief explanation of why the answer is correct",
            "difficulty": "easy | medium | hard"
        }}
    ]
}}
"""

        logger.info(f"Generating {count} questions for topic: '{topic}' using model '{model}'")

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
            # Fallback if parsing fails
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
