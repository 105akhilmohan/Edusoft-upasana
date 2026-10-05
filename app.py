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
Analyze the student's areas of growth based ONLY on the supplied data.

STUDENT PROFILE:
{json.dumps(student, indent=2)}

OVERALL SCORES:
{json.dumps(overall_score, indent=2)}

ATTENDANCE:
{json.dumps(attendance, indent=2)}

EXAMINATIONS & SUBJECT BREAKDOWN:
{json.dumps(examinations, indent=2)}

IMPORTANT DATA RULES:
- Use ONLY information present above.
- Do NOT invent any information.
- Do NOT mention fee dues unless fee information is explicitly provided.
- Do NOT create subjects that are not present.
- Do NOT create marks or percentages that are not present.
- Do NOT infer a student's financial status.
- Every percentage mentioned must match the supplied data.
- Focus areas must be based on actual weaknesses in the supplied academic
  or attendance data.
- Recommendations must be based on the actual student performance.

Generate:

{{
  "focus_areas": [
    "Specific subject or metric requiring improvement with exact percentage",
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
        
        context_str = "\n".join(context_parts)

        system_prompt = """
You are an expert school curriculum designer and examination question
generator for Edusoft.

Your most important rule is SUBJECT AND CHAPTER SCOPE CONTROL.

You MUST generate questions ONLY from the specified SUBJECT and
specified CHAPTER(S).

STRICT SCOPE RULES:

1. The provided subject_name is the ONLY allowed subject.

2. The provided chapter names are the ONLY allowed chapters.

3. Do NOT generate questions from another subject.

4. Do NOT introduce concepts that belong primarily to another subject.

5. Do NOT mix subjects.

6. Do NOT use general knowledge if it introduces content outside the
   specified subject/chapter.

7. Every question must be directly relevant to at least one of the
   specified chapters.

8. The question, options, correct answer, and marking scheme must all
   remain within the specified subject and chapter scope.

9. If a concept could belong to multiple subjects, interpret it only
   according to the specified subject and chapter.

10. Do not change the subject.

11. Do not invent a chapter that was not provided.

12. Do not silently replace the requested chapter with a related chapter.

13. If multiple chapters are provided, questions may come from any of
   those chapters, but they must remain inside the provided chapter list.

14. Questions must be academically appropriate for the specified class.

15. For MCQs, all options must also belong to the same subject and
   chapter context.

16. For Short/Long/Numerical questions, the model answer and marking
   scheme must also remain strictly within the same scope.

17. Do not include explanations unless explicitly requested.

18. Do not repeat the previously generated questions.

19. Return valid JSON only.

SUBJECT SCOPE HAS HIGHER PRIORITY THAN CREATIVITY.

If creativity conflicts with subject/chapter restrictions, ALWAYS follow
the subject/chapter restrictions.
"""

        user_prompt = f"""
Generate exactly {question_count} fresh and distinct educational
questions using ONLY the following information:

{context_str}

============================================================
ABSOLUTE SUBJECT RESTRICTION
============================================================

SUBJECT:
{subject_name}

Every generated question MUST belong to the subject:

"{subject_name}"

Do NOT generate content from:

- Mathematics
- Science
- Social Science
- English
- Malayalam
- Hindi
- Computer Science
- General Knowledge
- or any other subject

unless that content is explicitly part of the requested subject and
chapter.

============================================================
ABSOLUTE CHAPTER RESTRICTION
============================================================

ONLY these chapter(s) are allowed:

{chapters_formatted}

Every question MUST be traceable to one of these chapters.

If a question cannot clearly be associated with one of the above
chapters, DO NOT generate that question.

============================================================
QUESTION GENERATION RULES
============================================================

- Generate exactly {question_count} questions.
- Use only the requested subject.
- Use only the requested chapters.
- Keep the academic level appropriate for class {class_name}.
- Cover different concepts within the allowed chapters.
- Avoid repeating the same question pattern.
- Use different wording and scenarios.
- For MCQs, generate four plausible options.
- Do not create distractors from another subject.
- For Short questions, provide a correct model answer.
- For Long questions, provide a complete model answer.
- For Numerical questions, use appropriate calculations only if
  numerical problems are actually part of the requested subject/chapter.
- Do not add explanations.
- Do not add information from unrelated chapters.
- Do not add information from unrelated subjects.

============================================================
FINAL SCOPE CHECK
============================================================

Before returning the response, internally verify every question:

[ ] Correct subject
[ ] Correct chapter
[ ] Correct class level
[ ] No unrelated subject content
[ ] No unrelated chapter content
[ ] Answer belongs to the same subject
[ ] Options belong to the same subject
[ ] Marking scheme belongs to the same subject
[ ] No repeated question

If any question fails these checks, replace it with a valid question.

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

        logger.info(f"Generating {question_count} distinct questions for '{subject_name}' ({class_name}) [Nonce: {generation_nonce}]")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.88,
            top_p=0.95
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
