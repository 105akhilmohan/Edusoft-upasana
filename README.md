# Edusoft Question Generator API

A Flask microservice integrated with OpenAI to generate customized educational questions based on **topic**, **description**, and **count**.

---

## 📁 Project Structure

```
.
├── app.py                # Main Flask application and OpenAI integration
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container build specification
├── docker-compose.yml    # Docker Compose setup
├── .env                  # Environment secrets (OPENAI_API_KEY, etc.)
├── .env.example          # Sample environment file
├── .dockerignore         # Docker ignore rules
└── .gitignore            # Git ignore rules
```

---

## 🚀 Getting Started

### 1. Local Setup

#### Prerequisites
- Python 3.10+
- `pip`

#### Steps
```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```
The server will start at `http://localhost:5000`.

---

### 2. Run with Docker & Docker Compose

#### Build & Run
```bash
docker-compose up --build
```

#### Run in Background
```bash
docker-compose up -d
```

#### Stop Container
```bash
docker-compose down
```

---

## 📡 API Reference

### 1. Health Check
- **URL:** `/health`
- **Method:** `GET`
- **Response:**
```json
{
  "openai_configured": true,
  "status": "healthy"
}
```

---

### 2. Generate Questions
- **URL:** `/api/generate-questions` (or `/generate-questions`)
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`

#### Request Body
```json
{
  "topic": "Photosynthesis",
  "description": "Light-dependent and light-independent reactions in plant biology for grade 10 students",
  "count": 3,
  "difficulty": "medium",
  "question_type": "multiple_choice"
}
```

#### Request Parameters
| Parameter | Type | Required | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `topic` | string | **Yes** | Subject or topic for question generation | — |
| `description` | string | No | Additional context, target audience, or sub-topics | `""` |
| `count` | integer | No | Number of questions to generate (1–50) | `5` |
| `difficulty` | string | No | `easy`, `medium`, `hard`, or `mixed` | `medium` |
| `question_type`| string | No | `multiple_choice`, `true_false`, `short_answer`, `mixed` | `multiple_choice` |

---

#### Sample Response
```json
{
  "success": true,
  "data": {
    "topic": "Photosynthesis",
    "total_questions": 3,
    "difficulty": "medium",
    "questions": [
      {
        "id": 1,
        "type": "multiple_choice",
        "question": "Where do the light-dependent reactions of photosynthesis take place inside a plant cell?",
        "options": [
          "Thylakoid membrane",
          "Stroma",
          "Mitochondrial matrix",
          "Cytoplasm"
        ],
        "correct_answer": "Thylakoid membrane",
        "explanation": "Light-dependent reactions occur in the thylakoid membranes of chloroplasts where chlorophyll absorbs sunlight.",
        "difficulty": "medium"
      }
    ]
  }
}
```

---

### 3. Example `cURL` Command

```bash
curl -X POST http://localhost:5000/api/generate-questions \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Python Data Structures",
    "description": "Lists, Dictionaries, Sets, and Tuples",
    "count": 3,
    "difficulty": "medium",
    "question_type": "multiple_choice"
  }'
```
