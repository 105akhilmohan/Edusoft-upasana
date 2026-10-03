# Edusoft Question Generator API

A Flask microservice integrated with OpenAI to generate customized educational questions based on **Class**, **Subject**, **Chapters**, **Question Count**, **Question Types**, and **Suggestions**.

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

```bash
# Create and activate virtual environment (optional)
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```
The server starts at `http://localhost:5000`.

---

### 2. Run with Docker & Docker Compose

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# Stop container
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

#### Request Payload
```json
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
```

#### Parameters
| Parameter | Type | Required | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `class_name` | string | No | Class / Grade level (e.g., `"Class 10"`) | `""` |
| `subject_name` | string | **Yes** | Subject or Topic (e.g., `"Physics"`) | — |
| `chapters` | array[str] | No | List of chapters covered | `[]` |
| `question_count` | integer | No | Total questions to generate (1–50) | `5` |
| `difficulty` | string | No | `"Easy"`, `"Medium"`, `"Hard"`, or `"Mixed"` | `"Medium"` |
| `question_types` | array[str] | No | Types of questions (e.g., `["MCQ", "Short", "Long"]`) | `["MCQ", "Short", "Long"]` |
| `suggestions` | string | No | Additional requirements (e.g., `"Include numerical problems with step markings"`) | `""` |

---

#### Sample Output Response
```json
{
  "success": true,
  "data": {
    "class_name": "Class 10",
    "subject_name": "Physics",
    "chapters": [
      "Chapter 1: Force, Work, Energy and Power",
      "Chapter 2: Light and Refraction"
    ],
    "total_questions": 5,
    "difficulty": "Medium",
    "questions": [
      {
        "id": 1,
        "chapter": "Chapter 1: Force, Work, Energy and Power",
        "type": "MCQ",
        "marks": 1,
        "question": "If the velocity of a moving body is doubled, its kinetic energy becomes:",
        "options": [
          "A) Halved",
          "B) Doubled",
          "C) Four times",
          "D) Unchanged"
        ],
        "correct_answer": "C) Four times",
        "step_marking": [],
        "explanation": "Kinetic energy KE = 1/2 * m * v^2. Since KE is proportional to v^2, doubling the velocity increases KE by a factor of 4.",
        "difficulty": "Medium"
      },
      {
        "id": 2,
        "chapter": "Chapter 1: Force, Work, Energy and Power",
        "type": "Numerical",
        "marks": 3,
        "question": "A crane lifts a mass of 500 kg vertically upwards through a height of 20 m in 10 seconds. Calculate the power exerted by the crane. (Take g = 9.8 m/s²)",
        "options": [],
        "correct_answer": "9800 W (or 9.8 kW)",
        "step_marking": [
          {
            "step": "Calculate Work done: W = m * g * h = 500 * 9.8 * 20 = 98000 J",
            "marks": 1.5
          },
          {
            "step": "Calculate Power: P = Work / Time = 98000 / 10 = 9800 W",
            "marks": 1.5
          }
        ],
        "explanation": "Power is the rate of doing work. Total work is equal to the gravitational potential energy gained.",
        "difficulty": "Medium"
      }
    ]
  }
}
```
