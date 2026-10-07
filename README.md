# Edusoft AI Analytics & Question Generator API

Flask microservice integrated with OpenAI providing:
1. **Student Analytical Insights** (`POST /api/generate-insights`): Generates summary, strengths, focus areas, and recommendations from student academic and attendance records.
2. **Curriculum Question Generator** (`POST /api/generate-questions`): Generates educational questions (MCQ, Short, Long, Numerical) with step-by-step marking schemes.

---

## 📁 Project Structure

```
.
├── app.py                # Flask application with OpenAI endpoints
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container build specification with access logs
├── docker-compose.yml    # Docker Compose setup
├── .env                  # Environment secrets (OPENAI_API_KEY, etc.)
├── .env.example          # Sample environment file
├── .dockerignore         # Docker ignore rules
└── .gitignore            # Git ignore rules
```

---

## 🚀 Running the Project

### On Server (Docker)
```bash
docker-compose up --build -d
```
To view real-time HTTP access logs and status codes:
```bash
docker logs -f edusoft_question_generator
```

---

## 📡 API Reference

### 1. Generate Student Insights
- **URL:** `/api/generate-insights` (or `/generate-insights`, `/api/student-analysis`)
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`

#### Request Payload
```json
{
  "student": {
    "id": 825,
    "name": "Ram Mathew",
    "admission_no": "ADM-2024-00825",
    "roll_no": "12",
    "class": "Class 10",
    "section": "A",
    "gender": "Male"
  },
  "overall_score": {
    "composite_score": 84.6,
    "performance_label": "High Achiever",
    "academic_percentage": 82.5,
    "academic_grade": "A",
    "attendance_percentage": 89.5
  },
  "attendance": {
    "total_working_days": 190,
    "present": 170,
    "percentage": 89.47
  },
  "examinations": {
    "overall_percentage": 82.5,
    "overall_grade": "A",
    "top_subjects": [
      { "subject_name": "Mathematics", "percentage": 94.0, "grade": "A+" }
    ],
    "lowest_subjects": [
      { "subject_name": "History", "percentage": 68.0, "grade": "B" }
    ]
  }
}
```

#### Response Payload
```json
{
  "status": "success",
  "insights": {
    "summary": "Ram Mathew is maintaining an academic average of 82.50% (Grade A) with a Good attendance record of 89.47%.",
    "strengths": [
      "Consistent attendance above 85% helps retain academic continuity.",
      "Demonstrating top mastery in Mathematics (94.00%)."
    ],
    "focus_areas": [
      "Focus on improving scores in History (68.00%).",
      "Outstanding fee balance of 8,000.00 pending settlement."
    ],
    "recommendation": "Maintain academic rigor and review weaker subjects ahead of the upcoming term."
  }
}
```

---

### 2. Generate Educational Questions
- **URL:** `/api/generate-questions`
- **Method:** `POST`

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

---

### 3. Generate Timetable Schedule
- **URL:** `/api/generate-timetable` (or `/generate-timetable`, `/api/timetable/generate`)
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`

#### Request Payload
```json
{
  "class_id": 2,
  "section_id": 1,
  "week_dates": [
    "12/10/2026",
    "13/10/2026",
    "14/10/2026",
    "15/10/2026",
    "16/10/2026",
    "17/10/2026"
  ],
  "period_keys": [
    "eight_to_nine",
    "nine_to_ten",
    "ten_to_eleven",
    "eleven_to_twelve",
    "twelve_to_one",
    "two_to_three",
    "three_to_four",
    "four_to_five"
  ],
  "subject_teachers": [
    {
      "subject_id": 12,
      "name": "Medical Surgical Nursing",
      "teacher_id": 14,
      "teacher_name": "Dr. Anjali Sharma",
      "type": "Theory"
    },
    {
      "subject_id": 15,
      "name": "Pharmacology",
      "teacher_id": 22,
      "teacher_name": "Prof. Rajesh Kumar",
      "type": "Theory"
    }
  ],
  "unavailable_map": {
    "12/10/2026": {
      "14": "Dr. Anjali Sharma (Medical Leave)"
    },
    "13/10/2026": {
      "14": "Dr. Anjali Sharma (Medical Leave)"
    }
  },
  "teacher_busy_slots": {
    "12/10/2026": {
      "eight_to_nine": {
        "9": true
      }
    }
  },
  "user_prompt": "Substitute Dr. Anjali with same-subject faculty on Monday and Tuesday."
}
```

#### Response Payload
```json
{
  "status": "success",
  "message": "Weekly schedule optimized successfully with 0 teacher conflicts.",
  "schedule": {
    "12/10/2026": {
      "eight_to_nine": {
        "subject_id": 12,
        "subject_name": "Medical Surgical Nursing",
        "teacher_id": 22,
        "teacher_name": "Prof. Rajesh Kumar",
        "activity": "Theory Class",
        "is_substituted": true,
        "note": "Auto-substituted by AI: Same Subject Specialist (EMP-104)"
      },
      "nine_to_ten": {
        "subject_id": 15,
        "subject_name": "Pharmacology",
        "teacher_id": 22,
        "teacher_name": "Prof. Rajesh Kumar",
        "activity": "Theory Class",
        "is_substituted": false,
        "note": ""
      }
    }
  }
}
```

