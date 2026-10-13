# Edusoft AI Analytics, Evaluation & Question Generator API

Flask microservice integrated with OpenAI providing:
1. **Student Analytical Insights** (`POST /api/generate-insights`): Generates summary, strengths, focus areas, and recommendations from student academic and attendance records.
2. **Curriculum Question Generator** (`POST /api/generate-questions`): Generates educational questions strictly grounded in textbook content with step-by-step marking schemes.
3. **Automated AI Paper Evaluation** (`POST /api/evaluate-papers`, `GET /api/evaluation/<id>`): Evaluates student answer papers against model answers, producing question-by-question step markings, recognition confidence, letter grades, feedback, and diagnostic insights.
4. **Timetable Schedule Generator** (`POST /api/generate-timetable`): Multi-teacher conflict resolution, leave management, and schedule optimization.

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

---

### 4. Automated AI Paper Evaluation (POST)
- **URL:** `/api/evaluate-papers` (or `/api/evaluate-paper`, `/evaluate-papers`, `/api/evaluations`)
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`

Evaluates student answer sheets against model answers/rubrics using AI, producing question-level step-marking breakdowns, recognition confidence, letter grades, pass/fail status, AI feedback, and clinical/academic diagnostic insights. Automatically persists results for subsequent GET querying.

#### Request Payload
```json
{
  "title": "Medical Surgical Nursing - Mid Term Exam 2026",
  "class_id": 3,
  "class_name": "B.Sc. Nursing (Year II)",
  "section_id": 1,
  "section_name": "Section A",
  "subject_id": 12,
  "subject_name": "Medical Surgical Nursing",
  "question_paper_id": 14,
  "question_paper_title": "Medical Surgical Nursing - Mid Term Exam 2026",
  "max_marks": 100.0,
  "answer_key_file": "uploads/evaluation_papers/1741512300_key_model_answers.pdf",
  "questions": [
    {
      "question_no": 1,
      "question_text": "Describe the pathophysiology and nursing management of Acute Myocardial Infarction.",
      "question_type": "Descriptive",
      "max_marks": 20.0,
      "answer_key": "Plaque rupture, coronary thrombus, MONA protocol, 12-lead ECG monitoring."
    }
  ],
  "students": [
    {
      "student_id": 84,
      "student_name": "Anjali Nair",
      "admission_no": "AKG/2024/001",
      "roll_no": "101",
      "kuhs_reg_no": "210010482",
      "student_image": "uploads/student_images/1709123456.jpg",
      "file_name": "2 Pages (Anjali_Nair_p1.jpg, Anjali_Nair_p2.jpg)",
      "file_path": "uploads/evaluation_papers/1741512300_0_Anjali_Nair_p1.jpg",
      "file_type": "jpg",
      "page_count": 2,
      "student_response": "Plaque rupture leading to coronary thrombus formation and transmural ischemia. MONA protocol initiated with Oxygen, Aspirin, Nitroglycerin, Morphine, along with 12-lead ECG monitoring."
    }
  ]
}
```

#### Response Payload (`POST_RESPONSE`)
```json
{
  "status": "success",
  "total_sheets": 1,
  "passed_count": 1,
  "failed_count": 0,
  "average_score": 88.5,
  "max_marks": 100.0,
  "answer_key_file": "uploads/evaluation_papers/1741512300_key_model_answers.pdf",
  "students": [
    {
      "student_id": 84,
      "student_name": "Anjali Nair",
      "admission_no": "AKG/2024/001",
      "roll_no": "101",
      "kuhs_reg_no": "210010482",
      "student_image": "uploads/student_images/1709123456.jpg",
      "file_name": "2 Pages (Anjali_Nair_p1.jpg, Anjali_Nair_p2.jpg)",
      "file_path": "uploads/evaluation_papers/1741512300_0_Anjali_Nair_p1.jpg",
      "file_type": "jpg",
      "page_count": 2,
      "recognition_status": "matched",
      "recognition_confidence": 98.5,
      "total_marks": 100.0,
      "obtained_marks": 88.5,
      "percentage": 88.5,
      "grade": "A",
      "status": "Pass",
      "ai_feedback": "Student exhibited strong grasp of core theoretical concepts. Evaluated across 2 uploaded answer page(s) with A overall proficiency.",
      "ai_insights": {
        "recommendation": "Student demonstrated strong clinical logic in cardiac emergencies. Maintain consistent terminology and documentation.",
        "focus_areas": [
          "Pathophysiology biomarker curve timelines",
          "Emergency dosage documentation"
        ],
        "strengths": [
          "Accurate MONA protocol sequencing",
          "Adherence to aseptic standards"
        ]
      },
      "evaluation_breakdown": [
        {
          "question_no": 1,
          "question_text": "Describe the pathophysiology and nursing management of Acute Myocardial Infarction.",
          "question_type": "Descriptive",
          "max_marks": 20.0,
          "obtained_marks": 18.5,
          "answer_key": "Plaque rupture, coronary thrombus, MONA protocol, 12-lead ECG monitoring.",
          "student_response": "Student response extracted and verified across 2 uploaded answer page(s).",
          "step_marking": [
            {
              "step": "Pathophysiology & Mechanism",
              "max_marks": 6.0,
              "obtained_marks": 5.5,
              "status": "Full"
            },
            {
              "step": "Clinical Manifestations",
              "max_marks": 6.0,
              "obtained_marks": 5.5,
              "status": "Full"
            },
            {
              "step": "Nursing Interventions (MONA Protocol)",
              "max_marks": 8.0,
              "obtained_marks": 7.5,
              "status": "Full"
            }
          ],
          "feedback": "Demonstrated clear comprehension in written answer sheets. Key clinical terminology, diagnosis, and interventions were accurately stated."
        }
      ]
    }
  ]
}
```

---

### 5. Fetch Evaluation Details (GET)
- **URL:** `/api/evaluation/{id}` (or `/api/evaluations/{id}`, `/evaluation/{id}`)
- **Method:** `GET`
- **Headers:** `Accept: application/json`

Retrieves saved evaluation header and student results for institutional reporting and grade sheet generation.

#### Response Payload (`GET_RESPONSE`)
```json
{
  "status": "success",
  "data": {
    "id": "5",
    "title": "Medical Surgical Nursing - Mid Term Exam 2026",
    "class_id": "3",
    "class_name": "B.Sc. Nursing (Year II)",
    "section_id": "1",
    "section_name": "Section A",
    "subject_id": "12",
    "subject_name": "Medical Surgical Nursing",
    "question_paper_id": "14",
    "question_paper_title": "Medical Surgical Nursing - Mid Term Exam 2026",
    "total_students": "1",
    "passed_count": "1",
    "failed_count": "0",
    "average_score": "88.50",
    "max_marks": "100.00",
    "evaluation_date": "2026-03-15",
    "created_at": "2026-03-15 14:22:10",
    "students": [
      {
        "id": "18",
        "evaluation_id": "5",
        "student_id": "84",
        "student_name": "Anjali Nair",
        "admission_no": "AKG/2024/001",
        "roll_no": "101",
        "kuhs_reg_no": "210010482",
        "student_image": "uploads/student_images/1709123456.jpg",
        "file_name": "2 Pages (Anjali_Nair_p1.jpg, Anjali_Nair_p2.jpg)",
        "file_path": "uploads/evaluation_papers/1741512300_0_Anjali_Nair_p1.jpg",
        "total_marks": "100.00",
        "obtained_marks": "88.50",
        "percentage": "88.50",
        "grade": "A",
        "status": "Pass",
        "ai_feedback": "Student exhibited strong grasp of core theoretical concepts.",
        "ai_insights": "{\"recommendation\":\"Student demonstrated strong clinical logic.\",\"focus_areas\":[\"Pathophysiology timelines\"]}"
      }
    ]
  }
}
```

---

### 6. Additional Evaluation Endpoints
- **List All Evaluations:** `GET /api/evaluations` - Returns array of all evaluation summaries.
- **Detailed Step Breakdown:** `GET /api/evaluation/{id}/breakdown` - Returns the comprehensive question-level breakdown for the entire evaluation.
- **Student Specific Breakdown:** `GET /api/evaluation/{id}/student/{student_id}` - Returns the full evaluation sheet for a specific student.


