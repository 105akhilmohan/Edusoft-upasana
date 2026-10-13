import unittest
from unittest.mock import MagicMock, patch
import json
import os
import tempfile

from paper_evaluator import (
    process_paper_evaluation,
    evaluate_student_booklet_ai,
    parse_question_paper_and_answer_key,
    detect_booklet_boundaries,
    contains_forbidden_placeholder,
    evaluation_store,
    calculate_grade,
    determine_status
)


class TestPaperEvaluationAccuracy(unittest.TestCase):

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.chat = MagicMock()
        os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-tests"

    def test_missing_question_paper_returns_400(self):
        """Case: Question paper missing or unreadable returns clear 400 error."""
        payload = {
            "answer_key": "Model answers: Q1: Photosynthesis is...",
            "file_path": "non_existent_student_sheet.pdf"
        }
        resp, status_code = process_paper_evaluation(payload=payload, client=self.mock_client)
        self.assertEqual(status_code, 400)
        self.assertIn("Question paper is missing", resp["message"])

    def test_missing_answer_key_returns_400(self):
        """Case: Answer key missing returns clear 400 error."""
        payload = {
            "questions": [{"question_no": 1, "question_text": "Define force", "max_marks": 10.0}],
            "file_path": "student.pdf"
        }
        resp, status_code = process_paper_evaluation(payload=payload, client=self.mock_client)
        self.assertEqual(status_code, 400)
        self.assertIn("Answer key is missing", resp["message"])

    def test_missing_student_sheet_returns_400(self):
        """Case: Student answer sheet missing returns clear 400 error."""
        payload = {
            "questions": [{"question_no": 1, "question_text": "Define force", "max_marks": 10.0, "answer_key": "F = ma"}],
        }
        resp, status_code = process_paper_evaluation(payload=payload, client=self.mock_client)
        self.assertEqual(status_code, 400)
        self.assertIn("Student answer sheet is missing", resp["message"])

    def test_placeholder_rejection(self):
        """Case: Generic placeholders are rejected and flagged for manual review."""
        self.assertTrue(contains_forbidden_placeholder("Student response extracted and verified"))
        self.assertTrue(contains_forbidden_placeholder("[Handwritten answers on uploaded pages]"))
        self.assertFalse(contains_forbidden_placeholder("Photosynthesis is the process by which green plants convert light into chemical energy."))

    def test_accurate_handwritten_evaluation_and_metrics(self):
        """
        Cases 1, 2, 3, 4, 6:
        - Readable answers scored by rubric
        - Blank answer receives 0 and is_attempted=False
        - Illegible answer receives 0 and pending_review=True
        - Missing student identity does not zero out marks
        """
        questions = [
            {"question_no": 1, "question_text": "Define Photosynthesis.", "max_marks": 10.0, "model_answer": "Conversion of light into chemical energy using chlorophyll."},
            {"question_no": 2, "question_text": "State Newton's Second Law.", "max_marks": 10.0, "model_answer": "Force is rate of change of momentum, F=ma."},
            {"question_no": 3, "question_text": "Explain Krebs Cycle.", "max_marks": 10.0, "model_answer": "Aerobic cellular respiration producing ATP."},
            {"question_no": 4, "question_text": "Calculate kinetic energy for m=2, v=3.", "max_marks": 10.0, "model_answer": "KE = 1/2 m v^2 = 9 Joules."}
        ]

        # Simulate Vision model response:
        # Q1: Readable (8.5/10)
        # Q2: Blank/Unanswered (0/10)
        # Q3: Illegible handwriting (0/10, pending_review=True)
        # Q4: Readable with multi-page continuation (10/10)
        # Identity: missing on document
        mock_ai_response = {
            "detected_student_name": None,
            "detected_student_id": None,
            "recognition_status": "unmatched",
            "recognition_confidence": 0.0,
            "ai_feedback": "Student demonstrated strong grasp in mechanics and cellular biochemistry, but skipped Question 2 and handwriting was illegible in Question 3.",
            "ai_insights": {
                "recommendation": "Review cellular respiration and complete all questions.",
                "focus_areas": ["Newton's Second Law", "Handwriting legibility in descriptive biology"],
                "strengths": ["Formula application in kinetic energy", "Photosynthesis definition"]
            },
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "question_text": "Define Photosynthesis.",
                    "question_type": "Descriptive",
                    "max_marks": 10.0,
                    "obtained_marks": 8.5,
                    "is_attempted": True,
                    "pending_review": False,
                    "answer_key": "Conversion of light into chemical energy using chlorophyll.",
                    "student_response": "Photosynthesis is the synthesis of chemical energy from light by chloroplasts in green plants.",
                    "step_marking": [
                        {"step": "Core concept & energy conversion", "max_marks": 5.0, "obtained_marks": 4.5, "status": "Partial"},
                        {"step": "Organelle & pigments involved", "max_marks": 5.0, "obtained_marks": 4.0, "status": "Partial"}
                    ],
                    "feedback": "Accurate definition. Explicitly mentions chloroplasts and light conversion."
                },
                {
                    "question_no": 2,
                    "question_text": "State Newton's Second Law.",
                    "question_type": "Descriptive",
                    "max_marks": 10.0,
                    "obtained_marks": 0.0,
                    "is_attempted": False,
                    "pending_review": False,
                    "answer_key": "Force is rate of change of momentum, F=ma.",
                    "student_response": "[Unanswered]",
                    "step_marking": [
                        {"step": "Question attempted", "max_marks": 10.0, "obtained_marks": 0.0, "status": "Zero"}
                    ],
                    "feedback": "Question 2 was not attempted by the student."
                },
                {
                    "question_no": 3,
                    "question_text": "Explain Krebs Cycle.",
                    "question_type": "Descriptive",
                    "max_marks": 10.0,
                    "obtained_marks": 0.0,
                    "is_attempted": True,
                    "pending_review": True,
                    "answer_key": "Aerobic cellular respiration producing ATP.",
                    "student_response": "[Illegible handwriting - Flagged for manual review]",
                    "step_marking": [
                        {"step": "Manual Verification Required", "max_marks": 10.0, "obtained_marks": 0.0, "status": "Zero"}
                    ],
                    "feedback": "Handwritten response on Page 2 is smudged and illegible. Flagged for manual instructor review."
                },
                {
                    "question_no": 4,
                    "question_text": "Calculate kinetic energy for m=2, v=3.",
                    "question_type": "Descriptive",
                    "max_marks": 10.0,
                    "obtained_marks": 10.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "answer_key": "KE = 1/2 m v^2 = 9 Joules.",
                    "student_response": "KE = (1/2) * m * v^2 = (1/2) * 2 * 3^2 = 9 J. [Continued on Page 3: Final units Joules verified.]",
                    "step_marking": [
                        {"step": "Formula & Substitution", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"},
                        {"step": "Calculation & Units", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"}
                    ],
                    "feedback": "Perfect step-by-step calculation with correct units across page continuation."
                }
            ]
        }

        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(mock_ai_response)
        self.mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        booklet = {
            "booklet_index": 1,
            "page_numbers": [1, 2, 3],
            "combined_text": ""
        }

        result = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="student_answers.pdf",
            file_name="student_answers.pdf",
            questions=questions,
            total_marks=40.0,
            subject_title="Science Assessment",
            client=self.mock_client
        )

        # 1. Total questions, attempted, unanswered, evaluated, pending review counts
        self.assertEqual(result["total_questions"], 4)
        self.assertEqual(result["attempted_questions"], 3)  # Q1, Q3, Q4
        self.assertEqual(result["unanswered_questions"], 1) # Q2
        self.assertEqual(result["evaluated_questions"], 2)  # Q1, Q4
        self.assertEqual(result["pending_review_questions"], 1) # Q3

        # Mathematical consistency invariants:
        # total_questions == attempted + unanswered
        self.assertEqual(result["total_questions"], result["attempted_questions"] + result["unanswered_questions"])
        # attempted_questions == evaluated + pending_review
        self.assertEqual(result["attempted_questions"], result["evaluated_questions"] + result["pending_review_questions"])

        # 2. Marks verification
        self.assertEqual(result["total_marks"], 40.0)
        self.assertEqual(result["obtained_marks"], 18.5) # 8.5 + 0 + 0 + 10.0
        self.assertEqual(result["percentage"], 46.25)
        self.assertEqual(result["grade"], "D")
        self.assertEqual(result["status"], "Fail")

        # 3. Transcriptions and feedback are genuine
        q1_res = result["evaluation_breakdown"][0]
        self.assertIn("chloroplasts", q1_res["student_response"])
        self.assertFalse(contains_forbidden_placeholder(q1_res["student_response"]))
        self.assertFalse(q1_res["pending_review"])

        q2_res = result["evaluation_breakdown"][1]
        self.assertEqual(q2_res["student_response"], "[Unanswered]")
        self.assertFalse(q2_res["is_attempted"])

        q3_res = result["evaluation_breakdown"][2]
        self.assertTrue(q3_res["pending_review"])
        self.assertEqual(q3_res["obtained_marks"], 0.0)

        q4_res = result["evaluation_breakdown"][3]
        self.assertIn("Continued on Page 3", q4_res["student_response"])
        self.assertEqual(q4_res["obtained_marks"], 10.0)

        # 4. Identity decoupling: name and ID empty, but marks are awarded!
        self.assertEqual(result["student_name"], "")
        self.assertEqual(result["student_id"], "")
        self.assertEqual(result["recognition_status"], "unmatched")
        self.assertGreater(result["obtained_marks"], 0.0)

    def test_fallback_when_ai_fails_no_fabricated_scores(self):
        """Case 7: When AI call fails completely, NO fabricated multipliers or marks are awarded."""
        questions = [
            {"question_no": 1, "question_text": "Define Osmosis.", "max_marks": 20.0, "model_answer": "Movement of solvent molecules across semipermeable membrane."}
        ]

        # Simulate OpenAI failure
        self.mock_client.chat.completions.create.side_effect = Exception("OpenAI API rate limit exceeded")

        booklet = {
            "booklet_index": 1,
            "page_numbers": [1],
            "combined_text": "Some handwritten text on page"
        }

        result = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="osmosis_sheet.pdf",
            file_name="osmosis_sheet.pdf",
            questions=questions,
            total_marks=20.0,
            subject_title="Biology",
            client=self.mock_client
        )

        # Marks MUST be 0.0 - NO 0.80 or 0.65 multipliers!
        self.assertEqual(result["obtained_marks"], 0.0)
        self.assertEqual(result["total_questions"], 1)
        self.assertEqual(result["pending_review_questions"], 1)
        self.assertEqual(result["evaluated_questions"], 0)
        self.assertEqual(result["grade"], "Pending")
        self.assertEqual(result["status"], "Pending Review")

        q1_res = result["evaluation_breakdown"][0]
        self.assertTrue(q1_res["pending_review"])
        self.assertEqual(q1_res["obtained_marks"], 0.0)
        self.assertIn("manual review", q1_res["student_response"].lower())

    def test_marks_discrepancy_reported(self):
        """Case 9: Discrepancy between declared total marks and question paper sum is reported."""
        payload_questions = [
            {"question_no": 1, "question_text": "Q1", "max_marks": 25.0, "answer_key": "Ans 1"},
            {"question_no": 2, "question_text": "Q2", "max_marks": 49.0, "answer_key": "Ans 2"}
        ]
        # Declared total is 100, but sum of questions is 74
        qs, total, discrepancy, title = parse_question_paper_and_answer_key(
            qp_source=None,
            ak_source=None,
            payload_questions=payload_questions,
            declared_max_marks=100.0,
            client=None
        )
        self.assertEqual(total, 74.0)
        self.assertIsNotNone(discrepancy)
        self.assertIn("Declared total maximum marks (100.0) conflicts with the sum of question maximum marks (74.0)", discrepancy)

    def test_multiple_students_evaluated_independently(self):
        """Case 8: Multiple student sheets in payload are evaluated as distinct students."""
        questions = [
            {"question_no": 1, "question_text": "Define Work.", "max_marks": 10.0, "answer_key": "Work = Force * Displacement"}
        ]

        def ai_side_effect(**kwargs):
            mock_choice = MagicMock()
            mock_choice.message.content = json.dumps({
                "detected_student_name": None,
                "detected_student_id": None,
                "recognition_status": "unmatched",
                "ai_feedback": "Student answered definition of work.",
                "ai_insights": {"recommendation": "Good job", "focus_areas": [], "strengths": ["Clear formula"]},
                "evaluation_breakdown": [
                    {
                        "question_no": 1,
                        "question_text": "Define Work.",
                        "question_type": "Descriptive",
                        "max_marks": 10.0,
                        "obtained_marks": 9.0,
                        "is_attempted": True,
                        "pending_review": False,
                        "answer_key": "Work = Force * Displacement",
                        "student_response": "Work done is product of force and displacement.",
                        "step_marking": [{"step": "Formula", "max_marks": 10.0, "obtained_marks": 9.0, "status": "Partial"}],
                        "feedback": "Accurate definition and formula."
                    }
                ]
            })
            return MagicMock(choices=[mock_choice])

        self.mock_client.chat.completions.create.side_effect = ai_side_effect

        payload = {
            "questions": questions,
            "students": [
                {"file_name": "student_alice.jpg", "file_path": "student_alice.jpg", "page_count": 1},
                {"file_name": "student_bob.jpg", "file_path": "student_bob.jpg", "page_count": 1}
            ]
        }

        resp, status_code = process_paper_evaluation(payload=payload, client=self.mock_client)
        self.assertEqual(status_code, 200)
        self.assertEqual(resp["total_sheets"], 2)
        self.assertEqual(len(resp["students"]), 2)
        self.assertEqual(resp["students"][0]["id"], 1)
        self.assertEqual(resp["students"][1]["id"], 2)
        self.assertEqual(resp["students"][0]["file_name"], "student_alice.jpg")
        self.assertEqual(resp["students"][1]["file_name"], "student_bob.jpg")

    def test_vision_placeholder_response_rejected_and_flagged(self):
        """Case: Vision model returning generic placeholder is caught and flagged for manual review with 0 marks."""
        questions = [
            {"question_no": 1, "question_text": "Explain Doppler Effect.", "max_marks": 15.0, "model_answer": "Frequency shift due to relative motion."}
        ]

        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "detected_student_name": None,
            "detected_student_id": None,
            "recognition_status": "unmatched",
            "ai_feedback": "Placeholder output",
            "ai_insights": {"recommendation": "Review Doppler Effect", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "question_text": "Explain Doppler Effect.",
                    "question_type": "Descriptive",
                    "max_marks": 15.0,
                    "obtained_marks": 12.0,  # Fabricated marks accompanying placeholder
                    "is_attempted": True,
                    "pending_review": False,
                    "answer_key": "Frequency shift due to relative motion.",
                    "student_response": "Student response extracted and verified",  # FORBIDDEN PLACEHOLDER!
                    "step_marking": [{"step": "Concept", "max_marks": 15.0, "obtained_marks": 12.0, "status": "Full"}],
                    "feedback": "Addressed concepts."
                }
            ]
        })
        self.mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        booklet = {"booklet_index": 1, "page_numbers": [1], "combined_text": ""}
        result = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="doppler.jpg",
            file_name="doppler.jpg",
            questions=questions,
            total_marks=15.0,
            subject_title="Physics",
            client=self.mock_client
        )

        q1_res = result["evaluation_breakdown"][0]
        # Must be flagged for review and marks zeroed out!
        self.assertTrue(q1_res["pending_review"])
        self.assertEqual(q1_res["obtained_marks"], 0.0)
        self.assertIn("Unverified", q1_res["student_response"])
        self.assertEqual(result["obtained_marks"], 0.0)
        self.assertEqual(result["grade"], "Pending")
        self.assertEqual(result["status"], "Pending Review")

    def test_flask_api_post_and_get_privacy(self):
        """Cases: API compatibility, file references retention, and GET identity privacy rules."""
        from app import app

        test_client = app.test_client()

        # Mock OpenAI evaluation inside process_paper_evaluation
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "detected_student_name": None,  # Not on paper!
            "detected_student_id": None,
            "recognition_status": "unmatched",
            "ai_feedback": "Accurate response on photosynthesis.",
            "ai_insights": {"recommendation": "Well done", "focus_areas": [], "strengths": ["Clear definition"]},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "question_text": "What is photosynthesis?",
                    "question_type": "Descriptive",
                    "max_marks": 10.0,
                    "obtained_marks": 9.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "answer_key": "Process of turning light into glucose.",
                    "student_response": "Photosynthesis turns light energy into chemical energy in green plants.",
                    "step_marking": [{"step": "Definition", "max_marks": 10.0, "obtained_marks": 9.0, "status": "Full"}],
                    "feedback": "Clear explanation of light energy conversion."
                }
            ]
        })

        with patch("app.client") as mocked_app_client:
            mocked_app_client.chat = MagicMock()
            mocked_app_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

            # 1. POST request with database name injected by PHP backend
            post_payload = {
                "evaluation_id": 999,
                "student_name": "Database Fake Student",  # Sent by PHP backend
                "student_id": "PHP_DB_123",              # Sent by PHP backend
                "questions": [
                    {"question_no": 1, "question_text": "What is photosynthesis?", "max_marks": 10.0, "answer_key": "Process of turning light into glucose."}
                ],
                "file_path": "student_paper.pdf",
                "question_paper_file": "final_qp.pdf",
                "answer_key_file": "final_ak.pdf"
            }

            post_resp = test_client.post(
                "/api/evaluate-papers",
                data=json.dumps(post_payload),
                content_type="application/json"
            )
            self.assertEqual(post_resp.status_code, 200)
            post_json = post_resp.get_json()

            # Ensure file references are retained
            self.assertEqual(post_json["question_paper_file"], "final_qp.pdf")
            self.assertEqual(post_json["answer_key_file"], "final_ak.pdf")

            # In POST response, PHP DB name must be stripped because not detected on paper!
            student_obj = post_json["students"][0]
            self.assertEqual(student_obj["student_name"], "")
            self.assertEqual(student_obj["student_id"], "")
            self.assertEqual(student_obj["recognition_status"], "unmatched")
            self.assertEqual(student_obj["obtained_marks"], 9.0)
            self.assertEqual(student_obj["grade"], "A+")

            # 2. GET response verification
            get_resp = test_client.get("/api/evaluation/999")
            self.assertEqual(get_resp.status_code, 200)
            get_json = get_resp.get_json()
            get_student = get_json["data"]["students"][0]

            # Privacy rule: GET response strictly removes identity when unmatched on document
            self.assertEqual(get_student["student_name"], "")
            self.assertEqual(get_student["student_id"], "")
            self.assertEqual(get_student["recognition_status"], "unmatched")
            self.assertEqual(get_student["total_marks"], "10.00")
            self.assertEqual(get_student["obtained_marks"], "9.00")

            # File references in GET response
            self.assertEqual(get_json["data"]["question_paper_file"], "final_qp.pdf")
            self.assertEqual(get_json["data"]["answer_key_file"], "final_ak.pdf")

            # 3. GET student breakdown endpoint
            get_student_resp = test_client.get("/api/evaluation/999/student/1")
            self.assertEqual(get_student_resp.status_code, 200)
            get_student_json = get_student_resp.get_json()
            self.assertEqual(get_student_json["student"]["obtained_marks"], 9.0)


if __name__ == "__main__":
    unittest.main()
