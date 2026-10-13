"""
Comprehensive Unit Tests for Student Answer Extraction & AI Paper Evaluation
Validates:
1. Readable handwritten answer extraction & evaluation
2. Blank / unanswered questions
3. Illegible handwriting handling
4. Multi-page answer continuation
5. Missing or invalid answer key
6. Vision model / API failure handling (no fabricated scores)
7. Forbidden placeholder rejection & answer-key copy rejection
8. Multiple students in one upload
9. Marks exceeding maximum & step total inconsistencies
10. Feedback & insights sanitization (no attendance/enrollment text)
"""
import unittest
import json
import os
from unittest.mock import MagicMock, patch

from paper_evaluator import (
    evaluate_student_booklet_ai,
    process_paper_evaluation,
    parse_question_paper_and_answer_key,
    contains_forbidden_placeholder,
    collect_booklet_images,
    evaluation_store,
)


class MockOpenAIResponse:
    def __init__(self, content_dict):
        self.choices = [
            MagicMock(message=MagicMock(content=json.dumps(content_dict)))
        ]


class TestPaperEvaluationAccuracy(unittest.TestCase):

    def setUp(self):
        self.sample_questions = [
            {
                "question_no": 1,
                "question_text": "Describe the pathophysiology of acute myocardial infarction.",
                "question_type": "Descriptive",
                "max_marks": 10.0,
                "model_answer": "Plaque rupture leading to thrombus formation, coronary artery occlusion, and myocardial necrosis.",
                "marking_criteria": [
                    {"step": "Plaque rupture & thrombus", "max_marks": 5.0},
                    {"step": "Occlusion & necrosis", "max_marks": 5.0}
                ]
            },
            {
                "question_no": 2,
                "question_text": "State two clinical manifestations of heart failure.",
                "question_type": "Short Answer",
                "max_marks": 5.0,
                "model_answer": "Dyspnea and peripheral edema.",
                "marking_criteria": [
                    {"step": "Manifestations stated correctly", "max_marks": 5.0}
                ]
            }
        ]

    def test_01_readable_handwritten_answer(self):
        """1. Readable handwritten answer: extracts verbatim text and awards rubric marks."""
        ai_mock = {
            "detected_student_name": "Rahul Verma",
            "detected_student_id": "ST101",
            "recognition_status": "matched",
            "recognition_confidence": 95.0,
            "ai_feedback": "Student showed clear understanding of myocardial infarction pathophysiology and heart failure symptoms.",
            "ai_insights": {
                "recommendation": "Review diagnostic enzyme timelines.",
                "focus_areas": ["Cardiac biomarkers"],
                "strengths": ["Clear explanation of coronary thrombosis"]
            },
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "question_text": "Describe the pathophysiology of acute myocardial infarction.",
                    "question_type": "Descriptive",
                    "max_marks": 10.0,
                    "obtained_marks": 8.5,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Plaque rupture triggers thrombus in coronary artery causing ischemia and tissue necrosis.",
                    "step_marking": [
                        {"step": "Plaque rupture & thrombus", "max_marks": 5.0, "obtained_marks": 4.5, "status": "Partial"},
                        {"step": "Occlusion & necrosis", "max_marks": 5.0, "obtained_marks": 4.0, "status": "Partial"}
                    ],
                    "feedback": "Correctly identified plaque rupture and coronary necrosis."
                },
                {
                    "question_no": 2,
                    "question_text": "State two clinical manifestations of heart failure.",
                    "question_type": "Short Answer",
                    "max_marks": 5.0,
                    "obtained_marks": 5.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Dyspnea on exertion and bilateral ankle swelling / edema.",
                    "step_marking": [
                        {"step": "Manifestations stated correctly", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"}
                    ],
                    "feedback": "Both symptoms accurately identified."
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse(ai_mock)

        booklet = {
            "booklet_index": 1,
            "page_numbers": [1],
            "combined_text": "Student: Rahul Verma (ST101)\nQ1: Plaque rupture...\nQ2: Dyspnea..."
        }

        result = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=self.sample_questions,
            total_marks=15.0,
            subject_title="Pathology",
            client=mock_client
        )

        self.assertEqual(result["total_questions"], 2)
        self.assertEqual(result["attempted_questions"], 2)
        self.assertEqual(result["unanswered_questions"], 0)
        self.assertEqual(result["evaluated_questions"], 2)
        self.assertEqual(result["pending_review_questions"], 0)
        self.assertEqual(result["obtained_marks"], 13.5)
        self.assertEqual(result["total_marks"], 15.0)
        self.assertEqual(result["percentage"], 90.0)
        self.assertEqual(result["grade"], "A+")
        self.assertEqual(result["status"], "Pass")
        self.assertIn("Plaque rupture triggers thrombus", result["evaluation_breakdown"][0]["student_response"])
        self.assertFalse(contains_forbidden_placeholder(result["evaluation_breakdown"][0]["student_response"]))

    def test_02_blank_unanswered_question(self):
        """2. Blank answer: sets is_attempted=false, obtained_marks=0.0, student_response='[Unanswered]'."""
        ai_mock = {
            "detected_student_name": None,
            "detected_student_id": None,
            "recognition_status": "unmatched",
            "recognition_confidence": 0.0,
            "ai_feedback": "Question 2 was not attempted.",
            "ai_insights": {"recommendation": "Attempt all questions.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "question_text": "Describe the pathophysiology of acute myocardial infarction.",
                    "question_type": "Descriptive",
                    "max_marks": 10.0,
                    "obtained_marks": 7.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Artery blocked by clot causing myocardial necrosis.",
                    "step_marking": [{"step": "Pathology", "max_marks": 10.0, "obtained_marks": 7.0, "status": "Partial"}],
                    "feedback": "Core concept present."
                },
                {
                    "question_no": 2,
                    "question_text": "State two clinical manifestations of heart failure.",
                    "question_type": "Short Answer",
                    "max_marks": 5.0,
                    "obtained_marks": 0.0,
                    "is_attempted": False,
                    "pending_review": False,
                    "student_response": "[Unanswered]",
                    "step_marking": [{"step": "Question attempted", "max_marks": 5.0, "obtained_marks": 0.0, "status": "Zero"}],
                    "feedback": "Not attempted."
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse(ai_mock)

        booklet = {"booklet_index": 1, "page_numbers": [1], "combined_text": "Q1 answer only"}

        result = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=self.sample_questions,
            total_marks=15.0,
            subject_title="Pathology",
            client=mock_client
        )

        self.assertEqual(result["total_questions"], 2)
        self.assertEqual(result["attempted_questions"], 1)
        self.assertEqual(result["unanswered_questions"], 1)
        self.assertEqual(result["evaluated_questions"], 1)
        self.assertEqual(result["evaluation_breakdown"][1]["student_response"], "[Unanswered]")
        self.assertEqual(result["evaluation_breakdown"][1]["obtained_marks"], 0.0)

    def test_03_illegible_answer(self):
        """3. Illegible answer: flagged for manual review with 0.0 marks and no fabricated score."""
        ai_mock = {
            "detected_student_name": None,
            "detected_student_id": None,
            "recognition_status": "unmatched",
            "recognition_confidence": 0.0,
            "ai_feedback": "Question 1 handwriting is indecipherable.",
            "ai_insights": {"recommendation": "Manual review required.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "question_text": "Describe the pathophysiology of acute myocardial infarction.",
                    "question_type": "Descriptive",
                    "max_marks": 10.0,
                    "obtained_marks": 0.0,
                    "is_attempted": True,
                    "pending_review": True,
                    "student_response": "[Illegible handwriting - Flagged for manual review]",
                    "step_marking": [{"step": "Manual Verification Required", "max_marks": 10.0, "obtained_marks": 0.0, "status": "Zero"}],
                    "feedback": "Writing is blurred and cannot be deciphered."
                },
                {
                    "question_no": 2,
                    "question_text": "State two clinical manifestations of heart failure.",
                    "question_type": "Short Answer",
                    "max_marks": 5.0,
                    "obtained_marks": 5.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Dyspnea and swelling.",
                    "step_marking": [{"step": "Criteria", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"}],
                    "feedback": "Accurate."
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse(ai_mock)

        booklet = {"booklet_index": 1, "page_numbers": [1], "combined_text": "illegible scribbles"}

        result = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=self.sample_questions,
            total_marks=15.0,
            subject_title="Pathology",
            client=mock_client
        )

        q1 = result["evaluation_breakdown"][0]
        self.assertTrue(q1["pending_review"])
        self.assertEqual(q1["obtained_marks"], 0.0)
        self.assertEqual(q1["student_response"], "[Illegible handwriting - Flagged for manual review]")
        self.assertEqual(result["pending_review_questions"], 1)
        self.assertEqual(result["evaluated_questions"], 1)

    def test_04_multipage_continuation(self):
        """4. Multi-page continuation: collect_booklet_images handles multi-page answers."""
        booklet_multi = {
            "booklet_index": 1,
            "page_numbers": [1, 2],
            "pages": [
                {"page_no": 1, "file_path": "uploads/p1.jpg"},
                {"page_no": 2, "file_path": "uploads/p2.jpg"}
            ],
            "combined_text": "--- PAGE 1 ---\nQ1: Plaque rupture initiates\n--- PAGE 2 ---\ncoronary occlusion and tissue necrosis."
        }

        with patch("paper_evaluator.resolve_file_location", return_value=None):
            images = collect_booklet_images(booklet_multi)
            self.assertEqual(len(images), 0)

        ai_mock = {
            "detected_student_name": None,
            "detected_student_id": None,
            "recognition_status": "unmatched",
            "recognition_confidence": 0.0,
            "ai_feedback": "Student completed full explanation across pages.",
            "ai_insights": {"recommendation": "Good detail.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "question_text": self.sample_questions[0]["question_text"],
                    "question_type": "Descriptive",
                    "max_marks": 10.0,
                    "obtained_marks": 10.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Plaque rupture initiates coronary occlusion and tissue necrosis spanning across page 1 and page 2.",
                    "step_marking": [{"step": "Complete Mechanism", "max_marks": 10.0, "obtained_marks": 10.0, "status": "Full"}],
                    "feedback": "Full answer transcribed across multiple pages."
                },
                {
                    "question_no": 2,
                    "question_text": self.sample_questions[1]["question_text"],
                    "question_type": "Short Answer",
                    "max_marks": 5.0,
                    "obtained_marks": 5.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Shortness of breath and peripheral edema.",
                    "step_marking": [{"step": "Criteria", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"}],
                    "feedback": "Correct."
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse(ai_mock)

        result = evaluate_student_booklet_ai(
            booklet=booklet_multi,
            file_path="",
            file_name="multi.pdf",
            questions=self.sample_questions,
            total_marks=15.0,
            subject_title="Pathology",
            client=mock_client
        )

        self.assertEqual(result["page_count"], 2)
        self.assertIn("spanning across page 1 and page 2", result["evaluation_breakdown"][0]["student_response"])
        self.assertEqual(result["obtained_marks"], 15.0)

    def test_05_missing_or_invalid_answer_key(self):
        """5. Missing answer key: reports missing dependency and flags question with 0 marks."""
        questions_no_key = [
            {
                "question_no": 1,
                "question_text": "Explain something.",
                "max_marks": 10.0,
                "model_answer": "",  # Empty answer key
                "marking_criteria": []
            }
        ]

        ai_mock = {
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "question_text": "Explain something.",
                    "max_marks": 10.0,
                    "obtained_marks": 8.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Student answer here.",
                    "step_marking": []
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse(ai_mock)

        booklet = {"booklet_index": 1, "page_numbers": [1], "combined_text": "Student text"}

        result = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=questions_no_key,
            total_marks=10.0,
            subject_title="General",
            client=mock_client
        )

        q1 = result["evaluation_breakdown"][0]
        self.assertTrue(q1["pending_review"])
        self.assertEqual(q1["obtained_marks"], 0.0)
        self.assertIn("Missing answer key", q1["feedback"])

    def test_06_vision_model_api_failure(self):
        """6. Vision model failure: flags all questions for review with 0.0 marks, Status=Pending Review, Grade=Pending."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("OpenAI API rate limit exceeded")

        booklet = {"booklet_index": 1, "page_numbers": [1], "combined_text": "Some handwritten content"}

        result = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=self.sample_questions,
            total_marks=15.0,
            subject_title="Pathology",
            client=mock_client
        )

        self.assertEqual(result["total_questions"], 2)
        self.assertEqual(result["pending_review_questions"], 2)
        self.assertEqual(result["evaluated_questions"], 0)
        self.assertEqual(result["obtained_marks"], 0.0)
        self.assertEqual(result["status"], "Pending Review")
        self.assertEqual(result["grade"], "Pending")
        self.assertIn("manual instructor review", result["ai_feedback"].lower())

    def test_07_forbidden_placeholder_and_copied_key_rejection(self):
        """7. Placeholders and copied answer keys are strictly rejected and flagged for review."""
        placeholder_text = "Student response extracted and verified across 1 uploaded answer page(s)."
        self.assertTrue(contains_forbidden_placeholder(placeholder_text))

        copied_key = "Plaque rupture leading to thrombus formation, coronary artery occlusion, and myocardial necrosis."

        ai_mock = {
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 8.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": placeholder_text,
                    "step_marking": [{"step": "Crit", "max_marks": 10.0, "obtained_marks": 8.0, "status": "Partial"}]
                },
                {
                    "question_no": 2,
                    "max_marks": 5.0,
                    "obtained_marks": 5.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": copied_key,
                    "step_marking": [{"step": "Crit", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"}]
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse(ai_mock)

        booklet = {"booklet_index": 1, "page_numbers": [1], "combined_text": "Handwritten text"}

        sample_q2_copied = [
            self.sample_questions[0],
            {
                "question_no": 2,
                "question_text": "Question with key.",
                "max_marks": 5.0,
                "model_answer": copied_key,
                "marking_criteria": []
            }
        ]

        result = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=sample_q2_copied,
            total_marks=15.0,
            subject_title="Pathology",
            client=mock_client
        )

        q1 = result["evaluation_breakdown"][0]
        self.assertTrue(q1["pending_review"])
        self.assertEqual(q1["obtained_marks"], 0.0)
        self.assertEqual(q1["student_response"], "[Unverified transcription - Flagged for manual review]")

        q2 = result["evaluation_breakdown"][1]
        self.assertTrue(q2["pending_review"])
        self.assertEqual(q2["obtained_marks"], 0.0)
        self.assertEqual(q2["student_response"], "[Flagged for review - Answer key duplication detected]")

    def test_08_step_marking_inconsistency_and_max_marks_clamping(self):
        """8. Step marks sum must equal question marks; obtained marks cannot exceed max marks."""
        ai_mock = {
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 15.0,  # Exceeds max_marks 10.0!
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Correct detailed answer.",
                    "step_marking": [
                        {"step": "Step 1", "max_marks": 5.0, "obtained_marks": 6.0},  # Exceeds step max!
                        {"step": "Step 2", "max_marks": 5.0, "obtained_marks": 4.0}
                    ]
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse(ai_mock)

        booklet = {"booklet_index": 1, "page_numbers": [1], "combined_text": "Answer text"}

        result = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=[self.sample_questions[0]],
            total_marks=10.0,
            subject_title="Pathology",
            client=mock_client
        )

        q1 = result["evaluation_breakdown"][0]
        self.assertLessEqual(q1["obtained_marks"], 10.0)
        step_sum = sum(st["obtained_marks"] for st in q1["step_marking"])
        self.assertAlmostEqual(q1["obtained_marks"], step_sum, places=1)
        self.assertLessEqual(q1["step_marking"][0]["obtained_marks"], 5.0)

    def test_09_multiple_students_in_one_upload(self):
        """9. Multiple students in one payload: evaluates each student independently."""
        payload = {
            "title": "Pathology Exam",
            "max_marks": 15.0,
            "questions": self.sample_questions,
            "students": [
                {"file_path": "student1.jpg", "file_name": "student1.jpg", "page_count": 1},
                {"file_path": "student2.jpg", "file_name": "student2.jpg", "page_count": 1}
            ]
        }

        ai_mock = {
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 7.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Plaque rupture response.",
                    "step_marking": [{"step": "S1", "max_marks": 10.0, "obtained_marks": 7.0, "status": "Partial"}]
                },
                {
                    "question_no": 2,
                    "max_marks": 5.0,
                    "obtained_marks": 3.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Dyspnea only.",
                    "step_marking": [{"step": "S2", "max_marks": 5.0, "obtained_marks": 3.0, "status": "Partial"}]
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse(ai_mock)

        resp, status = process_paper_evaluation(payload=payload, client=mock_client)
        self.assertEqual(status, 200)
        self.assertEqual(resp["total_sheets"], 2)
        self.assertEqual(len(resp["students"]), 2)
        self.assertEqual(resp["students"][0]["id"], 1)
        self.assertEqual(resp["students"][1]["id"], 2)

    def test_10_sanitizes_attendance_and_pending_assessment_feedback(self):
        """10. Strips out attendance/enrollment terms and provides subject-relevant feedback."""
        ai_mock = {
            "ai_feedback": "Student attendance logging has just begun and academic assessments are pending for the term.",
            "ai_insights": {
                "recommendation": "Maintain regular attendance of working days.",
                "focus_areas": ["enrollment procedures"],
                "strengths": ["present days consistency"]
            },
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 8.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Valid response on plaque rupture.",
                    "step_marking": [{"step": "S1", "max_marks": 10.0, "obtained_marks": 8.0, "status": "Partial"}],
                    "feedback": "Student attended all sessions."
                },
                {
                    "question_no": 2,
                    "max_marks": 5.0,
                    "obtained_marks": 4.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Valid heart failure symptoms.",
                    "step_marking": [{"step": "S2", "max_marks": 5.0, "obtained_marks": 4.0, "status": "Partial"}],
                    "feedback": "Enrollment active."
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse(ai_mock)

        booklet = {"booklet_index": 1, "page_numbers": [1], "combined_text": "Student exam"}

        result = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=self.sample_questions,
            total_marks=15.0,
            subject_title="Pathology",
            client=mock_client
        )

        fb_lower = result["ai_feedback"].lower()
        self.assertNotIn("academic assessments are pending", fb_lower)
        self.assertNotIn("attendance", fb_lower)
        self.assertNotIn("enrollment", fb_lower)
        self.assertNotIn("working days", fb_lower)

        rec_lower = result["ai_insights"]["recommendation"].lower()
        self.assertNotIn("attendance", rec_lower)


if __name__ == "__main__":
    unittest.main()
