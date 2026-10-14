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
    extract_identity_from_document_text,
    normalize_extracted_text,
    detect_booklet_boundaries,
    match_student_identity_with_backend,
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

    def test_11_identity_extraction_label_variations(self):
        """11. Test all identity label variations and separator formats."""
        # Test variations of Student Name
        cases_name = [
            ("Student Name: Rahul Verma\nQ1: ...", "Rahul Verma"),
            ("Candidate Name - John Doe\nQ1: ...", "John Doe"),
            ("Name of Student: Priya Sharma\nQ1: ...", "Priya Sharma"),
            ("Name of Candidate: Anita Roy\nQ1: ...", "Anita Roy"),
            ("Name of the Student = David Miller\nQ1: ...", "David Miller"),
            ("Student's Name: Emily Davis\nQ1: ...", "Emily Davis"),
            ("Candidate Name Rahul Verma\nQ1: ...", "Rahul Verma"),  # space only
            ("Student Name:\nRahul Verma\nQ1: ...", "Rahul Verma"),  # newline
            ("Candidate Name\nJohn Doe\nQ1: ...", "John Doe"),      # newline without colon
        ]
        for text, expected in cases_name:
            name, _, _, _, _ = extract_identity_from_document_text(text)
            self.assertEqual(name, expected, f"Failed for: {text!r}")

        # Test variations of Student ID / Number / Roll / Adm / Reg
        cases_id = [
            ("Student ID: ST101\n...", "ST101"),
            ("Student ID - ST-101/A\n...", "ST-101/A"),
            ("Student Number: 84\n...", "84"),
            ("Student No.: 0042\n...", "0042"),
            ("Student No: 0042\n...", "0042"),
            ("Candidate ID: CID-9988\n...", "CID-9988"),
            ("Candidate Number: 456\n...", "456"),
            ("Candidate No: 456\n...", "456"),
            ("Roll Number: 12\n...", "12"),
            ("Roll No.: 007\n...", "007"),
            ("Roll No: 101\n...", "101"),
            ("Roll No 0042\n...", "0042"),  # space only
            ("Roll No:\n0042\n...", "0042"), # newline
            ("Admission Number: ADM/2023/042\n...", "ADM/2023/042"),
            ("Admission No.: ADM-01\n...", "ADM-01"),
            ("Adm No: 2023-01\n...", "2023-01"),
            ("Register Number: KUHS202301\n...", "KUHS202301"),
            ("Register No.: 12345\n...", "12345"),
            ("Reg Number: 12345\n...", "12345"),
            ("Reg No.: 0055\n...", "0055"),
            ("Registration Number: 987654\n...", "987654"),
            ("Registration No.: 00987\n...", "00987"),
            ("KUHS Registration Number: 2023001\n...", "2023001"),
            ("KUHS Reg No: 2023001\n...", "2023001"),
            ("KUHS Reg. No.: KUHS-2023-042\n...", "KUHS-2023-042"),
            ("KUHS No: 2023001\n...", "2023001"),
        ]
        for text, expected in cases_id:
            _, det_id, _, _, _ = extract_identity_from_document_text(text)
            self.assertEqual(det_id, expected, f"Failed for: {text!r}")

    def test_12_preserve_leading_zeros_and_symbols_in_evaluation(self):
        """12. Preserves leading zeros and symbols in final_student_id."""
        booklet_leading_zero = {
            "booklet_index": 1,
            "page_numbers": [1],
            "combined_text": "Student Name: Rahul Verma\nRoll No: 0042\nQ1: Plaque rupture..."
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse({
            "detected_student_name": "Rahul Verma",
            "detected_student_id": "0042",
            "recognition_status": "matched",
            "recognition_confidence": 98.0,
            "ai_feedback": "Good work.",
            "ai_insights": {"recommendation": "Study more.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 8.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Plaque rupture triggers thrombus.",
                    "step_marking": [{"step": "Step 1", "max_marks": 10.0, "obtained_marks": 8.0, "status": "Partial"}]
                }
            ]
        })
        res = evaluate_student_booklet_ai(
            booklet=booklet_leading_zero,
            file_path="",
            file_name="sheet.pdf",
            questions=[self.sample_questions[0]],
            total_marks=10.0,
            subject_title="Pathology",
            client=mock_client
        )
        self.assertEqual(res["student_name"], "Rahul Verma")
        self.assertEqual(res["student_id"], "0042")
        self.assertEqual(res["roll_no"], "0042")

    def test_13_reject_forbidden_words_and_metadata(self):
        """13. Rejects exam names, subject names, question numbers, marks, dates, and institutions."""
        invalid_texts = [
            "Student Name: Annual Examination 2026\nRoll No: 101",
            "Student Name: Pathology\nRoll No: 101",
            "Student Name: Question 1\nRoll No: 101",
            "Candidate Name: Government Medical College\nRoll No: 101",
            "Candidate Name: Maximum Marks 100\nRoll No: 101",
            "Student Name: 14/10/2026\nRoll No: 101",
        ]
        for txt in invalid_texts:
            name, det_id, _, _, _ = extract_identity_from_document_text(txt)
            self.assertIsNone(name, f"Expected name to be None for {txt!r}, but got {name!r}")
            self.assertEqual(det_id, "101")

    def test_14_multipage_booklet_recovery_and_no_header_splitting(self):
        """14. Inspects Page 1 first, recovers missing ID from Page 2 header, does NOT split on running header."""
        page_1 = "Candidate Information:\nStudent Name: Rahul Verma\nInstructions to Candidates..."
        page_2 = "--- PAGE 2 ---\nRahul Verma | Roll No: 101\nQ1: Plaque rupture initiates coronary occlusion..."
        combined = f"{page_1}\n\n{page_2}"

        # Page 1 had name, page 2 had roll no -> extract_identity recovers both
        name, det_id, _, roll, _ = extract_identity_from_document_text(combined)
        self.assertEqual(name, "Rahul Verma")
        self.assertEqual(det_id, "101")
        self.assertEqual(roll, "101")

    def test_15_multistudent_boundary_detection_and_independent_identity(self):
        """15. Independent extraction per booklet in combined document, without leaking across booklets."""
        p1 = "Examination Answer Sheet\nStudent Name: Student Alpha\nRoll No: 001\nQ1: Answer A"
        p2 = "Page 2 of 2\nStudent Alpha | Roll No: 001\nQ2: Answer A cont."
        p3 = "Examination Answer Sheet\nStudent Name: Student Beta\nRoll No: 002\nQ1: Answer B"
        p4 = "Page 2 of 2\nStudent Beta | Roll No: 002\nQ2: Answer B cont."
        p5 = "Unidentified Sheet\nNo cover page or name here\nQ1: Answer C"

        mock_pages = [
            MagicMock(extract_text=MagicMock(return_value=p1), extract_tables=MagicMock(return_value=[])),
            MagicMock(extract_text=MagicMock(return_value=p2), extract_tables=MagicMock(return_value=[])),
            MagicMock(extract_text=MagicMock(return_value=p3), extract_tables=MagicMock(return_value=[])),
            MagicMock(extract_text=MagicMock(return_value=p4), extract_tables=MagicMock(return_value=[])),
            MagicMock(extract_text=MagicMock(return_value=p5), extract_tables=MagicMock(return_value=[])),
        ]
        mock_pdf = MagicMock(pages=mock_pages)
        mock_pdf.__enter__.return_value = mock_pdf

        with patch("pdfplumber.open", return_value=mock_pdf):
            booklets = detect_booklet_boundaries("dummy.pdf")

        self.assertGreaterEqual(len(booklets), 2)
        # First booklet: Student Alpha, Roll 001
        self.assertEqual(booklets[0]["detected_name"], "Student Alpha")
        self.assertEqual(booklets[0]["detected_id"], "001")
        # Second booklet: Student Beta, Roll 002 (never reusing Student Alpha)
        self.assertEqual(booklets[1]["detected_name"], "Student Beta")
        self.assertEqual(booklets[1]["detected_id"], "002")

        # Third booklet (if segmented at p5) must NOT borrow Student Beta's identity
        if len(booklets) >= 3:
            self.assertIsNone(booklets[2]["detected_name"])
            self.assertIsNone(booklets[2]["detected_id"])

    def test_16_vision_unverified_identity_flagged_for_manual_review(self):
        """16. Ambiguous handwritten identity returns recognition_status='unverified' and identity_manual_review=True."""
        booklet = {
            "booklet_index": 1,
            "page_numbers": [1],
            "combined_text": ""  # Scanned image with unreadable handwriting
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse({
            "detected_student_name": "R???l V?rma",
            "detected_student_id": "ST???",
            "recognition_status": "unverified",
            "recognition_confidence": 45.0,
            "identity_manual_review": True,
            "ai_feedback": "Student identity is partially smudged.",
            "ai_insights": {"recommendation": "Review cover sheet.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 7.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Plaque rupture triggers thrombosis.",
                    "step_marking": [{"step": "Step 1", "max_marks": 10.0, "obtained_marks": 7.0, "status": "Partial"}]
                }
            ]
        })
        res = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=[self.sample_questions[0]],
            total_marks=10.0,
            subject_title="Pathology",
            client=mock_client
        )
        self.assertEqual(res["recognition_status"], "unverified")
        self.assertTrue(res["identity_manual_review"])
        self.assertIn("Flagged for manual review", res["recognized_text"])

    def test_17_text_identity_not_overwritten_by_null_vision(self):
        """17. Preserves valid text-extracted identity even if vision model returns null for identity."""
        booklet = {
            "booklet_index": 1,
            "page_numbers": [1],
            "detected_name": "Rahul Verma",
            "detected_id": "101",
            "combined_text": "Student Name: Rahul Verma\nRoll No: 101\nQ1: Answer text..."
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse({
            "detected_student_name": None,  # Vision didn't re-extract
            "detected_student_id": None,
            "recognition_status": "unmatched",
            "recognition_confidence": 0.0,
            "ai_feedback": "Evaluated successfully.",
            "ai_insights": {"recommendation": "Good.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 8.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Plaque rupture answer.",
                    "step_marking": [{"step": "Step 1", "max_marks": 10.0, "obtained_marks": 8.0, "status": "Partial"}]
                }
            ]
        })
        res = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=[self.sample_questions[0]],
            total_marks=10.0,
            subject_title="Pathology",
            client=mock_client
        )
        # Identity extracted from text must be preserved!
        self.assertEqual(res["student_name"], "Rahul Verma")
        self.assertEqual(res["student_id"], 101)
        self.assertEqual(res["recognition_status"], "matched")

    def test_18_handwritten_code_matches_php_record(self):
        """Regression Test 1: A handwritten student code that matches the PHP record."""
        # Answer booklet has handwritten code 'AKN-25/1110/' in header
        booklet = {
            "booklet_index": 1,
            "page_numbers": [1],
            "combined_text": "",
            "detected_adm": "AKN-25/1110/",
            "detected_id": "AKN-25/1110/"
        }
        # PHP backend supplies authoritative student record with DB id=1 and admission_no='AKN-25/1110/'
        backend_rec = {
            "student_id": 1,
            "admission_no": "AKN-25/1110/",
            "student_name": "AADHISREE M L"
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse({
            "detected_student_name": None,
            "detected_admission_no": "AKN-25/1110/",
            "detected_student_id": "AKN-25/1110/",
            "recognition_status": "matched",
            "recognition_confidence": 98.0,
            "identity_manual_review": False,
            "ai_feedback": "Accurate explanation of pathophysiology.",
            "ai_insights": {"recommendation": "Great understanding.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 10.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Homeostasis maintains internal physiological equilibrium.",
                    "step_marking": [
                        {"step": "Plaque rupture & thrombus", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"},
                        {"step": "Occlusion & necrosis", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"}
                    ]
                }
            ]
        })
        res = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=[self.sample_questions[0]],
            total_marks=10.0,
            subject_title="Pathology",
            backend_rec=backend_rec,
            client=mock_client
        )
        self.assertEqual(res["recognition_status"], "matched")
        self.assertGreaterEqual(res["recognition_confidence"], 95.0)
        self.assertFalse(res["identity_manual_review"])
        self.assertEqual(res["admission_no"], "AKN-25/1110/")
        self.assertEqual(res["evaluation_breakdown"][0]["obtained_marks"], 10.0)
        self.assertTrue(res["evaluation_breakdown"][0]["is_attempted"])
        self.assertFalse(res["evaluation_breakdown"][0]["pending_review"])

    def test_19_handwritten_code_differs_from_php_record(self):
        """Regression Test 2: A handwritten code that differs from the PHP record."""
        # Answer booklet has handwritten code 'AKN-25/1110/'
        booklet = {
            "booklet_index": 1,
            "page_numbers": [1],
            "combined_text": "",
            "detected_adm": "AKN-25/1110/",
            "detected_id": "AKN-25/1110/"
        }
        # PHP backend authoritative record expects 'AKN-25/1067'
        backend_rec = {
            "student_id": 1,
            "admission_no": "AKN-25/1067",
            "student_name": "AADHISREE M L"
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse({
            "detected_student_name": None,
            "detected_admission_no": "AKN-25/1110/",
            "detected_student_id": "AKN-25/1110/",
            "recognition_status": "unmatched",
            "recognition_confidence": 0.0,
            "identity_manual_review": False,
            "ai_feedback": "Evaluated handwritten answers.",
            "ai_insights": {"recommendation": "Review cardiology.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 9.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Plaque rupture triggers thrombosis and myocardial necrosis.",
                    "step_marking": [
                        {"step": "Plaque rupture & thrombus", "max_marks": 5.0, "obtained_marks": 4.5, "status": "Partial"},
                        {"step": "Occlusion & necrosis", "max_marks": 5.0, "obtained_marks": 4.5, "status": "Partial"}
                    ]
                }
            ]
        })
        res = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=[self.sample_questions[0]],
            total_marks=10.0,
            subject_title="Pathology",
            backend_rec=backend_rec,
            client=mock_client
        )
        # Must be unmatched because AKN-25/1110/ != AKN-25/1067
        self.assertEqual(res["recognition_status"], "unmatched")
        self.assertEqual(res["recognition_confidence"], 0.0)
        # Authoritative backend student name 'AADHISREE M L' must NOT be leaked
        self.assertNotEqual(res["student_name"], "AADHISREE M L")
        self.assertEqual(res["admission_no"], "AKN-25/1110/")
        self.assertIn("mismatch", res["recognized_text"].lower())
        # Grading continues independently
        self.assertEqual(res["obtained_marks"], 9.0)

    def test_20_student_name_and_code_both_readable(self):
        """Regression Test 3: A student name and code that are both readable."""
        booklet = {
            "booklet_index": 1,
            "page_numbers": [1],
            "combined_text": "Student Name: Pooja Nair\nAdmission No: AKN-25/2042\nQ1: Homeostasis..."
        }
        backend_rec = {
            "student_id": 42,
            "admission_no": "AKN-25/2042",
            "student_name": "Pooja Nair"
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse({
            "detected_student_name": "Pooja Nair",
            "detected_admission_no": "AKN-25/2042",
            "detected_student_id": "AKN-25/2042",
            "recognition_status": "matched",
            "recognition_confidence": 99.0,
            "identity_manual_review": False,
            "ai_feedback": "Excellent response.",
            "ai_insights": {"recommendation": "Maintain consistency.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 10.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Plaque rupture triggers coronary thrombosis.",
                    "step_marking": [
                        {"step": "Plaque rupture & thrombus", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"},
                        {"step": "Occlusion & necrosis", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"}
                    ]
                }
            ]
        })
        res = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=[self.sample_questions[0]],
            total_marks=10.0,
            subject_title="Pathology",
            backend_rec=backend_rec,
            client=mock_client
        )
        self.assertEqual(res["recognition_status"], "matched")
        self.assertEqual(res["student_name"], "Pooja Nair")
        self.assertEqual(res["admission_no"], "AKN-25/2042")
        self.assertGreaterEqual(res["recognition_confidence"], 98.0)
        self.assertFalse(res["identity_manual_review"])

    def test_21_unreadable_or_ambiguous_code(self):
        """Regression Test 4: An unreadable or ambiguous code."""
        booklet = {
            "booklet_index": 1,
            "page_numbers": [1],
            "combined_text": ""
        }
        backend_rec = {
            "student_id": 1,
            "admission_no": "AKN-25/1067",
            "student_name": "AADHISREE M L"
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse({
            "detected_student_name": None,
            "detected_admission_no": "AKN-25/???",
            "detected_student_id": "AKN-25/???",
            "recognition_status": "unverified",
            "recognition_confidence": 50.0,
            "identity_manual_review": True,
            "ai_feedback": "Header student code is smudged.",
            "ai_insights": {"recommendation": "Review cover page.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 8.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Plaque rupture triggers thrombosis.",
                    "step_marking": [
                        {"step": "Plaque rupture & thrombus", "max_marks": 5.0, "obtained_marks": 4.0, "status": "Partial"},
                        {"step": "Occlusion & necrosis", "max_marks": 5.0, "obtained_marks": 4.0, "status": "Partial"}
                    ]
                }
            ]
        })
        res = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=[self.sample_questions[0]],
            total_marks=10.0,
            subject_title="Pathology",
            backend_rec=backend_rec,
            client=mock_client
        )
        self.assertEqual(res["recognition_status"], "unverified")
        self.assertEqual(res["recognition_confidence"], 50.0)
        self.assertTrue(res["identity_manual_review"])
        self.assertIn("manual review", res["recognized_text"].lower())

    def test_22_correct_handwritten_answer_closely_matching_model_answer(self):
        """Regression Test 5: A correct handwritten answer closely matching the model answer is NOT flagged as duplicate."""
        booklet = {
            "booklet_index": 1,
            "page_numbers": [1],
            "combined_text": "Q1: Plaque rupture leading to thrombus formation, coronary artery occlusion, and myocardial necrosis."
        }
        # Student writes textbook explanation that closely follows model answer
        student_text = "Plaque rupture leading to thrombus formation, coronary artery occlusion, and myocardial necrosis."
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse({
            "detected_student_name": "Pooja Nair",
            "detected_student_id": "AKN-25/2042",
            "recognition_status": "matched",
            "recognition_confidence": 98.0,
            "identity_manual_review": False,
            "ai_feedback": "Perfect explanation.",
            "ai_insights": {"recommendation": "Excellent.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 10.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": student_text,
                    "step_marking": [
                        {"step": "Plaque rupture & thrombus", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"},
                        {"step": "Occlusion & necrosis", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"}
                    ],
                    "feedback": "Flawless description matching pathophysiology criteria."
                }
            ]
        })
        res = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=[self.sample_questions[0]],
            total_marks=10.0,
            subject_title="Pathology",
            client=mock_client
        )
        q1 = res["evaluation_breakdown"][0]
        # Must NOT be flagged as duplicated answer key!
        self.assertNotIn("Answer key duplication detected", q1["student_response"])
        self.assertEqual(q1["obtained_marks"], 10.0)
        self.assertTrue(q1["is_attempted"])
        self.assertFalse(q1["pending_review"])

    def test_23_answer_genuinely_missing(self):
        """Regression Test 6: An answer that is genuinely missing has is_attempted=False, marks=0.0."""
        booklet = {
            "booklet_index": 1,
            "page_numbers": [1],
            "combined_text": "Q1: Plaque rupture triggers myocardial necrosis."
        }
        # Student answered Q1 but omitted Q2
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse({
            "detected_student_name": "Pooja Nair",
            "detected_student_id": "AKN-25/2042",
            "recognition_status": "matched",
            "recognition_confidence": 98.0,
            "ai_feedback": "Evaluated.",
            "ai_insights": {"recommendation": "Answer all questions.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 8.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Plaque rupture triggers myocardial necrosis.",
                    "step_marking": [
                        {"step": "Plaque rupture & thrombus", "max_marks": 5.0, "obtained_marks": 4.0, "status": "Partial"},
                        {"step": "Occlusion & necrosis", "max_marks": 5.0, "obtained_marks": 4.0, "status": "Partial"}
                    ]
                },
                {
                    "question_no": 2,
                    "max_marks": 5.0,
                    "obtained_marks": 0.0,
                    "is_attempted": False,
                    "pending_review": False,
                    "student_response": "[Unanswered]",
                    "feedback": "Question 2 was not attempted by the student."
                }
            ]
        })
        res = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=self.sample_questions,
            total_marks=15.0,
            subject_title="Pathology",
            client=mock_client
        )
        q2 = res["evaluation_breakdown"][1]
        self.assertFalse(q2["is_attempted"])
        self.assertFalse(q2["pending_review"])
        self.assertEqual(q2["obtained_marks"], 0.0)
        self.assertEqual(q2["student_response"], "[Unanswered]")
        self.assertEqual(res["attempted_questions"], 1)
        self.assertEqual(res["unanswered_questions"], 1)
        self.assertEqual(res["total_questions"], 2)

    def test_24_answer_present_on_later_page(self):
        """Regression Test 7: Multi-page answer continuation across pages."""
        booklet = {
            "booklet_index": 1,
            "page_numbers": [1, 2],
            "combined_text": "Page 1: Plaque rupture in coronary artery...\nPage 2: ...leads to occlusion, ischemia, and tissue necrosis."
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse({
            "detected_student_name": "Pooja Nair",
            "detected_student_id": "AKN-25/2042",
            "recognition_status": "matched",
            "recognition_confidence": 98.0,
            "ai_feedback": "Comprehensive multi-page answer.",
            "ai_insights": {"recommendation": "Great work.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 10.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Plaque rupture in coronary artery leads to occlusion, ischemia, and tissue necrosis.",
                    "step_marking": [
                        {"step": "Plaque rupture & thrombus", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"},
                        {"step": "Occlusion & necrosis", "max_marks": 5.0, "obtained_marks": 5.0, "status": "Full"}
                    ]
                }
            ]
        })
        res = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=[self.sample_questions[0]],
            total_marks=10.0,
            subject_title="Pathology",
            client=mock_client
        )
        q1 = res["evaluation_breakdown"][0]
        self.assertTrue(q1["is_attempted"])
        self.assertFalse(q1["pending_review"])
        self.assertEqual(q1["obtained_marks"], 10.0)
        self.assertEqual(res["page_count"], 2)

    def test_25_extraction_failure_flagged_for_review(self):
        """Regression Test 8: An answer that cannot be located because of extraction failure is flagged for review (NOT unattempted)."""
        booklet = {
            "booklet_index": 1,
            "page_numbers": [1],
            "combined_text": ""
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse({
            "detected_student_name": "Pooja Nair",
            "detected_student_id": "AKN-25/2042",
            "recognition_status": "matched",
            "recognition_confidence": 95.0,
            "ai_feedback": "Page scan unreadable for Question 1.",
            "ai_insights": {"recommendation": "Manual review required.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 0.0,
                    "is_attempted": True,
                    "pending_review": True,
                    "student_response": "Extraction failed due to corrupted page scan.",
                    "feedback": "Extraction failed for Question 1. Flagged for review.",
                    "step_marking": [
                        {"step": "Criteria", "max_marks": 10.0, "obtained_marks": 0.0, "status": "Zero"}
                    ]
                }
            ]
        })
        res = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=[self.sample_questions[0]],
            total_marks=10.0,
            subject_title="Pathology",
            client=mock_client
        )
        q1 = res["evaluation_breakdown"][0]
        # Must be treated as pending review, NOT as an unattempted question!
        self.assertTrue(q1["pending_review"])
        self.assertTrue(q1["is_attempted"])
        self.assertEqual(q1["obtained_marks"], 0.0)
        self.assertIn("Extraction failure", q1["student_response"])
        self.assertEqual(res["pending_review_questions"], 1)
        self.assertEqual(res["evaluated_questions"], 0)

    def test_26_multistudent_pdf_different_identities(self):
        """Regression Test 9: A multi-student PDF with different identities segmented and matched correctly."""
        p1 = "Admission No: AKN-25/1110/\nCandidate: Student Alpha\nQ1: MI answer"
        p2 = "Page 2: continuation of Alpha"
        p3 = "Admission No: AKN-25/2042\nCandidate: Student Beta\nQ1: MI answer"
        p4 = "Page 2: continuation of Beta"

        mock_pages = [
            MagicMock(extract_text=MagicMock(return_value=p1), extract_tables=MagicMock(return_value=[])),
            MagicMock(extract_text=MagicMock(return_value=p2), extract_tables=MagicMock(return_value=[])),
            MagicMock(extract_text=MagicMock(return_value=p3), extract_tables=MagicMock(return_value=[])),
            MagicMock(extract_text=MagicMock(return_value=p4), extract_tables=MagicMock(return_value=[])),
        ]
        mock_pdf = MagicMock(pages=mock_pages)
        mock_pdf.__enter__.return_value = mock_pdf

        with patch("pdfplumber.open", return_value=mock_pdf):
            booklets = detect_booklet_boundaries("dummy.pdf")

        self.assertEqual(len(booklets), 2)
        self.assertEqual(booklets[0]["detected_adm"], "AKN-25/1110/")
        self.assertEqual(booklets[1]["detected_adm"], "AKN-25/2042")
        # Identities must remain independent and not cross-contaminate
        self.assertNotEqual(booklets[0]["detected_adm"], booklets[1]["detected_adm"])

    def test_27_pending_review_and_consistent_metrics(self):
        """Regression Test 10: Consistent evaluation metrics with mixed attempted, pending-review, and unanswered questions."""
        three_questions = [
            {"question_no": 1, "question_text": "Q1", "max_marks": 10.0, "model_answer": "Key 1"},
            {"question_no": 2, "question_text": "Q2", "max_marks": 5.0, "model_answer": "Key 2"},
            {"question_no": 3, "question_text": "Q3", "max_marks": 5.0, "model_answer": "Key 3"}
        ]
        booklet = {
            "booklet_index": 1,
            "page_numbers": [1],
            "combined_text": "Q1 answered, Q2 blurred handwriting, Q3 blank"
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockOpenAIResponse({
            "detected_student_name": "Pooja Nair",
            "detected_student_id": "AKN-25/2042",
            "recognition_status": "matched",
            "recognition_confidence": 95.0,
            "ai_feedback": "Partial evaluation completed.",
            "ai_insights": {"recommendation": "Review pending questions.", "focus_areas": [], "strengths": []},
            "evaluation_breakdown": [
                {
                    "question_no": 1,
                    "max_marks": 10.0,
                    "obtained_marks": 10.0,
                    "is_attempted": True,
                    "pending_review": False,
                    "student_response": "Complete correct answer for Q1.",
                    "step_marking": [{"step": "Criteria", "max_marks": 10.0, "obtained_marks": 10.0, "status": "Full"}]
                },
                {
                    "question_no": 2,
                    "max_marks": 5.0,
                    "obtained_marks": 0.0,
                    "is_attempted": True,
                    "pending_review": True,
                    "student_response": "[Illegible handwriting - Flagged for manual review]",
                    "feedback": "Handwriting is smudged."
                },
                {
                    "question_no": 3,
                    "max_marks": 5.0,
                    "obtained_marks": 0.0,
                    "is_attempted": False,
                    "pending_review": False,
                    "student_response": "[Unanswered]",
                    "feedback": "Not attempted."
                }
            ]
        })
        res = evaluate_student_booklet_ai(
            booklet=booklet,
            file_path="",
            file_name="sheet.pdf",
            questions=three_questions,
            total_marks=20.0,
            subject_title="Pathology",
            client=mock_client
        )
        # Invariants enforcement:
        # total_questions = attempted_questions + unanswered_questions
        self.assertEqual(res["total_questions"], res["attempted_questions"] + res["unanswered_questions"])
        self.assertEqual(res["total_questions"], 3)
        self.assertEqual(res["attempted_questions"], 2)
        self.assertEqual(res["unanswered_questions"], 1)

        # attempted_questions = evaluated_questions + pending_review_questions
        self.assertEqual(res["attempted_questions"], res["evaluated_questions"] + res["pending_review_questions"])
        self.assertEqual(res["evaluated_questions"], 1)
        self.assertEqual(res["pending_review_questions"], 1)

        # Marks and totals
        self.assertEqual(res["total_marks"], 20.0)
        self.assertEqual(res["obtained_marks"], 10.0)
        self.assertEqual(res["percentage"], 50.0)
        self.assertEqual(res["status"], "Pass")


if __name__ == "__main__":
    unittest.main()
