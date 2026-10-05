"""
Edusoft Textbook & Curriculum Content Repository and Validation Engine.
Provides textbook content retrieval and post-generation multi-stage validation.
"""
import re
import string
import logging

logger = logging.getLogger("edusoft_service.textbook_repo")

# Curated Textbook Content Knowledge Base
# Schema: [class_key][subject_key][chapter_key] = Content Text
TEXTBOOK_DATABASE = {
    "1": {
        "maths": {
            "arithmetic_operations": (
                "CLASS 1 MATHEMATICS - CHAPTER: ARITHMETIC OPERATIONS (ADDITION & SUBTRACTION)\n"
                "Overview:\n"
                "This chapter teaches young children (Age 5-6) the fundamental concepts of counting, single-digit addition, "
                "and single-digit subtraction using small whole numbers from 1 to 20.\n\n"
                "Core Concepts & Syllabus Topics:\n"
                "1. Concept of Addition (Putting Together):\n"
                "   - Combining two groups of objects (e.g., 3 apples + 2 apples = 5 apples).\n"
                "   - Plus sign (+) and Equals sign (=).\n"
                "   - Single-digit addition facts up to 10 and 20 (e.g., 1+1=2, 2+3=5, 4+4=8, 5+5=10, 6+3=9, 7+2=9, 8+1=9, 9+1=10).\n"
                "   - Adding zero: Adding 0 to any number leaves the number unchanged (e.g., 5 + 0 = 5).\n"
                "   - One more than a number (e.g., 1 more than 6 is 7).\n\n"
                "2. Concept of Subtraction (Taking Away):\n"
                "   - Removing objects from a group (e.g., 5 birds on a tree, 2 fly away, 3 birds are left).\n"
                "   - Minus sign (-) and Equals sign (=).\n"
                "   - Single-digit subtraction facts (e.g., 5 - 2 = 3, 7 - 4 = 3, 8 - 3 = 5, 9 - 6 = 3, 10 - 5 = 5).\n"
                "   - Subtracting zero: Subtracting 0 leaves the number unchanged (e.g., 6 - 0 = 6).\n"
                "   - Subtracting a number from itself equals zero (e.g., 4 - 4 = 0).\n"
                "   - One less than a number (e.g., 1 less than 8 is 7).\n\n"
                "3. Counting & Number Comparisons:\n"
                "   - Counting forwards (1, 2, 3, 4, 5, 6, 7, 8, 9, 10... 20).\n"
                "   - Counting backwards (10, 9, 8, 7, 6, 5, 4, 3, 2, 1).\n"
                "   - Basic word problems involving daily objects like pencils, toys, fruits, balloons, and flowers.\n"
                "   - Finding missing numbers in simple operations: e.g., 3 + ___ = 5 or 6 - ___ = 4.\n\n"
                "Strict Constraints:\n"
                "- Only positive integers between 0 and 20.\n"
                "- No negative numbers, decimals, fractions, algebra, variables (x, y), multiplication, or division.\n"
                "- Absolutely no secondary science, physics, velocity, momentum, forces, or lenses."
            ),
            "shapes_and_space": (
                "CLASS 1 MATHEMATICS - CHAPTER: SHAPES AND SPACE\n"
                "Core Concepts: Inside-Outside, Bigger-Smaller, Top-Bottom, Nearer-Farther, Shapes (Circle, Square, Triangle, Rectangle)."
            ),
            "numbers_from_one_to_nine": (
                "CLASS 1 MATHEMATICS - CHAPTER: NUMBERS FROM ONE TO NINE\n"
                "Core Concepts: Counting objects 1 to 9, more or less, number names (one, two, three... nine), matching numbers."
            )
        },
        "science": {
            "living_and_non_living": (
                "CLASS 1 SCIENCE / EVS - CHAPTER: LIVING AND NON-LIVING THINGS\n"
                "Core Concepts: Plants, animals, human body parts, clean habits, food and water."
            )
        },
        "english": {
            "alphabet_and_words": (
                "CLASS 1 ENGLISH - CHAPTER: ALPHABETS AND SIMPLE WORDS\n"
                "Core Concepts: Phonics, A-Z letter identification, three-letter rhyming words (cat, bat, mat, sun, run)."
            )
        }
    },
    "2": {
        "maths": {
            "counting_in_groups": (
                "CLASS 2 MATHEMATICS - CHAPTER: COUNTING IN GROUPS\n"
                "Core Concepts: 2-digit numbers up to 100, place value (tens and ones), addition and subtraction of 2-digit numbers."
            )
        }
    },
    "10": {
        "physics": {
            "light_reflection_and_refraction": (
                "CLASS 10 PHYSICS - CHAPTER: LIGHT - REFLECTION AND REFRACTION\n"
                "Core Concepts: Spherical mirrors, mirror formula, refraction, Snell's law, convex and concave lenses, lens formula, optical power."
            ),
            "force_and_laws_of_motion": (
                "CLASS 10 PHYSICS - CHAPTER: FORCE AND LAWS OF MOTION\n"
                "Core Concepts: Newton's laws of motion, inertia, momentum, conservation of momentum, F=ma."
            )
        },
        "maths": {
            "arithmetic_progressions": (
                "CLASS 10 MATHEMATICS - CHAPTER: ARITHMETIC PROGRESSIONS\n"
                "Core Concepts: nth term of an AP (an = a + (n-1)d), sum of first n terms (Sn = n/2[2a + (n-1)d]), common difference."
            )
        }
    }
}

# Cross-Subject Forbidden Keywords Dictionary
FORBIDDEN_KEYWORDS_BY_SUBJECT = {
    "maths": [
        "newton", "kinetic energy", "potential energy", "velocity", "momentum", "acceleration",
        "lens", "lenses", "convex lens", "concave lens", "focal length", "dioptre", "diopter",
        "snell's law", "refraction", "reflection", "optical power", "prism", "light ray",
        "electric current", "resistance", "ohm's law", "voltage", "joule", "watt", "pascal",
        "gravitational", "force of friction", "photosynthesis", "cell membrane", "dna",
        "chromosome", "chemical reaction", "acid and base", "dynasty", "mughal", "constitution",
        "monarchy", "democracy", "parliament", "ecosystem"
    ],
    "physics": [
        "photosynthesis", "chlorophyll", "cell division", "dna", "mitosis", "bacteria",
        "dynasty", "mughal empire", "indus valley", "grammar", "adjective", "verb",
        "shakespeare", "sonnet"
    ],
    "chemistry": [
        "convex lens", "snell's law", "focal length", "velocity of light", "newton's second law",
        "dynasty", "sonnet", "rhyme scheme"
    ],
    "biology": [
        "snell's law", "lens formula", "dioptre", "focal length", "newton's second law",
        "ohm's law", "resistor in series", "algebraic expression"
    ]
}

# High-School / Secondary Concepts Forbidden for Primary School (Classes 1 - 5)
PRIMARY_FORBIDDEN_CONCEPTS = [
    "sin(", "cos(", "tan(", "theta", "derivative", "integral", "matrix", "vector",
    "f = ma", "v = u + at", "ke = 1/2", "quadratic", "polynomial", "logarithm",
    "speed of light", "atomic mass", "molar mass", "refractive index", "dioptre",
    "momentum", "wavelength", "frequency", "electromagnetism", "thermodynamics"
]


def normalize_class(class_name: str) -> str:
    """Normalize class name to clean standard digit string (e.g. 'Class 1' -> '1')."""
    if not class_name:
        return "1"
    clean = str(class_name).lower().strip()
    clean = re.sub(r"^(class|grade|std|standard)\s*", "", clean)
    clean = clean.strip()
    roman_map = {
        "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
        "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10",
        "xi": "11", "xii": "12", "first": "1", "second": "2",
        "third": "3", "fourth": "4", "fifth": "5"
    }
    return roman_map.get(clean, clean)


def normalize_subject(subject_name: str) -> str:
    """Normalize subject name to canonical key (maths, physics, chemistry, etc.)."""
    if not subject_name:
        return "general"
    clean = str(subject_name).lower().strip()
    # Remove subject codes like '(MTH1)', '(PHY101)', etc.
    clean = re.sub(r"\(.*?\)", "", clean).strip()
    
    if any(m in clean for m in ["math", "maths", "mathematics", "arithmetic"]):
        return "maths"
    if "physic" in clean:
        return "physics"
    if "chem" in clean:
        return "chemistry"
    if any(b in clean for b in ["bio", "biology", "botany", "zoology"]):
        return "biology"
    if any(s in clean for s in ["science", "gen science", "general science", "evs"]):
        return "science"
    if any(soc in clean for soc in ["social", "sst", "history", "geography", "civics"]):
        return "social_science"
    if "english" in clean:
        return "english"
    if "hindi" in clean:
        return "hindi"
    if "malayalam" in clean:
        return "malayalam"
    if any(c in clean for c in ["computer", "cs", "it", "informatics"]):
        return "computer_science"
    
    return clean.replace(" ", "_")


def normalize_chapter_key(chapter_name: str) -> str:
    """Normalize chapter string to lookup key (e.g. '1: Arithmetic Operations' -> 'arithmetic_operations')."""
    if not chapter_name:
        return ""
    clean = str(chapter_name).lower().strip()
    # Strip leading chapter numbers e.g. "1:", "Chapter 1:", "Ch 1 - "
    clean = re.sub(r"^(chapter|ch)?\s*\d+\s*[:\-\.]\s*", "", clean)
    clean = re.sub(r"[^\w\s]", "", clean)
    clean = re.sub(r"\s+", "_", clean).strip("_")
    return clean


def get_textbook_content(class_name: str, subject_name: str, chapter_name: str, payload_content: str = None) -> tuple:
    """
    Retrieve exact textbook content using Class + Subject + Chapter.
    Returns (content_text, content_identifier) or (None, None).
    """
    # 1. If explicit content was provided in request payload, use it as primary
    if payload_content and isinstance(payload_content, str) and len(payload_content.strip()) > 20:
        logger.info(f"Using request-supplied textbook content (length={len(payload_content.strip())})")
        return payload_content.strip(), f"payload_supplied_content:{subject_name}:{chapter_name}"

    cls_key = normalize_class(class_name)
    subj_key = normalize_subject(subject_name)
    chap_key = normalize_chapter_key(chapter_name)

    logger.info(f"Looking up textbook repo: Class='{cls_key}', Subject='{subj_key}', Chapter='{chap_key}'")

    # 2. Check local textbook database
    if cls_key in TEXTBOOK_DATABASE:
        subj_dict = TEXTBOOK_DATABASE[cls_key].get(subj_key, {})
        
        # Direct chapter key match
        if chap_key in subj_dict:
            return subj_dict[chap_key], f"textbook_db:class_{cls_key}:{subj_key}:{chap_key}"
        
        # Fuzzy / partial chapter match
        for k, text in subj_dict.items():
            if k in chap_key or chap_key in k or any(part in k for part in chap_key.split("_") if len(part) > 3):
                return text, f"textbook_db:class_{cls_key}:{subj_key}:{k}"

    # 3. Dynamic generic fallback for valid class-subject curriculum standards
    if cls_key == "1" and subj_key == "maths":
        # Any Class 1 Maths arithmetic/numbers chapter
        fallback_text = TEXTBOOK_DATABASE["1"]["maths"]["arithmetic_operations"]
        return fallback_text, f"textbook_db:class_1:maths:standard_arithmetic_syllabus"

    return None, None


def normalize_text_for_comparison(text: str) -> str:
    """Normalize question text for duplicate detection."""
    if not text:
        return ""
    # Lowercase, remove punctuation and extra spaces
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def validate_question(q_obj: dict, class_name: str, subject_name: str, chapter_name: str, textbook_content: str) -> tuple:
    """
    Strict validation of a single generated question against Class, Subject, Chapter, and Textbook Content.
    Returns (is_valid: bool, reason: str).
    """
    if not isinstance(q_obj, dict):
        return False, "Question is not a valid JSON dictionary."

    q_text = str(q_obj.get("question", "")).strip()
    if not q_text:
        return False, "Question text is empty."

    subj_key = normalize_subject(subject_name)
    cls_key = normalize_class(class_name)
    q_lower = q_text.lower()
    options_lower = " ".join([str(opt).lower() for opt in q_obj.get("options", [])])
    answer_lower = str(q_obj.get("correct_answer", "")).lower()
    full_text = f"{q_lower} {options_lower} {answer_lower}"

    # 1. Subject Contamination Check
    forbidden_terms = FORBIDDEN_KEYWORDS_BY_SUBJECT.get(subj_key, [])
    for term in forbidden_terms:
        # Check whole words / phrases
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, full_text):
            return False, f"Subject contamination detected: contains forbidden term '{term}' for subject '{subject_name}'."

    # 2. Primary Grade Level Constraint Check (Classes 1 - 5)
    try:
        cls_int = int(cls_key)
    except ValueError:
        cls_int = 1

    if cls_int <= 5:
        for concept in PRIMARY_FORBIDDEN_CONCEPTS:
            if concept in full_text:
                return False, f"Grade mismatch: contains secondary-level concept '{concept}' forbidden for Class {class_name}."

    # 3. Class 1 Specific Verification
    if cls_int == 1:
        if subj_key == "maths":
            # Check for high-school math or physics
            if any(term in full_text for term in ["triangle prism", "refractive", "focal", "joule", "acceleration", "linear momentum", "conservation of energy"]):
                return False, f"Class 1 Maths violation: advanced physics/mechanics concepts detected."

    # 4. Question Structure Verification
    q_type = str(q_obj.get("type", "")).strip()
    if q_type.upper() == "MCQ":
        opts = q_obj.get("options", [])
        if not isinstance(opts, list) or len(opts) < 2:
            return False, "MCQ question missing valid options array."

    return True, "VALID"
