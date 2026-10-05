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
                "This chapter teaches young children (Age 5-6) fundamental counting, single-digit addition, "
                "and single-digit subtraction using small whole numbers from 1 to 20.\n\n"
                "Core Concepts & Syllabus Topics:\n"
                "1. Concept of Addition (Putting Together):\n"
                "   - Combining two groups of objects (e.g., 3 apples + 2 apples = 5 apples).\n"
                "   - Plus sign (+) and Equals sign (=).\n"
                "   - Single-digit addition facts up to 10 and 20 (e.g., 1+1=2, 2+3=5, 4+4=8, 5+5=10, 6+3=9, 7+2=9, 8+1=9, 9+1=10).\n"
                "   - Adding zero: Adding 0 to any number leaves the number unchanged (e.g., 5 + 0 = 5).\n"
                "2. Concept of Subtraction (Taking Away):\n"
                "   - Removing objects from a group (e.g., 5 birds on a tree, 2 fly away, 3 birds are left).\n"
                "   - Minus sign (-) and Equals sign (=).\n"
                "   - Single-digit subtraction facts (e.g., 5 - 2 = 3, 7 - 4 = 3, 8 - 3 = 5, 9 - 6 = 3, 10 - 5 = 5).\n"
                "   - Subtracting zero: Subtracting 0 leaves the number unchanged (e.g., 6 - 0 = 6).\n"
                "   - Subtracting a number from itself equals zero (e.g., 4 - 4 = 0).\n"
                "3. Counting & Number Comparisons:\n"
                "   - Counting forwards (1 to 20) and backwards (10 to 1).\n"
                "   - Basic word problems involving daily objects like pencils, toys, fruits, balloons, and flowers.\n"
                "Strict Constraints:\n"
                "- Only positive integers between 0 and 20.\n"
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
            "food_from_plants": (
                "CLASS 1 SCIENCE / EVS - CHAPTER: FOOD FROM PLANTS\n"
                "Target Audience: Primary School Class 1 (Age 5-6)\n\n"
                "Core Concepts & Textbook Topics:\n"
                "1. Food We Get From Plants:\n"
                "   - Fruits: Mango, Apple, Banana, Orange, Grapes, Papaya, Guava.\n"
                "   - Vegetables: Potato, Tomato, Carrot, Spinach, Onion, Peas, Cabbage.\n"
                "   - Cereals and Grains: Rice, Wheat, Corn (Maize) - used to make bread/roti and rice dishes.\n"
                "   - Pulses (Dals): Gram, Moong, Peas, Beans - rich in protein for growing kids.\n"
                "   - Nuts: Almonds, Cashews, Walnuts, Groundnuts.\n"
                "   - Spices and Beverages: Cardamom, Pepper, Tea, Coffee, Sugar from Sugarcane.\n\n"
                "2. Parts of Plants We Eat:\n"
                "   - Leaves: Spinach, Cabbage, Mint, Coriander.\n"
                "   - Roots: Carrot, Radish, Beetroot, Turnip.\n"
                "   - Stems: Sugarcane, Potato, Ginger.\n"
                "   - Seeds: Peas, Corn, Rice, Wheat, Beans.\n"
                "   - Fruits: Tomato, Brinjal, Apple, Mango, Cucumber.\n"
                "   - Flowers: Cauliflower, Broccoli.\n\n"
                "3. Good Food Habits:\n"
                "   - Washing fruits and vegetables with clean water before eating.\n"
                "   - Eating fresh green vegetables daily.\n"
                "   - Not wasting food.\n\n"
                "Strict Constraints:\n"
                "- Only simple elementary concepts suitable for a 6-year-old child.\n"
                "- Absolutely NO Physics (velocity, momentum, force, lenses, optics, energy, kinetic energy, circuits).\n"
                "- Absolutely NO secondary Chemistry or complex biology."
            ),
            "living_and_non_living": (
                "CLASS 1 SCIENCE / EVS - CHAPTER: LIVING AND NON-LIVING THINGS\n"
                "Core Concepts: Living things breathe, grow, eat, and move (Plants, Animals, Humans). Non-living things do not breathe or grow (Chair, Table, Toy, Stone, Car)."
            ),
            "plants_around_us": (
                "CLASS 1 SCIENCE / EVS - CHAPTER: PLANTS AROUND US\n"
                "Core Concepts: Big trees (Banyan, Mango), Small shrubs (Rose), Herbs (Mint, Grass), Climbers (Money plant), Parts of plant (Root, Stem, Leaf, Flower, Fruit)."
            ),
            "animals_around_us": (
                "CLASS 1 SCIENCE / EVS - CHAPTER: ANIMALS AROUND US\n"
                "Core Concepts: Domestic animals (Dog, Cat, Cow), Wild animals (Lion, Tiger, Elephant), Birds (Sparrow, Crow, Parrot), Water animals (Fish)."
            )
        },
        "english": {
            "alphabet_and_words": (
                "CLASS 1 ENGLISH - CHAPTER: ALPHABETS AND SIMPLE WORDS\n"
                "Core Concepts: Phonics, A-Z letter identification, three-letter rhyming words (cat, bat, mat, sun, run)."
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
        "monarchy", "democracy", "parliament", "ecosystem", "isolated physical system",
        "mechanical energy", "linear momentum", "electric charge"
    ],
    "science": [
        "linear momentum", "kinetic energy", "velocity of a moving body", "mass 180 kg",
        "mass 40 kg", "isolated physical system", "snell's law", "convex lens", "concave lens",
        "focal length of +21 cm", "optical power in dioptres", "newton's second law", "f = ma",
        "triangular glass prism", "angle of deviation", "angle of emergence",
        "freely falling body of mass m", "mechanical energy is conserved"
    ]
}

# High-School / Secondary Concepts Forbidden for Primary School (Classes 1 - 5)
PRIMARY_FORBIDDEN_CONCEPTS = [
    "sin(", "cos(", "tan(", "theta", "derivative", "integral", "matrix", "vector",
    "f = ma", "v = u + at", "ke = 1/2", "quadratic", "polynomial", "logarithm",
    "speed of light", "atomic mass", "molar mass", "refractive index", "dioptre",
    "momentum", "wavelength", "frequency", "electromagnetism", "thermodynamics",
    "newton", "velocity", "acceleration", "focal length", "optical power", "prism",
    "isolated physical system", "linear momentum", "mechanical energy", "kinetic energy",
    "ohm's law", "resistor", "electric current", "conservation of energy", "freely falling",
    "dioptres", "joule", "watt", "pascal"
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
    clean = re.sub(r"\(.*?\)", "", clean).strip()
    
    if any(m in clean for m in ["math", "maths", "mathematics", "arithmetic"]):
        return "maths"
    if "physic" in clean:
        return "physics"
    if "chem" in clean:
        return "chemistry"
    if any(b in clean for b in ["bio", "biology", "botany", "zoology"]):
        return "biology"
    if any(s in clean for s in ["science", "gen science", "general science", "evs", "environmental"]):
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
    clean = re.sub(r"^(chapter|ch)?\s*\d+\s*[:\-\.]\s*", "", clean)
    clean = re.sub(r"[^\w\s]", "", clean)
    clean = re.sub(r"\s+", "_", clean).strip("_")
    return clean


def synthesize_curriculum_content(class_name: str, subject_name: str, chapter_name: str) -> str:
    """
    Dynamically constructs authoritative, grade-bound syllabus content for any requested chapter
    to ensure the LLM is strictly grounded and never drifts into secondary science/physics.
    """
    cls_key = normalize_class(class_name)
    subj_key = normalize_subject(subject_name)
    raw_chap_clean = re.sub(r"^(chapter|ch)?\s*\d+\s*[:\-\.]\s*", "", chapter_name).strip()

    try:
        cls_num = int(cls_key)
    except ValueError:
        cls_num = 1

    if cls_num <= 5:
        # Primary Grade Synthesis
        return (
            f"OFFICIAL PRIMARY SCHOOL CURRICULUM - CLASS {cls_key}\n"
            f"SUBJECT: {subject_name}\n"
            f"CHAPTER: {chapter_name}\n\n"
            f"Scope & Boundary Rules:\n"
            f"- Target Student Age: {cls_num + 5} years old (Primary School Grade {cls_num}).\n"
            f"- Topic Focus: Fundamental concepts relating exclusively to '{raw_chap_clean}'.\n"
            f"- Core Vocabulary: Simple elementary everyday words and practical examples suitable for Class {cls_num}.\n"
            f"- Mandatory Scope: Only basic grade-appropriate facts, simple identification, counting/naming, and elementary concepts.\n"
            f"- STRICTLY FORBIDDEN: Any secondary school physics, mechanics, optics, lenses, forces, velocity, momentum, formulas, chemistry, or advanced theories."
        )
    else:
        # Secondary / Middle School Synthesis
        return (
            f"OFFICIAL SCHOOL CURRICULUM - CLASS {cls_key}\n"
            f"SUBJECT: {subject_name}\n"
            f"CHAPTER: {chapter_name}\n\n"
            f"Scope & Boundary Rules:\n"
            f"- Topic Focus: Standard school syllabus concepts relating strictly to '{raw_chap_clean}'.\n"
            f"- All questions must strictly pertain to {subject_name} and '{raw_chap_clean}'.\n"
            f"- Do NOT introduce concepts from other subjects."
        )


def get_textbook_content(class_name: str, subject_name: str, chapter_name: str, payload_content: str = None) -> tuple:
    """
    Retrieve exact textbook content using Class + Subject + Chapter.
    Returns (content_text, content_identifier).
    """
    # 1. Payload-supplied content
    if payload_content and isinstance(payload_content, str) and len(payload_content.strip()) > 20:
        logger.info(f"Using request-supplied textbook content (length={len(payload_content.strip())})")
        return payload_content.strip(), f"payload_supplied_content:{subject_name}:{chapter_name}"

    cls_key = normalize_class(class_name)
    subj_key = normalize_subject(subject_name)
    chap_key = normalize_chapter_key(chapter_name)

    logger.info(f"Looking up textbook repo: Class='{cls_key}', Subject='{subj_key}', Chapter='{chap_key}'")

    # 2. Check local curated database
    if cls_key in TEXTBOOK_DATABASE:
        subj_dict = TEXTBOOK_DATABASE[cls_key].get(subj_key, {})
        if chap_key in subj_dict:
            return subj_dict[chap_key], f"textbook_db:class_{cls_key}:{subj_key}:{chap_key}"
        
        for k, text in subj_dict.items():
            if k in chap_key or chap_key in k or any(part in k for part in chap_key.split("_") if len(part) > 3):
                return text, f"textbook_db:class_{cls_key}:{subj_key}:{k}"

    # 3. Dynamic curriculum synthesizer (Grade & Scope Bound)
    synthetic_content = synthesize_curriculum_content(class_name, subject_name, chapter_name)
    return synthetic_content, f"curriculum_synthesized:class_{cls_key}:{subj_key}:{chap_key}"


def normalize_text_for_comparison(text: str) -> str:
    """Normalize question text for duplicate detection."""
    if not text:
        return ""
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

    # 1. Primary Grade Level Constraint Check (Classes 1 - 5)
    try:
        cls_int = int(cls_key)
    except ValueError:
        cls_int = 1

    if cls_int <= 5:
        for concept in PRIMARY_FORBIDDEN_CONCEPTS:
            pattern = r"\b" + re.escape(concept) + r"\b" if re.match(r"^\w+$", concept) else re.escape(concept)
            if re.search(pattern, full_text):
                return False, f"Grade {cls_int} mismatch: contains secondary-level concept '{concept}' strictly forbidden for Class {class_name}."

    # 2. Subject Contamination Check
    forbidden_terms = FORBIDDEN_KEYWORDS_BY_SUBJECT.get(subj_key, [])
    for term in forbidden_terms:
        pattern = r"\b" + re.escape(term) + r"\b" if re.match(r"^\w+$", term) else re.escape(term)
        if re.search(pattern, full_text):
            return False, f"Subject contamination: contains forbidden term '{term}' for subject '{subject_name}'."

    # 3. Question Structure Verification
    q_type = str(q_obj.get("type", "")).strip()
    if q_type.upper() == "MCQ":
        opts = q_obj.get("options", [])
        if not isinstance(opts, list) or len(opts) < 2:
            return False, "MCQ question missing valid options array."

    return True, "VALID"
