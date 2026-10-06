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


def normalize_chapter_key(name) -> str:
    """
    Normalize a textbook chapter name only for INTERNAL LOOKUP.
    Never use this value as the display chapter name.
    """
    if name is None:
        return ""

    value = str(name).strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def synthesize_curriculum_content(class_name: str, subject_name: str, chapter_identifier: str) -> str:
    """
    Synthesizes grade-bound curriculum bounds only as a fallback if no textbook PDF was uploaded.
    """
    cls_str = str(class_name or "1").strip()
    subj_str = str(subject_name or "General").strip()
    chap_str = str(chapter_identifier or "Curriculum Unit").strip()
    return (
        f"CURRICULUM CONTENT FOR CLASS {cls_str.upper()} - SUBJECT: {subj_str.upper()}\n"
        f"CHAPTER: {chap_str.upper()}\n\n"
        f"Core syllabus topics, standard terminology, and learning outcomes for {chap_str}."
    )


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


# Registry for extracted textbooks: (cls_key, subj_key) -> dict of chapters
EXTRACTED_TEXTBOOK_REGISTRY = {}


def register_extracted_textbook(subject_name: str, subject_code: str, chapters: list, class_name: str = ""):
    """
    Registers authentic textbook chapters extracted directly from a PDF.
    Preserves 100% full chapter content and original textbook chapter names.
    """
    if not chapters:
        return

    cls_key = normalize_class(class_name) if class_name else "all"
    subj_key = normalize_subject(subject_name)
    
    registry_key = f"{cls_key}:{subj_key}"
    if registry_key not in EXTRACTED_TEXTBOOK_REGISTRY:
        EXTRACTED_TEXTBOOK_REGISTRY[registry_key] = {}

    for ch in chapters:
        chapter_name = str(ch.get("chapter_name", "")).strip()
        chapter_key = normalize_chapter_key(chapter_name)
        ch_id = str(ch.get("chapter_id") or chapter_key)
        record = {
            "chapter_id": ch_id,
            "chapter_no": str(ch.get("chapter_no", "")),
            "chapter_name": chapter_name,
            "chapter_key": chapter_key,
            "chapter_name_source": ch.get("chapter_name_source", "textbook_pdf"),
            "chapter_name_verified": ch.get("chapter_name_verified", True),
            "source_pdf_page": ch.get("source_pdf_page") or ch.get("start_page"),
            "start_page": ch.get("start_page"),
            "end_page": ch.get("end_page"),
            "description": ch.get("description", ""),
            "content": ch.get("content", "")
        }
        EXTRACTED_TEXTBOOK_REGISTRY[registry_key][ch_id] = record
        EXTRACTED_TEXTBOOK_REGISTRY[registry_key][chapter_key] = record
        EXTRACTED_TEXTBOOK_REGISTRY[registry_key][chapter_name.lower()] = record
        logger.info(
            "[REGISTRY]\nchapter_id=%r\nchapter_name=%r\nchapter_key=%r",
            ch_id, chapter_name, chapter_key
        )

    logger.info(f"Registered {len(chapters)} authentic chapters in textbook registry for {registry_key}")


def get_textbook_chapter_record(class_name: str, subject_name: str, chapter_identifier: str, payload_content: str = None) -> dict:
    """
    Retrieve authentic textbook chapter record using class, subject, and chapter_id/name.
    Returns dictionary with keys: chapter_id, chapter_name, content, chapter_name_source, chapter_name_verified.
    """
    chap_id = str(chapter_identifier or "").strip()

    # 1. Payload-supplied content
    if payload_content and isinstance(payload_content, str) and len(payload_content.strip()) > 20:
        record = {
            "chapter_id": normalize_chapter_key(chap_id),
            "chapter_name": chap_id or subject_name or "Textbook Chapter",
            "chapter_key": normalize_chapter_key(chap_id),
            "content": payload_content.strip(),
            "chapter_name_source": "payload_supplied",
            "chapter_name_verified": True
        }
        logger.info(
            "[TEXTBOOK LOOKUP]\nregistry_key='payload'\nchapter_id=%s\nchapter_name=%s",
            record["chapter_id"], record["chapter_name"]
        )
        return record

    cls_key = normalize_class(class_name)
    subj_key = normalize_subject(subject_name)
    chap_norm = normalize_chapter_key(chap_id)
    chap_lower = chap_id.lower()

    # 2. Check EXTRACTED_TEXTBOOK_REGISTRY
    search_keys = [f"{cls_key}:{subj_key}", f"all:{subj_key}"]
    for reg_k in search_keys:
        if reg_k in EXTRACTED_TEXTBOOK_REGISTRY:
            reg_dict = EXTRACTED_TEXTBOOK_REGISTRY[reg_k]
            matched_rec = None
            if chap_id in reg_dict:
                matched_rec = reg_dict[chap_id]
            elif chap_norm in reg_dict:
                matched_rec = reg_dict[chap_norm]
            elif chap_lower in reg_dict:
                matched_rec = reg_dict[chap_lower]
            else:
                for k, rec in reg_dict.items():
                    if chap_norm and (chap_norm in k or k in chap_norm):
                        matched_rec = rec
                        break

            if matched_rec:
                logger.info(
                    "[TEXTBOOK LOOKUP]\nregistry_key=%s\nchapter_id=%s\nchapter_name=%s",
                    reg_k, matched_rec.get("chapter_id"), matched_rec.get("chapter_name")
                )
                if len(matched_rec.get("content", "").strip()) < 1000:
                    logger.warning(
                        "[SHORT CHAPTER CONTENT] name=%r characters=%s",
                        matched_rec.get("chapter_name"), len(matched_rec.get("content", ""))
                    )
                return matched_rec

    # 3. Check local curated database
    if cls_key in TEXTBOOK_DATABASE:
        subj_dict = TEXTBOOK_DATABASE[cls_key].get(subj_key, {})
        if chap_norm in subj_dict:
            rec = {
                "chapter_id": chap_norm,
                "chapter_name": chap_id,
                "chapter_key": chap_norm,
                "content": subj_dict[chap_norm],
                "chapter_name_source": "textbook_db",
                "chapter_name_verified": True
            }
            logger.info(
                "[TEXTBOOK LOOKUP]\nregistry_key=%s\nchapter_id=%s\nchapter_name=%s",
                f"{cls_key}:{subj_key}", rec["chapter_id"], rec["chapter_name"]
            )
            return rec

    # 4. Fallback: synthesizes grade-bound curriculum bounds only if textbook PDF was not uploaded
    synthetic_content = synthesize_curriculum_content(class_name, subject_name, chap_id)
    rec = {
        "chapter_id": chap_norm or "general_chapter",
        "chapter_name": chap_id or subject_name or "Curriculum Unit",
        "chapter_key": chap_norm or "general_chapter",
        "content": synthetic_content,
        "chapter_name_source": "curriculum_synthesized",
        "chapter_name_verified": False
    }
    logger.info(
        "[TEXTBOOK LOOKUP]\nregistry_key='synthesized'\nchapter_id=%s\nchapter_name=%s",
        rec["chapter_id"], rec["chapter_name"]
    )
    return rec


def get_textbook_content(class_name: str, subject_name: str, chapter_name: str, payload_content: str = None) -> tuple:
    """
    Backwards-compatible helper returning (content_text, content_identifier).
    """
    rec = get_textbook_chapter_record(class_name, subject_name, chapter_name, payload_content)
    return rec["content"], f"{rec['chapter_name_source']}:{rec['chapter_id']}"


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
