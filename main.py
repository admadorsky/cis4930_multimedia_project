from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import spacy
from typing import Dict, List, Tuple, Optional
import statistics

app = Flask(__name__)
CORS(app)

# Load NLP models once per process so requests stay fast.
sentiment_model = pipeline("sentiment-analysis")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

CATEGORY_RULES = [
    {"id": "emotion", "label": "Emotion", "rule": "Emotion First Rule", "max_points": 2},
    {"id": "character-depth", "label": "Character Depth", "rule": "Character Depth Rule", "max_points": 2},
    {"id": "human-experience", "label": "Human Experience", "rule": "Human Experience Rule", "max_points": 2},
    {"id": "symbolism", "label": "Symbolism & Motifs", "rule": "Symbolism Rule", "max_points": 2},
    {"id": "audience-fit", "label": "Audience Fit", "rule": "Contextual Adaptation Rule", "max_points": 2},
    {"id": "conflict", "label": "Conflict & Resolution", "rule": "Conflict & Resolution Rule", "max_points": 2},
    {"id": "sensory", "label": "Sensory Immersion", "rule": "Sensory Immersion Rule", "max_points": 2},
    {"id": "authenticity", "label": "Authenticity / Originality", "rule": "Authenticity Rule", "max_points": 2},
    {"id": "rhythm", "label": "Dynamic Rhythm / Pacing", "rule": "Dynamic Rhythm Rule", "max_points": 2},
    {"id": "moral", "label": "Moral Resonance / Thematic Impact", "rule": "Moral Resonance Rule", "max_points": 2},
]

METADATA_FIELDS = ["genre", "length", "mood", "audience", "setting"]

AUDIENCE_BASELINES = {
    None: {"optimal": (10, 22), "acceptable": (7, 28)},
    "kids": {"optimal": (6, 14), "acceptable": (4, 18)},
    "ya": {"optimal": (8, 18), "acceptable": (6, 22)},
    "adult": {"optimal": (13, 28), "acceptable": (9, 32)},
    "general": {"optimal": (10, 22), "acceptable": (7, 28)},
}

LENGTH_ADJUSTMENTS = {
    "short film": (-2, -1),
    "feature": (3, 3),
    "pilot": (2, 2),
    "episode": (1, 2),
}

SENSORY_GENRES = {"fantasy", "sci-fi", "thriller"}
SENSORY_SETTING_HINTS = ["forest", "ocean", "space", "city", "desert", "mountain", "rain", "storm", "battlefield"]
REFLECTIVE_MOOD_CUES = ["reflect", "healing", "coming-of-age", "lesson", "meditative", "introspective"]


def _add_breakdown_entry(breakdown: Dict[str, Dict[str, str]], rule: str, score: int, max_points: int, message: str):
    breakdown[rule] = {
        "score": score,
        "max_points": max_points,
        "message": message,
    }


def _normalize_field(value: Optional[str]) -> Optional[str]:
    if value is None or not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "n/a":
        return None
    return cleaned


def _normalize_metadata(raw_metadata: Optional[Dict[str, object]]) -> Dict[str, Optional[str]]:
    raw_metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    normalized: Dict[str, Optional[str]] = {}
    for field in METADATA_FIELDS:
        field_value = _normalize_field(raw_metadata.get(field))  # type: ignore[arg-type]
        normalized[field] = field_value
        normalized[f"{field}_norm"] = field_value.lower() if field_value else None
    return normalized


def _audience_length_ranges(metadata: Dict[str, Optional[str]]):
    audience_key = metadata.get("audience_norm")
    baseline = AUDIENCE_BASELINES.get(audience_key, AUDIENCE_BASELINES[None])
    delta_min, delta_max = LENGTH_ADJUSTMENTS.get(metadata.get("length_norm"), (0, 0))
    optimal_min = baseline["optimal"][0] + delta_min
    optimal_max = baseline["optimal"][1] + delta_max
    if optimal_min > optimal_max:
        optimal_min, optimal_max = optimal_max - 1, optimal_max

    acceptable_min = baseline["acceptable"][0] + delta_min
    acceptable_max = baseline["acceptable"][1] + delta_max
    if acceptable_min > acceptable_max:
        acceptable_min, acceptable_max = acceptable_max - 1, acceptable_max

    optimal = (max(2, optimal_min), max(optimal_min + 1, optimal_max))
    acceptable = (max(2, acceptable_min), max(acceptable_min + 1, acceptable_max))
    return optimal, acceptable


def _describe_format(metadata: Dict[str, Optional[str]]) -> str:
    return metadata.get("length") or metadata.get("genre") or "this format"


def _sensory_expectation(metadata: Dict[str, Optional[str]]) -> int:
    expectation = 4
    if metadata.get("genre_norm") in SENSORY_GENRES:
        expectation += 1
    setting_norm = metadata.get("setting_norm") or ""
    if any(hint in setting_norm for hint in SENSORY_SETTING_HINTS):
        expectation += 1
    return expectation


def _expects_full_conflict_arc(metadata: Dict[str, Optional[str]]) -> bool:
    return metadata.get("length_norm") in {"feature", "pilot", "episode"} or metadata.get("genre_norm") in {"thriller", "sci-fi"}


def _expects_reflective_ending(metadata: Dict[str, Optional[str]]) -> bool:
    if metadata.get("genre_norm") in {"drama"}:
        return True
    mood_norm = metadata.get("mood_norm") or ""
    return any(cue in mood_norm for cue in REFLECTIVE_MOOD_CUES)


def _score_story(text: str, metadata: Optional[Dict[str, object]] = None) -> Tuple[int, Dict[str, Dict[str, str]]]:
    metadata = _normalize_metadata(metadata)
    breakdown: Dict[str, Dict[str, str]] = {}
    score = 0

    doc = nlp(text)
    lower_text = text.lower()
    sentence_spans = [sent for sent in doc.sents if sent.text.strip()]
    sentences = [sent.text.strip() for sent in sentence_spans]

    if not sentences:
        _add_breakdown_entry(
            breakdown,
            "Input Validation",
            0,
            0,
            "Please provide a story with at least one complete sentence.",
        )
        return score, breakdown

    # --------------------------
    # 1. Emotion First Rule
    dominant_emotions = {"joy": 0, "fear": 0, "hope": 0, "sorrow": 0, "love": 0, "loss": 0}
    for sent in sentences:
        result = sentiment_model(sent[:512])[0]
        magnitude = result["score"]
        if result["label"] == "POSITIVE":
            dominant_emotions["joy"] += magnitude
            dominant_emotions["hope"] += magnitude * 0.8
            dominant_emotions["love"] += magnitude * 0.9
        else:
            dominant_emotions["sorrow"] += magnitude
            dominant_emotions["fear"] += magnitude * 0.8
            dominant_emotions["loss"] += magnitude * 0.9

    emotion_strength = max(dominant_emotions.values()) / max(len(sentences), 1)
    if emotion_strength >= 0.65:
        emotion_points = 2
        message = "Strong emotional presence anchors the narrative."
    elif emotion_strength >= 0.35:
        emotion_points = 1
        message = "Moderate emotional cues detected; consider amplifying them."
    else:
        emotion_points = 0
        message = "No dominant emotions detected."
    _add_breakdown_entry(breakdown, "Emotion First Rule", emotion_points, 2, message)
    score += emotion_points

    # --------------------------
    # 2. Character Depth Rule
    characters = []
    seen_character_keys = set()
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        name = ent.text.strip()
        if not name:
            continue
        key = name.lower()
        if key in seen_character_keys:
            continue
        seen_character_keys.add(key)
        characters.append(name)

    min_characters = 1
    if metadata.get("length_norm") in {"feature", "pilot", "episode"}:
        min_characters = 2
    if metadata.get("genre_norm") in {"drama", "fantasy", "sci-fi"}:
        min_characters = max(min_characters, 2)

    format_desc = _describe_format(metadata)

    if characters:
        actions_detected = any(any(tok.pos_ == "VERB" for tok in sent) for sent in sentence_spans)
        if len(characters) < min_characters:
            if actions_detected:
                points = 1
                message = (
                    f"Only {len(characters)} distinct character(s); {format_desc} stories typically need "
                    f"{min_characters} active characters."
                )
            else:
                points = 0
                message = (
                    f"Introduce at least {min_characters} distinct, goal-driven characters for {format_desc} pieces."
                )
        elif actions_detected:
            points = 2
            message = f"Characters ({', '.join(characters)}) pursue goals and take action."
        else:
            points = 1
            message = "Characters detected but no clear goals/actions."
        _add_breakdown_entry(
            breakdown,
            "Character Depth Rule",
            points,
            2,
            message,
        )
        score += points
    else:
        _add_breakdown_entry(
            breakdown,
            "Character Depth Rule",
            0,
            2,
            f"No characters detected; {format_desc} stories rely on at least {min_characters}.",
        )

    # --------------------------
    # 3. Human Experience Rule
    human_keywords = [
        "love",
        "loss",
        "family",
        "friendship",
        "identity",
        "ambition",
        "courage",
        "compassion",
    ]
    human_hits = sum(1 for word in human_keywords if word in lower_text)
    if human_hits >= 3:
        human_points = 2
        message = "Story strongly grounded in universal human experiences."
    elif human_hits >= 1:
        human_points = 1
        message = "Some human-centric themes detected; could be emphasized more."
    else:
        human_points = 0
        message = "No universal human experiences detected."
    _add_breakdown_entry(
        breakdown, "Human Experience Rule", human_points, 2, message
    )
    score += human_points

    # --------------------------
    # 4. Symbolism Rule
    embeddings = embed_model.encode(sentences, convert_to_tensor=True)
    sim_scores = util.pytorch_cos_sim(embeddings, embeddings)
    repetition_count = max((sim_scores > 0.8).sum().item() - len(sentences), 0)
    repetition_ratio = repetition_count / max(len(sentences), 1)
    if repetition_ratio >= 0.4:
        symbolism_points = 2
        message = "Multiple recurring motifs suggest rich symbolism."
    elif repetition_ratio > 0:
        symbolism_points = 1
        message = "Some motifs repeat; consider reinforcing them."
    else:
        symbolism_points = 0
        message = "No symbolic motifs detected."
    _add_breakdown_entry(breakdown, "Symbolism Rule", symbolism_points, 2, message)
    score += symbolism_points

    # --------------------------
    # 5. Contextual Adaptation Rule
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    optimal_range, acceptable_range = _audience_length_ranges(metadata)
    if optimal_range[0] <= avg_len <= optimal_range[1]:
        adaptation_points = 2
        message = (
            f"Sentence length aligns with expectations for {_describe_format(metadata).lower()} "
            f"({optimal_range[0]}–{optimal_range[1]} words)."
        )
    elif acceptable_range[0] <= avg_len <= acceptable_range[1]:
        adaptation_points = 1
        message = (
            f"Pacing is close to the target range ({optimal_range[0]}–{optimal_range[1]} words); "
            "minor adjustments could improve fit."
        )
    else:
        adaptation_points = 0
        message = (
            f"Average sentence length ({avg_len:.1f} words) sits outside the expected range "
            f"for {_describe_format(metadata).lower()} stories."
        )
    _add_breakdown_entry(
        breakdown,
        "Contextual Adaptation Rule",
        adaptation_points,
        2,
        message,
    )
    score += adaptation_points

    # --------------------------
    # 6. Conflict and Resolution Rule
    conflict_words = ["struggle", "challenge", "obstacle", "battle", "threat", "enemy"]
    resolution_words = ["resolve", "finally", "peace", "overcome", "heal", "end", "victory", "conclusion"]
    conflict_hits = sum(1 for word in conflict_words if word in lower_text)
    resolution_hits = sum(1 for word in resolution_words if word in lower_text)
    requires_full_arc = _expects_full_conflict_arc(metadata)
    if conflict_hits and resolution_hits:
        conflict_points = 2
        message = "Conflict and resolution cues detected."
    elif conflict_hits or resolution_hits:
        if requires_full_arc:
            conflict_points = 1
            message = "Conflict detected, but long-form stories benefit from an explicit resolution."
        else:
            conflict_points = 2
            message = "Open conflict is acceptable for shorter formats; tension is clearly established."
    else:
        conflict_points = 0
        message = "No clear conflict or resolution found."
    _add_breakdown_entry(
        breakdown,
        "Conflict & Resolution Rule",
        conflict_points,
        2,
        message,
    )
    score += conflict_points

    # --------------------------
    # 7. Sensory Immersion Rule
    sensory_words = ["see", "hear", "touch", "smell", "taste", "bright", "dark", "cold", "warm", "shimmer", "flicker", "rough", "fragrant", "echo", "glow"]
    sensory_hits = sum(1 for word in sensory_words if word in lower_text)
    sensory_expectation = _sensory_expectation(metadata)
    if sensory_hits >= sensory_expectation:
        sensory_points = 2
        message = "Sensory detail matches the vivid tone implied by the metadata."
    elif sensory_hits >= max(1, sensory_expectation - 2):
        sensory_points = 1
        message = "Some sensory detail present; consider richer imagery to match the intended genre/setting."
    else:
        sensory_points = 0
        message = "Sensory immersion falls short of the expectations set by the metadata."
    _add_breakdown_entry(
        breakdown, "Sensory Immersion Rule", sensory_points, 2, message
    )
    score += sensory_points

    # --------------------------
    # 8. Authenticity Rule
    clichés = ["once upon a time", "happily ever after", "it was a dark and stormy night"]
    cliche_hits = sum(1 for c in clichés if c in lower_text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
    lexical_diversity = len(set(tokens)) / max(len(tokens), 1)
    if cliche_hits > 0:
        authenticity_points = 0
        message = "Story leans on well-known clichés; try fresh phrasing."
    elif lexical_diversity < 0.45:
        authenticity_points = 1
        message = "Language feels repetitive; consider varied imagery."
    else:
        authenticity_points = 2
        message = "Original language choices keep the story fresh."
    _add_breakdown_entry(
        breakdown, "Authenticity Rule", authenticity_points, 2, message
    )
    score += authenticity_points

    # --------------------------
    # 9. Dynamic Rhythm Rule
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) <= 1:
        rhythm_points = 1
        message = "Single-sentence sample; unable to assess rhythm."
    else:
        avg_length = statistics.mean(lengths)
        stdev = statistics.pstdev(lengths)
        variation_ratio = stdev / max(avg_length, 1)
        if variation_ratio >= 0.45:
            rhythm_points = 2
            message = "Sentence length variation provides dynamic rhythm."
        elif variation_ratio >= 0.25:
            rhythm_points = 1
            message = "Rhythm is steady; add more variation for dramatic beats."
        else:
            rhythm_points = 0
            message = "Sentence lengths are very uniform, reducing momentum."
    _add_breakdown_entry(
        breakdown, "Dynamic Rhythm Rule", rhythm_points, 2, message
    )
    score += rhythm_points

    # --------------------------
    # 10. Moral Resonance Rule
    moral_keywords = ["lesson", "learn", "realize", "understand", "reflect", "question", "truth", "courage", "compassion"]
    moral_hits = sum(1 for word in moral_keywords if word in lower_text)
    expects_reflection = _expects_reflective_ending(metadata)
    if moral_hits >= 2:
        moral_points = 2
        message = "Reflective lesson or insight detected."
    elif moral_hits == 1:
        moral_points = 1
        message = "Some reflective language; clarify the takeaway."
    else:
        moral_points = 0
        if expects_reflection:
            message = f"{_describe_format(metadata)} stories often end with a clear realization; consider highlighting one."
        else:
            message = "No moral or reflective ending detected."
    _add_breakdown_entry(
        breakdown, "Moral Resonance Rule", moral_points, 2, message
    )
    score += moral_points

    _add_breakdown_entry(
        breakdown, "Random Variation", 0, 0, "Random variation disabled for deterministic testing."
    )

    return score, breakdown


def evaluate_story(text: str, metadata: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    raw_score, breakdown = _score_story(text, metadata)
    category_scores: List[Dict[str, object]] = []

    for rule in CATEGORY_RULES:
        result = breakdown.get(
            rule["rule"],
            {"score": 0, "max_points": rule["max_points"], "message": "Not evaluated."},
        )
        max_points = rule["max_points"] or 1
        normalized = round((result["score"] / max_points) * 10, 1)
        category_scores.append(
            {
                "id": rule["id"],
                "category": rule["label"],
                "score": normalized,
                "raw_score": result["score"],
                "max_points": rule["max_points"],
                "message": result["message"],
            }
        )

    overall = round(
        sum(category["score"] for category in category_scores) / len(category_scores), 1
    ) if category_scores else 0

    return {
        "overall_score": overall,
        "raw_score": raw_score,
        "max_score": sum(rule["max_points"] for rule in CATEGORY_RULES),
        "categories": category_scores,
        "breakdown": breakdown,
    }


@app.route("/health", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze_story():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    metadata = payload.get("metadata")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Script text is required."}), 400

    result = evaluate_story(text.strip(), metadata)
    return jsonify(result)


def run_demo():
    sample_story = """
    The village slept under a sky of shimmering stars, yet Mira could not close her eyes.
    She thought about the promise she made to her brother, about the fear in his eyes, and she finally understood the courage it took to leave.
    """
    sample_metadata = {"genre": "Drama", "audience": "Adult", "length": "Feature"}
    result = evaluate_story(sample_story, sample_metadata)
    print(f"Overall Score: {result['overall_score']}")
    for category in result["categories"]:
        print(
            f"{category['category']}: {category['raw_score']} / {category['max_points']} -> {category['message']}"
        )
    print("\nRule Breakdown:")
    for rule_name, details in result["breakdown"].items():
        print(f" - {rule_name}: {details['score']} / {details['max_points']} ({details['message']})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Story Insight expert system API/server.")
    parser.add_argument("--demo", action="store_true", help="Run the console demo and exit.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the API server.")
    parser.add_argument("--port", type=int, default=5000, help="Port for the API server.")
    parser.add_argument("--debug", action="store_true", help="Run the Flask server in debug mode.")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)
