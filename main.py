from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import spacy
from typing import Dict, List, Tuple
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


def _add_breakdown_entry(breakdown: Dict[str, Dict[str, str]], rule: str, score: int, max_points: int, message: str):
    breakdown[rule] = {
        "score": score,
        "max_points": max_points,
        "message": message,
    }


def _score_story(text: str) -> Tuple[int, Dict[str, Dict[str, str]]]:
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
    characters = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if characters:
        actions_detected = any(any(tok.pos_ == "VERB" for tok in sent) for sent in sentence_spans)
        if actions_detected:
            _add_breakdown_entry(
                breakdown,
                "Character Depth Rule",
                2,
                2,
                f"Characters ({', '.join(characters)}) pursue goals and take action.",
            )
            score += 2
        else:
            _add_breakdown_entry(
                breakdown,
                "Character Depth Rule",
                1,
                2,
                "Characters detected but no clear goals/actions.",
            )
            score += 1
    else:
        _add_breakdown_entry(breakdown, "Character Depth Rule", 0, 2, "No characters detected.")

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
    if 10 <= avg_len <= 22:
        adaptation_points = 2
        message = "Sentence length and tone feel well-calibrated for a broad audience."
    elif 7 <= avg_len < 10 or 22 < avg_len <= 28:
        adaptation_points = 1
        message = "Pacing is close to the sweet spot; minor adjustments could improve flow."
    else:
        adaptation_points = 0
        message = "Sentence length suggests the tone may miss the intended audience."
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
    if conflict_hits and resolution_hits:
        conflict_points = 2
        message = "Conflict and resolution cues detected."
    elif conflict_hits or resolution_hits:
        conflict_points = 1
        message = "Partial conflict arc detected; consider clarifying resolution."
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
    if sensory_hits >= 4:
        sensory_points = 2
        message = "Rich sensory details build vivid immersion."
    elif sensory_hits >= 2:
        sensory_points = 1
        message = "Some sensory details present; more specificity could enhance immersion."
    else:
        sensory_points = 0
        message = "No sensory immersion detected."
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
    if moral_hits >= 2:
        moral_points = 2
        message = "Reflective lesson or insight detected."
    elif moral_hits == 1:
        moral_points = 1
        message = "Some reflective language; clarify the takeaway."
    else:
        moral_points = 0
        message = "No moral or reflective ending detected."
    _add_breakdown_entry(
        breakdown, "Moral Resonance Rule", moral_points, 2, message
    )
    score += moral_points

    _add_breakdown_entry(
        breakdown, "Random Variation", 0, 0, "Random variation disabled for deterministic testing."
    )

    return score, breakdown


def evaluate_story(text: str) -> Dict[str, object]:
    raw_score, breakdown = _score_story(text)
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

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Script text is required."}), 400

    result = evaluate_story(text.strip())
    return jsonify(result)


def run_demo():
    sample_story = """
    The village slept under a sky of shimmering stars, yet Mira could not close her eyes.
    She thought about the promise she made to her brother, about the fear in his eyes, and she finally understood the courage it took to leave.
    """
    result = evaluate_story(sample_story)
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
