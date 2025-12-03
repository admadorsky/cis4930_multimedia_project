from experta import KnowledgeEngine, Fact, MATCH, Rule
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import spacy

# ---------- Facts ----------
class StoryRequest(Fact):
    """User request / prompt for a story."""
    pass

class StoryPlan(Fact):
    """Planned story parameters decided by the expert system."""
    pass

# ---------- Knowledge Engine ----------
class StoryRatingEngine(KnowledgeEngine):
    @Rule(StoryRequest(text=MATCH.text))
    def rate_story(self, text):
        breakdown = {}
        score = 0

        # --- Load models ---
        sentiment_model = pipeline("sentiment-analysis")
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        nlp = spacy.load("en_core_web_sm")

        # Split text into sentences
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # --------------------------
        # 1. Emotion First Rule
        dominant_emotions = {"joy": 0, "fear": 0, "hope": 0, "sorrow": 0, "love": 0, "loss": 0}
        for sent in sentences:
            result = sentiment_model(sent[:512])[0]
            if result['label'] == "POSITIVE":
                dominant_emotions["joy"] += result['score']
                dominant_emotions["hope"] += result['score'] * 0.8
                dominant_emotions["love"] += result['score'] * 0.9
            else:
                dominant_emotions["sorrow"] += result['score']
                dominant_emotions["fear"] += result['score'] * 0.8
                dominant_emotions["loss"] += result['score'] * 0.9

        if max(dominant_emotions.values()) > 0.5:
            breakdown['Emotion First Rule'] = (2, "Strong emotional presence detected")
            score += 2
        else:
            breakdown['Emotion First Rule'] = (0, "No dominant emotions detected")

        # --------------------------
        # 2. Character Depth Rule
        characters = [ent.text for ent in doc.ents if ent.label_ in ["PERSON"]]
        if characters:
            actions_detected = any(
                any(tok.pos_ == "VERB" for tok in sent_doc)
                for sent_doc in [nlp(s) for s in sentences]
            )
            if actions_detected:
                breakdown['Character Depth Rule'] = (2, f"Characters ({', '.join(characters)}) have goals and actions")
                score += 2
            else:
                breakdown['Character Depth Rule'] = (1, "Characters detected but no clear goals/actions")
                score += 1
        else:
            breakdown['Character Depth Rule'] = (0, "No characters detected")

        # --------------------------
        # 3. Human Experience Rule
        human_keywords = ["love", "loss", "family", "friendship", "identity", "ambition", "courage", "compassion"]
        if any(word.lower() in text.lower() for word in human_keywords):
            breakdown['Human Experience Rule'] = (2, "Story grounded in universal human experiences")
            score += 2
        else:
            breakdown['Human Experience Rule'] = (0, "No universal human experiences detected")

        # --------------------------
        # 4. Symbolism Rule
        embeddings = embed_model.encode(sentences, convert_to_tensor=True)
        sim_scores = util.pytorch_cos_sim(embeddings, embeddings)
        repetition_count = (sim_scores > 0.8).sum().item() - len(sentences)
        if repetition_count > 0:
            breakdown['Symbolism Rule'] = (2, "Recurring motifs or symbolic concepts detected")
            score += 2
        else:
            breakdown['Symbolism Rule'] = (0, "No symbolic motifs detected")

        # --------------------------
        # 5. Contextual Adaptation Rule
        avg_len = sum(len(s.split()) for s in sentences)/len(sentences)
        if avg_len >= 8:
            breakdown['Contextual Adaptation Rule'] = (1, "Tone and pacing appropriate for general audience")
            score += 1
        else:
            breakdown['Contextual Adaptation Rule'] = (0, "Story may not adapt well to audience")

        # --------------------------
        # 6. Conflict and Resolution Rule
        conflict_words = ["struggle", "challenge", "obstacle", "finally", "resolve", "end", "overcome"]
        if any(word.lower() in text.lower() for word in conflict_words):
            breakdown['Conflict & Resolution Rule'] = (2, "Conflict and resolution detected")
            score += 2
        else:
            breakdown['Conflict & Resolution Rule'] = (0, "No clear conflict or resolution detected")

        # --------------------------
        # 7. Sensory Immersion Rule
        sensory_words = ["see", "hear", "touch", "smell", "taste", "bright", "dark", "cold", "warm", "shimmer", "flicker"]
        if any(word.lower() in text.lower() for word in sensory_words):
            breakdown['Sensory Immersion Rule'] = (1, "Sensory details present, immersive experience")
            score += 1
        else:
            breakdown['Sensory Immersion Rule'] = (0, "No sensory immersion detected")

        # --------------------------
        # 8. Authenticity Rule
        clichés = ["once upon a time", "happily ever after", "it was a dark and stormy night"]
        if not any(c in text.lower() for c in clichés):
            breakdown['Authenticity Rule'] = (1, "Story is original, no clichés detected")
            score += 1
        else:
            breakdown['Authenticity Rule'] = (0, "Story contains clichés")

        # --------------------------
        # 9. Dynamic Rhythm Rule
        lengths = [len(s.split()) for s in sentences]
        if max(lengths) - min(lengths) > 5:
            breakdown['Dynamic Rhythm Rule'] = (1, "Sentence length variation provides dynamic rhythm")
            score += 1
        else:
            breakdown['Dynamic Rhythm Rule'] = (1, "Balanced sentence lengths, rhythm acceptable")
            score += 1

        # --------------------------
        # 10. Moral Resonance Rule
        moral_keywords = ["lesson", "learn", "realize", "understand", "reflect", "question", "truth", "courage", "compassion"]
        if any(word.lower() in text.lower() for word in moral_keywords):
            breakdown['Moral Resonance Rule'] = (2, "Reflective lesson or insight detected")
            score += 2
        else:
            breakdown['Moral Resonance Rule'] = (0, "No moral or reflective ending detected")

        # --------------------------
        breakdown['Random Variation'] = (0, "Random variation disabled for testing")

        # Declare the result
        self.declare(StoryPlan(score=score, breakdown=breakdown))

# ---------- Example Usage ----------
if __name__ == "__main__":
    engine = StoryRatingEngine()
    engine.reset()
    engine.declare(StoryRequest(text="""
The village slept under a sky of shimmering stars, yet Mira could not close her eyes...
"""))
    engine.run()

    for fact in engine.facts.values():
        if isinstance(fact, StoryPlan):
            print(f"Score: {fact['score']}")
            for rule, result in fact['breakdown'].items():
                print(f"{rule}: {result}")
