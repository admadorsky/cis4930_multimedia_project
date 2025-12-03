import sys, types, pytest, json

class _Tok:
    def __init__(self, t):
        self.text = t
        self.is_alpha = t.isalpha()
        self.lemma_ = t.lower()
        self.pos_ = "VERB" if t.lower().endswith(("s", "ed")) else "NOUN"

class _Span:
    def __init__(self, text):
        self.text = text
        self._toks = [_Tok(t) for t in text.split() if t]
    def __iter__(self):
        return iter(self._toks)

class _Ent:
    def __init__(self, text):
        self.text = text
        self.label_ = "PERSON"

class _Doc:
    def __init__(self, text):
        self.text = text
        parts = [s.strip() for s in text.split(".") if s.strip()]
        self.sents = [_Span(s) for s in parts]
        seen, ents = set(), []
        for w in text.replace("\n", " ").split():
            if w[:1].isupper():
                k = w.strip(",. ").lower()
                if k and k not in seen:
                    seen.add(k)
                    ents.append(_Ent(w.strip(",. ")))
        self.ents = ents
    def __iter__(self):
        for s in self.sents:
            for t in s:
                yield t

def _fake_spacy_load(_name):
    return lambda txt: _Doc(txt)

def _fake_pipeline(_task):
    def run(sentence):
        lab = "POSITIVE" if "joy" in sentence.lower() else "NEGATIVE"
        score = 0.9 if lab == "POSITIVE" else 0.6
        return [{"label": lab, "score": score}]
    return run

class _FakeMatrix:
    def __init__(self, n):
        self.n = n
    def __gt__(self, _thresh):
        return self
    def sum(self):
        class _V:
            def __init__(self, v): self.v = v
            def item(self): return self.v
        return _V(self.n)

class _FakeEmbedModel:
    def encode(self, sentences, convert_to_tensor=True):
        return list(range(len(sentences)))

spacy_mod = types.ModuleType("spacy")
spacy_mod.load = _fake_spacy_load
transformers_mod = types.ModuleType("transformers")
transformers_mod.pipeline = _fake_pipeline
st_util_mod = types.ModuleType("sentence_transformers.util")
st_util_mod.pytorch_cos_sim = lambda a, b: _FakeMatrix(len(a))
st_mod = types.ModuleType("sentence_transformers")
st_mod.SentenceTransformer = lambda _name: _FakeEmbedModel()
st_mod.util = st_util_mod

sys.modules["spacy"] = spacy_mod
sys.modules["transformers"] = transformers_mod
sys.modules["sentence_transformers"] = st_mod
sys.modules["sentence_transformers.util"] = st_util_mod

import main as appmod

@pytest.fixture
def client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

def _cat(data, name):
    return next(c for c in data["categories"] if c["category"] == name)

def test_health_endpoint_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}

def test_analyze_requires_text(client):
    r = client.post("/analyze", json={"text": "   "})
    assert r.status_code == 400
    assert "error" in r.get_json()

def test_character_depth_feature_length_requires_two_people(client):
    payload = {"text": "Alice runs to help. Bob waits and decides.", "metadata": {"length": "feature", "genre": "drama"}}
    r = client.post("/analyze", json=payload)
    data = r.get_json()
    assert r.status_code == 200
    assert _cat(data, "Character Depth")["raw_score"] == 2

def test_contextual_adaptation_kids_shortfilm_optimal(client):
    payload = {"text": "The fox jumps quickly. Friends laugh and share joy together.", "metadata": {"audience": "kids", "length": "short film"}}
    r = client.post("/analyze", json=payload)
    data = r.get_json()
    assert _cat(data, "Audience Fit")["raw_score"] == 2

def test_authenticity_penalizes_cliche_and_low_diversity(client):
    payload = {"text": "Once upon a time a hero walks and walks. The crowd watches.", "metadata": {"length": "episode", "genre": "fantasy"}}
    r = client.post("/analyze", json=payload)
    data = r.get_json()
    assert _cat(data, "Authenticity / Originality")["raw_score"] == 0

def test_conflict_open_is_ok_for_short_form_when_no_full_arc_required(client):
    payload = {"text": "Mara faces a challenge and a threat in the city. Tension rises.", "metadata": {"length": "short film", "genre": "drama"}}
    r = client.post("/analyze", json=payload)
    data = r.get_json()
    assert _cat(data, "Conflict & Resolution")["raw_score"] == 2

def test_human_experience_scores_when_keywords_present(client):
    payload = {"text": "We face loss and find love through family with courage and compassion. Friends learn together.", "metadata": {"genre": "drama", "length": "short film"}}
    r = client.post("/analyze", json=payload)
    data = r.get_json()
    assert _cat(data, "Human Experience")["raw_score"] == 2

def test_sensory_immersion_matches_expectation_for_scifi_space_setting(client):
    text = "I see bright stars and dark voids, hear echoes, feel cold metal, smell oil, taste dust, and touch rough panels as lights shimmer and engines glow."
    payload = {"text": text, "metadata": {"genre": "sci-fi", "setting": "deep space station", "length": "short film"}}
    r = client.post("/analyze", json=payload)
    data = r.get_json()
    assert _cat(data, "Sensory Immersion")["raw_score"] == 2

def test_audience_fit_acceptably_close_not_optimal(client):
    s = "This story moves with steady steps as simple ideas guide a young hero across a small town, meeting friends and making choices that shape the day."
    text = f"{s} {s}"
    payload = {"text": text, "metadata": {"audience": "general"}}
    r = client.post("/analyze", json=payload)
    data = r.get_json()
    assert _cat(data, "Audience Fit")["raw_score"] == 1

def test_moral_resonance_detected_with_clear_takeaway(client):
    payload = {"text": "After many trials they learn to reflect and realize the truth about compassion and courage.", "metadata": {"genre": "drama"}}
    r = client.post("/analyze", json=payload)
    data = r.get_json()
    assert _cat(data, "Moral Resonance / Thematic Impact")["raw_score"] == 2
