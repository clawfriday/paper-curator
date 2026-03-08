from src.backend import rag


def test_extract_keywords_filters_stopwords_and_dedupes():
    q = "What does GRPO do for RL training with DeepSeek GRPO and RLHF?"
    kws = rag._extract_keywords(q)
    assert "grpo" in kws
    assert len(kws) == len(set(kws))
    assert "what" not in kws


def test_lexical_overlap_score_increases_with_hits():
    keywords = ["grpo", "reward", "policy"]
    low = rag._lexical_overlap_score("This discusses optimization.", keywords)
    high = rag._lexical_overlap_score("GRPO reward model and policy updates are described.", keywords)
    assert high > low


def test_mmr_select_respects_k_and_keeps_top_first():
    candidates = [
        {"final_score": 0.95, "embedding": [0.95, 0.1, 0.1, 0.1]},
        {"final_score": 0.90, "embedding": [0.94, 0.1, 0.1, 0.1]},
        {"final_score": 0.80, "embedding": [0.2, 0.9, 0.1, 0.1]},
    ]
    out = rag._mmr_select(candidates, [1, 0, 0, 0], k=2, lambda_mult=0.7)
    assert len(out) == 2
    assert out[0]["final_score"] == 0.95


def test_recency_score_prefers_recent_years():
    s_recent = rag._recency_score(2025, now_year=2026)
    s_old = rag._recency_score(2018, now_year=2026)
    assert s_recent > s_old
