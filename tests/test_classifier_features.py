from app.classifier.features import extract_features, needs_current_knowledge


def test_detects_compare_keyword():
    f = extract_features("Compare and contrast cats and dogs.")
    assert f["has_compare_keyword"] == 1


def test_detects_analyze_keyword():
    f = extract_features("Analyze this dataset for trends.")
    assert f["has_analyze_keyword"] == 1


def test_no_keywords_on_simple_prompt():
    f = extract_features("What is the capital of France?")
    assert f["has_analyze_keyword"] == 0
    assert f["has_compare_keyword"] == 0
    assert f["question_mark_count"] == 1


def test_has_context_flag():
    with_ctx = extract_features("What is the name?", context="John Smith is the CEO.")
    without_ctx = extract_features("What is the name?", context=None)
    assert with_ctx["has_context"] == 1
    assert without_ctx["has_context"] == 0


def test_output_format_complexity_levels():
    plain = extract_features("Tell me a story about a dog.")
    listy = extract_features("List the steps to bake a cake.")
    structured = extract_features("Return the result as a JSON object.")
    assert plain["output_format_complexity"] == 0
    assert listy["output_format_complexity"] == 1
    assert structured["output_format_complexity"] == 2


def test_num_constraints_counts_constraint_words():
    f = extract_features("The summary must be under 50 words and should avoid jargon.")
    assert f["num_constraints"] >= 2


def test_sentence_count_at_least_one():
    f = extract_features("Hello there")
    assert f["sentence_count"] == 1
    multi = extract_features("First sentence. Second sentence! Third one?")
    assert multi["sentence_count"] == 3


def test_needs_current_knowledge_detects_recency_words():
    assert needs_current_knowledge("What is happening with inflation right now?") is False
    assert needs_current_knowledge("What is the current state of inflation?") is True
    assert needs_current_knowledge("What's the latest on the trade deal?") is True


def test_needs_current_knowledge_detects_near_current_year():
    import datetime

    year = datetime.datetime.now().year
    assert needs_current_knowledge(f"What happened in {year}?") is True
    assert needs_current_knowledge(f"This isn't 2018, this is {year - 1}-{year}") is True
    assert needs_current_knowledge("What happened in 1969?") is False


def test_needs_current_knowledge_catches_past_cutoff_year_not_just_near_today():
    # Regression: a naive "years near today" window missed "the 2024 election" once today was 2026, since what matters is the model's cutoff, not proximity to today.
    import datetime

    year = datetime.datetime.now().year
    assert needs_current_knowledge(f"What were the results of the {year - 2} election?") is True


def test_needs_current_knowledge_false_for_plain_factual_question():
    assert needs_current_knowledge("What is the capital of France?") is False
