import datetime
import re

_INSTRUCTION_WORDS = [
    "extract", "summarize", "summarise", "classify", "list", "write", "explain",
    "evaluate", "compare", "contrast", "analyze", "analyse", "design", "convert",
    "reformat", "identify", "categorize", "categorise", "describe", "translate",
    "recommend", "critique", "assess",
]
_CONSTRAINT_WORDS = [
    "must", "should", "only", "exactly", "at least", "no more than", "within",
    "given that", "constraint", "limit", "cannot", "avoid", "ensure",
]
_LIST_FORMAT_WORDS = ["list", "bullet", "steps", "enumerate"]
_STRUCTURED_FORMAT_WORDS = ["json", "table", "structured", "key-value", "csv", "schema"]


def _count_word_occurrences(text: str, words: list[str]) -> int:
    return sum(len(re.findall(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)) for word in words)


def extract_features(prompt: str, context: str | None = None) -> dict:
    text = prompt.lower()

    if any(re.search(rf"\b{word}\b", text) for word in _STRUCTURED_FORMAT_WORDS):
        output_format_complexity = 2
    elif any(re.search(rf"\b{word}\b", text) for word in _LIST_FORMAT_WORDS):
        output_format_complexity = 1
    else:
        output_format_complexity = 0

    sentences = [s for s in re.split(r"[.!?]+", prompt) if s.strip()]

    return {
        "token_count": len(re.findall(r"\S+", prompt)),
        "char_count": len(prompt),
        "has_analyze_keyword": int(bool(re.search(r"\banaly[sz]e\b", text))),
        "has_compare_keyword": int(bool(re.search(r"\b(compare|contrast)\b", text))),
        "instruction_word_count": _count_word_occurrences(prompt, _INSTRUCTION_WORDS),
        "num_constraints": _count_word_occurrences(prompt, _CONSTRAINT_WORDS),
        "has_context": int(bool(context and context.strip())),
        "output_format_complexity": output_format_complexity,
        "question_mark_count": prompt.count("?"),
        "sentence_count": max(len(sentences), 1),
    }


_RECENCY_WORDS = [
    "current", "currently", "recent", "recently", "latest", "today", "tonight",
    "this week", "this month", "this year", "up to date", "up-to-date", "as of now",
    "most recent", "last election", "who won",
]

# Years from here on are the risk zone: what matters is whether an event predates or postdates a cheap model's cutoff (~late 2023 for Llama 3.1 8B), not whether the year is close to today.
_STALENESS_THRESHOLD_YEAR_OFFSET = -2


def needs_current_knowledge(prompt: str) -> bool:
    """Detects whether a prompt needs current information the complexity classifier has no way to see, via explicit recency language or a year at/after a cheap model's approximate training cutoff."""
    text = prompt.lower()
    if any(re.search(rf"\b{re.escape(word)}\b", text) for word in _RECENCY_WORDS):
        return True
    current_year = datetime.datetime.now().year
    threshold_year = current_year + _STALENESS_THRESHOLD_YEAR_OFFSET
    return any(str(year) in prompt for year in range(threshold_year, current_year + 2))


FEATURE_COLUMNS = [
    "token_count", "char_count", "has_analyze_keyword", "has_compare_keyword",
    "instruction_word_count", "num_constraints", "has_context",
    "output_format_complexity", "question_mark_count", "sentence_count",
]
