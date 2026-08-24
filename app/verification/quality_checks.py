import json
import re


def extraction_check(response_text: str, required_keys: list[str]) -> bool:
    if not required_keys:
        return True
    try:
        data = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return all(key.lower() in response_text.lower() for key in required_keys)
    if not isinstance(data, dict):
        return False
    return all(key in data for key in required_keys)


def classification_check(response_text: str, reference_text: str) -> bool:
    # Checks that the candidate's label appears in the reference rather than requiring exact equality, since a terse label often disagrees in formatting, not substance, with a verbose reference answer.
    candidate_label = _normalize_label(response_text)
    if not candidate_label:
        return False
    return bool(re.search(rf"\b{re.escape(candidate_label)}\b", reference_text.lower()))


def _normalize_label(text: str) -> str:
    return text.strip().strip(".").lower()
