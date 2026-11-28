_BRACKETS = {"}": "{", ")": "(", "]": "[", ">": "<"}


def extract_answer(text: str, open_marker: str, close_marker: str) -> str | None:
    r"""Return the content of the last `open_marker ... close_marker` span.

    Only the last occurrence of `open_marker` is considered. The answer runs
    from there to the first `close_marker` after it, except when the open
    marker ends with an opening bracket and the close marker is the matching
    closing bracket (e.g. `\boxed{` and `}`): then brackets are balanced, so
    `\boxed{\frac{1}{2}}` gives `\frac{1}{2}`. An empty close marker takes the
    rest of the text. When both markers are the same string, the content
    between its last two occurrences is used.

    Args:
        text: Model output (final answer only, reasoning excluded).
        open_marker: Opening marker (non-empty).
        close_marker: Closing marker (may be empty).

    Returns:
        The answer, or None if there is no complete marker span.
    """
    if open_marker == close_marker:
        parts = text.split(open_marker)
        return parts[-2] if len(parts) >= 3 else None

    start = text.rfind(open_marker)
    if start < 0:
        return None
    start += len(open_marker)

    if close_marker == "":
        return text[start:]

    opener = _BRACKETS[close_marker] if close_marker in _BRACKETS else None
    if opener is not None and open_marker.endswith(opener):
        depth = 1
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == close_marker:
                depth -= 1
                if depth == 0:
                    return text[start:i]
        return None

    end = text.find(close_marker, start)
    if end < 0:
        return None
    return text[start:end]
