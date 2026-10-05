"""Unit tests for compaction.py. Stub tokenizer (1 token per word), no model."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compaction as c  # noqa: E402


def count(text: str) -> int:
    return len(text.split())


def msgs(n: int, words: int = 5) -> list[dict]:
    """n alternating messages starting with user."""
    out = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"{role} {i} " + "word " * words})
    return out


# --- needs_compaction / context_usage ------------------------------------------


def test_needs_compaction_threshold():
    assert not c.needs_compaction(74, 100, 0.75, True)
    assert c.needs_compaction(75, 100, 0.75, True)
    assert c.needs_compaction(120, 100, 0.75, True)


def test_needs_compaction_requires_something_to_fold_and_a_budget():
    assert not c.needs_compaction(500, 100, 0.75, False)
    assert not c.needs_compaction(500, 0, 0.75, True)


def test_context_usage_matches_plan_example():
    used, budget = c.context_usage(
        summary_tokens=100, history_tokens=1000, answer_tokens=520,
        instruction_tokens=750, n_ctx=4096, max_generation_tokens=1400,
        max_history_tokens=2500, margin_tokens=64,
    )
    assert used == 1620
    assert budget == 1882  # 4096 - 1400 - 750 - 64


def test_context_usage_budget_is_capped_and_never_negative():
    _, capped = c.context_usage(
        summary_tokens=0, history_tokens=0, answer_tokens=0, instruction_tokens=750,
        n_ctx=6144, max_generation_tokens=1400, max_history_tokens=2500, margin_tokens=64,
    )
    assert capped == 2500
    _, floor = c.context_usage(
        summary_tokens=0, history_tokens=0, answer_tokens=0, instruction_tokens=9000,
        n_ctx=4096, max_generation_tokens=1400, max_history_tokens=2500, margin_tokens=64,
    )
    assert floor == 0


# --- select_fold_range ----------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_nothing_to_fold_in_short_lists(n):
    start, end = c.select_fold_range(msgs(n), 0, 4)
    assert end == start


def test_keeps_recent_messages_and_aligns_to_a_user_turn():
    m = msgs(10)  # u a u a u a u a u a
    start, end = c.select_fold_range(m, 0, 4)
    assert (start, end) == (0, 6)
    assert m[end]["role"] == "user"
    assert len(m) - end >= 4


def test_never_splits_an_exchange_for_odd_keep_values():
    m = msgs(10)
    for keep in (2, 3, 4, 5, 6):
        start, end = c.select_fold_range(m, 0, keep)
        if end > start:
            assert m[end]["role"] == "user"
            assert len(m) - end >= keep


def test_respects_existing_coverage():
    m = msgs(12)
    assert c.select_fold_range(m, 6, 4) == (6, 8)
    assert c.select_fold_range(m, 8, 4) == (8, 8)  # only the recent window is left
    assert c.select_fold_range(m, 99, 4) == (12, 12)  # absurd coverage is harmless


def test_keep_recent_has_a_floor_of_two():
    m = msgs(6)
    start, end = c.select_fold_range(m, 0, 0)
    assert len(m) - end >= 2


def test_misaligned_history_still_ends_on_user_boundary():
    m = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
         {"role": "assistant", "content": "c"}, {"role": "user", "content": "d"},
         {"role": "assistant", "content": "e"}, {"role": "user", "content": "f"},
         {"role": "assistant", "content": "g"}]
    start, end = c.select_fold_range(m, 0, 2)
    assert end == 0 or m[end]["role"] == "user"


# --- slice / resolve ------------------------------------------------------------


def test_slice_after_summary_uses_raw_indices():
    m = msgs(6)
    assert c.slice_after_summary(m, 0) == m
    assert c.slice_after_summary(m, 4) == m[4:]


def test_slice_never_removes_the_last_message():
    m = msgs(5)
    assert c.slice_after_summary(m, 50) == m[-1:]
    assert c.slice_after_summary(m, -3) == m
    assert c.slice_after_summary([], 3) == []


def test_resolve_summary():
    assert c.resolve_summary(None, 4, 10, 2500) == (None, 0)
    assert c.resolve_summary("   ", 4, 10, 2500) == (None, 0)  # blank: coverage ignored
    assert c.resolve_summary("  hello  ", 4, 10, 2500) == ("hello", 4)
    assert c.resolve_summary("hello", 99, 10, 2500) == ("hello", 9)
    with pytest.raises(ValueError):
        c.resolve_summary("x" * 2501, 4, 10, 2500)


# --- validate_summary -----------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   \n\t", "x" * 2501, None, 5, ["a"], {"summary": "a"}])
def test_validate_summary_rejects(bad):
    with pytest.raises(ValueError):
        c.validate_summary(bad, 2500)


def test_validate_summary_trims_and_accepts_the_limit():
    assert c.validate_summary("  ok \n", 2500) == "ok"
    assert len(c.validate_summary("x" * 2500, 2500)) == 2500


# --- build_summary_prompt / plan_chunks ----------------------------------------


def test_prompt_contains_previous_summary_and_every_message():
    m = msgs(6)
    p = c.build_summary_prompt("Budget is 5k.", m)
    assert "PREVIOUS SUMMARY:\nBudget is 5k." in p
    for item in m:
        assert item["content"] in p
    assert 'Respond with ONLY a JSON object: {"summary"' in p
    assert "exactly as stated" in p


def test_prompt_without_previous_summary_says_none_and_survives_braces():
    p = c.build_summary_prompt(None, [{"role": "user", "content": "use {curly} and {0} braces"}])
    assert "PREVIOUS SUMMARY:\n(none)" in p
    assert "use {curly} and {0} braces" in p


def test_one_chunk_when_everything_fits():
    m = msgs(6)
    chunks = c.plan_chunks(m, None, count, input_max_tokens=3000, summary_max_tokens=320)
    assert len(chunks) == 1 and chunks[0] == m


def test_chunks_respect_the_input_cap_and_keep_order():
    m = msgs(20, words=60)
    chunks = c.plan_chunks(m, "prev " * 50, count, input_max_tokens=700, summary_max_tokens=100)
    assert len(chunks) > 1
    assert [x["content"] for ch in chunks for x in ch] == [x["content"] for x in m]
    for ch in chunks:
        assert count(c.build_summary_prompt("prev " * 100, ch)) <= 700 + 40  # label/newline slack


def test_oversized_single_message_is_truncated_not_dropped():
    big = {"role": "user", "content": "alpha " * 100 + "middle " * 2000 + "omega " * 100}
    chunks = c.plan_chunks([big], None, count, input_max_tokens=800, summary_max_tokens=100)
    assert len(chunks) == 1
    text = chunks[0][0]["content"]
    assert text.startswith("alpha") and text.rstrip().endswith("omega")
    assert "[…]" in text
    assert count(text) < 700


def test_plan_chunks_empty():
    assert c.plan_chunks([], "x", count, input_max_tokens=3000, summary_max_tokens=320) == []


def test_truncate_to_tokens_noop_when_it_fits():
    assert c.truncate_to_tokens("a b c", 10, count) == "a b c"


# --- reading model output -------------------------------------------------------


def test_extract_summary_from_valid_json():
    assert c.extract_summary(json.dumps({"summary": "  The user runs a bakery.\nBudget 5k. "})) == (
        "The user runs a bakery.\nBudget 5k."
    )


def test_extract_summary_salvages_truncated_output():
    raw = '{"summary": "User runs a bakery in Lagos. Budget is 5k. Asked abo'
    assert c.extract_summary(raw) == "User runs a bakery in Lagos. Budget is 5k. Asked abo"


def test_extract_summary_handles_split_escapes():
    assert c.extract_summary('{"summary": "line one\\') == "line one"
    assert c.extract_summary('{"summary": "caf\\u00') == "caf"
    assert c.extract_summary('{"summary": "a\\nb\\"c') == 'a\nb"c'


def test_extract_summary_returns_empty_when_nothing_usable():
    assert c.extract_summary("") == ""
    assert c.extract_summary("not json at all") == ""
    assert c.extract_summary('{"other": "x"}') == ""
    assert c.extract_summary('{"summary": ""}') == ""


def test_trim_to_sentence():
    assert c.trim_to_sentence("One. Two. Thr") == "One. Two."
    assert c.trim_to_sentence("No terminator here at all") == "No terminator here at all"
    # cutting would throw away most of the text: keep it
    assert c.trim_to_sentence("A. " + "long fragment " * 20) .startswith("A. long")
