"""Unit tests for documents.py: cleaning, chunking, outline, retrieval, validation.
Stub tokenizer (1 token per 4 characters), no model."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import documents as D  # noqa: E402


def count(text: str) -> int:
    return max(1, len(text) // 4)


def wrap(s: str, width: int = 40) -> str:
    """Hard-wrap like a PDF text extractor does."""
    lines, cur = [], ""
    for word in s.split():
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    lines.append(cur)
    return "\n".join(lines)


SECTIONS = [
    ("1. INTRODUCTION", "The blue economy covers sustainable use of ocean resources and protects coastal ecosystems. " * 6),
    ("2. MARKET OVERVIEW", "Global markets for marine services keep growing and investors are watching ports closely. " * 6),
    ("12. VERIFICATION AND NOTIFICATIONS", "Every \ufb01nancial transaction needs Veri\ufb01cation before settlement. Noti\ufb01cations go to all parties. " * 5),
    ("24. RISKS", "Regulatory change and weather are the main risks to coastal operators and shipping lanes. " * 6),
]


def sample_raw() -> str:
    pages = [f"BLUE ECONOMY REPORT\n{h}\n{wrap(body)}\n{i + 1}" for i, (h, body) in enumerate(SECTIONS)]
    return "\n\n".join(pages)


def make_docs(raw: str | None = None, name: str = "Blue.pdf") -> list[dict]:
    built = D.build_document(name, raw if raw is not None else sample_raw(), pages=4)
    return D.validate_documents(
        [{"id": built["doc_id"], "name": built["name"], "chunks": built["chunks"],
          "overlaps": built["overlaps"], "outline": built["outline"]}]
    )


# --- cleaning ---------------------------------------------------------------------


def test_ligatures_become_letters_so_words_match():
    out = D.clean_text("The Veri\ufb01cation of \ufb01nancial Noti\ufb01cations")
    assert out == "The Verification of financial Notifications"


def test_wrapped_lines_are_rejoined_and_hyphens_removed():
    out = D.clean_text("This sentence was wrapped\nby the pdf extractor and the veri-\nfication step is split.")
    assert out == "This sentence was wrapped by the pdf extractor and the verification step is split."


def test_hyphen_before_uppercase_is_kept():
    assert D.clean_text("a well-\nKnown name") == "a well- Known name"


def test_headings_and_list_items_keep_their_own_lines():
    raw = "intro text that\ncontinues here\n5. BLUE AI\nbody of the section\nwraps too\n- first item\n- second item"
    out = D.clean_text(raw)
    assert out.split("\n") == [
        "intro text that continues here",
        "5. BLUE AI",
        "body of the section wraps too",
        "- first item",
        "- second item",
    ]


def test_control_and_zero_width_characters_and_page_numbers_go():
    out = D.clean_text("a\x00b\u200bc\u00ad" + "d\n\n12\n\nPage 3 of 10\n\nreal text")
    assert out == "abcd\n\nreal text"


def test_empty_input():
    assert D.clean_text("") == "" and D.clean_text("  \n\n ") == ""


# --- chunking ---------------------------------------------------------------------


def test_chunks_respect_target_and_never_split_words():
    text = D.clean_text(sample_raw())
    chunks = D.chunk_text(text)
    assert len(chunks) >= 3
    for c in chunks:
        core = c["text"][c["ov"] :]
        assert len(core) <= D.CHUNK_TARGET + 5
        assert core == text[c["start"] : c["end"]]
        assert core[0] != " " and core[-1] not in "-"
        # core starts and ends on a word boundary of the original text
        assert c["start"] == 0 or text[c["start"] - 1].isspace()
        assert c["end"] == len(text) or text[c["end"]].isspace() or text[c["end"] - 1] in ".!?;:,"


def test_overlap_prefix_comes_from_previous_chunk_on_a_word_boundary():
    text = D.clean_text(sample_raw())
    chunks = D.chunk_text(text)
    for prev, cur in zip(chunks, chunks[1:]):
        prefix = cur["text"][: cur["ov"] - 1]
        assert prefix and prev["text"][prev["ov"] :].endswith(prefix)
        assert cur["text"][cur["ov"] - 1] == " "


def test_one_long_token_is_hard_cut_not_dropped():
    url = "https://example.com/" + "a" * 2500
    chunks = D.chunk_text(D.clean_text(f"see {url} for details"))
    joined = "".join(c["text"][c["ov"] :].replace(" ", "") for c in chunks)
    assert "a" * 2500 in joined


def test_short_text_is_one_chunk_without_overlap():
    chunks = D.chunk_text("Just one short paragraph.")
    assert len(chunks) == 1 and chunks[0]["ov"] == 0 and chunks[0]["text"] == "Just one short paragraph."


# --- outline ----------------------------------------------------------------------


def test_outline_maps_headings_to_chunks_and_drops_running_headers():
    built = D.build_document("Blue.pdf", sample_raw(), pages=4)
    titles = [o["t"] for o in built["outline"]]
    assert titles == [s[0] for s in SECTIONS]  # "BLUE ECONOMY REPORT" repeats 4x: a page header, not a section
    for item in built["outline"]:
        core = built["chunks"][item["c"]][built["overlaps"][item["c"]] :]
        assert item["t"] in core  # the heading really is inside the chunk it points to
    assert [o["c"] for o in built["outline"]] == sorted(o["c"] for o in built["outline"])


def test_outline_ignores_sentences_and_numbered_sentences():
    out = D.build_document("x", "Intro.\n\n1. This is a full sentence item that ends with a period.\n\n2 apples are fine", pages=None)
    assert out["outline"] == []


# --- document building --------------------------------------------------------------


def test_doc_id_is_stable_so_reuploading_is_a_noop():
    a = D.build_document("A.pdf", sample_raw())
    b = D.build_document("copy of A.pdf", sample_raw())
    assert a["doc_id"] == b["doc_id"] and a["chunks"] == b["chunks"]
    assert D.build_document("A.pdf", sample_raw() + " changed")["doc_id"] != a["doc_id"]


def test_scanned_pdf_gets_a_clear_message():
    with pytest.raises(D.DocumentError) as e:
        D.build_document("scan.pdf", "  \n\n ", pages=5)
    assert e.value.status == 400 and "no selectable text" in e.value.message and "OCR" in e.value.message


def test_text_poor_pdf_warns():
    out = D.build_document("thin.pdf", "A tiny bit of text on a big pdf", pages=40)
    assert out["warnings"] and "40 page" in out["warnings"][0]


def test_oversized_document_is_413():
    with pytest.raises(D.DocumentError) as e:
        D.build_document("big.txt", "word " * 90_000)
    assert e.value.status == 413 and "too large" in e.value.message


def test_empty_text_file():
    with pytest.raises(D.DocumentError) as e:
        D.build_document("x.txt", "")
    assert e.value.status == 400


# --- validation -------------------------------------------------------------------------


def good_doc(i: int = 1, chunks=None) -> dict:
    return {"id": f"d{i}", "name": f"Doc {i}", "chunks": chunks or ["alpha beta gamma"], "overlaps": [0], "outline": []}


def test_validate_accepts_and_normalizes():
    docs = D.validate_documents([good_doc(1), good_doc(1), good_doc(2)])
    assert [d["id"] for d in docs] == ["d1", "d2"]  # duplicate id dropped
    assert D.validate_documents(None) == [] and D.validate_documents([]) == []


@pytest.mark.parametrize(
    "bad",
    [
        "nope",
        [5],
        [{"name": "x", "chunks": ["a"]}],  # no id
        [{"id": "x" * 65, "chunks": ["a"]}],
        [{"id": "a", "chunks": []}],
        [{"id": "a", "chunks": [1, 2]}],
    ],
)
def test_validate_rejects_bad_shapes_with_400(bad):
    with pytest.raises(D.DocumentError) as e:
        D.validate_documents(bad)
    assert e.value.status == 400


def test_validate_limits_give_413():
    with pytest.raises(D.DocumentError) as e:
        D.validate_documents([good_doc(i) for i in range(D.MAX_DOCS + 1)])
    assert e.value.status == 413
    with pytest.raises(D.DocumentError) as e:
        D.validate_documents([good_doc(1, ["x" * (D.MAX_CHUNK_CHARS + 1)])])
    assert e.value.status == 413
    big = ["y" * 2900] * 150  # 435k characters
    with pytest.raises(D.DocumentError) as e:
        D.validate_documents([{"id": "big", "name": "b", "chunks": big}])
    assert e.value.status == 413 and "too large" in e.value.message


def test_a_document_at_the_per_chat_limit_passes_validation_despite_overlaps():
    """The client counts cleaned characters; the server must count the same way, not the chunk text
    with each chunk's duplicated overlap prefix (about 11% more), or a chat the UI allows would be
    refused with a 413 on every message."""
    words, n = [], 0
    for i in range(10**6):
        w = f"w{i}"
        words.append(w)
        n += len(w) + 1
        if n >= 395_000:
            break
    built = D.build_document("big.txt", " ".join(words))
    assert built["chars"] <= D.MAX_DOC_CHARS
    raw_chars = sum(len(c) for c in built["chunks"])
    assert raw_chars > D.MAX_TOTAL_CHARS  # with overlaps the chunk text alone WOULD exceed the request cap
    docs = D.validate_documents(
        [{"id": built["doc_id"], "name": "big", "chunks": built["chunks"], "overlaps": built["overlaps"], "outline": []}]
    )
    assert len(docs) == 1


def test_two_documents_adding_up_to_the_limit_pass_too():
    halves = []
    for tag in ("a", "b"):
        words, n = [], 0
        for i in range(10**6):
            w = f"{tag}{i}"
            words.append(w)
            n += len(w) + 1
            if n >= 196_000:
                break
        halves.append(D.build_document(tag, " ".join(words)))
    D.validate_documents(
        [{"id": b["doc_id"], "name": b["name"], "chunks": b["chunks"], "overlaps": b["overlaps"], "outline": []} for b in halves]
    )


def test_validate_repairs_bad_overlaps_and_outline_entries():
    d = good_doc(1, ["abc def", "ghi jkl"])
    d["overlaps"] = [0, 99]  # 99 is longer than the chunk
    d["outline"] = [{"t": "ok", "c": 1}, {"t": "bad", "c": 9}, {"t": 3, "c": 0}, "junk"]
    out = D.validate_documents([d])[0]
    assert out["overlaps"] == [0, 0] and out["outline"] == [{"t": "ok", "c": 1}]


# --- question routing ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "q",
    [
        "Summarize this for me",
        "summarize everything",
        "Summarise the document",
        "What is this about?",
        "what is the document about",
        "Give me the main points",
        "explain this document",
        "what does this pdf say?",
        "What's in this file",
        "tl;dr",
        "Describe the attached report",
    ],
)
def test_overview_questions_are_recognized(q):
    assert D.is_overview_question(q)


@pytest.mark.parametrize(
    "q",
    ["What is the verification process?", "Who is responsible for notifications?", "Hi", "how many risks are listed", "what is a blue economy"],
)
def test_specific_questions_are_not_overviews(q):
    assert not D.is_overview_question(q)


def select(docs, q, prev="", budget=1100):
    return D.select_context(docs, q, prev_question=prev, budget_tokens=budget, count_tokens=count)


def test_overview_covers_start_middle_and_end():
    big = "\n\n".join(
        f"{n}. TOPIC {n}\n{wrap(f'Distinct marker{n} content about subject{n}. ' + 'filler words go here for volume. ' * 25)}"
        for n in range(1, 21)
    )
    docs = make_docs(big)
    r = select(docs, "Summarize this for me", budget=1100)
    assert r["mode"] == "overview" and count(r["text"]) <= 1100
    assert "marker1 " in r["text"] and "marker20 " in r["text"]
    mids = [n for n in range(8, 14) if f"marker{n} " in r["text"]]
    assert mids, "something from the middle of the document must be sampled"
    assert "TOPIC 1" in r["text"] and "excerpts" in r["text"]  # outline + the honesty note


def test_overview_of_a_short_document_includes_it_all():
    docs = make_docs("Tiny doc about bees.\n\nBees make honey.")
    r = select(docs, "summarize")
    assert "Bees make honey" in r["text"] and "Tiny doc about bees" in r["text"]


def test_specific_question_finds_the_ligature_text():
    docs = make_docs()
    assert "\ufb01" not in "".join(docs[0]["chunks"])
    r = select(docs, "How does verification work?")
    assert r["mode"] == "search" and "Verification" in r["text"] and "12. VERIFICATION" in r["text"]


def test_plural_and_verb_forms_match():
    docs = make_docs("The notifications are delivered by email.\n\nPayments are verified daily.")
    assert select(docs, "which notification channel")["mode"] == "search"
    assert "verified" in select(docs, "who verifies payments")["text"]


def test_best_hit_brings_its_neighbour():
    docs = make_docs()
    r = select(docs, "what are the main weather risks")
    assert r["selected"] >= 2
    assert "24. RISKS" in r["text"] and "12. VERIFICATION" in r["text"]  # best chunk plus the one before it (it is the last)


def test_adjacent_chunks_are_merged_without_their_duplicated_overlap():
    text = D.clean_text(sample_raw())
    chunks = D.chunk_text(text)
    doc = {"id": "x", "name": "x", "chunks": [c["text"] for c in chunks], "overlaps": [c["ov"] for c in chunks], "outline": []}
    norm = lambda t: " ".join(t.split())  # noqa: E731
    assert norm(D._join(doc, [1, 2])) == norm(text[chunks[1]["start"] : chunks[2]["end"]])
    assert "[…]" in D._join(doc, [0, 2])  # a gap is marked, not silently joined


def test_section_lookup_jumps_to_the_heading():
    docs = make_docs()
    r = select(docs, "what does section 12 say?")
    assert r["mode"] == "section" and "12. VERIFICATION" in r["text"] and "24. RISKS" not in r["text"]
    r2 = select(docs, "tell me about part 24")
    assert r2["mode"] == "section" and "24. RISKS" in r2["text"]


def test_section_number_does_not_match_a_longer_number():
    docs = make_docs()
    # "section 1" must find "1. INTRODUCTION", not "12. ..."
    r = select(docs, "section 1")
    assert r["mode"] == "section" and "1. INTRODUCTION" in r["text"] and "VERIFICATION" not in r["text"].split("1. INTRODUCTION")[0]


def test_unknown_section_falls_back_to_search():
    docs = make_docs()
    assert select(docs, "what does section 99 say about risks")["mode"] == "search"


def test_greetings_add_no_document_context():
    docs = make_docs()
    for q in ["Hi", "thanks!", "ok cool", "hello there"]:
        r = select(docs, q)
        assert r["mode"] == "none" and r["text"] == "", q


def test_nothing_matching_says_so():
    docs = make_docs()
    r = select(docs, "what is the capital of Mongolia")
    assert r["mode"] == "nomatch" and "no passage matched" in r["text"] and "Blue.pdf" in r["text"]


def test_followup_borrows_the_previous_question():
    docs = make_docs()
    alone = select(docs, "tell me more")
    assert alone["mode"] in ("none", "nomatch") or "weather" not in alone["text"]
    follow = select(docs, "tell me more about that", prev="what are the weather risks?")
    assert follow["mode"] == "followup" and "weather" in follow["text"]


def test_budget_is_respected_and_tiny_budget_gives_a_note():
    docs = make_docs()
    for q in ["Summarize this", "how does verification work", "section 12"]:
        r = select(docs, q, budget=200)
        assert count(r["text"]) <= 200 + 5, q
    r = select(docs, "how does verification work", budget=50)
    assert r["mode"] == "nobudget" and "no room" in r["text"]


def test_multiple_documents_are_all_searched_and_all_summarized():
    a = make_docs("Alpha document about volcanoes and lava flows.", "A.txt")[0]
    b = make_docs("Beta document about glaciers and ice sheets.", "B.txt")[0]
    docs = [a, b]
    r = select(docs, "tell me about glaciers")
    assert "[B.txt]" in r["text"] and "glaciers" in r["text"]
    o = select(docs, "summarize both")
    assert "volcanoes" in o["text"] and "glaciers" in o["text"] and "[A.txt]" in o["text"] and "[B.txt]" in o["text"]


def test_no_documents_means_no_context():
    r = select([], "summarize this")
    assert r["mode"] == "none" and r["text"] == ""


# --- extraction ------------------------------------------------------------------------------


def tiny_pdf(text: str) -> bytes:
    """A one-page PDF with real selectable text, built by hand."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def test_extract_a_real_pdf_end_to_end():
    raw, meta = D.extract_file("report.PDF", tiny_pdf("Verification of notifications is mandatory"))
    assert meta == {"pages": 1}
    built = D.build_document("report.pdf", raw, pages=meta["pages"])
    assert "Verification of notifications is mandatory" in " ".join(built["chunks"])


def test_extract_text_and_markdown_and_rejects_unknown_types():
    assert D.extract_file("a.txt", "héllo".encode())[0] == "héllo"
    assert D.extract_file("a.MD", b"# T")[0] == "# T"
    with pytest.raises(ValueError):
        D.extract_file("a.exe", b"x")


def test_garbage_pdf_raises_for_the_caller_to_turn_into_400():
    with pytest.raises(Exception):
        D.extract_file("bad.pdf", b"this is not a pdf")
