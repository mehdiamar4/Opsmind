import json
from typing import Optional

import pytest

from app.embeddings import EMBEDDING_DIMENSIONS
from app.retrieval import RetrievedChunk
from evals import run as eval_run
from evals.run import DEFAULT_QUESTIONS, evaluate_cases, load_cases, save_report


def case(case_id: str, document_id: int) -> dict:
    return {
        "id": case_id,
        "question": f"Question {case_id}?",
        "expected_document_id": document_id,
        "expected_fact": "An example fact",
        "expected_section": "Example Section",
        "category": "fact_lookup",
    }


def chunk(
    document_id: int,
    distance: float = 0.1,
    section_title: Optional[str] = "Example Section",
) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=document_id,
        filename=f"document-{document_id}.pdf",
        page_number=1,
        section_title=section_title,
        chunk_index=0,
        content="Example text",
        distance=distance,
    )


def embed(texts):
    return [
        [float(index) / 10] + [0.0] * (EMBEDDING_DIMENSIONS - 1)
        for index, _ in enumerate(texts)
    ]


def answer(question: str, context: str) -> str:
    return "AN EXAMPLE FACT!"


def test_expected_document_in_top_five_is_a_hit() -> None:
    questions = case("match", 2)

    def embedder(texts):
        assert texts == ["Question match?"]
        return [[0.1] + [0.0] * (EMBEDDING_DIMENSIONS - 1)]

    def retriever(vector, *, top_k):
        assert vector[0] == 0.1
        assert top_k == 5
        return [chunk(1, section_title="Other Section"), chunk(2, 0.2)]

    report = evaluate_cases(
        [questions], embedder=embedder, retriever=retriever, answerer=answer
    )

    assert report["recall_at_5"] == 1.0
    assert report["recall_at_1"] == 0.0
    assert report["section_at_1"] == 0.0
    assert report["results"][0]["hit"] is True
    assert report["results"][0]["hit_at_1"] is False
    assert report["results"][0]["section_at_1"] is False
    assert report["results"][0]["answer"] == "AN EXAMPLE FACT!"
    assert report["results"][0]["fact_match"] is True
    assert [
        source["document_id"] for source in report["results"][0]["retrieved_sources"]
    ] == [1, 2]


def test_expected_document_beyond_top_five_is_a_miss() -> None:
    report = evaluate_cases(
        [case("miss", 9)],
        embedder=lambda texts: [[0.1] + [0.0] * (EMBEDDING_DIMENSIONS - 1)],
        retriever=lambda vector, *, top_k: [chunk(1)] * 5 + [chunk(9)],
        answerer=answer,
    )

    assert report["recall_at_5"] == 0.0
    assert report["recall_at_1"] == 0.0
    assert report["section_at_1"] == 1.0
    assert report["results"][0]["hit"] is False
    assert report["results"][0]["hit_at_1"] is False
    assert report["results"][0]["section_at_1"] is True
    assert len(report["results"][0]["retrieved_sources"]) == 5


def test_recall_at_five_and_json_output(tmp_path) -> None:
    questions = [case("one", 1), case("two", 2), case("three", 3)]
    question_indices = {item["question"]: index for index, item in enumerate(questions)}

    def indexed_embedder(texts):
        index = question_indices[texts[0]]
        return [[float(index)] + [0.0] * (EMBEDDING_DIMENSIONS - 1)]

    def retriever(vector, *, top_k):
        results = {
            0: [chunk(8, section_title="Other Section"), chunk(1)],
            1: [chunk(8)],
            2: [chunk(3)],
        }
        return results[vector[0]]

    report = evaluate_cases(
        questions,
        embedder=indexed_embedder,
        retriever=retriever,
        answerer=lambda question, context: (
            "an example fact." if "two" not in question else "No answer here."
        ),
    )
    output = tmp_path / "results.json"
    save_report(report, output)

    assert report["metric"] == "Recall@5"
    assert (report["hits_at_1"], report["recall_at_1"]) == (1, pytest.approx(1 / 3))
    assert (report["hits"], report["total"]) == (2, 3)
    assert report["recall_at_5"] == pytest.approx(2 / 3)
    assert report["section_at_1"] == pytest.approx(2 / 3)
    assert report["fact_match"] == pytest.approx(2 / 3)
    assert report["fact_matches"] == 2
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["results"][0]["retrieved_sources"][0]["rank"] == 1
    assert report["results"][0]["expected_fact"] == "An example fact"
    assert report["results"][0]["answer"] == "an example fact."
    assert report["results"][1]["fact_match"] is False


def test_fact_match_normalizes_case_punctuation_and_whitespace() -> None:
    question = case("fact-normalization", 7)
    question["expected_fact"] = "EUR 160 per night"
    report = evaluate_cases(
        [question],
        embedder=embed,
        retriever=lambda vector, *, top_k: [chunk(7)],
        answerer=lambda question, context: "Eligible amount: eur, 160 per NIGHT.",
    )

    assert report["fact_matches"] == 1
    assert report["fact_match"] == 1.0
    assert report["results"][0]["fact_match"] is True


def test_optional_llm_judge_records_result_reason_and_accuracy(tmp_path) -> None:
    questions = [case("paraphrase", 1), case("unsupported", 2)]
    question_indices = {item["question"]: index for index, item in enumerate(questions)}
    judged_inputs = []

    def fake_judge(question, expected_fact, generated_answer, retrieved_context):
        judged_inputs.append(
            (question, expected_fact, generated_answer, retrieved_context)
        )
        correct = question.endswith("paraphrase?")
        return {
            "correct": correct,
            "reason": "Accurate paraphrase." if correct else "Adds unsupported detail.",
        }

    def indexed_embedder(texts):
        index = question_indices[texts[0]]
        return [[float(index)] + [0.0] * (EMBEDDING_DIMENSIONS - 1)]

    report = evaluate_cases(
        questions,
        embedder=indexed_embedder,
        retriever=lambda vector, *, top_k: [chunk(int(vector[0]) + 1)],
        answerer=lambda question, context: "Answer text.",
        judge=fake_judge,
    )
    output = tmp_path / "results.json"
    save_report(report, output)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert [entry[:3] for entry in judged_inputs] == [
        (question["question"], question["expected_fact"], "Answer text.")
        for question in questions
    ]
    assert all("Example text" in entry[3] for entry in judged_inputs)
    assert "document_id=1" in judged_inputs[0][3]
    assert "document_id=2" in judged_inputs[1][3]
    assert report["recall_at_1"] == 1.0
    assert report["recall_at_5"] == 1.0
    assert report["section_at_1"] == 1.0
    assert report["fact_match"] == 0.0
    assert report["llm_judge_correct"] == 1
    assert report["llm_judge_accuracy"] == 0.5
    assert saved["results"][0]["llm_judge"] == {
        "correct": True,
        "reason": "Accurate paraphrase.",
    }
    assert saved["results"][1]["llm_judge"]["reason"] == "Adds unsupported detail."


def test_judge_answer_requests_structured_openai_output(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Response:
                output_text = '{"correct":true,"reason":"Accurate paraphrase."}'

            return Response()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_options"] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setattr(eval_run, "OpenAI", FakeOpenAI)
    result = eval_run.judge_answer("Q?", "expected", "answer", "source context")

    assert result == {"correct": True, "reason": "Accurate paraphrase."}
    assert captured["model"] == "test-model"
    assert captured["store"] is False
    assert captured["text"]["format"]["type"] == "json_schema"
    assert captured["text"]["format"]["schema"] == eval_run.JUDGE_SCHEMA
    assert json.loads(captured["input"]) == {
        "question": "Q?",
        "expected_fact": "expected",
        "generated_answer": "answer",
        "retrieved_context": "source context",
    }


def test_section_at_one_requires_exact_expected_section() -> None:
    question = case("section-match", 7)
    question["expected_section"] = "2. Initial Response Targets"

    report = evaluate_cases(
        [question],
        embedder=lambda texts: [[0.1] + [0.0] * (EMBEDDING_DIMENSIONS - 1)],
        retriever=lambda vector, *, top_k: [
            chunk(7, section_title="2. Initial Response Targets")
        ],
        answerer=answer,
    )

    assert report["section_hits_at_1"] == 1
    assert report["section_at_1"] == 1.0
    assert report["results"][0]["section_at_1"] is True


def test_sample_question_file_has_required_fields() -> None:
    questions = load_cases(DEFAULT_QUESTIONS)

    assert len(questions) == 10
    assert {question["id"] for question in questions} == {
        "hr-vacation-days",
        "hr-security-training",
        "hr-probation",
        "it-p1-response",
        "it-error-e47",
        "it-p1-updates",
        "finance-hotel-limit",
        "finance-receipt",
        "cloud-cutover",
        "cloud-rollback",
    }


def test_cli_prints_results_and_saves_json(tmp_path, monkeypatch, capsys) -> None:
    output = tmp_path / "results.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evals.run",
            "--questions",
            str(DEFAULT_QUESTIONS),
            "--output",
            str(output),
            "--llm-judge",
        ],
    )

    def mocked_evaluation(cases, *, judge=None):
        expected_facts = {case["question"]: case["expected_fact"] for case in cases}
        case_indices = {case["question"]: index for index, case in enumerate(cases)}
        return evaluate_cases(
            cases,
            embedder=lambda texts: [
                [float(case_indices[texts[0]])]
                + [0.0] * (EMBEDDING_DIMENSIONS - 1)
            ],
            retriever=lambda vector, *, top_k: [
                chunk(
                    cases[int(vector[0])]["expected_document_id"],
                    section_title=cases[int(vector[0])]["expected_section"],
                )
            ],
            answerer=lambda question, context: expected_facts[question],
            judge=judge,
        )

    monkeypatch.setattr(
        eval_run,
        "judge_answer",
        lambda question, expected_fact, answer, retrieved_context: {
            "correct": True,
            "reason": "Mocked semantic match.",
        },
    )
    monkeypatch.setattr(eval_run, "evaluate_cases", mocked_evaluation)
    eval_run.main()

    printed = capsys.readouterr().out
    assert "hr-vacation-days: Recall@1=HIT Recall@5=HIT Section@1=HIT" in printed
    assert "Recall@1: 10/10 (100.0%)" in printed
    assert "Recall@5: 10/10 (100.0%)" in printed
    assert "Section@1: 10/10 (100.0%)" in printed
    assert "FactMatch: 10/10 (100.0%)" in printed
    assert "LLMJudge accuracy: 10/10 (100.0%)" in printed
    saved_report = json.loads(output.read_text(encoding="utf-8"))
    assert saved_report["hits_at_1"] == 10
    assert saved_report["hits"] == 10
    assert saved_report["section_hits_at_1"] == 10
    assert saved_report["fact_matches"] == 10
    assert saved_report["llm_judge_correct"] == 10
    assert saved_report["results"][0]["llm_judge"]["reason"] == "Mocked semantic match."
