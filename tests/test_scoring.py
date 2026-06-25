"""Tests for scoring.py — scorecard structure and aggregate metrics."""

import pytest

from rag_shootout.scoring import (
    DimensionScore,
    QuestionScore,
    compute_summary,
    make_empty_scorecard,
    scores_to_dataframe,
)


# ─── DimensionScore ───────────────────────────────────────────────────────────


class TestDimensionScore:
    def test_default_is_none(self):
        ds = DimensionScore()
        assert ds.accuracy is None
        assert ds.completeness is None
        assert ds.faithfulness is None

    def test_is_complete_all_set(self):
        ds = DimensionScore(accuracy=4, completeness=3, faithfulness=5)
        assert ds.is_complete()

    def test_is_complete_partial(self):
        ds = DimensionScore(accuracy=4)
        assert not ds.is_complete()

    def test_total_none_when_incomplete(self):
        ds = DimensionScore(accuracy=4, completeness=3)
        assert ds.total() is None

    def test_total_sums_dimensions(self):
        ds = DimensionScore(accuracy=4, completeness=3, faithfulness=5)
        assert ds.total() == 12.0

    def test_validate_out_of_range(self):
        ds = DimensionScore(accuracy=6, completeness=3, faithfulness=3)
        with pytest.raises(ValueError):
            ds.validate()

    def test_validate_valid(self):
        ds = DimensionScore(accuracy=1, completeness=5, faithfulness=3)
        ds.validate()  # should not raise


# ─── QuestionScore ────────────────────────────────────────────────────────────


class TestQuestionScore:
    def _scored_question(self, v=(4, 4, 4), p=(3, 3, 3)) -> QuestionScore:
        q = QuestionScore(question_id=1)
        q.vector = DimensionScore(*v)
        q.pageindex = DimensionScore(*p)
        return q

    def test_winner_vector(self):
        q = self._scored_question(v=(5, 5, 5), p=(3, 3, 3))
        assert q.winner() == "Vector RAG"

    def test_winner_pageindex(self):
        q = self._scored_question(v=(3, 3, 3), p=(5, 5, 5))
        assert q.winner() == "PageIndex"

    def test_winner_tie(self):
        q = self._scored_question(v=(4, 4, 4), p=(4, 4, 4))
        assert q.winner() == "Tie"

    def test_winner_unscored(self):
        q = QuestionScore(question_id=1)
        assert q.winner() == "not scored yet"


# ─── make_empty_scorecard ─────────────────────────────────────────────────────


class TestMakeEmptyScorecard:
    def test_length(self):
        sc = make_empty_scorecard(n_questions=10)
        assert len(sc) == 10

    def test_ids_one_indexed(self):
        sc = make_empty_scorecard(n_questions=5)
        ids = [q.question_id for q in sc]
        assert ids == [1, 2, 3, 4, 5]

    def test_all_unscored(self):
        sc = make_empty_scorecard(n_questions=3)
        for q in sc:
            assert not q.vector.is_complete()
            assert not q.pageindex.is_complete()


# ─── compute_summary ──────────────────────────────────────────────────────────


class TestComputeSummary:
    def _full_scorecard(self) -> list[QuestionScore]:
        sc = make_empty_scorecard(n_questions=4)
        # q1: vector wins
        sc[0].vector = DimensionScore(5, 5, 5)
        sc[0].pageindex = DimensionScore(3, 3, 3)
        # q2: pageindex wins
        sc[1].vector = DimensionScore(3, 3, 3)
        sc[1].pageindex = DimensionScore(5, 5, 5)
        # q3: tie
        sc[2].vector = DimensionScore(4, 4, 4)
        sc[2].pageindex = DimensionScore(4, 4, 4)
        # q4: partially scored — should be excluded
        sc[3].vector = DimensionScore(5, 5, 5)
        # pageindex left unscored
        return sc

    def test_error_when_nothing_scored(self):
        sc = make_empty_scorecard()
        result = compute_summary(sc)
        assert "error" in result

    def test_n_scored_excludes_partial(self):
        sc = self._full_scorecard()
        result = compute_summary(sc)
        assert result["n_scored"] == 3  # q4 excluded

    def test_win_counts(self):
        sc = self._full_scorecard()
        result = compute_summary(sc)
        assert result["wins"]["Vector RAG"] == 1
        assert result["wins"]["PageIndex"] == 1
        assert result["wins"]["Tie"] == 1

    def test_total_score_vector(self):
        sc = self._full_scorecard()
        result = compute_summary(sc)
        # q1: 15, q2: 9, q3: 12 — total: 36
        assert result["vector"]["total"] == 36

    def test_total_score_pageindex(self):
        sc = self._full_scorecard()
        result = compute_summary(sc)
        # q1: 9, q2: 15, q3: 12 — total: 36
        assert result["pageindex"]["total"] == 36


# ─── scores_to_dataframe ──────────────────────────────────────────────────────


class TestScoresToDataframe:
    def test_returns_dataframe_shape(self):
        sc = make_empty_scorecard(n_questions=10)
        df = scores_to_dataframe(sc)
        assert len(df) == 10

    def test_columns_present(self):
        sc = make_empty_scorecard(n_questions=2)
        df = scores_to_dataframe(sc)
        for col in ["vector_accuracy", "pageindex_accuracy", "winner"]:
            assert col in df.columns

    def test_unscored_winner(self):
        sc = make_empty_scorecard(n_questions=1)
        df = scores_to_dataframe(sc)
        assert df["winner"].iloc[0] == "not scored yet"
