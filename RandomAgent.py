from __future__ import annotations
import random

from Agent import (
    Agent,
    CONTINUOUS_CLAIM,
    CONTINUOUS_RESEARCH,
    CONTINUOUS_RESEARCH_FINISH,
    CONTINUOUS_REVIEW,
)
from config import SIM
from Paper import (
    BAD_FAITH_REVIEW,
    GOOD_FAITH_REVIEW,
    REVIEW_PARADIGM_DISCRETE,
    Paper,
)


class RandomAgent(Agent):

    def __init__(
        self,
        intrinsic_talent: float,
        academic_capital: float = 0.0,
        paper_progress: float = 0.0,
        review_progress: float = 0.0,
        name: str | None = None,
        *,
        claim_probability: float = SIM.random_claim_probability,
        good_faith_probability: float = SIM.random_good_faith_probability,
    ):
        super().__init__(
            intrinsic_talent=intrinsic_talent,
            academic_capital=academic_capital,
            paper_progress=paper_progress,
            review_progress=review_progress,
            name=name,
        )
        self.claim_probability = claim_probability
        self.good_faith_probability = good_faith_probability

    def choose_marketplace_action(self) -> Paper | None:
        reviewable = [
            p
            for p in Agent.all_papers
            if self._can_review(p)
        ]
        if not reviewable:
            return None
        if random.random() >= self.claim_probability:
            return None
        return random.choice(reviewable)

    def choose_review_kind(self, paper: Paper) -> str:
        if random.random() < self.good_faith_probability:
            return GOOD_FAITH_REVIEW
        return BAD_FAITH_REVIEW

    def choose_work_action(self) -> tuple[str, Paper | None]:
        if self.active_review_paper is not None:
            return random.choice(
                [
                    ("peer_review", self.active_review_paper),
                    ("finish_review_write_paper", None),
                ]
            )
        return "write_paper", None

    def choose_continuous_action(self) -> tuple[str, Paper | None]:
        reviewable = [
            p
            for p in Agent.all_papers
            if self._can_review(p)
        ]
        if reviewable and random.random() < self.claim_probability:
            return (CONTINUOUS_CLAIM, random.choice(reviewable))

        if self.active_review_paper is not None:
            return random.choice([
                (CONTINUOUS_REVIEW, None),
                (CONTINUOUS_RESEARCH, None),
            ])
        options = [(CONTINUOUS_RESEARCH, None)]
        if not self.continuous_publish_by_threshold():
            options.append((CONTINUOUS_RESEARCH_FINISH, None))
        return random.choice(options)

    def _can_review(self, paper: Paper) -> bool:
        helper = getattr(paper, "can_start_review", None)
        if helper is not None:
            return bool(helper(self))
        return paper.author is not self


class ProbabilisticDiscreteAgent(RandomAgent):
    """Discrete-only baseline with a fixed probability over good/bad reviews."""

    requires_review_paradigm = REVIEW_PARADIGM_DISCRETE

    def __init__(
        self,
        intrinsic_talent: float,
        academic_capital: float = 0.0,
        paper_progress: float = 0.0,
        review_progress: float = 0.0,
        name: str | None = None,
        *,
        claim_probability: float = SIM.probabilistic_claim_probability,
        good_faith_probability: float = SIM.probabilistic_good_faith_probability,
    ):
        super().__init__(
            intrinsic_talent=intrinsic_talent,
            academic_capital=academic_capital,
            paper_progress=paper_progress,
            review_progress=review_progress,
            name=name,
            claim_probability=claim_probability,
            good_faith_probability=good_faith_probability,
        )
