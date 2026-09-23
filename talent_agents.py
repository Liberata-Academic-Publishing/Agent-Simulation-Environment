"""Equal-sized talent cohorts with the same fixed write/review strategy."""
from __future__ import annotations

from Agent import Agent, CONTINUOUS_CLAIM, CONTINUOUS_RESEARCH, CONTINUOUS_REVIEW
from Paper import GOOD_FAITH_REVIEW, fixed_review_effort
from config import SIM


class TalentAgent(Agent):
    """Alternate publishing a manuscript and completing one good-faith review.

    If no review is available, keep writing. No learning or economic forecasts
    are used, making this a control strategy for comparing talents.
    """

    def __init__(self, quality: float, rate: float, name: str):
        super().__init__(quality, name=name)
        self.configure_talents(quality, rate)
        self.configure_continuous_publishing("threshold")
        self.wants_review = False

    def publish_paper(self):
        paper = super().publish_paper()
        self.wants_review = True
        return paper

    def finish_research_paper(self):
        paper = super().finish_research_paper()
        self.wants_review = True
        return paper

    def choose_marketplace_action(self):
        if self.active_review_paper is not None or not self.wants_review:
            return None
        return next((p for p in self.all_papers if p.can_start_review(self)), None)

    def claim_review(self, paper, review_kind=None):
        result = super().claim_review(paper, review_kind)
        self.wants_review = False
        return result

    def choose_work_action(self):
        if self.active_review_paper is not None:
            if self.active_review_effort < fixed_review_effort(GOOD_FAITH_REVIEW):
                return "peer_review", None
            return "finish_review_write_paper", None
        return "write_paper", None

    def choose_continuous_action(self):
        paper = self.choose_marketplace_action()
        if paper is not None:
            return CONTINUOUS_CLAIM, paper
        action, _ = self.choose_work_action()
        return (CONTINUOUS_REVIEW if action == "peer_review" else CONTINUOUS_RESEARCH), None


def build_talent_cohorts(count_per_group: int = SIM.talent_agents_per_group):
    """Return 20 agents in each of four cohorts by default (80 total)."""
    if isinstance(count_per_group, bool) or not isinstance(count_per_group, int) or count_per_group < 1:
        raise ValueError("count_per_group must be a positive integer")
    groups = {}
    for quality_label, quality in (("low", SIM.talent_low), ("high", SIM.talent_high)):
        for rate_label, rate in (("low", SIM.talent_low), ("high", SIM.talent_high)):
            label = f"quality_{quality_label}_rate_{rate_label}"
            groups[label] = [TalentAgent(quality, rate, f"{label}_{i}") for i in range(count_per_group)]
    return groups
