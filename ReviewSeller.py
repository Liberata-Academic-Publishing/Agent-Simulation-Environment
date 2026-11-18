"""A quoting component, NOT an additional worker or capital account."""
from __future__ import annotations

import math


class ReviewSeller:
    def __init__(self, owner):
        self.owner = owner

    def quote_ticks(self, config):
        if self.owner.active_review_paper is not None:
            return None
        share = float(self.owner.review_ask_share(config))
        if not math.isfinite(share) or not 0 < share <= 1:
            raise ValueError("review ask must be finite and in (0, 1]")
        # Round asks UP: never promise a lower price than the seller requested.
        return max(1, math.ceil(share / config.market_share_tick - 1e-12))
