from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from ..structures import FiniteMagma
from ..witnesses import Witness
from ..theories.groups import associativity_witness, identity_witness, inverse_witness, commutativity_witness


@dataclass
class HornGroupOpponent:
    """Bounded Horn opponent for finite group theory.

    In this P0 version the opponent checks full finite tables, but the public
    interface already carries depth and budget. Later versions should sample
    instances according to k,b and return the first witness found.
    """

    k: int = 2
    budget: int = 100
    seed: int = 0
    include_commutativity: bool = False

    def challenge(self, A: FiniteMagma) -> List[Witness]:
        rng = random.Random(self.seed)
        checkers = [associativity_witness, identity_witness, inverse_witness]
        if self.include_commutativity:
            checkers.append(commutativity_witness)
        rng.shuffle(checkers)
        witnesses: List[Witness] = []
        for checker in checkers[: max(1, min(len(checkers), self.budget))]:
            w = checker(A)
            if w is not None:
                witnesses.append(w)
        return witnesses
