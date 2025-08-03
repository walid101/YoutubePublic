from dataclasses import dataclass
from typing import TypedDict


@dataclass
class Timestamp:
    start: float
    end: float
    duration: float
