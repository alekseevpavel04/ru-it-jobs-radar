"""The VRAM and step-time guard of a training run, kept free of torch so it is unit tested.

On Windows the driver may move CUDA memory that does not fit into shared system RAM instead of
failing. The run then goes on many times slower, so the guard watches two signs: torch's reserved
memory above the budget, and the time per token rising far above that of the first steps. Time is
taken per token seen, so a micro-batch of long examples does not look like a slowdown.
"""

import statistics
from dataclasses import dataclass, field


@dataclass
class StepGuard:
    limit_mib: float | None  # None: no VRAM check (CPU dry run)
    factor: float
    reference_steps: int
    patience: int
    seen: list[float] = field(default_factory=list)
    reference: float | None = None
    slow_in_a_row: int = 0
    peak_reserved_mib: float = 0.0

    def check(self, seconds: float, tokens: int, reserved_mib: float) -> str | None:
        """Record one optimizer step; the reason to stop, or None."""
        self.peak_reserved_mib = max(self.peak_reserved_mib, reserved_mib)
        if self.limit_mib is not None and reserved_mib > self.limit_mib:
            return f"reserved VRAM {reserved_mib:.0f} MiB is above the limit {self.limit_mib:.0f} MiB"
        cost = seconds / tokens * 1000 if tokens > 0 else seconds
        if self.reference is None:
            self.seen.append(cost)
            # the first step pays for compilation and allocator warm-up, the next ones set the pace
            if len(self.seen) == self.reference_steps + 1:
                self.reference = statistics.median(self.seen[1:])
            return None
        self.slow_in_a_row = self.slow_in_a_row + 1 if cost > self.factor * self.reference else 0
        if self.slow_in_a_row >= self.patience:
            unit = "s per 1000 tokens" if tokens > 0 else "s per step"
            return (
                f"{self.patience} steps in a row at {cost:.2f} {unit}, more than {self.factor:g} times "
                f"the reference {self.reference:.2f}: memory is probably spilling into system RAM"
            )
        return None


def vram_limit_mib(total_mib: float, used_at_start_mib: float, margin_mib: float) -> float:
    """What torch may reserve: the card minus what was in use before the model loaded (other
    processes, the desktop, this process's CUDA context) minus a margin."""
    return total_mib - used_at_start_mib - margin_mib
