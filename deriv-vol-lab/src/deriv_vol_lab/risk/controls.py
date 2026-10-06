"""UTC-day loss limits and consecutive-loss circuit breakers."""

from __future__ import annotations

from datetime import UTC, date, datetime
from math import isfinite

from pydantic import BaseModel, ConfigDict


class CircuitBreakerStatus(BaseModel):
    """Snapshot of realized losses and whether new trades are permitted."""

    model_config = ConfigDict(frozen=True)

    utc_day: date
    daily_net_pnl: float
    consecutive_losses: int
    halted: bool
    reasons: tuple[str, ...]


class LossCircuitBreaker:
    """Track daily net losses and loss streaks using UTC epoch-second timestamps.

    Daily PnL resets at UTC midnight. A consecutive-loss halt remains active
    across days until :meth:`reset` is explicitly called.
    """

    def __init__(self, *, daily_loss_limit: float, max_consecutive_losses: int) -> None:
        if not isfinite(daily_loss_limit) or daily_loss_limit <= 0:
            raise ValueError("daily_loss_limit must be finite and greater than zero")
        if (
            isinstance(max_consecutive_losses, bool)
            or not isinstance(max_consecutive_losses, int)
            or max_consecutive_losses <= 0
        ):
            raise ValueError("max_consecutive_losses must be a positive integer")
        self.daily_loss_limit = daily_loss_limit
        self.max_consecutive_losses = max_consecutive_losses
        self._utc_day: date | None = None
        self._daily_net_pnl = 0.0
        self._consecutive_losses = 0
        self._last_epoch: int | None = None

    def can_trade(self, epoch: int) -> bool:
        """Return whether a trade is permitted at this UTC epoch second."""
        self._advance_day(epoch)
        return not self._reasons()

    def record_trade(self, epoch: int, net_pnl: float) -> CircuitBreakerStatus:
        """Record settled net PnL and return the resulting breaker state."""
        if not isfinite(net_pnl):
            raise ValueError("net_pnl must be finite")
        if not self.can_trade(epoch):
            raise RuntimeError("Cannot record a trade while the circuit breaker is halted")
        self._daily_net_pnl += net_pnl
        self._consecutive_losses = self._consecutive_losses + 1 if net_pnl < 0 else 0
        return self.status(epoch)

    def status(self, epoch: int) -> CircuitBreakerStatus:
        """Return a status snapshot, advancing/resetting the UTC-day state."""
        self._advance_day(epoch)
        reasons = self._reasons()
        return CircuitBreakerStatus(
            utc_day=self._utc_day or datetime.fromtimestamp(epoch, UTC).date(),
            daily_net_pnl=self._daily_net_pnl,
            consecutive_losses=self._consecutive_losses,
            halted=bool(reasons),
            reasons=tuple(reasons),
        )

    def reset(self) -> None:
        """Manually clear accumulated losses and the consecutive-loss halt."""
        self._utc_day = None
        self._daily_net_pnl = 0.0
        self._consecutive_losses = 0
        self._last_epoch = None

    def _advance_day(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise ValueError("epoch must be an integer UTC epoch second")
        if self._last_epoch is not None and epoch < self._last_epoch:
            raise ValueError("epochs must be non-decreasing")
        current_day = datetime.fromtimestamp(epoch, UTC).date()
        if current_day != self._utc_day:
            self._utc_day = current_day
            self._daily_net_pnl = 0.0
        self._last_epoch = epoch

    def _reasons(self) -> list[str]:
        reasons: list[str] = []
        if self._daily_net_pnl <= -self.daily_loss_limit:
            reasons.append("daily-loss-limit")
        if self._consecutive_losses >= self.max_consecutive_losses:
            reasons.append("consecutive-loss-limit")
        return reasons
