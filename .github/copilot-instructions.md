Project: institutional-grade analytics and research platform for Deriv Volatility Indices (synthetic, algorithmic random processes).
Facts that must shape all code:
- Synthetic indices have NO volume and NO order flow. Never implement RVOL or volume-based logic.
- Volatility indices are designed with constant volatility (R_75 = 75% annualised, 1HZ75V = 75% annualised at 1 tick per second). Market trades 24/7, UTC only.
- Any claimed trading edge must be tested against the null hypothesis that the price process is a driftless random walk.
Engineering standards:
- Python 3.11+, full type hints, pydantic models for all data contracts, pytest with >85% coverage, ruff + mypy strict.
- Structured logging (structlog), no print() in library code. Config via environment variables and a typed settings class. No secrets in code.
- All timestamps UTC epoch seconds internally. Deterministic outputs: every random process takes an explicit seed.
- Use only free data (Deriv public WebSocket API). Market-data access only; never request trade or payment API scopes.
- Every function that computes a statistic must have a unit test with a known-answer fixture.
