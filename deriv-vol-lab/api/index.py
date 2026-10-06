"""Vercel serverless entry point; serves snapshots without background workers."""

from deriv_vol_lab.api.app import app

__all__ = ["app"]
