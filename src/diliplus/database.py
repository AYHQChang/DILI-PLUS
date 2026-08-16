"""Read-only access to the external medical DuckDB."""

from __future__ import annotations

import duckdb

from .config import Settings


def connect_source_database(settings: Settings):
    """Open the configured source database with an enforced read-only connection."""
    if settings.database_read_only is not True:
        raise ValueError("Source DuckDB writes are forbidden")
    return duckdb.connect(str(settings.database_path), read_only=True)

