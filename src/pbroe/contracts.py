"""Reusable DataFrame contracts for point-in-time tables."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import pandas as pd
from pandas.api import types as ptypes


class ContractError(ValueError):
    """Raised when a table violates its declared contract."""


class ColumnKind(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    DATETIME = "datetime"


@dataclass(frozen=True)
class ValidationReport:
    table_name: str
    row_count: int
    column_count: int
    key_columns: tuple[str, ...]
    date_columns: tuple[str, ...]


@dataclass(frozen=True)
class TableContract:
    name: str
    required_columns: tuple[str, ...]
    key_columns: tuple[str, ...]
    date_columns: tuple[str, ...] = ()
    dtype_rules: Mapping[str, ColumnKind] | None = None

    def validate(self, frame: pd.DataFrame) -> ValidationReport:
        if not isinstance(frame, pd.DataFrame):
            raise ContractError(f"{self.name}: expected pandas DataFrame")

        missing = sorted(set(self.required_columns) - set(frame.columns))
        if missing:
            raise ContractError(f"{self.name}: missing required columns: {missing}")

        missing_keys = sorted(set(self.key_columns) - set(frame.columns))
        if missing_keys:
            raise ContractError(f"{self.name}: missing key columns: {missing_keys}")

        if self.key_columns:
            null_key_mask = frame.loc[:, list(self.key_columns)].isna().any(axis=1)
            if bool(null_key_mask.any()):
                raise ContractError(
                    f"{self.name}: null values in key columns on "
                    f"{int(null_key_mask.sum())} rows"
                )
            duplicate_mask = frame.duplicated(subset=list(self.key_columns), keep=False)
            if bool(duplicate_mask.any()):
                raise ContractError(
                    f"{self.name}: duplicate keys on {int(duplicate_mask.sum())} rows"
                )

        for column in self.date_columns:
            if column not in frame.columns:
                raise ContractError(f"{self.name}: missing date column: {column}")
            if not _valid_dates(frame[column]):
                raise ContractError(
                    f"{self.name}: {column} contains null or invalid YYYYMMDD dates"
                )

        for column, kind in (self.dtype_rules or {}).items():
            if column not in frame.columns:
                raise ContractError(f"{self.name}: missing typed column: {column}")
            if not _dtype_matches(frame[column], kind):
                raise ContractError(
                    f"{self.name}: {column} dtype {frame[column].dtype} "
                    f"does not satisfy {kind.value}"
                )

        return ValidationReport(
            table_name=self.name,
            row_count=len(frame),
            column_count=len(frame.columns),
            key_columns=self.key_columns,
            date_columns=self.date_columns,
        )


def _valid_dates(series: pd.Series) -> bool:
    if bool(series.isna().any()):
        return False
    if ptypes.is_datetime64_any_dtype(series.dtype):
        return True
    normalized = series.astype("string").str.strip()
    if not bool(normalized.str.fullmatch(r"\d{8}").all()):
        return False
    parsed = pd.to_datetime(normalized, format="%Y%m%d", errors="coerce")
    return not bool(parsed.isna().any())


def _dtype_matches(series: pd.Series, kind: ColumnKind) -> bool:
    if kind is ColumnKind.STRING:
        return ptypes.is_string_dtype(series.dtype) or ptypes.is_object_dtype(series.dtype)
    if kind is ColumnKind.INTEGER:
        return ptypes.is_integer_dtype(series.dtype)
    if kind is ColumnKind.NUMERIC:
        return ptypes.is_numeric_dtype(series.dtype) and not ptypes.is_bool_dtype(
            series.dtype
        )
    if kind is ColumnKind.BOOLEAN:
        return ptypes.is_bool_dtype(series.dtype)
    if kind is ColumnKind.DATETIME:
        return ptypes.is_datetime64_any_dtype(series.dtype)
    raise ContractError(f"unsupported column kind: {kind}")
