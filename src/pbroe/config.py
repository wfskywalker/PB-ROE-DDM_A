"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


class ConfigError(ValueError):
    """Raised when configuration is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class PathConfig:
    asset_pricing_root: Path
    output_root: Path
    run_log_root: Path
    raw_data_root: Path


@dataclass(frozen=True)
class FactorConfig:
    broker_lookback_days: int
    min_brokers_per_year: int
    forecast_year_count: int
    min_industry_peer_count: int
    roe_history_years: int
    min_coe_growth_spread: float
    growth_buckets: tuple[float, ...]


@dataclass(frozen=True)
class BacktestConfig:
    history_start_date: int
    history_end_date: int
    group_count: int
    top_pool_first_group: int
    buy_commission: float
    sell_commission: float
    stamp_tax: float


@dataclass(frozen=True)
class ProjectConfig:
    paths: PathConfig
    factor: FactorConfig
    backtest: BacktestConfig

    @classmethod
    def load(cls, config_path: str | Path) -> "ProjectConfig":
        path = Path(config_path).expanduser().resolve()
        if not path.is_file():
            raise ConfigError(f"configuration file does not exist: {path}")

        with path.open("rb") as handle:
            raw = tomllib.load(handle)

        base = path.parent
        paths_raw = _required_section(raw, "paths")
        factor_raw = _required_section(raw, "factor")
        backtest_raw = _required_section(raw, "backtest")

        paths = PathConfig(
            asset_pricing_root=_resolve_path(
                base, _required_value(paths_raw, "asset_pricing_root")
            ),
            output_root=_resolve_path(base, _required_value(paths_raw, "output_root")),
            run_log_root=_resolve_path(base, _required_value(paths_raw, "run_log_root")),
            raw_data_root=_resolve_path(base, _required_value(paths_raw, "raw_data_root")),
        )
        factor = FactorConfig(
            broker_lookback_days=int(_required_value(factor_raw, "broker_lookback_days")),
            min_brokers_per_year=int(
                _required_value(factor_raw, "min_brokers_per_year")
            ),
            forecast_year_count=int(_required_value(factor_raw, "forecast_year_count")),
            min_industry_peer_count=int(
                _required_value(factor_raw, "min_industry_peer_count")
            ),
            roe_history_years=int(_required_value(factor_raw, "roe_history_years")),
            min_coe_growth_spread=float(
                _required_value(factor_raw, "min_coe_growth_spread")
            ),
            growth_buckets=tuple(
                float(value) for value in _required_value(factor_raw, "growth_buckets")
            ),
        )
        backtest = BacktestConfig(
            history_start_date=int(_required_value(backtest_raw, "history_start_date")),
            history_end_date=int(_required_value(backtest_raw, "history_end_date")),
            group_count=int(_required_value(backtest_raw, "group_count")),
            top_pool_first_group=int(
                _required_value(backtest_raw, "top_pool_first_group")
            ),
            buy_commission=float(_required_value(backtest_raw, "buy_commission")),
            sell_commission=float(_required_value(backtest_raw, "sell_commission")),
            stamp_tax=float(_required_value(backtest_raw, "stamp_tax")),
        )

        config = cls(paths=paths, factor=factor, backtest=backtest)
        config.validate()
        return config

    def validate(self) -> None:
        root = self.paths.asset_pricing_root
        if not root.is_dir():
            raise ConfigError(f"asset_pricing_root is not a directory: {root}")
        if not (root / "data").is_dir():
            raise ConfigError(f"asset_pricing_root has no data directory: {root}")
        for label, path in (
            ("output_root", self.paths.output_root),
            ("run_log_root", self.paths.run_log_root),
            ("raw_data_root", self.paths.raw_data_root),
        ):
            if not path.is_dir():
                raise ConfigError(f"{label} is not an existing directory: {path}")

        factor = self.factor
        if factor.broker_lookback_days <= 0:
            raise ConfigError("broker_lookback_days must be positive")
        if factor.min_brokers_per_year < 2:
            raise ConfigError("min_brokers_per_year must be at least 2")
        if factor.forecast_year_count != 3:
            raise ConfigError("V1 forecast_year_count must equal 3")
        if factor.roe_history_years != 3:
            raise ConfigError("V1 roe_history_years must equal 3")
        if factor.min_industry_peer_count <= 0:
            raise ConfigError("min_industry_peer_count must be positive")
        if factor.min_coe_growth_spread <= 0:
            raise ConfigError("min_coe_growth_spread must be positive")
        if factor.growth_buckets != (0.0, 0.01, 0.02, 0.03, 0.04):
            raise ConfigError("V1 growth_buckets must be 0%, 1%, 2%, 3%, and 4%")

        backtest = self.backtest
        if backtest.history_start_date > backtest.history_end_date:
            raise ConfigError("history_start_date must not exceed history_end_date")
        if backtest.group_count != 10:
            raise ConfigError("V1 group_count must equal 10")
        if backtest.top_pool_first_group != 8:
            raise ConfigError("V1 top 30% must start at group 8")
        for name, value in (
            ("buy_commission", backtest.buy_commission),
            ("sell_commission", backtest.sell_commission),
            ("stamp_tax", backtest.stamp_tax),
        ):
            if not 0 <= value < 0.1:
                raise ConfigError(f"{name} must be in [0, 0.1)")


def _required_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid [{name}] section")
    return value


def _required_value(section: dict[str, Any], name: str) -> Any:
    if name not in section:
        raise ConfigError(f"missing required configuration key: {name}")
    return section[name]


def _resolve_path(base: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()
