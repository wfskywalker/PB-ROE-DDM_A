from pathlib import Path
import tomllib

import pytest

from pbroe.config import ConfigError, ProjectConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_config_loads_and_resolves_paths(tmp_path):
    upstream = tmp_path / "vendor_data"
    (upstream / "data").mkdir(parents=True)
    for name in ("artifacts", "logs/runs", "data/raw"):
        (tmp_path / name).mkdir(parents=True)
    example = tomllib.loads((PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8"))
    assert example["factor"]["min_brokers_per_year"] == 2
    template = (PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        template.replace("/absolute/path/to/your/point-in-time-A-share-data", str(upstream)),
        encoding="utf-8",
    )
    config = ProjectConfig.load(config_path)
    assert config.paths.asset_pricing_root == upstream
    assert config.factor.growth_buckets == (0.0, 0.01, 0.02, 0.03, 0.04)
    assert config.backtest.top_pool_first_group == 8


def test_missing_config_is_rejected():
    with pytest.raises(ConfigError, match="does not exist"):
        ProjectConfig.load(PROJECT_ROOT / "does_not_exist.toml")


def test_invalid_required_path_is_rejected(tmp_path):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
[paths]
asset_pricing_root = "/definitely/not/a/real/path"
output_root = "."
run_log_root = "."
raw_data_root = "."
[factor]
broker_lookback_days = 180
min_brokers_per_year = 2
forecast_year_count = 3
min_industry_peer_count = 10
roe_history_years = 3
min_coe_growth_spread = 0.02
growth_buckets = [0.00, 0.01, 0.02, 0.03, 0.04]
[backtest]
history_start_date = 20120330
history_end_date = 20260630
group_count = 10
top_pool_first_group = 8
buy_commission = 0.0001
sell_commission = 0.0001
stamp_tax = 0.0005
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="not a directory"):
        ProjectConfig.load(config_path)
