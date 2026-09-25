import pandas as pd
import pytest

from pbroe.contracts import ColumnKind, ContractError, TableContract


@pytest.fixture
def price_contract():
    return TableContract(
        name="prices",
        required_columns=("trade_date", "ts_code", "close"),
        key_columns=("trade_date", "ts_code"),
        date_columns=("trade_date",),
        dtype_rules={
            "trade_date": ColumnKind.INTEGER,
            "ts_code": ColumnKind.STRING,
            "close": ColumnKind.NUMERIC,
        },
    )


@pytest.fixture
def valid_prices():
    return pd.DataFrame(
        {
            "trade_date": pd.Series([20260130, 20260227], dtype="int64"),
            "ts_code": pd.Series(["000001.SZ", "000001.SZ"], dtype="string"),
            "close": [10.0, 11.0],
        }
    )


def test_valid_frame_returns_report(price_contract, valid_prices):
    report = price_contract.validate(valid_prices)
    assert report.row_count == 2
    assert report.key_columns == ("trade_date", "ts_code")


def test_missing_column_rejected(price_contract, valid_prices):
    with pytest.raises(ContractError, match="missing required columns"):
        price_contract.validate(valid_prices.drop(columns="close"))


def test_duplicate_key_rejected(price_contract, valid_prices):
    duplicated = pd.concat([valid_prices, valid_prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(ContractError, match="duplicate keys"):
        price_contract.validate(duplicated)


def test_null_key_rejected(price_contract, valid_prices):
    invalid = valid_prices.copy()
    invalid.loc[0, "ts_code"] = pd.NA
    with pytest.raises(ContractError, match="null values"):
        price_contract.validate(invalid)


@pytest.mark.parametrize(
    ("bad_date", "expected_error"),
    [
        (20260230, "invalid YYYYMMDD"),
        (20261301, "invalid YYYYMMDD"),
        ("2026-01-30", "invalid YYYYMMDD"),
        (None, "null values in key columns"),
    ],
)
def test_invalid_date_rejected(
    price_contract, valid_prices, bad_date, expected_error
):
    invalid = valid_prices.copy()
    invalid["trade_date"] = invalid["trade_date"].astype("object")
    invalid.loc[0, "trade_date"] = bad_date
    with pytest.raises(ContractError, match=expected_error):
        price_contract.validate(invalid)


def test_wrong_dtype_rejected(price_contract, valid_prices):
    invalid = valid_prices.copy()
    invalid["close"] = pd.Series(["10", "11"], dtype="string")
    with pytest.raises(ContractError, match="does not satisfy numeric"):
        price_contract.validate(invalid)
