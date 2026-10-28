from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from attribution import io
from attribution.schemas import (
    BENCHMARK,
    HOLDINGS,
    PRICES,
    REGIME_SERIES,
    SECTORS,
    SECURITY_MASTER,
    Policy,
    SchemaValidationError,
    check_holdings_against_master,
    is_valid_isin,
)


def _policy_kwargs() -> dict:
    neutral = {s: 1.0 / 15 for s in SECTORS}
    neutral = {s: round(w, 6) for s, w in neutral.items()}
    neutral[SECTORS[0]] += 1.0 - sum(neutral.values())
    neutral["Cash"] = 0.0
    return {
        "neutral_weights": neutral,
        "bands": {k: (max(0.0, v - 0.05), v + 0.05) for k, v in neutral.items()},
        "derisk_mode": "cash",
        "review_date": date(2026, 9, 30),
        "benchmark_name": "BSE 500",
    }


# --- enumerations ---------------------------------------------------------


def test_fifteen_unique_sectors() -> None:
    assert len(SECTORS) == len(set(SECTORS)) == 15


# --- holdings -------------------------------------------------------------


def test_valid_holdings_pass_and_are_typed(holdings_frame: pd.DataFrame) -> None:
    out = HOLDINGS.validate(holdings_frame)
    assert str(out["date"].dtype) == "datetime64[ns]"
    assert out["date"].dt.tz is None
    assert out["weight"].dtype == float


def test_weights_not_summing_to_one_names_date_and_rows(holdings_frame: pd.DataFrame) -> None:
    holdings_frame.loc[3, "weight"] = 0.49
    with pytest.raises(SchemaValidationError) as exc:
        HOLDINGS.validate(holdings_frame)
    msg = str(exc.value)
    assert "2025-02-28" in msg and "0.990000000" in msg
    assert "index 2" in msg and "index 3" in msg
    assert "2025-01-31" not in msg


def test_weight_sum_within_tolerance_passes(holdings_frame: pd.DataFrame) -> None:
    holdings_frame.loc[3, "weight"] = 0.5 + 5e-7
    HOLDINGS.validate(holdings_frame)


def test_unknown_sector_rejected(holdings_frame: pd.DataFrame) -> None:
    holdings_frame.loc[1, "sector"] = "Banks"
    with pytest.raises(SchemaValidationError) as exc:
        HOLDINGS.validate(holdings_frame)
    assert "column 'sector'" in str(exc.value) and "'Banks'" in str(exc.value)


def test_sector_case_not_coerced(holdings_frame: pd.DataFrame) -> None:
    holdings_frame.loc[0, "sector"] = "auto"
    with pytest.raises(SchemaValidationError):
        HOLDINGS.validate(holdings_frame)


def test_tz_aware_dates_rejected(holdings_frame: pd.DataFrame) -> None:
    holdings_frame["date"] = pd.to_datetime(holdings_frame["date"]).dt.tz_localize("Asia/Kolkata")
    with pytest.raises(SchemaValidationError, match="tz-naive"):
        HOLDINGS.validate(holdings_frame)


def test_tz_offset_string_rejected(holdings_frame: pd.DataFrame) -> None:
    holdings_frame.loc[0, "date"] = "2025-01-31T00:00:00+05:30"
    with pytest.raises(SchemaValidationError, match="tz-naive"):
        HOLDINGS.validate(holdings_frame)


def test_time_of_day_rejected(holdings_frame: pd.DataFrame) -> None:
    holdings_frame.loc[0, "date"] = "2025-01-31 15:30"
    with pytest.raises(SchemaValidationError, match="time of day"):
        HOLDINGS.validate(holdings_frame)


def test_negative_weight_rejected(holdings_frame: pd.DataFrame) -> None:
    holdings_frame.loc[0, "weight"] = -0.1
    holdings_frame.loc[1, "weight"] = 1.1
    with pytest.raises(SchemaValidationError, match=r"within \[0, 1\]"):
        HOLDINGS.validate(holdings_frame)


def test_missing_and_unexpected_columns(holdings_frame: pd.DataFrame) -> None:
    bad = holdings_frame.drop(columns=["sector"]).assign(extra=1)
    with pytest.raises(SchemaValidationError) as exc:
        HOLDINGS.validate(bad)
    assert "missing column(s): sector" in str(exc.value)
    assert "unexpected column(s): extra" in str(exc.value)


def test_duplicate_key_rejected(holdings_frame: pd.DataFrame) -> None:
    holdings_frame.loc[1, "security_id"] = "A"
    with pytest.raises(SchemaValidationError, match="duplicate key"):
        HOLDINGS.validate(holdings_frame)


def test_whitespace_padded_id_not_stripped(holdings_frame: pd.DataFrame) -> None:
    holdings_frame.loc[0, "security_id"] = " A"
    with pytest.raises(SchemaValidationError, match="whitespace"):
        HOLDINGS.validate(holdings_frame)


def test_all_issues_reported_together(holdings_frame: pd.DataFrame) -> None:
    holdings_frame.loc[0, "sector"] = "Nope"
    holdings_frame["weight"] = holdings_frame["weight"].astype(object)
    holdings_frame.loc[1, "weight"] = "abc"
    with pytest.raises(SchemaValidationError) as exc:
        HOLDINGS.validate(holdings_frame)
    assert len(exc.value.issues) == 2


def test_empty_table_rejected(holdings_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaValidationError, match="no rows"):
        HOLDINGS.validate(holdings_frame.iloc[0:0])


# --- returns --------------------------------------------------------------


def _prices(returns: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [f"2025-{m:02d}-01" for m in range(1, len(returns) + 1)],
            "security_id": ["A"] * len(returns),
            "total_return": returns,
        }
    )


def test_percent_scale_returns_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="look like percent"):
        PRICES.validate(_prices([1.2, -0.8, 2.5, 0.4]))


def test_percent_string_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="plain decimal"):
        PRICES.validate(_prices(["1.2%", 0.01]))


def test_return_of_minus_one_or_less_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="greater than -1"):
        PRICES.validate(_prices([0.01, -1.0, 0.02]))


def test_non_finite_and_missing_returns_rejected() -> None:
    with pytest.raises(SchemaValidationError) as exc:
        PRICES.validate(_prices([0.01, float("inf"), None]))
    msg = str(exc.value)
    assert "finite" in msg and "missing value" in msg


def test_bool_not_accepted_as_number() -> None:
    with pytest.raises(SchemaValidationError):
        PRICES.validate(_prices([True, 0.01]))


def test_decimal_returns_pass() -> None:
    PRICES.validate(_prices([0.012, -0.034, 0.5, -0.2]))


# --- other tables ---------------------------------------------------------


def test_benchmark_weights_must_sum_to_one() -> None:
    df = pd.DataFrame(
        {"date": ["2025-01-31"] * 2, "sector": ["Auto", "IT"], "weight": [0.5, 0.5], "total_return": [0.01, 0.02]}
    )
    with pytest.raises(SchemaValidationError, match="'IT'"):
        BENCHMARK.validate(df)
    df["sector"] = ["Auto", "Information Technology"]
    df.loc[1, "weight"] = 0.4
    with pytest.raises(SchemaValidationError, match="sum to"):
        BENCHMARK.validate(df)


def test_regime_rejects_unknown_state_and_exposure_out_of_range() -> None:
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "raw_state": ["normal", "panic"],
            "confirmed_state": ["normal", "normal"],
            "equity_exposure": [1.0, 1.2],
        }
    )
    with pytest.raises(SchemaValidationError) as exc:
        REGIME_SERIES.validate(df)
    assert len(exc.value.issues) == 2


def test_isin_validation() -> None:
    assert is_valid_isin("INE002A01018")
    assert not is_valid_isin("INE002A01019")  # wrong check digit
    assert not is_valid_isin("US0378331005")  # not Indian
    master = pd.DataFrame(
        {"security_id": ["X"], "name": ["X Ltd"], "sector": ["Auto"], "isin": ["INE002A01019"]}
    )
    with pytest.raises(SchemaValidationError, match="invalid ISIN"):
        SECURITY_MASTER.validate(master)


def test_holdings_cross_check_against_master(holdings_frame: pd.DataFrame) -> None:
    holdings = HOLDINGS.validate(holdings_frame)
    master = SECURITY_MASTER.validate(
        pd.DataFrame(
            {"security_id": ["A"], "name": ["A Ltd"], "sector": ["FMCG"], "isin": ["INE002A01018"]}
        )
    )
    with pytest.raises(SchemaValidationError) as exc:
        check_holdings_against_master(holdings, master)
    msg = str(exc.value)
    assert "B" in msg and "not in SecurityMaster" in msg
    assert "Auto vs master FMCG" in msg


# --- policy ---------------------------------------------------------------


def test_valid_policy() -> None:
    p = Policy(**_policy_kwargs())
    assert p.neutral_equity_exposure == pytest.approx(1.0)
    assert Policy.from_dict(p.to_dict()) == p


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda k: k.update(derisk_mode="sell_everything"), "derisk_mode"),
        (lambda k: k["bands"].update({"Auto": (0.2, 0.3)}), "outside its band"),
        (lambda k: k["bands"].update({"Auto": (0.3, 0.2)}), "low <= high"),
        (lambda k: k["neutral_weights"].update({"Banks": 0.0}), "unknown key"),
        (lambda k: k["neutral_weights"].pop("Realty"), "missing sector"),
        (lambda k: k["neutral_weights"].update({"Auto": 0.5}), "sum to"),
        (lambda k: k.update(review_date=datetime(2026, 9, 30, 10, 0)), "review_date"),
        (lambda k: k.update(benchmark_name=" "), "benchmark_name"),
    ],
)
def test_invalid_policy(mutate, fragment: str) -> None:
    kwargs = _policy_kwargs()
    mutate(kwargs)
    with pytest.raises(SchemaValidationError, match=fragment):
        Policy(**kwargs)


def test_cash_derisk_requires_cash_key() -> None:
    kwargs = _policy_kwargs()
    kwargs["neutral_weights"].pop("Cash")
    kwargs["bands"].pop("Cash")
    with pytest.raises(SchemaValidationError, match="requires a 'Cash'"):
        Policy(**kwargs)
    kwargs["derisk_mode"] = "within_equity"
    Policy(**kwargs)


# --- loaders --------------------------------------------------------------


def test_csv_error_names_file_column_and_line(tmp_path: Path) -> None:
    path = tmp_path / "holdings.csv"
    path.write_text(
        "# a header comment\n"
        "# another\n"
        "date,security_id,weight,sector\n"
        "2025-01-31,A,0.6,Auto\n"
        "2025-01-31,B,0.4,Banking\n",
        encoding="utf-8",
    )
    with pytest.raises(SchemaValidationError) as exc:
        io.load_holdings(path)
    msg = str(exc.value)
    assert str(path) in msg
    assert "column 'sector'" in msg
    assert "line 5" in msg  # physical line of the bad row


def test_csv_round_trip_with_header(tmp_path: Path, holdings_frame: pd.DataFrame) -> None:
    path = io.write_table(holdings_frame, tmp_path / "h.csv", HOLDINGS, ["synthetic"])
    assert io.read_header_comment(path) == ["# synthetic"]
    loaded = io.load_holdings(path)
    pd.testing.assert_frame_equal(loaded, HOLDINGS.validate(holdings_frame))


def test_parquet_round_trip_and_error_rows(tmp_path: Path, holdings_frame: pd.DataFrame) -> None:
    path = io.write_table(holdings_frame, tmp_path / "h.parquet", HOLDINGS, ["synthetic"])
    assert io.read_header_comment(path) == ["synthetic"]
    pd.testing.assert_frame_equal(io.load_holdings(path), HOLDINGS.validate(holdings_frame))

    bad = holdings_frame.copy()
    bad.loc[2, "weight"] = 1.5
    bad_path = tmp_path / "bad.parquet"
    bad.to_parquet(bad_path, index=False)
    with pytest.raises(SchemaValidationError) as exc:
        io.load_holdings(bad_path)
    msg = str(exc.value)
    assert "bad.parquet" in msg and "column 'weight'" in msg and "row 3" in msg


def test_loader_rejects_wrong_suffix_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        io.load_prices(tmp_path / "nope.csv")
    other = tmp_path / "x.xlsx"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported file type"):
        io.load_prices(other)


def test_policy_json_round_trip_and_bad_json(tmp_path: Path) -> None:
    p = Policy(**_policy_kwargs())
    path = io.write_policy(p, tmp_path / "policy.json", ["synthetic"])
    assert io.load_policy(path) == p
    path.write_text('{"neutral_weights": {}, "bands": {}, "derisk_mode": "cash", "extra": 1}', encoding="utf-8")
    with pytest.raises(SchemaValidationError) as exc:
        io.load_policy(path)
    assert "unexpected key(s): extra" in str(exc.value)
    assert "missing key(s): benchmark_name, review_date" in str(exc.value)
