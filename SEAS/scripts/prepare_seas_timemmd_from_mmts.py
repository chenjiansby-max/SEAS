#!/usr/bin/env python3
"""Build SEAS-readable Time-MMD files from MM-TSFlib rich CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATA_SPECS = {
    "Agriculture": ("Algriculture", "US_RetailBroilerComposite_Month.csv"),
    "Climate": ("Climate", "US_precipitation_month.csv"),
    "Economy": ("Economy", "US_TradeBalance_Month.csv"),
    "Energy": ("Energy", "US_GasolinePrice_Week.csv"),
    "Environment": ("Environment", "NewYork_AQI_Day.csv"),
    "Health": ("Public_Health", "US_FLURATIO_Week.csv"),
    "Security": ("Security", "US_FEMAGrant_Month.csv"),
    "SocialGood": ("SocialGood", "Unadj_UnemploymentRate_ALL_processed.csv"),
    "Traffic": ("Traffic", "US_VMT_Month.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmts-data-root",
        default="./data/raw/timemmd_mmts",
        help="Root directory of the MM-TSFlib Time-MMD data files.",
    )
    parser.add_argument(
        "--fallback-data-root",
        default="./data/raw/timemmd_fallback",
        help="Fallback root used only when the MM-TSFlib root is missing a CSV.",
    )
    parser.add_argument(
        "--output-root",
        default="./data/timemmd_mmts_tats",
        help="Output directory for SEAS-readable CSV files.",
    )
    parser.add_argument(
        "--text-col",
        default="Final_Search_4",
        help="Rich text column to map into SEAS's fact field.",
    )
    return parser.parse_args()


def source_path(domain: str, mmts_root: Path, fallback_root: Path) -> tuple[Path, str]:
    subdir, filename = DATA_SPECS[domain]
    primary = mmts_root / subdir / filename
    if primary.exists():
        return primary, "mmts"

    fallback = fallback_root / subdir / filename
    if fallback.exists():
        return fallback, "fallback"

    raise FileNotFoundError(
        f"Missing {domain} CSV. Tried {primary} and {fallback}."
    )


def choose_text(df: pd.DataFrame, requested: str) -> pd.Series:
    for col in (requested, "Final_Search_4", "Final_Search_2", "Final_Search_6", "Final_Output", "fact"):
        if col in df.columns:
            return df[col].fillna("No information available").astype(str)
    return pd.Series(["No information available"] * len(df), index=df.index)


def convert_one(domain: str, src: Path, out: Path, text_col: str) -> None:
    df = pd.read_csv(src)
    required = ["date", "OT", "start_date", "end_date", "prior_history_avg"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{src} is missing required columns: {missing}")

    converted = pd.DataFrame(
        {
            "date": df["date"],
            "OT": df["OT"],
            "start_date": df["start_date"],
            "end_date": df["end_date"],
            "prior_history_avg": df["prior_history_avg"],
            "fact": choose_text(df, text_col),
            "preds": df["Final_Output"].fillna("").astype(str)
            if "Final_Output" in df.columns
            else choose_text(df, text_col),
        }
    )

    if converted["OT"].isna().any():
        bad = int(converted["OT"].isna().sum())
        raise ValueError(f"{domain} has {bad} missing OT values after conversion.")

    out.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(out, index=False)
    print(f"{domain}: {len(converted)} rows -> {out}")


def main() -> int:
    args = parse_args()
    mmts_root = Path(args.mmts_data_root)
    fallback_root = Path(args.fallback_data_root)
    output_root = Path(args.output_root)

    print(f"MM-TSFlib data root: {mmts_root}")
    print(f"Output root: {output_root}")
    for domain in DATA_SPECS:
        src, source_kind = source_path(domain, mmts_root, fallback_root)
        if source_kind == "fallback":
            print(f"[warn] {domain}: using fallback source {src}")
        convert_one(domain, src, output_root / f"{domain}.csv", args.text_col)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
