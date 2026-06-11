#!/usr/bin/env python3
"""Prepare supplementary weak-text datasets for ICDE Experiment 2.

Default datasets follow the protocol already used in MEMOIR_V2:
1. TTC_Climate: climate_2014_2023_final.csv from Multimodal_Forecasting
2. FNF_AULoad_NSW: Australia load series + aligned news from From_News_to_Forecast

Optional dataset:
3. BitcoinPrice: raw bitcoin price series + weakly aligned rolling news context

Why not medical by default?
The medical data are patient-wise short independent series. Directly concatenating
them into one long series would create cross-patient windows and an unfair protocol
for the current seven-model comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="./data/icde_experiment2_extra",
    )
    parser.add_argument(
        "--climate-csv",
        default="./data/raw/extra_datasets/climate_2014_2023_final.csv",
    )
    parser.add_argument(
        "--fnf-ts-csv",
        default="./data/raw/extra_datasets/weather_load_2019-2022.csv",
    )
    parser.add_argument(
        "--fnf-news-csv",
        default="./data/raw/extra_datasets/AU_load_news_dataframe_2019-2020_iteration_0.csv",
    )
    parser.add_argument(
        "--bitcoin-csv",
        default="./data/raw/extra_datasets/bitcoin_daily.csv",
    )
    parser.add_argument(
        "--bitcoin-news-json",
        default="./data/raw/extra_datasets/bitcoin_news.json",
    )
    parser.add_argument(
        "--bitcoin-lookback-days",
        type=int,
        default=7,
        help="Number of past days used to collect weakly aligned news context.",
    )
    parser.add_argument(
        "--bitcoin-max-articles",
        type=int,
        default=5,
        help="Maximum number of recent articles concatenated for each date.",
    )
    parser.add_argument(
        "--bitcoin-max-chars-per-article",
        type=int,
        default=320,
        help="Maximum characters kept per article snippet.",
    )
    parser.add_argument(
        "--include-bitcoin",
        action="store_true",
        help="Also export BitcoinPrice as an optional third supplementary dataset.",
    )
    return parser.parse_args()


def build_prior(series: pd.Series, window: int = 7) -> pd.Series:
    prior = series.shift(1).rolling(window=window, min_periods=1).mean()
    if prior.isna().any():
        prior = prior.fillna(series.expanding().mean())
    if prior.isna().any():
        prior = prior.fillna(series.iloc[0])
    return prior


def build_prior_std(series: pd.Series, window: int = 7) -> pd.Series:
    prior_std = series.shift(1).rolling(window=window, min_periods=2).std()
    if prior_std.isna().any():
        prior_std = prior_std.fillna(0.0)
    return prior_std


def finalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["start_date"] = out["date"]
    out["end_date"] = out["date"]
    out["fact"] = out["fact"].fillna("No information available").astype(str)
    out["preds"] = out["fact"]
    if "prior_history_std" not in out.columns:
        out["prior_history_std"] = 0.0
    out["prior_history_std"] = pd.to_numeric(out["prior_history_std"], errors="coerce").fillna(0.0)
    # Keep compatibility with MM-TSFlib and other baselines that still look for Time-MMD-style text columns.
    for col in ("Final_Search_2", "Final_Search_4", "Final_Search_6", "Final_Output"):
        if col not in out.columns:
            out[col] = out["fact"]
    cols = [
        "date",
        "OT",
        "start_date",
        "end_date",
        "prior_history_avg",
        "prior_history_std",
        "fact",
        "preds",
        "Final_Search_2",
        "Final_Search_4",
        "Final_Search_6",
        "Final_Output",
    ]
    out = out[cols]
    if out["OT"].isna().any():
        bad = int(out["OT"].isna().sum())
        raise ValueError(f"Found {bad} missing OT values.")
    return out


def prepare_climate(climate_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(climate_csv)
    required = {"date", "temp", "text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Climate CSV missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "OT": pd.to_numeric(df["temp"], errors="coerce"),
            "prior_history_avg": build_prior(pd.to_numeric(df["temp"], errors="coerce")),
            "prior_history_std": build_prior_std(pd.to_numeric(df["temp"], errors="coerce")),
            "fact": df["text"].fillna("No information available").astype(str),
        }
    )
    return finalize_frame(out)


def prepare_fnf_auload(ts_path: Path, news_path: Path) -> pd.DataFrame:
    ts = pd.read_csv(ts_path)
    ts = ts[ts["State"].astype(str).str.upper() == "NSW"].copy()
    ts["date"] = pd.to_datetime(ts["SETTLEMENTDATE"])
    ts["news_date"] = ts["date"].dt.strftime("%Y-%m-%d")

    news = pd.read_csv(news_path)
    news["news_date"] = pd.to_datetime(news["time"]).dt.strftime("%Y-%m-%d")
    news_map = dict(zip(news["news_date"], news["news"].fillna("No information available").astype(str)))

    out = pd.DataFrame(
        {
            "date": ts["date"],
            "OT": pd.to_numeric(ts["TOTALDEMAND"], errors="coerce"),
            "prior_history_avg": build_prior(pd.to_numeric(ts["TOTALDEMAND"], errors="coerce"), window=48),
            "prior_history_std": build_prior_std(pd.to_numeric(ts["TOTALDEMAND"], errors="coerce"), window=48),
            "fact": ts["news_date"].map(news_map).fillna("No information available"),
        }
    )
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["start_date"] = out["date"]
    out["end_date"] = out["date"]
    out["preds"] = out["fact"]
    for col in ("Final_Search_2", "Final_Search_4", "Final_Search_6", "Final_Output"):
        out[col] = out["fact"]
    cols = [
        "date",
        "OT",
        "start_date",
        "end_date",
        "prior_history_avg",
        "prior_history_std",
        "fact",
        "preds",
        "Final_Search_2",
        "Final_Search_4",
        "Final_Search_6",
        "Final_Output",
    ]
    out = out[cols].dropna(subset=["OT"]).reset_index(drop=True)
    return out


def truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text[:max_chars]


def prepare_bitcoin(
    bitcoin_csv: Path,
    bitcoin_news_json: Path,
    lookback_days: int,
    max_articles: int,
    max_chars_per_article: int,
) -> pd.DataFrame:
    df = pd.read_csv(bitcoin_csv)
    price_df = df[df["ID"] == "price"].copy()
    if price_df.empty:
        raise ValueError("No `price` series found in bitcoin_daily.csv.")
    price_df["date"] = pd.to_datetime(price_df["TIME"])
    price_df["OT"] = pd.to_numeric(price_df["VALUE"], errors="coerce")
    price_df = price_df.sort_values("date").reset_index(drop=True)
    price_df = price_df.loc[price_df["OT"].first_valid_index() :].reset_index(drop=True)
    price_df = price_df.dropna(subset=["OT"]).reset_index(drop=True)

    with open(bitcoin_news_json, "r", encoding="utf-8") as f:
        news = json.load(f)

    news_df = pd.DataFrame(news)
    if "publication_time" not in news_df.columns:
        raise ValueError("bitcoin_news.json missing `publication_time`.")
    news_df["publication_time"] = pd.to_datetime(news_df["publication_time"], errors="coerce")
    news_df = news_df.dropna(subset=["publication_time"]).sort_values("publication_time").reset_index(drop=True)
    news_df["snippet"] = (
        news_df["title"].fillna("").astype(str)
        + ". "
        + news_df["full_article"].fillna(news_df.get("summary", "")).astype(str)
    ).map(lambda x: truncate_text(x, max_chars_per_article))

    facts = []
    for current_date in price_df["date"]:
        mask = (
            (news_df["publication_time"] <= current_date)
            & (news_df["publication_time"] > current_date - pd.Timedelta(days=lookback_days))
        )
        recent = news_df.loc[mask].tail(max_articles)
        if recent.empty:
            facts.append("No information available")
            continue
        parts = [
            f"{row.publication_time.strftime('%Y-%m-%d')}: {row.snippet}"
            for row in recent.itertuples(index=False)
        ]
        facts.append("Available facts are as follows: " + " ".join(parts))

    out = pd.DataFrame(
        {
            "date": price_df["date"],
            "OT": price_df["OT"],
            "prior_history_avg": build_prior(price_df["OT"]),
            "prior_history_std": build_prior_std(price_df["OT"]),
            "fact": facts,
        }
    )
    return finalize_frame(out)


def write_metadata(output_root: Path) -> None:
    meta = {
        "datasets": {
            "TTC_Climate": {
                "frequency": "daily",
                "recommended_seq_len": 24,
                "recommended_pred_lens": [6, 8, 10, 12],
                "target": "temp",
                "text_column": "fact",
            },
            "FNF_AULoad_NSW": {
                "frequency": "hourly",
                "recommended_seq_len": 48,
                "recommended_pred_lens": [12, 24, 36, 48],
                "target": "TOTALDEMAND",
                "text_column": "fact",
            },
            "BitcoinPrice": {
                "frequency": "daily",
                "recommended_seq_len": 30,
                "recommended_pred_lens": [7, 14, 30],
                "target": "price",
                "text_column": "fact",
            },
        },
        "note": "Medical is excluded from the default benchmark because it consists of many short patient-wise independent series.",
    }
    (output_root / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    climate_df = prepare_climate(Path(args.climate_csv))
    auload_df = prepare_fnf_auload(Path(args.fnf_ts_csv), Path(args.fnf_news_csv))

    climate_path = output_root / "TTC_Climate.csv"
    auload_path = output_root / "FNF_AULoad_NSW.csv"
    climate_df.to_csv(climate_path, index=False)
    auload_df.to_csv(auload_path, index=False)

    bitcoin_path = None
    if args.include_bitcoin:
        bitcoin_df = prepare_bitcoin(
            Path(args.bitcoin_csv),
            Path(args.bitcoin_news_json),
            lookback_days=args.bitcoin_lookback_days,
            max_articles=args.bitcoin_max_articles,
            max_chars_per_article=args.bitcoin_max_chars_per_article,
        )
        bitcoin_path = output_root / "BitcoinPrice.csv"
        bitcoin_df.to_csv(bitcoin_path, index=False)
    write_metadata(output_root)

    print(f"wrote {climate_path} rows={len(climate_df)}")
    print(f"wrote {auload_path} rows={len(auload_df)}")
    if bitcoin_path is not None:
        print(f"wrote {bitcoin_path} rows={len(bitcoin_df)}")
    print(f"wrote {output_root / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
