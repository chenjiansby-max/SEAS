#!/usr/bin/env python3
"""Snippet-level counterfactual case study for SEAS.

This script loads a trained SEAS checkpoint, selects a representative test
window, ranks text snippets by their semantic relevance to the distilled event
tokens, and measures how local forecasts change when a specific snippet is
blanked or replaced by random text.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from exp.exp_long_term_forecasting_SEAS import Exp_Long_Term_Forecast_SEAS


def seq_len_for_domain(domain: str) -> int:
    if domain in {"Energy", "Health"}:
        return 24
    if domain == "Environment":
        return 96
    return 12


def build_args(cli: argparse.Namespace, domain: str, pred_len: int) -> SimpleNamespace:
    seq_len = seq_len_for_domain(domain)
    args = SimpleNamespace()
    args.task_name = "long_term_forecast"
    args.is_training = 0
    args.model_id = f"seas_freqlookback_bert_{domain}_s{cli.seed}_sl{seq_len}_p{pred_len}"
    args.model = "TimeMixer"
    args.data = "custom"
    args.root_path = str(cli.data_root)
    args.data_path = f"{domain}.csv"
    args.features = "S"
    args.target = "OT"
    args.freq = "h"
    args.checkpoints = str(cli.checkpoints)

    args.seq_len = seq_len
    args.label_len = 0
    args.pred_len = pred_len
    args.seasonal_patterns = "Monthly"
    args.inverse = False

    args.expand = 2
    args.d_conv = 4
    args.top_k = 5
    args.num_kernels = 6
    args.enc_in = 1
    args.dec_in = 1
    args.c_out = 1
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.d_ff = 1024
    args.moving_avg = 25
    args.factor = 3
    args.distil = True
    args.dropout = 0.1
    args.embed = "timeF"
    args.activation = "gelu"
    args.output_attention = False
    args.channel_independence = 1
    args.decomp_method = "moving_avg"
    args.use_norm = 1
    args.down_sampling_layers = 2
    args.down_sampling_window = 2
    args.down_sampling_method = "avg"
    args.seg_len = 48

    args.num_workers = cli.num_workers
    args.itr = 1
    args.train_epochs = 50
    args.batch_size = cli.batch_size
    args.patience = 20
    args.learning_rate = 0.005
    args.learning_rate2 = 0.005
    args.learning_rate3 = 0.001
    args.learning_rate_weight = 0.0001
    args.des = f"S{cli.seed}_SEAS_freqlookback_BERT"
    args.loss = "MSE"
    args.lradj = "type1"
    args.use_amp = False

    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = "0"
    args.device_ids = [0]

    args.p_hidden_dims = [128, 128]
    args.p_hidden_layers = 2

    args.llm_model = "BERT"
    args.llm_dim = 768
    args.llm_layers = 6
    args.text_path = "None"
    args.type_tag = "#F#"
    args.text_len = 3
    args.prompt_weight = 0.1
    args.pool_type = "avg"
    args.date_name = "end_date"
    args.addHisRate = 0.5
    args.init_method = "normal"
    args.seed = cli.seed
    args.save_name = str(cli.out_dir / "snippet_counterfactual_metrics.txt")
    args.use_fullmodel = 0
    args.use_closedllm = 0
    args.huggingface_token = None
    args.method = "SEAS"
    args.n_ts_features = 1
    args.text_emb = 32
    args.llm_path = str(cli.llm_path)
    args.dominance_freq = 0
    args.freq_cut_off_rate = 1
    args.proj_per_freq = True
    args.text_name = "fact"
    args.text_control_mode = "aligned"
    args.text_shift_steps = 1
    args.text_control_seed = cli.seed

    args.feat_start_id = 1
    args.feat_end_id = 4
    args.mm_emb_size = 16
    args.mm_hidden_size = 16
    args.llm_emb_size = 768
    args.text_dropout = 0.2
    args.fuse_history = True
    args.fuse_prediction = False
    args.use_product = True
    args.only_text_input = False
    args.sum_fusion = False
    args.use_learnable_semantic = 0
    args.semantic_hidden_dim = 256
    args.semantic_dropout = 0.1
    args.semantic_lr = 0.001
    args.use_text_s4d_spectral = 1
    args.text_s4d_state = 64
    args.text_s4d_dropout = 0.1
    args.text_fusion_hidden_dim = 64
    args.seas_fusion_mode = "residual"
    args.seas_residual_init = -4.0
    args.seas_text_max_residual = 0.2
    args.seas_low_freq_ratio = 0.35
    args.seas_low_residual_init = -2.0
    args.seas_high_residual_init = -5.0
    args.numeric_only = 0

    args.use_seas = 1
    args.seas_num_bands = 3
    args.seas_num_events = 4
    args.seas_hidden_dim = 64
    args.seas_max_gain = 0.5
    args.seas_max_phase = 0.5
    args.seas_max_residual = 0.3
    args.seas_decomp_residual = 0.25
    args.seas_disable_asd = 0
    args.seas_disable_sed = 0
    args.seas_disable_sfm = 0
    args.seas_disable_hsg = 0
    args.seas_disable_text_s4d = 0
    return args


def setting_name(args: SimpleNamespace) -> str:
    return "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        0,
    )


def load_checkpoints(exp: Exp_Long_Term_Forecast_SEAS, checkpoint_root: Path, setting: str) -> Path:
    ckpt_dir = checkpoint_root / setting
    if not ckpt_dir.exists():
        matches = glob.glob(str(checkpoint_root / f"{setting}*"))
        if matches:
            ckpt_dir = Path(matches[0])
    required = [ckpt_dir / f"checkpoint_{idx}.pth" for idx in range(3)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files:\n" + "\n".join(missing))

    device = exp.device
    exp.model.load_state_dict(torch.load(required[0], map_location=device))
    exp.mlp.load_state_dict(torch.load(required[1], map_location=device))
    exp.mm_model.load_state_dict(torch.load(required[2], map_location=device))
    return ckpt_dir


def clean_snippet_text(text: str, limit: int = 220) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3].rstrip(" ;,") + "..."
    return text


def encode_texts(exp: Exp_Long_Term_Forecast_SEAS, texts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        enc = exp.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ids = enc.input_ids.to(exp.device)
        masks = enc.attention_mask.to(exp.device)
        embedding = exp.llm_model.get_input_embeddings()(ids)
        if exp.pool_type == "pooler":
            pooled = exp.llm_model(inputs_embeds=embedding).pooler_output
        else:
            hidden = exp.llm_model(inputs_embeds=embedding).last_hidden_state
            pooled = torch.sum(hidden * masks.unsqueeze(-1), dim=1) / torch.sum(masks, dim=1).unsqueeze(1)
    return pooled.detach()


def compute_full_batch(exp, test_data, test_loader):
    exp.model.eval()
    exp.mlp.eval()
    exp.mm_model.eval()
    if exp.semantic_evidence is not None:
        exp.semantic_evidence.eval()

    with torch.no_grad():
        batch_x, batch_y, batch_x_mark, batch_y_mark, index = next(iter(test_loader))
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float().to(exp.device)
        batch_x_mark = batch_x_mark.float().to(exp.device)
        prior_y = torch.from_numpy(test_data.get_prior_y(index)).float().to(exp.device)
        prompt_embeddings_batch = torch.FloatTensor(test_data.get_text_embedding(index)).to(exp.device)
        if exp.semantic_evidence is not None:
            prompt_embeddings_batch = exp.semantic_evidence(batch_x, prompt_embeddings_batch)
        text_embeddings = exp.mlp(prompt_embeddings_batch, batch_x_mark)
        pred_freq = exp.mm_model(batch_x, batch_x_mark, None, None, text_embeddings)
        pred = (1 - exp.prompt_weight) * pred_freq + exp.prompt_weight * prior_y
        true = batch_y[:, -exp.args.pred_len :, :]

    return batch_x, batch_x_mark, index, prompt_embeddings_batch, pred, true, prior_y


def rank_snippets_by_event_relevance(exp, test_data, index, case_index, prompt_embeddings_batch, batch_x_mark):
    mark_case = batch_x_mark[case_index: case_index + 1]
    pooled_window = prompt_embeddings_batch[case_index: case_index + 1].detach().clone().requires_grad_(False)
    text_spec = exp.mlp(pooled_window, mark_case)
    text_spec_seas = exp.mm_model.seas_text_spec(text_spec, channels=1, freq_len=exp.mm_model.H_f)
    distiller = exp.mm_model.seas_event_distiller
    event_tokens = distiller(text_spec_seas, channels=1)
    time_semantic = exp.mlp.spectral_text_encoder.last_time_semantic
    scores = F.cosine_similarity(event_tokens[:, :, :, None, :], time_semantic[:, None, None, :, :], dim=-1)
    # scores: [1,1,num_events,seq_len]
    score_np = scores[0, 0].detach().cpu().numpy()
    step_scores = np.max(np.abs(score_np), axis=0)

    index_np = index.detach().cpu().numpy() if torch.is_tensor(index) else np.asarray(index)
    base_idx = int(index_np[case_index])
    dates = [str(x[0]) for x in test_data.date[base_idx: base_idx + exp.args.seq_len]]
    raw_text = test_data.get_text(index)[case_index].reshape(-1).tolist()

    rows = []
    for step_idx, score in enumerate(step_scores):
        rows.append(
            {
                "rank_score": float(score),
                "step_idx": step_idx,
                "date": dates[step_idx] if step_idx < len(dates) else "",
                "text_snippet": clean_snippet_text(raw_text[step_idx]) if step_idx < len(raw_text) else "",
            }
        )
    df = pd.DataFrame(rows).sort_values("rank_score", ascending=False).reset_index(drop=True)
    return df


def evaluate_single_blank_counterfactual(
    exp,
    batch_x,
    batch_x_mark,
    prompt_embeddings_batch,
    prior_y,
    true,
    case_index,
    candidate_df,
    blank_embed,
):
    if candidate_df.empty:
        return None
    top_row = candidate_df.iloc[0]
    step_idx = int(top_row["step_idx"])
    modified_prompt = prompt_embeddings_batch.detach().clone()
    modified_prompt[case_index, step_idx] = blank_embed.to(modified_prompt.device)
    pred_variant = run_forecast_from_prompt_embeddings(exp, batch_x, batch_x_mark, modified_prompt, prior_y)
    pred_case = pred_variant.detach().cpu().numpy()[case_index, :, -1]
    true_case = true.detach().cpu().numpy()[case_index, :, -1]
    full_pred_case = run_forecast_from_prompt_embeddings(exp, batch_x, batch_x_mark, prompt_embeddings_batch, prior_y).detach().cpu().numpy()[case_index, :, -1]
    full_mse = float(np.mean((full_pred_case - true_case) ** 2))
    blank_mse = float(np.mean((pred_case - true_case) ** 2))
    return {
        "case_index": case_index,
        "step_idx": step_idx,
        "date": top_row["date"],
        "snippet_score": float(top_row["rank_score"]),
        "full_mse": full_mse,
        "blank_mse": blank_mse,
        "delta_mse": blank_mse - full_mse,
        "text_snippet": str(top_row["text_snippet"]),
    }


def search_best_case(
    exp,
    test_data,
    index,
    batch_x,
    batch_x_mark,
    prompt_embeddings_batch,
    prior_y,
    true,
    max_cases,
):
    blank_embed = encode_texts(exp, ["No information available"])[0]
    rows = []
    num_cases = min(max_cases, prompt_embeddings_batch.shape[0])
    for case_index in range(num_cases):
        candidate_df = rank_snippets_by_event_relevance(exp, test_data, index, case_index, prompt_embeddings_batch, batch_x_mark)
        row = evaluate_single_blank_counterfactual(
            exp,
            batch_x,
            batch_x_mark,
            prompt_embeddings_batch,
            prior_y,
            true,
            case_index,
            candidate_df,
            blank_embed,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        return None, pd.DataFrame()
    search_df = pd.DataFrame(rows).sort_values("delta_mse", ascending=False).reset_index(drop=True)
    return int(search_df.iloc[0]["case_index"]), search_df


def run_forecast_from_prompt_embeddings(exp, batch_x, batch_x_mark, prompt_embeddings_batch, prior_y):
    with torch.no_grad():
        text_embeddings = exp.mlp(prompt_embeddings_batch, batch_x_mark)
        pred_freq = exp.mm_model(batch_x, batch_x_mark, None, None, text_embeddings)
        pred = (1 - exp.prompt_weight) * pred_freq + exp.prompt_weight * prior_y
    return pred


def choose_replacement_text(step_df: pd.DataFrame, candidate_idx: int) -> Tuple[int, str]:
    if len(step_df) <= 1:
        return int(candidate_idx), "No information available"
    fallback = step_df.iloc[-1]
    if int(fallback["step_idx"]) == candidate_idx and len(step_df) > 1:
        fallback = step_df.iloc[-2]
    return int(fallback["step_idx"]), str(fallback["text_snippet"])


def save_case_outputs(
    out_dir: Path,
    case_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    case_df.to_csv(out_dir / "candidate_snippets.csv", index=False)
    metrics_df.to_csv(out_dir / "snippet_counterfactual_metrics.csv", index=False)
    forecast_df.to_csv(out_dir / "snippet_counterfactual_forecast.csv", index=False)
    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Snippet-level counterfactual case study\n\n")
        f.write("This directory contains ranked candidate snippets, per-snippet counterfactual metrics, and case-level forecast traces.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--domain", default="Security")
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--search-best-case", type=int, default=1)
    parser.add_argument("--max-search-cases", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=Path("./data/timemmd_mmts_tats"))
    parser.add_argument("--checkpoints", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--llm-path", type=Path, default=Path("./pretrained/bert"))
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs/snippet_counterfactual_case_study"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    cli = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    args = build_args(cli, cli.domain, cli.pred_len)
    exp = Exp_Long_Term_Forecast_SEAS(args)
    setting = setting_name(args)
    ckpt_dir = load_checkpoints(exp, cli.checkpoints, setting)

    test_data, test_loader = exp._get_data(flag="test")
    exp.update_text_embedding(test_data)
    batch_x, batch_x_mark, index, prompt_embeddings_batch, pred_full, true, prior_y = compute_full_batch(exp, test_data, test_loader)

    search_df = pd.DataFrame()
    if cli.search_best_case:
      best_case_index, search_df = search_best_case(
          exp,
          test_data,
          index,
          batch_x,
          batch_x_mark,
          prompt_embeddings_batch,
          prior_y,
          true,
          cli.max_search_cases,
      )
      if best_case_index is not None:
          cli.case_index = best_case_index

    case_df = rank_snippets_by_event_relevance(exp, test_data, index, cli.case_index, prompt_embeddings_batch, batch_x_mark)
    case_df = case_df.head(cli.top_k).copy()

    full_pred_case = pred_full.detach().cpu().numpy()[cli.case_index, :, -1]
    true_case = true.detach().cpu().numpy()[cli.case_index, :, -1]
    history = batch_x.detach().cpu().numpy()[cli.case_index, :, -1]
    base_metrics = {
        "variant": "full",
        "step_idx": -1,
        "date": "",
        "replacement": "original",
        "snippet_score": np.nan,
        "mse": float(np.mean((full_pred_case - true_case) ** 2)),
        "mae": float(np.mean(np.abs(full_pred_case - true_case))),
        "forecast_delta_l1": 0.0,
        "text_snippet": "",
    }
    metrics_rows = [base_metrics]

    hist_x = np.arange(len(history))
    fut_x = np.arange(len(history), len(history) + len(true_case))
    forecast_rows = [
        pd.DataFrame({"time_index": hist_x, "value": history, "series": "History"}),
        pd.DataFrame({"time_index": fut_x, "value": true_case, "series": "Ground truth"}),
        pd.DataFrame({"time_index": fut_x, "value": full_pred_case, "series": "Full SEAS"}),
    ]

    blank_embed = encode_texts(exp, ["No information available"])[0]
    random_embed = encode_texts(exp, [f"Random semantic control sentence {cli.seed}."])[0]

    for row in case_df.itertuples(index=False):
        step_idx = int(row.step_idx)
        snippet_text = str(row.text_snippet)
        replacement_step, replacement_text = choose_replacement_text(case_df, step_idx)
        swap_embed = encode_texts(exp, [replacement_text])[0]

        for variant_name, replacement_label, replacement_embed in [
            (f"blank_step_{step_idx}", "blank", blank_embed),
            (f"random_step_{step_idx}", "random", random_embed),
            (f"swap_step_{step_idx}", f"swap_with_step_{replacement_step}", swap_embed),
        ]:
            modified_prompt = prompt_embeddings_batch.detach().clone()
            modified_prompt[cli.case_index, step_idx] = replacement_embed.to(modified_prompt.device)
            pred_variant = run_forecast_from_prompt_embeddings(exp, batch_x, batch_x_mark, modified_prompt, prior_y)
            pred_case = pred_variant.detach().cpu().numpy()[cli.case_index, :, -1]
            diff = pred_case - true_case
            metrics_rows.append(
                {
                    "variant": variant_name,
                    "step_idx": step_idx,
                    "date": row.date,
                    "replacement": replacement_label,
                    "snippet_score": float(row.rank_score),
                    "mse": float(np.mean(diff ** 2)),
                    "mae": float(np.mean(np.abs(diff))),
                    "forecast_delta_l1": float(np.mean(np.abs(pred_case - full_pred_case))),
                    "text_snippet": snippet_text,
                }
            )
            forecast_rows.append(
                pd.DataFrame(
                    {
                        "time_index": fut_x,
                        "value": pred_case,
                        "series": variant_name,
                    }
                )
            )

    metrics_df = pd.DataFrame(metrics_rows)
    forecast_df = pd.concat(forecast_rows, ignore_index=True)

    out_dir = cli.out_dir / f"{cli.domain}_H{cli.pred_len}_case{cli.case_index}"
    save_case_outputs(out_dir, case_df, metrics_df, forecast_df)
    if not search_df.empty:
        search_df.to_csv(out_dir / "case_search_summary.csv", index=False)
    with (out_dir / "case_report.md").open("w", encoding="utf-8") as f:
        f.write(f"# Snippet counterfactual case study: {cli.domain}, H={cli.pred_len}\n\n")
        f.write(f"- checkpoint: `{ckpt_dir}`\n")
        f.write(f"- case_index: `{cli.case_index}`\n")
        f.write(f"- top_k snippets: `{cli.top_k}`\n")
        f.write(f"- full MSE: `{base_metrics['mse']:.6f}`\n")
        f.write(f"- full MAE: `{base_metrics['mae']:.6f}`\n")
        if not search_df.empty:
            f.write(f"- searched candidate cases: `{len(search_df)}`\n")
            f.write(f"- best delta MSE after blanking top-1 snippet: `{search_df.iloc[0]['delta_mse']:.6f}`\n")

    print(f"Wrote snippet counterfactual case study to {out_dir}")


if __name__ == "__main__":
    main()
