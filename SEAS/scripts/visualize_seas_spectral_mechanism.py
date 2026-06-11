#!/usr/bin/env python
"""Visualize SEAS spectral modulation and representative case forecasts.

This script loads trained SEAS checkpoints, runs one test batch per selected
dataset/horizon, hooks the semantic frequency modulator, and saves paper-ready
mechanism plots plus CSV summaries.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from exp.exp_long_term_forecasting_SEAS import Exp_Long_Term_Forecast_SEAS


STOP_TOKENS = {
    "the", "of", "and", "to", "in", "a", "is", "for", "on", "that", "with",
    "as", "are", "was", "were", "be", "by", "this", "it", "from", "at",
    "an", "or", "which", "has", "have", "had", "its", "their", "into",
    "across", "much", "more", "most", "all", "week", "state", "states",
    "available", "facts", "follows", "source", "about", "much", "lower",
}


def seq_len_for_domain(domain: str) -> int:
    if domain in {"Energy", "Health"}:
        return 24
    if domain == "Environment":
        return 96
    return 12


def parse_cases(case_text: str) -> List[Tuple[str, int]]:
    cases = []
    for item in case_text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Case must be Domain:pred_len, got {item!r}")
        domain, pred_len = item.split(":", 1)
        cases.append((domain.strip(), int(pred_len)))
    return cases


def build_args(cli: argparse.Namespace, domain: str, pred_len: int) -> SimpleNamespace:
    seq_len = seq_len_for_domain(domain)
    args = SimpleNamespace()
    variant_tag = getattr(cli, "variant_tag", "full")
    variant_model_prefix = "seas_freqlookback_bert"
    variant_des_suffix = "SEAS_freqlookback_BERT"
    if variant_tag == "wo_sfm":
        variant_model_prefix = "seas_wo_sfm_freqlookback_bert"
        variant_des_suffix = "SEAS_wo_sfm_freqlookback_BERT"
    elif variant_tag != "full":
        variant_model_prefix = f"seas_{variant_tag}_freqlookback_bert"
        variant_des_suffix = f"SEAS_{variant_tag}_freqlookback_BERT"

    args.task_name = "long_term_forecast"
    args.is_training = 0
    args.model_id = f"{variant_model_prefix}_{domain}_s{cli.seed}_sl{seq_len}_p{pred_len}"
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
    args.des = f"S{cli.seed}_{variant_des_suffix}"
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
    args.save_name = str(cli.out_dir / "mechanism_metrics.txt")
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
    args.seas_disable_asd = int(getattr(cli, "disable_asd", 0))
    args.seas_disable_sed = int(getattr(cli, "disable_sed", 0))
    args.seas_disable_sfm = int(getattr(cli, "disable_sfm", 0))
    args.seas_disable_hsg = int(getattr(cli, "disable_hsg", 0))
    args.seas_disable_text_s4d = int(getattr(cli, "disable_text_s4d", 0))
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


def install_sfm_capture(exp: Exp_Long_Term_Forecast_SEAS) -> Dict[str, torch.Tensor]:
    mod = exp.mm_model.seas_modulator
    cache: Dict[str, torch.Tensor] = {}

    def capture_forward(spec, band_tokens, event_tokens, band_weights):
        if event_tokens.size(2) != mod.num_bands:
            event_tokens_local = F.interpolate(
                event_tokens.permute(0, 1, 3, 2).reshape(-1, event_tokens.size(-1), event_tokens.size(2)),
                size=mod.num_bands,
                mode="linear",
                align_corners=False,
            ).reshape(event_tokens.size(0), event_tokens.size(1), event_tokens.size(-1), mod.num_bands).permute(0, 1, 3, 2)
        else:
            event_tokens_local = event_tokens

        cond = torch.cat([band_tokens.real, band_tokens.imag, event_tokens_local], dim=-1)
        gain, phase, alpha, delta_real, delta_imag = mod.modulator(cond).chunk(5, dim=-1)
        gain = torch.tanh(gain) * mod.max_gain
        phase = torch.tanh(phase) * mod.max_phase
        alpha = torch.sigmoid(alpha) * mod.max_residual

        amp = 1.0 + gain
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        rotated_real = amp * (band_tokens.real * cos_phase - band_tokens.imag * sin_phase)
        rotated_imag = amp * (band_tokens.real * sin_phase + band_tokens.imag * cos_phase)
        enhanced_real = mod.real_norm(rotated_real + alpha * delta_real)
        enhanced_imag = mod.imag_norm(rotated_imag + alpha * delta_imag)

        delta_band_real = enhanced_real - band_tokens.real
        delta_band_imag = enhanced_imag - band_tokens.imag
        delta_real_full = torch.einsum("bckd,kf->bcfd", delta_band_real, band_weights)
        delta_imag_full = torch.einsum("bckd,kf->bcfd", delta_band_imag, band_weights)
        out = torch.complex(spec.real + delta_real_full, spec.imag + delta_imag_full)

        cache.clear()
        cache.update(
            spec_before=spec.detach().cpu(),
            spec_after=out.detach().cpu(),
            band_tokens_abs=band_tokens.abs().detach().cpu(),
            event_tokens_raw=event_tokens.detach().cpu(),
            event_tokens=event_tokens_local.detach().cpu(),
            band_weights=band_weights.detach().cpu(),
            gain=gain.detach().cpu(),
            phase=phase.detach().cpu(),
            alpha=alpha.detach().cpu(),
            delta_abs=torch.complex(delta_real_full, delta_imag_full).abs().detach().cpu(),
        )
        return out

    mod.forward = capture_forward
    return cache


def install_asd_capture(exp: Exp_Long_Term_Forecast_SEAS, cache: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    decomposer = getattr(exp.mm_model, "seas_decomposer", None)
    if decomposer is None:
        return cache

    def capture_forward(spec):
        decomposed, band_tokens, band_weights = decomposer.__class__.forward(decomposer, spec)
        cache["spec_before"] = decomposed.detach().cpu()
        cache["band_tokens_abs"] = band_tokens.abs().detach().cpu()
        cache["band_weights"] = band_weights.detach().cpu()
        return decomposed, band_tokens, band_weights

    decomposer.forward = capture_forward
    return cache


def tensor_mean_by_band(x: torch.Tensor) -> np.ndarray:
    return x.float().mean(dim=tuple(i for i in range(x.dim()) if i != 2)).numpy()


def spectrum_curve(x: torch.Tensor) -> np.ndarray:
    return x.abs().float().mean(dim=(0, 1, 3)).numpy()


def clone_cache(cache: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in cache.items()}


def restore_cache(cache: Dict[str, torch.Tensor], saved: Dict[str, torch.Tensor]) -> None:
    cache.clear()
    cache.update(clone_cache(saved))


def save_sed_frequency_attention(case_dir: Path, exp: Exp_Long_Term_Forecast_SEAS, case_index: int) -> None:
    distiller = getattr(exp.mm_model, "seas_event_distiller", None)
    attn = getattr(distiller, "last_attention_weights", None)
    if attn is None:
        return

    attn_np = attn.detach().cpu().float().numpy()
    if attn_np.ndim == 4:
        attn_np = attn_np.mean(axis=1)
    case_idx = min(case_index, attn_np.shape[0] - 1)
    rows = []
    for event_idx in range(attn_np.shape[1]):
        for freq_idx in range(attn_np.shape[2]):
            rows.append(
                {
                    "event": f"Event {event_idx + 1}",
                    "event_id": event_idx + 1,
                    "freq_bin": freq_idx,
                    "attention": float(attn_np[case_idx, event_idx, freq_idx]),
                }
            )
    pd.DataFrame(rows).to_csv(case_dir / "sed_event_frequency_attention.csv", index=False)


def clean_snippet_text(text: str, limit: int = 180) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3].rstrip(" ;,") + "..."
    return text


def save_sed_text_alignment(
    case_dir: Path,
    exp: Exp_Long_Term_Forecast_SEAS,
    test_data,
    index: torch.Tensor,
    case_index: int,
    seq_len: int,
) -> None:
    distiller = getattr(exp.mm_model, "seas_event_distiller", None)
    adapter = getattr(getattr(exp.mlp, "spectral_text_encoder", None), "last_time_semantic", None)
    event_tokens = getattr(distiller, "last_event_tokens", None)
    if adapter is None or event_tokens is None:
        return

    time_semantic = adapter.detach().cpu().float()
    event_tokens = event_tokens.detach().cpu().float()
    scores = F.cosine_similarity(
        event_tokens.unsqueeze(2),
        time_semantic.unsqueeze(1),
        dim=-1,
    ).numpy()

    index_np = index.detach().cpu().numpy() if torch.is_tensor(index) else np.asarray(index)
    base_idx = int(index_np[case_index])
    dates = [str(x[0]) for x in test_data.date[base_idx: base_idx + seq_len]]
    text_window = test_data.get_text(index)[case_index].reshape(-1).tolist()
    text_window = [clean_snippet_text(x) for x in text_window]

    case_idx = min(case_index, scores.shape[0] - 1)
    rows = []
    for event_idx in range(scores.shape[1]):
        for step_idx in range(scores.shape[2]):
            rows.append(
                {
                    "event": f"Event {event_idx + 1}",
                    "event_id": event_idx + 1,
                    "step_idx": step_idx,
                    "date": dates[step_idx] if step_idx < len(dates) else "",
                    "score": float(scores[case_idx, event_idx, step_idx]),
                    "text_snippet": text_window[step_idx] if step_idx < len(text_window) else "",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(case_dir / "sed_event_text_alignment.csv", index=False)

    top_df = (
        df.sort_values(["event_id", "score"], ascending=[True, False])
        .groupby("event_id", as_index=False)
        .head(2)
        .copy()
    )
    top_df.to_csv(case_dir / "sed_event_top_snippets.csv", index=False)


def save_sed_token_attribution(
    case_dir: Path,
    exp: Exp_Long_Term_Forecast_SEAS,
    test_data,
    index: torch.Tensor,
    batch_x_mark: torch.Tensor,
    case_index: int,
    seq_len: int,
) -> None:
    top_path = case_dir / "sed_event_top_snippets.csv"
    if not top_path.exists():
        return

    top_df = pd.read_csv(top_path)
    if top_df.empty:
        return

    index_np = index.detach().cpu().numpy() if torch.is_tensor(index) else np.asarray(index)
    text_window = test_data.get_text(index)[case_index].reshape(-1).tolist()
    pooled_window = np.asarray(test_data.get_text_embedding(index)[case_index], dtype=np.float32)
    mark_case = batch_x_mark[case_index: case_index + 1].detach()

    rows = []
    summary_rows = []
    event_groups = top_df.sort_values(["event_id", "score"], ascending=[True, False]).groupby("event_id", as_index=False).head(1)
    distiller = exp.mm_model.seas_event_distiller
    adapter = exp.mlp.spectral_text_encoder

    for item in event_groups.itertuples(index=False):
        event_id = int(item.event_id)
        step_idx = int(item.step_idx)
        prompt_var = torch.tensor(pooled_window[None, :, :], device=exp.device, dtype=torch.float32, requires_grad=True)
        exp.mlp.zero_grad(set_to_none=True)
        exp.mm_model.zero_grad(set_to_none=True)
        text_spec = exp.mlp(prompt_var, mark_case)
        text_spec_seas = exp.mm_model.seas_text_spec(text_spec, channels=1, freq_len=exp.mm_model.H_f)
        event_tokens = distiller(text_spec_seas, channels=1)
        time_semantic = adapter.last_time_semantic
        event_vec = event_tokens[0, 0, event_id - 1]
        step_vec = time_semantic[0, step_idx]
        score = F.cosine_similarity(event_vec.unsqueeze(0), step_vec.unsqueeze(0), dim=-1).sum()
        score.backward()
        grad = prompt_var.grad[0, step_idx].detach().cpu()

        snippet = str(text_window[step_idx])
        enc = exp.tokenizer([snippet], return_tensors="pt", padding=True, truncation=True, max_length=256)
        ids = enc.input_ids.to(exp.device)
        masks = enc.attention_mask.to(exp.device)
        with torch.no_grad():
            embeds = exp.llm_model.get_input_embeddings()(ids)
            token_hidden = exp.llm_model(inputs_embeds=embeds).last_hidden_state[0].detach().cpu()
        tokens = exp.tokenizer.convert_ids_to_tokens(ids[0].detach().cpu().tolist())
        denom = max(int(masks[0].sum().item()), 1)
        token_scores = (token_hidden * grad.unsqueeze(0)).sum(dim=-1) / float(denom)

        valid = []
        for token, mask_value, tok_score in zip(tokens, masks[0].detach().cpu().tolist(), token_scores.tolist()):
            if int(mask_value) == 0:
                continue
            if token in {"[CLS]", "[SEP]", "[PAD]"}:
                continue
            clean = token.replace("##", "")
            if not any(ch.isalnum() for ch in clean):
                continue
            if clean.lower() in STOP_TOKENS:
                continue
            if len(clean) <= 2:
                continue
            valid.append((token, float(tok_score)))

        merged = {}
        for token, tok_score in valid:
            clean = token.replace("##", "")
            merged[clean] = merged.get(clean, 0.0) + float(tok_score)
        merged_items = sorted(merged.items(), key=lambda x: abs(x[1]), reverse=True)
        top_tokens = merged_items[:8]
        for rank, (token, tok_score) in enumerate(top_tokens, start=1):
            rows.append(
                {
                    "event": f"Event {event_id}",
                    "event_id": event_id,
                    "step_idx": step_idx,
                    "date": str(item.date),
                    "token_rank": rank,
                    "token": token,
                    "token_score": tok_score,
                    "text_snippet": clean_snippet_text(snippet, limit=220),
                }
            )
        summary_rows.append(
            {
                "event": f"Event {event_id}",
                "event_id": event_id,
                "date": str(item.date),
                "step_idx": step_idx,
                "text_snippet": clean_snippet_text(snippet, limit=180),
                "top_tokens": ", ".join(token for token, _ in top_tokens[:5]),
            }
        )

    if rows:
        pd.DataFrame(rows).to_csv(case_dir / "sed_event_top_tokens.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(case_dir / "sed_event_token_summary.csv", index=False)


def band_weighted_values(freq_values: np.ndarray, band_weights: np.ndarray) -> np.ndarray:
    denom = band_weights.sum(axis=1).clip(min=1e-8)
    return (band_weights * freq_values.reshape(1, -1)).sum(axis=1) / denom


def run_event_counterfactuals(
    exp: Exp_Long_Term_Forecast_SEAS,
    cache: Dict[str, torch.Tensor],
    full_cache: Dict[str, torch.Tensor],
    batch_x: torch.Tensor,
    batch_x_mark: torch.Tensor,
    text_embeddings: torch.Tensor,
    prior_y: torch.Tensor,
    true: torch.Tensor,
    case_index: int,
    history: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    event_tokens = full_cache.get("event_tokens_raw")
    if event_tokens is None:
        return pd.DataFrame(), pd.DataFrame()

    num_events = int(event_tokens.shape[2])
    band_weights = full_cache["band_weights"].float().numpy()
    full_delta = full_cache["delta_abs"].float().numpy()
    case_idx = min(case_index, full_delta.shape[0] - 1)
    band_names = ["Low", "Mid", "High"][: band_weights.shape[0]]

    true_case = true.detach().cpu().numpy()[case_idx, :, -1]
    forecast_rows = [
        pd.DataFrame(
            {
                "time_index": np.arange(len(history)),
                "value": history,
                "series": "History",
            }
        ),
        pd.DataFrame(
            {
                "time_index": np.arange(len(history), len(history) + len(true_case)),
                "value": true_case,
                "series": "Ground truth",
            }
        ),
    ]
    event_rows = []

    exp.mm_model._seas_event_mask = torch.ones(num_events, device=exp.device)
    with torch.no_grad():
        pred_full_freq = exp.mm_model(batch_x, batch_x_mark, None, None, text_embeddings)
        pred_full = (1 - exp.prompt_weight) * pred_full_freq + exp.prompt_weight * prior_y
    full_pred_case = pred_full.detach().cpu().numpy()[case_idx, :, -1]
    forecast_rows.append(
        pd.DataFrame(
            {
                "time_index": np.arange(len(history), len(history) + len(full_pred_case)),
                "value": full_pred_case,
                "series": "Full SEAS",
            }
        )
    )

    for event_idx in range(num_events):
        mask = torch.ones(num_events, device=exp.device)
        mask[event_idx] = 0.0
        exp.mm_model._seas_event_mask = mask
        with torch.no_grad():
            pred_freq = exp.mm_model(batch_x, batch_x_mark, None, None, text_embeddings)
            pred = (1 - exp.prompt_weight) * pred_freq + exp.prompt_weight * prior_y

        ablated_cache = clone_cache(cache)
        ablated_delta = ablated_cache["delta_abs"].float().numpy()
        freq_impact = np.abs(full_delta[case_idx].mean(axis=(0, 2)) - ablated_delta[case_idx].mean(axis=(0, 2)))
        band_impact = band_weighted_values(freq_impact, band_weights)
        pred_case = pred.detach().cpu().numpy()[case_idx, :, -1]
        diff = pred_case - true_case

        for band_name, value in zip(band_names, band_impact):
            event_rows.append(
                {
                    "event": f"Event {event_idx + 1}",
                    "event_id": event_idx + 1,
                    "band": band_name,
                    "band_impact": float(value),
                    "case_mse": float(np.mean(diff**2)),
                    "case_mae": float(np.mean(np.abs(diff))),
                }
            )
        forecast_rows.append(
            pd.DataFrame(
                {
                    "time_index": np.arange(len(history), len(history) + len(pred_case)),
                    "value": pred_case,
                    "series": f"w/o Event {event_idx + 1}",
                }
            )
        )

    exp.mm_model._seas_event_mask = None
    restore_cache(cache, full_cache)
    return pd.DataFrame(event_rows), pd.concat(forecast_rows, ignore_index=True)


def summarize_cache(cache: Dict[str, torch.Tensor]) -> Dict[str, float]:
    before = spectrum_curve(cache["spec_before"])
    after = spectrum_curve(cache["spec_after"])
    gain = tensor_mean_by_band(cache["gain"])
    phase = cache["phase"].abs().float().mean(dim=(0, 1, 3)).numpy()
    alpha = tensor_mean_by_band(cache["alpha"])
    weights = cache["band_weights"].float().numpy()
    event_norm = cache["event_tokens_raw"].float().norm(dim=-1).mean(dim=(0, 1)).numpy()

    row: Dict[str, float] = {
        "amp_before_mean": float(before.mean()),
        "amp_after_mean": float(after.mean()),
        "amp_change_ratio": float((after.mean() + 1e-8) / (before.mean() + 1e-8)),
    }
    for idx, name in enumerate(["low", "mid", "high"]):
        if idx < len(gain):
            row[f"gain_{name}"] = float(gain[idx])
            row[f"phase_abs_{name}"] = float(phase[idx])
            row[f"alpha_{name}"] = float(alpha[idx])
            row[f"band_weight_{name}_mean"] = float(weights[idx].mean())
    for idx, value in enumerate(event_norm, start=1):
        row[f"event_{idx}_norm"] = float(value)
    return row


def save_mechanism_figure(
    out_path: Path,
    domain: str,
    pred_len: int,
    history: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    cache: Dict[str, torch.Tensor],
) -> None:
    if "spec_after" not in cache and "spec_before" in cache:
        cache["spec_after"] = cache["spec_before"]
    if "gain" not in cache and "band_weights" in cache:
        num_bands = cache["band_weights"].shape[0]
        zeros = torch.zeros(1, 1, num_bands, 1)
        cache["gain"] = zeros.clone()
        cache["phase"] = zeros.clone()
        cache["alpha"] = zeros.clone()
    if "event_tokens_raw" not in cache:
        cache["event_tokens_raw"] = torch.zeros(1, 1, 4, 1)
    if "delta_abs" not in cache and "band_weights" in cache:
        cache["delta_abs"] = torch.zeros(1, 1, cache["band_weights"].shape[1], 1)

    before = spectrum_curve(cache["spec_before"])
    after = spectrum_curve(cache["spec_after"])
    weights = cache["band_weights"].float().numpy()
    gain = tensor_mean_by_band(cache["gain"])
    phase = cache["phase"].abs().float().mean(dim=(0, 1, 3)).numpy()
    alpha = tensor_mean_by_band(cache["alpha"])
    event_norm = cache["event_tokens_raw"].float().norm(dim=-1).mean(dim=(0, 1)).numpy()

    fig, axes = plt.subplots(2, 3, figsize=(17, 8.8))
    fig.suptitle(f"SEAS Spectral Modulation Case: {domain}, H={pred_len}", fontsize=16, fontweight="bold")

    hist_x = np.arange(len(history))
    fut_x = np.arange(len(history), len(history) + len(true))
    axes[0, 0].plot(hist_x, history, color="#2f6fed", linewidth=2.0, label="History")
    axes[0, 0].plot(fut_x, true, color="#222222", linewidth=2.0, label="Ground truth")
    axes[0, 0].plot(fut_x, pred, color="#e95f32", linewidth=2.0, linestyle="--", label="SEAS forecast")
    axes[0, 0].axvline(len(history) - 1, color="#999999", linewidth=1.0)
    axes[0, 0].set_title("Case forecast")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.25)

    freq = np.arange(len(before))
    axes[0, 1].plot(freq, before, color="#3b78d8", linewidth=2.2, label="Before SFM")
    axes[0, 1].plot(freq, after, color="#2f9e44", linewidth=2.2, label="After SFM")
    axes[0, 1].fill_between(freq, before, after, color="#2f9e44", alpha=0.16)
    axes[0, 1].set_title("Historical spectrum before/after modulation")
    axes[0, 1].set_xlabel("Frequency bin")
    axes[0, 1].set_ylabel("Mean amplitude")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.25)

    band_names = ["Low trend", "Mid periodic", "High shock"][: weights.shape[0]]
    band_colors = ["#3b78d8", "#f08c00", "#e03131"]
    for idx, name in enumerate(band_names):
        axes[0, 2].plot(freq, weights[idx], linewidth=2.0, color=band_colors[idx], label=name)
    axes[0, 2].set_title("ASD soft band weights")
    axes[0, 2].set_xlabel("Frequency bin")
    axes[0, 2].set_ylabel("Weight")
    axes[0, 2].legend(frameon=False)
    axes[0, 2].grid(alpha=0.25)

    x = np.arange(len(band_names))
    width = 0.26
    axes[1, 0].bar(x - width, gain[: len(band_names)], width, color="#2f9e44", label="Amplitude gain")
    axes[1, 0].bar(x, phase[: len(band_names)], width, color="#845ef7", label="|Phase shift|")
    axes[1, 0].bar(x + width, alpha[: len(band_names)], width, color="#f08c00", label="Residual alpha")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(["Low", "Mid", "High"][: len(band_names)])
    axes[1, 0].set_title("SFM semantic modulation strength")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.25)

    event_x = np.arange(len(event_norm))
    axes[1, 1].bar(event_x, event_norm, color=["#4263eb", "#f76707", "#37b24d", "#e03131"][: len(event_norm)])
    axes[1, 1].set_xticks(event_x)
    axes[1, 1].set_xticklabels([f"Event {i + 1}" for i in event_x])
    axes[1, 1].set_title("SED semantic event token norms")
    axes[1, 1].set_ylabel("Mean L2 norm")
    axes[1, 1].grid(axis="y", alpha=0.25)

    delta = cache["delta_abs"].float().mean(dim=(0, 1, 3)).numpy()
    axes[1, 2].bar(freq, delta, color="#2f9e44", alpha=0.9)
    axes[1, 2].set_title("Injected semantic residual by frequency")
    axes[1, 2].set_xlabel("Frequency bin")
    axes[1, 2].set_ylabel("Mean |delta|")
    axes[1, 2].grid(axis="y", alpha=0.25)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_arrays(
    case_dir: Path,
    cache: Dict[str, torch.Tensor],
    history: np.ndarray,
    true_case: np.ndarray,
    pred_case: np.ndarray,
) -> None:
    if "spec_after" not in cache and "spec_before" in cache:
        cache["spec_after"] = cache["spec_before"]
    if "gain" not in cache and "band_weights" in cache:
        num_bands = cache["band_weights"].shape[0]
        zeros = torch.zeros(1, 1, num_bands, 1)
        cache["gain"] = zeros.clone()
        cache["phase"] = zeros.clone()
        cache["alpha"] = zeros.clone()
        cache["delta_abs"] = torch.zeros(1, 1, cache["band_weights"].shape[1], 1)
    if "event_tokens_raw" not in cache:
        cache["event_tokens_raw"] = torch.zeros(1, 1, 4, 1)

    before = spectrum_curve(cache["spec_before"])
    after = spectrum_curve(cache["spec_after"])
    pd.DataFrame({"freq_bin": np.arange(len(before)), "before_sfm_amp": before, "after_sfm_amp": after}).to_csv(
        case_dir / "spectrum_before_after.csv", index=False
    )
    pd.DataFrame(cache["band_weights"].float().numpy()).to_csv(case_dir / "asd_band_weights.csv", index=False)

    band_names = ["low", "mid", "high"]
    mod_df = pd.DataFrame(
        {
            "band": band_names[: cache["gain"].shape[2]],
            "gain_mean": tensor_mean_by_band(cache["gain"]),
            "phase_abs_mean": cache["phase"].abs().float().mean(dim=(0, 1, 3)).numpy(),
            "residual_alpha_mean": tensor_mean_by_band(cache["alpha"]),
        }
    )
    mod_df.to_csv(case_dir / "sfm_band_modulation.csv", index=False)

    hist_x = np.arange(len(history))
    fut_x = np.arange(len(history), len(history) + len(true_case))
    forecast_df = pd.DataFrame(
        {
            "time_index": np.concatenate([hist_x, fut_x, fut_x]),
            "value": np.concatenate([history, true_case, pred_case]),
            "series": ["History"] * len(history) + ["Ground truth"] * len(true_case) + ["SEAS forecast"] * len(pred_case),
        }
    )
    forecast_df.to_csv(case_dir / "forecast_case.csv", index=False)

    event_norm = cache["event_tokens_raw"].float().norm(dim=-1).mean(dim=(0, 1)).numpy()
    pd.DataFrame(
        {
            "event": [f"Event {idx}" for idx in range(1, len(event_norm) + 1)],
            "mean_l2_norm": event_norm,
        }
    ).to_csv(case_dir / "sed_event_norms.csv", index=False)

    delta = cache["delta_abs"].float().mean(dim=(0, 1, 3)).numpy()
    pd.DataFrame(
        {
            "freq_bin": np.arange(len(delta)),
            "delta_abs_mean": delta,
        }
    ).to_csv(case_dir / "semantic_residual_by_frequency.csv", index=False)

    # Save full case-level tensors for downstream figure building.
    np.save(case_dir / "history.npy", history.astype(np.float32))
    np.save(case_dir / "true_case.npy", true_case.astype(np.float32))
    np.save(case_dir / "pred_case.npy", pred_case.astype(np.float32))
    np.save(case_dir / "spec_before.npy", cache["spec_before"].numpy())
    np.save(case_dir / "spec_after.npy", cache["spec_after"].numpy())
    np.save(case_dir / "band_weights.npy", cache["band_weights"].float().numpy())
    np.save(case_dir / "gain.npy", cache["gain"].float().numpy())
    np.save(case_dir / "phase.npy", cache["phase"].float().numpy())
    np.save(case_dir / "alpha.npy", cache["alpha"].float().numpy())
    np.save(case_dir / "delta_abs.npy", cache["delta_abs"].float().numpy())
    np.save(case_dir / "event_tokens_raw.npy", cache["event_tokens_raw"].float().numpy())


def run_case(cli: argparse.Namespace, domain: str, pred_len: int) -> Dict[str, float]:
    args = build_args(cli, domain, pred_len)
    setting = setting_name(args)
    case_dir = cli.out_dir / f"{domain}_H{pred_len}"
    case_dir.mkdir(parents=True, exist_ok=True)

    exp = Exp_Long_Term_Forecast_SEAS(args)
    ckpt_dir = load_checkpoints(exp, cli.checkpoints, setting)
    cache: Dict[str, torch.Tensor] = {}
    cache = install_asd_capture(exp, cache)
    if not bool(getattr(args, "seas_disable_sfm", 0)):
        cache = install_sfm_capture(exp)

    test_data, test_loader = exp._get_data(flag="test")
    exp.update_text_embedding(test_data)

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
        true = batch_y[:, -args.pred_len :, :]

    if not cache:
        raise RuntimeError("Mechanism capture cache is empty. Check that SEAS modules are enabled.")
    full_cache = clone_cache(cache)

    pred_np = pred.detach().cpu().numpy()
    true_np = true.detach().cpu().numpy()
    hist_np = batch_x.detach().cpu().numpy()
    mse = float(np.mean((pred_np - true_np) ** 2))
    mae = float(np.mean(np.abs(pred_np - true_np)))

    history = hist_np[cli.case_index, :, -1]
    true_case = true_np[cli.case_index, :, -1]
    pred_case = pred_np[cli.case_index, :, -1]

    event_band, event_forecast = run_event_counterfactuals(
        exp=exp,
        cache=cache,
        full_cache=full_cache,
        batch_x=batch_x,
        batch_x_mark=batch_x_mark,
        text_embeddings=text_embeddings,
        prior_y=prior_y,
        true=true,
        case_index=cli.case_index,
        history=history,
    )
    if not event_band.empty:
        event_band.to_csv(case_dir / "sed_event_band_impact.csv", index=False)
    if not event_forecast.empty:
        event_forecast.to_csv(case_dir / "sed_event_counterfactual_forecast.csv", index=False)
    save_sed_frequency_attention(case_dir, exp, cli.case_index)
    save_sed_text_alignment(case_dir, exp, test_data, index, cli.case_index, args.seq_len)
    save_sed_token_attribution(case_dir, exp, test_data, index, batch_x_mark, cli.case_index, args.seq_len)

    save_mechanism_figure(case_dir / "spectral_modulation_case.png", domain, pred_len, history, true_case, pred_case, cache)
    save_mechanism_figure(case_dir / "spectral_modulation_case.pdf", domain, pred_len, history, true_case, pred_case, cache)
    save_arrays(case_dir, cache, history, true_case, pred_case)

    text_window = test_data.get_text(index)[cli.case_index].reshape(-1).tolist()
    text_tail = [str(x) for x in text_window[-min(3, len(text_window)) :]]
    with open(case_dir / "case_report.md", "w", encoding="utf-8") as f:
        f.write(f"# {domain}, H={pred_len} 频谱调制案例\n\n")
        f.write(f"- checkpoint: `{ckpt_dir}`\n")
        f.write(f"- case_index: `{cli.case_index}`\n")
        f.write(f"- MSE: `{mse:.6f}`\n")
        f.write(f"- MAE: `{mae:.6f}`\n\n")
        f.write("## 最近文本上下文\n\n")
        for item in text_tail:
            f.write(f"- {item[:500]}\n")
        f.write("\n## 读图要点\n\n")
        f.write("- `ASD soft band weights` 展示模型把历史频谱分配到低频趋势、中频周期和高频冲击的比例。\n")
        f.write("- `SFM semantic modulation strength` 展示文本事件对不同频带的幅值、相位和残差注入强度。\n")
        f.write("- `Injected semantic residual by frequency` 展示语义信息主要注入到哪些频率位置。\n")

    summary = summarize_cache(cache)
    summary.update(
        domain=domain,
        pred_len=pred_len,
        seq_len=args.seq_len,
        mse=mse,
        mae=mae,
        checkpoint=str(ckpt_dir),
        variant=getattr(cli, "variant_tag", "full"),
    )
    print(f"[OK] {domain} H={pred_len}: MSE={mse:.6f}, MAE={mae:.6f}, output={case_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="6", help="Physical GPU id exposed through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--cases",
        default="Security:12,Climate:12,Traffic:12,Health:24",
        help="Comma-separated Domain:pred_len list.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("./data/timemmd_mmts_tats"))
    parser.add_argument("--checkpoints", type=Path, default=Path("./checkpoints"))
    parser.add_argument(
        "--llm-path",
        type=Path,
        default=Path("./pretrained/bert"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./outputs/spectral_mechanism"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--variant-tag", default="full")
    parser.add_argument("--disable-asd", type=int, default=0)
    parser.add_argument("--disable-sed", type=int, default=0)
    parser.add_argument("--disable-sfm", type=int, default=0)
    parser.add_argument("--disable-hsg", type=int, default=0)
    parser.add_argument("--disable-text-s4d", type=int, default=0)
    cli = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    cli.out_dir.mkdir(parents=True, exist_ok=True)
    cases = parse_cases(cli.cases)
    rows = []
    for domain, pred_len in cases:
        rows.append(run_case(cli, domain, pred_len))

    summary_path = cli.out_dir / "mechanism_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    with open(cli.out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("# SEAS 频谱调制可视化与案例分析\n\n")
        f.write("本目录用于 ICDE 论文中的机制分析图：ASD 频带分解、SED 事件蒸馏、SFM 语义频谱调制、预测案例。\n\n")
        f.write(f"- cases: `{cli.cases}`\n")
        f.write(f"- seed: `{cli.seed}`\n")
        f.write(f"- data_root: `{cli.data_root}`\n")
        f.write(f"- summary: `{summary_path}`\n")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
