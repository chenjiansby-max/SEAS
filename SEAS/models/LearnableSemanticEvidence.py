import torch
import torch.nn as nn


class LearnableSemanticEvidence(nn.Module):
    """Forecast-aware refinement for per-step text embeddings."""

    def __init__(self, text_dim, ts_dim=1, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.ts_proj = nn.Sequential(
            nn.Linear(ts_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score = nn.Linear(hidden_dim, 1)
        self.value = nn.Linear(text_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, text_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, text_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, ts_tokens, text_tokens):
        if ts_tokens.dim() != 3 or text_tokens.dim() != 3:
            raise ValueError("Expected ts_tokens [B,L,C] and text_tokens [B,L,D].")

        if ts_tokens.size(1) != text_tokens.size(1):
            min_len = min(ts_tokens.size(1), text_tokens.size(1))
            ts_tokens = ts_tokens[:, :min_len]
            text_tokens = text_tokens[:, :min_len]

        query = self.ts_proj(ts_tokens)
        key = self.text_proj(text_tokens)
        evidence_logits = self.score(torch.tanh(query + key))
        evidence_weight = torch.softmax(evidence_logits, dim=1)

        evidence = torch.sum(evidence_weight * self.value(text_tokens), dim=1, keepdim=True)
        evidence = evidence.expand(-1, text_tokens.size(1), -1)
        refined = self.out(evidence)
        gate = self.gate(query)

        return self.norm(text_tokens + gate * refined)
