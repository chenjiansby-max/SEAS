import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class S4DKernel(nn.Module):
    """Diagonal SSM kernel used by the lightweight text S4D encoder."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        c = torch.randn(d_model, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(c))
        self.log_dt = nn.Parameter(log_dt)

        log_a_real = torch.log(0.5 * torch.ones(d_model, N // 2))
        a_imag = math.pi * torch.arange(N // 2, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1)
        self.log_A_real = nn.Parameter(log_a_real)
        self.A_imag = nn.Parameter(a_imag)

    def forward(self, length):
        dt = torch.exp(self.log_dt)
        c = torch.view_as_complex(self.C)
        a = -torch.exp(self.log_A_real) + 1j * self.A_imag

        positions = torch.arange(length, device=a.device, dtype=a.real.dtype)
        dt_a = a * dt.unsqueeze(-1)
        kernel = dt_a.unsqueeze(-1) * positions
        c = c * (torch.exp(dt_a) - 1.0) / a
        kernel = 2 * torch.einsum("hn,hnl->hl", c, torch.exp(kernel)).real
        return kernel


class S4D(nn.Module):
    """Minimal S4D block for sequence modeling on text embeddings."""

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True):
        super().__init__()
        self.h = d_model
        self.transposed = transposed
        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DKernel(d_model=self.h, N=d_state)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, x):
        if not self.transposed:
            x = x.transpose(-1, -2)

        length = x.size(-1)
        kernel = self.kernel(length)

        kernel_f = torch.fft.rfft(kernel, n=2 * length)
        x_f = torch.fft.rfft(x, n=2 * length)
        y = torch.fft.irfft(x_f * kernel_f, n=2 * length)[..., :length]
        y = y + x * self.D.unsqueeze(-1)
        y = self.output_linear(self.dropout(self.activation(y)))

        if not self.transposed:
            y = y.transpose(-1, -2)
        return y


class TextS4DSpectralAdapter(nn.Module):
    """Turns per-step text embeddings into a low-frequency semantic spectrum."""

    def __init__(self, llm_emb_size, text_emb, mm_emb_size, seq_len, freq_cut_off_rate=1.0,
                 d_state=64, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.h_f = seq_len // 2 + 1
        self.low_freq = max(1, int(self.h_f * freq_cut_off_rate))
        self.last_time_semantic = None
        self.last_text_spectrum = None

        self.input_proj = nn.Sequential(
            nn.Linear(llm_emb_size, text_emb),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.s4d = S4D(d_model=text_emb, d_state=d_state, dropout=dropout, transposed=True)
        self.norm = nn.LayerNorm(text_emb)
        self.real_proj = nn.Linear(text_emb, mm_emb_size)
        self.imag_proj = nn.Linear(text_emb, mm_emb_size)
        self.real_norm = nn.LayerNorm(mm_emb_size)
        self.imag_norm = nn.LayerNorm(mm_emb_size)
        self.sparsity_threshold = 0.01

    def forward(self, text_embeddings):
        text_hidden = self.input_proj(text_embeddings)
        text_hidden = text_hidden.transpose(1, 2)
        text_hidden = self.s4d(text_hidden).transpose(1, 2)
        text_hidden = self.norm(text_hidden)

        text_freq = torch.fft.rfft(text_hidden, dim=1, norm="ortho")[:, :self.low_freq]
        real = self.real_proj(text_freq.real) - self.imag_proj(text_freq.imag)
        imag = self.real_proj(text_freq.imag) + self.imag_proj(text_freq.real)
        real = self.real_norm(real)
        imag = self.imag_norm(imag)

        text_spec = torch.stack([real, imag], dim=-1)
        text_spec = F.softshrink(text_spec, lambd=self.sparsity_threshold)
        text_spec = torch.view_as_complex(text_spec)
        self.last_text_spectrum = text_spec.detach()
        self.last_time_semantic = torch.fft.irfft(text_spec, n=self.seq_len, dim=1, norm="ortho").detach()
        return text_spec.unsqueeze(1)
