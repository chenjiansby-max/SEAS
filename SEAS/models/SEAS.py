import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Embed import TextEmbedding_wo_pos, PositionalEmbedding
from layers.FreqAttention_Family import FullFreqAttention, FreqAttentionLayer
from models.SpectralTextS4D import TextS4DSpectralAdapter
from utils.masking import TriangularCausalMask, ProbMask, ConstantMask


class my_Layernorm(nn.Module):
    def __init__(self, eps=1e-5, elementwise_affine=True):
        """
        Custom LayerNorm for complex vectors with 4D input.
        
        Args:
            normalized_shape: Shape of the last dimension to normalize (e.g., d_model).
            eps: A value added to the denominator for numerical stability.
            elementwise_affine: A boolean value that when set to True, this module
                                has learnable per-element affine parameters.
        """
        super(my_Layernorm, self).__init__()
        # self.normalized_shape = normalized_shape
        self.eps = eps
        # self.elementwise_affine = elementwise_affine


        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

    def forward(self, input):
        # Compute the magnitude of the complex tensor
        magnitude = torch.abs(input)

        # Compute mean and variance along the last dimension (d_model)
        mean = magnitude.mean(dim=-1, keepdim=True)
        var = magnitude.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize the input based on the magnitude
        normalized_input = (input - mean) / torch.sqrt(var + self.eps)

        # Apply learnable affine transformation if enabled
        # if self.elementwise_affine:
        #     normalized_input = normalized_input * self.weight + self.bias

        return normalized_input



class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)  
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  
                x = F.relu(x)
                x = self.dropout(x)  
            x = F.relu(x)
            x = self.dropout(x)
        return x

class TextEncoder(nn.Module):
    """Build either the original SEAS text spectrum or the SEAS text spectrum."""

    def __init__(self, configs):
        super(TextEncoder, self).__init__()
        self.configs = configs
        self.text_emb = configs.text_emb
        disable_text_s4d = bool(getattr(configs, "seas_disable_text_s4d", 0))
        self.use_text_s4d_spectral = bool(
            (getattr(configs, "use_text_s4d_spectral", 0) or getattr(configs, "use_seas", 0)) and
            not disable_text_s4d
        )

        self.embed_size = configs.mm_emb_size  # embed_size
        self.hidden_size = configs.mm_hidden_size  # hidden_size
        self.pred_len = configs.pred_len
        # self.feature_size = configs.enc_in  # channels
        self.feature_size = configs.n_ts_features
        self.seq_len = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 1
        self.T_f = self.pred_len//2+1
        self.H_f = self.seq_len//2+1

        self.dominance_freq = int(self.H_f)
        self.dominance_freq_pred = int(self.T_f)
        
        if self.use_text_s4d_spectral:
            self.spectral_text_encoder = TextS4DSpectralAdapter(
                llm_emb_size=configs.llm_emb_size,
                text_emb=self.text_emb,
                mm_emb_size=self.embed_size,
                seq_len=self.seq_len,
                freq_cut_off_rate=configs.freq_cut_off_rate,
                d_state=getattr(configs, "text_s4d_state", 64),
                dropout=getattr(configs, "text_s4d_dropout", configs.text_dropout),
            )
        else:
            self.text_proj_layer = MLP(layer_sizes=[configs.llm_emb_size, self.text_emb], dropout_rate=configs.text_dropout)
            self.text_embed_layer = TextEmbedding_wo_pos(configs.llm_emb_size, configs.llm_emb_size, configs.embed, configs.freq, configs.dropout)
            self.r1 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.text_emb, self.dominance_freq*self.embed_size))
            self.i1 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.text_emb, self.dominance_freq*self.embed_size))
            self.rb1 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.dominance_freq, self.embed_size))
            self.ib1 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.dominance_freq, self.embed_size))
        
        
    def forward(self, text_embeddings, x_mark_enc):
        if self.use_text_s4d_spectral:
            return self.spectral_text_encoder(text_embeddings)

        B = text_embeddings.shape[0]
        N_c = 1
        text_embeddings = self.text_embed_layer(text_embeddings, x_mark_enc)
        text_embeddings = self.text_proj_layer(text_embeddings) 

        text_embeddings = text_embeddings.unsqueeze(1).repeat(1, N_c, 1, 1) # B x N_c x N_t x D
        
        r1 = self.r1.unsqueeze(0).repeat(B,1,1,1) # 16 x 3 x 6 x 1200
        text_real = torch.matmul(text_embeddings, r1).reshape(B, N_c, self.seq_len, self.dominance_freq, self.embed_size)
        # text_real = text_real[:,:,:,0,:].unsqueeze(3).repeat(1,1,1,self.dominance_freq,1)
        # text_real = torch.sum(text_real, dim=2)
        # text_real = F.relu(text_real+self.rb1)

        i1 = self.i1.unsqueeze(0).repeat(B,1,1,1) # 16 x 3 x 6 x 1200
        text_imag = torch.matmul(text_embeddings, i1).reshape(B, N_c, self.seq_len, self.dominance_freq, self.embed_size)
        # text_imag = text_imag[:,:,:,0,:].unsqueeze(3).repeat(1,1,1,self.dominance_freq,1)
        # text_imag = torch.sum(text_imag, dim=2)
        # text_imag = F.relu(text_imag+self.ib1)

        # if self.dominance_freq != self.H_f:
        #     real_padding = torch.zeros(B, N_c, self.H_f - self.dominance_freq, self.embed_size).to(text_imag.device)
        #     imag_padding = torch.zeros(B, N_c, self.H_f - self.dominance_freq, self.embed_size).to(text_imag.device)
        #     text_real = torch.cat([text_real, real_padding], dim=2)
        #     text_imag = torch.cat([text_imag, imag_padding], dim=2)
        
        text = torch.stack([text_real, text_imag], dim=-1)
        text = F.softshrink(text, lambd=self.sparsity_threshold)
        text = torch.view_as_complex(text)
        
        return text

class FreqMLP(nn.Module):
    def __init__(self, d_in, d_out):
        super(FreqMLP, self).__init__()
        self.freq_upsampler_real = nn.Linear(d_in, d_out) # complex layer for frequency upcampling]
        self.freq_upsampler_imag = nn.Linear(d_in, d_out) 
        self.sparsity_threshold = 0.01
        self.real_norm = nn.InstanceNorm2d(1)
        self.imag_norm = nn.InstanceNorm2d(1)
    def forward(self, x):
        real = self.freq_upsampler_real(x.real)-self.freq_upsampler_imag(x.imag)
        imag = self.freq_upsampler_real(x.imag)+self.freq_upsampler_imag(x.real)
        real = F.relu(real)
        imag = F.relu(imag)
        # low_specxy_real = self.real_norm_3(low_specxy_real.permute(0,1,3,2)).permute(0,1,3,2)
        # low_specxy_imag = self.imag_norm_3(low_specxy_imag.permute(0,1,3,2)).permute(0,1,3,2)
        real = real + x.real
        imag = imag + x.imag

        real = self.real_norm(real)
        imag = self.imag_norm(imag)
        
        x = torch.stack([real, imag], dim=-1)
        # text = torch.stack([real, imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        return x


class AdaptiveSpectralDecomposition(nn.Module):
    """Learn soft low/mid/high frequency bands and return band-level spectrum tokens."""

    def __init__(self, freq_len, embed_size, num_bands=3, max_residual=0.25):
        super().__init__()
        self.freq_len = freq_len
        self.num_bands = num_bands
        self.max_residual = max_residual

        centers = torch.linspace(0.0, 1.0, steps=num_bands).unsqueeze(1)
        freq_grid = torch.linspace(0.0, 1.0, steps=freq_len).unsqueeze(0)
        init_logits = -((freq_grid - centers) ** 2) * max(4.0, float(num_bands * num_bands))
        self.band_logits = nn.Parameter(init_logits)
        self.residual_logit = nn.Parameter(torch.tensor(-2.0))
        self.real_proj = nn.Sequential(nn.Linear(embed_size, embed_size), nn.GELU(), nn.Linear(embed_size, embed_size))
        self.imag_proj = nn.Sequential(nn.Linear(embed_size, embed_size), nn.GELU(), nn.Linear(embed_size, embed_size))
        self.real_norm = nn.LayerNorm(embed_size)
        self.imag_norm = nn.LayerNorm(embed_size)

    def forward(self, spec):
        weights = torch.softmax(self.band_logits, dim=0)  # [K, F], each frequency softly assigned to bands
        normalizer = weights.sum(dim=1).clamp_min(1e-6).view(1, 1, self.num_bands, 1)

        band_real = torch.einsum("bcfd,kf->bckd", spec.real, weights) / normalizer
        band_imag = torch.einsum("bcfd,kf->bckd", spec.imag, weights) / normalizer
        band_real = self.real_norm(self.real_proj(band_real))
        band_imag = self.imag_norm(self.imag_proj(band_imag))
        band_tokens = torch.complex(band_real, band_imag)

        recon_real = torch.einsum("bckd,kf->bcfd", band_real, weights)
        recon_imag = torch.einsum("bckd,kf->bcfd", band_imag, weights)
        residual_scale = torch.sigmoid(self.residual_logit) * self.max_residual
        decomposed = torch.complex(
            spec.real + residual_scale * recon_real,
            spec.imag + residual_scale * recon_imag,
        )
        return decomposed, band_tokens, weights


class SemanticEventDistiller(nn.Module):
    """Distill text spectra into a small set of semantic event tokens."""

    def __init__(self, embed_size, num_events=4, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.num_events = num_events
        self.event_queries = nn.Parameter(torch.randn(num_events, embed_size) * 0.02)
        self.last_attention_weights = None
        self.last_text_hidden = None
        self.last_event_tokens = None
        self.input_proj = nn.Sequential(
            nn.Linear(embed_size * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_size),
        )
        self.event_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=1, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_size),
        )

    def forward(self, text_spec, channels):
        if text_spec is None or text_spec.dim() != 4:
            raise ValueError("SEAS expects complex text spectrum [B,C,F,D].")

        text_hidden = torch.cat([text_spec.real, text_spec.imag], dim=-1).mean(dim=1)
        text_hidden = self.input_proj(text_hidden)
        queries = self.event_queries.unsqueeze(0).expand(text_hidden.size(0), -1, -1)
        events, attn_weights = self.event_attn(
            queries,
            text_hidden,
            text_hidden,
            need_weights=True,
            average_attn_weights=True,
        )
        events = self.norm(events + self.ffn(events))
        self.last_attention_weights = attn_weights.detach()
        self.last_text_hidden = text_hidden.detach()
        self.last_event_tokens = events.detach()
        return events.unsqueeze(1).expand(-1, channels, -1, -1)


class SemanticFrequencyModulatorV4(nn.Module):
    """Apply semantic-conditioned amplitude, phase, and residual modulation to spectral bands."""

    def __init__(self, embed_size, num_bands=3, hidden_dim=64, max_gain=0.5, max_phase=0.5, max_residual=0.3):
        super().__init__()
        self.num_bands = num_bands
        self.max_gain = max_gain
        self.max_phase = max_phase
        self.max_residual = max_residual
        self.modulator = nn.Sequential(
            nn.Linear(embed_size * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_size * 5),
        )
        self.real_norm = nn.LayerNorm(embed_size)
        self.imag_norm = nn.LayerNorm(embed_size)

    def forward(self, spec, band_tokens, event_tokens, band_weights):
        if event_tokens.size(2) != self.num_bands:
            event_tokens = F.interpolate(
                event_tokens.permute(0, 1, 3, 2).reshape(-1, event_tokens.size(-1), event_tokens.size(2)),
                size=self.num_bands,
                mode="linear",
                align_corners=False,
            ).reshape(event_tokens.size(0), event_tokens.size(1), event_tokens.size(-1), self.num_bands).permute(0, 1, 3, 2)

        cond = torch.cat([band_tokens.real, band_tokens.imag, event_tokens], dim=-1)
        gain, phase, alpha, delta_real, delta_imag = self.modulator(cond).chunk(5, dim=-1)
        gain = torch.tanh(gain) * self.max_gain
        phase = torch.tanh(phase) * self.max_phase
        alpha = torch.sigmoid(alpha) * self.max_residual

        amp = 1.0 + gain
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        rotated_real = amp * (band_tokens.real * cos_phase - band_tokens.imag * sin_phase)
        rotated_imag = amp * (band_tokens.real * sin_phase + band_tokens.imag * cos_phase)
        enhanced_real = self.real_norm(rotated_real + alpha * delta_real)
        enhanced_imag = self.imag_norm(rotated_imag + alpha * delta_imag)

        delta_band_real = enhanced_real - band_tokens.real
        delta_band_imag = enhanced_imag - band_tokens.imag
        delta_real = torch.einsum("bckd,kf->bcfd", delta_band_real, band_weights)
        delta_imag = torch.einsum("bckd,kf->bcfd", delta_band_imag, band_weights)
        return torch.complex(spec.real + delta_real, spec.imag + delta_imag)


class HorizonAwareSpectralGenerator(nn.Module):
    """Generate future spectra conditioned on the target horizon length."""

    def __init__(self, hist_freq_len, fut_freq_len, embed_size):
        super().__init__()
        self.hist_freq_len = hist_freq_len
        self.fut_freq_len = fut_freq_len
        self.horizon_proj = nn.Sequential(nn.Linear(1, embed_size), nn.GELU(), nn.Linear(embed_size, embed_size))
        self.freq_real = nn.Linear(hist_freq_len, fut_freq_len)
        self.freq_imag = nn.Linear(hist_freq_len, fut_freq_len)
        self.real_norm = nn.LayerNorm(embed_size)
        self.imag_norm = nn.LayerNorm(embed_size)

    def forward(self, real, imag, pred_len):
        horizon = real.new_full((real.size(0), 1, 1, 1), float(pred_len) / 512.0)
        horizon_emb = self.horizon_proj(horizon)
        real = real + horizon_emb
        imag = imag + horizon_emb
        out_real = self.freq_real(real.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) - \
            self.freq_imag(imag.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_imag = self.freq_real(imag.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) + \
            self.freq_imag(real.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return self.real_norm(out_real), self.imag_norm(out_imag)

        
class FreqModelHistPred(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2311.06184.pdf

    SEAS extension: Semantic Spectral State Fusion adds an S4D text semantic
    sequence encoder and fuses its spectrum with the time-series spectrum.
    """

    def __init__(self, configs):
        super(FreqModelHistPred, self).__init__()
        self.task_name = configs.task_name
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.embed_size = configs.mm_emb_size  # embed_size
        self.hidden_size = configs.mm_hidden_size  # hidden_size
        self.pred_len = configs.pred_len
        # self.feature_size = configs.enc_in  # channels
        self.feature_size = configs.n_ts_features
        self.seq_len = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 1
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        
        self.configs = configs
        self.use_seas = bool(getattr(configs, "use_seas", 0))
        self.seas_disable_text_s4d = bool(getattr(configs, "seas_disable_text_s4d", 0))
        self.use_text_s4d_spectral = bool(
            (getattr(configs, "use_text_s4d_spectral", 0) or self.use_seas) and
            not self.seas_disable_text_s4d
        )
        if not configs.proj_per_freq:
            self.fc = nn.Sequential(
                nn.Linear(self.pred_len * self.embed_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.pred_len)
            )
        else:
            print('Project per frequency')
            self.fc = nn.Linear(self.embed_size, 1)
            # self.fc = nn.Sequential(
            #     nn.Linear(self.embed_size, self.embed_size//2),
            #     nn.LeakyReLU(),
            #     nn.Dropout(0.1),
            #     nn.Linear(self.embed_size//2, 1)
            # )
        self.text_emb = configs.text_emb
        # self.text_emb = self.embed_size
        llm_dim=4096
        self.text_proj_layer = MLP(layer_sizes=[llm_dim, int(llm_dim/8), self.text_emb], dropout_rate=0.3)
        # mlp_sizes = [768, int(768/8), self.text_emb]
        # self.text_proj_layer = nn.Sequential(
        #         nn.Linear(mlp_sizes[0], mlp_sizes[1]),
        #         nn.ReLU(),
        #         nn.Linear(mlp_sizes[1], mlp_sizes[2]),
        #         nn.ReLU(),
        #         nn.Dropout(0.3)
        #     )
        self.text_embed_layer = TextEmbedding_wo_pos(768, 768, configs.embed, configs.freq, configs.dropout)
        self.T_f = self.pred_len//2+1
        self.H_f = self.seq_len//2+1

        self.dominance_freq = int(self.H_f * configs.freq_cut_off_rate)
        self.dominance_freq_pred = int(self.T_f * configs.freq_cut_off_rate)

        self.r1 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.text_emb, self.dominance_freq*self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.text_emb, self.dominance_freq*self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.dominance_freq, self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.dominance_freq, self.embed_size))

        self.r2 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.text_emb, self.dominance_freq_pred*self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.text_emb, self.dominance_freq_pred*self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.dominance_freq_pred, self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.dominance_freq_pred, self.embed_size))

        self.freq_upsampler_real = nn.Linear(self.H_f, self.T_f) # complex layer for frequency upcampling]
        self.freq_upsampler_imag = nn.Linear(self.H_f, self.T_f) 
        if self.configs.only_text_input:
            self.freq_upsampler_real = nn.Linear(self.seq_len, self.T_f) # complex layer for frequency upcampling]
            self.freq_upsampler_imag = nn.Linear(self.seq_len, self.T_f)     
        # self.freq_upsampler_real = nn.Sequential(
        #         nn.Linear(self.H_f, self.H_f),
        #         nn.ReLU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(self.H_f, self.T_f),
        #     )
        # self.freq_upsampler_imag = nn.Sequential(
        #         nn.Linear(self.H_f, self.H_f),
        #         nn.ReLU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(self.H_f, self.T_f),
        #     )
        self.real_norm_1 = nn.LayerNorm(self.embed_size)
        self.imag_norm_1 = nn.LayerNorm(self.embed_size)
        self.real_norm_2 = nn.InstanceNorm2d(1)
        self.imag_norm_2 = nn.InstanceNorm2d(1)
        self.real_norm_3 = nn.InstanceNorm2d(1)
        self.imag_norm_3 = nn.InstanceNorm2d(1)
        self.real_norm_4 = nn.BatchNorm2d(1)
        self.imag_norm_4 = nn.BatchNorm2d(1)

        self.norm_1 = my_Layernorm()
        self.norm_2 = my_Layernorm()

        scale = 0.1
        self.embedding_real = nn.Parameter(scale * torch.randn(1, self.embed_size))
        self.embedding_imag = nn.Parameter(scale * torch.randn(1, self.embed_size))

        self.embedding_real_ffd = nn.Parameter(torch.randn(self.embed_size, self.embed_size))
        self.embedding_imag_ffd = nn.Parameter(torch.randn(self.embed_size, self.embed_size))
        
        self.embedding_real_dec = nn.Parameter(scale * torch.randn(self.embed_size, 1))
        self.embedding_imag_dec = nn.Parameter(scale * torch.randn(self.embed_size, 1))
        
        self.freq_attn_layer = FreqAttentionLayer(
                        FullFreqAttention(False, 5, attention_dropout=configs.dropout,
                                      output_attention=False),
                        d_model=self.embed_size, n_heads=1)

        self.freq_attn_layer_2 = FreqAttentionLayer(
                        FullFreqAttention(False, 5, attention_dropout=configs.dropout,
                                      output_attention=False),
                        d_model=self.embed_size, n_heads=1)

        self.position_emb = PositionalEmbedding(self.embed_size)
        self.dropout = nn.Dropout(0.1)
        self.hist_ffn = FreqMLP(self.embed_size, self.embed_size)
        self.fut_ffn = FreqMLP(self.embed_size, self.embed_size)
        if self.use_seas:
            num_bands = getattr(configs, "seas_num_bands", 3)
            self.seas_num_bands = num_bands
            self.seas_disable_asd = bool(getattr(configs, "seas_disable_asd", 0))
            self.seas_disable_sed = bool(getattr(configs, "seas_disable_sed", 0))
            self.seas_disable_sfm = bool(getattr(configs, "seas_disable_sfm", 0))
            self.seas_disable_hsg = bool(getattr(configs, "seas_disable_hsg", 0))
            hidden_dim = getattr(configs, "seas_hidden_dim", 64)
            self.seas_decomposer = AdaptiveSpectralDecomposition(
                freq_len=self.H_f,
                embed_size=self.embed_size,
                num_bands=num_bands,
                max_residual=getattr(configs, "seas_decomp_residual", 0.25),
            )
            self.seas_event_distiller = SemanticEventDistiller(
                embed_size=self.embed_size,
                num_events=getattr(configs, "seas_num_events", num_bands),
                hidden_dim=hidden_dim,
                dropout=getattr(configs, "text_s4d_dropout", configs.text_dropout),
            )
            self.seas_modulator = SemanticFrequencyModulatorV4(
                embed_size=self.embed_size,
                num_bands=num_bands,
                hidden_dim=hidden_dim,
                max_gain=getattr(configs, "seas_max_gain", 0.5),
                max_phase=getattr(configs, "seas_max_phase", 0.5),
                max_residual=getattr(configs, "seas_max_residual", 0.3),
            )
            self.seas_generator = HorizonAwareSpectralGenerator(
                hist_freq_len=self.H_f,
                fut_freq_len=self.T_f,
                embed_size=self.embed_size,
            )
        if self.use_text_s4d_spectral:
            fusion_hidden = getattr(configs, "text_fusion_hidden_dim", self.embed_size * 2)
            self.seas_fusion_mode = getattr(configs, "seas_fusion_mode", "residual")
            self.seas_text_max_residual = getattr(configs, "seas_text_max_residual", 0.2)
            self.seas_low_freq_ratio = getattr(configs, "seas_low_freq_ratio", 0.35)
            self.text_freq_gate = nn.Sequential(
                nn.Linear(self.embed_size * 4, fusion_hidden),
                nn.GELU(),
                nn.Dropout(getattr(configs, "text_s4d_dropout", configs.text_dropout)),
                nn.Linear(fusion_hidden, self.embed_size),
                nn.Sigmoid(),
            )
            self.text_residual_logit = nn.Parameter(torch.tensor(getattr(configs, "seas_residual_init", -4.0)))
            low_init = getattr(configs, "seas_low_residual_init", -2.0)
            high_init = getattr(configs, "seas_high_residual_init", -5.0)
            band_init = torch.full((self.H_f,), high_init)
            low_count = max(1, int(self.H_f * self.seas_low_freq_ratio))
            band_init[:low_count] = low_init
            self.text_band_residual_logit = nn.Parameter(band_init.view(1, 1, self.H_f, 1))
            self.text_freq_prior = nn.Parameter(torch.zeros(1, 1, self.H_f, self.embed_size))
            self.text_real_norm = nn.LayerNorm(self.embed_size)
            self.text_imag_norm = nn.LayerNorm(self.embed_size)

    def spectral_text_fusion(self, ts_spec, text_spec):
        if text_spec is None:
            return ts_spec.real, ts_spec.imag

        if text_spec.dim() != 4:
            raise ValueError("Text S4D spectral fusion expects complex text tensor [B,C,F,D].")

        if text_spec.size(1) == 1 and ts_spec.size(1) > 1:
            text_spec = text_spec.expand(-1, ts_spec.size(1), -1, -1)
        elif text_spec.size(1) != ts_spec.size(1):
            raise ValueError("Text spectrum channel count must match time-series channels.")

        if text_spec.size(2) < ts_spec.size(2):
            pad_size = ts_spec.size(2) - text_spec.size(2)
            real_pad = torch.zeros(
                text_spec.size(0), text_spec.size(1), pad_size, text_spec.size(3),
                device=text_spec.device, dtype=text_spec.real.dtype
            )
            imag_pad = torch.zeros_like(real_pad)
            real = torch.cat([text_spec.real, real_pad], dim=2)
            imag = torch.cat([text_spec.imag, imag_pad], dim=2)
            text_spec = torch.complex(real, imag)
        elif text_spec.size(2) > ts_spec.size(2):
            text_spec = text_spec[:, :, :ts_spec.size(2), :]

        gate_input = torch.cat([ts_spec.real, ts_spec.imag, text_spec.real, text_spec.imag], dim=-1)
        gate = self.text_freq_gate(gate_input)

        if self.seas_fusion_mode == "strong":
            fused_real = self.text_real_norm(ts_spec.real + gate * text_spec.real)
            fused_imag = self.text_imag_norm(ts_spec.imag + gate * text_spec.imag)
            return fused_real, fused_imag

        ts_mag = ts_spec.abs().mean(dim=-1, keepdim=True)
        text_mag = text_spec.abs().mean(dim=-1, keepdim=True)
        mag_scale = torch.where(
            ts_mag > 1e-6,
            (ts_mag / (text_mag + 1e-6)).clamp(0.1, 10.0),
            torch.ones_like(ts_mag),
        )
        band_gate = torch.sigmoid(self.text_freq_prior[:, :, :ts_spec.size(2), :])
        if self.seas_fusion_mode == "band_residual":
            residual_scale = torch.sigmoid(
                self.text_band_residual_logit[:, :, :ts_spec.size(2), :]
            ) * self.seas_text_max_residual
        else:
            residual_scale = torch.sigmoid(self.text_residual_logit) * self.seas_text_max_residual
        delta_real = gate * band_gate * text_spec.real * mag_scale
        delta_imag = gate * band_gate * text_spec.imag * mag_scale
        fused_real = ts_spec.real + residual_scale * delta_real
        fused_imag = ts_spec.imag + residual_scale * delta_imag
        return fused_real, fused_imag

    def seas_plain_band_tokens(self, spec):
        num_bands = self.seas_num_bands
        centers = torch.linspace(0.0, 1.0, steps=num_bands, device=spec.device).unsqueeze(1)
        freq_grid = torch.linspace(0.0, 1.0, steps=spec.size(2), device=spec.device).unsqueeze(0)
        weights = torch.softmax(-((freq_grid - centers) ** 2) * max(4.0, float(num_bands * num_bands)), dim=0)
        normalizer = weights.sum(dim=1).clamp_min(1e-6).view(1, 1, num_bands, 1)
        band_real = torch.einsum("bcfd,kf->bckd", spec.real, weights) / normalizer
        band_imag = torch.einsum("bcfd,kf->bckd", spec.imag, weights) / normalizer
        return torch.complex(band_real, band_imag), weights

    def seas_text_spec(self, text_spec, channels, freq_len):
        if text_spec is None:
            return None
        if text_spec.dim() == 5:
            text_spec = text_spec.mean(dim=2)
        elif text_spec.dim() != 4:
            return None

        if text_spec.size(1) == 1 and channels > 1:
            text_spec = text_spec.expand(-1, channels, -1, -1)
        elif text_spec.size(1) != channels:
            return None

        if text_spec.size(2) < freq_len:
            pad_size = freq_len - text_spec.size(2)
            real_pad = torch.zeros(
                text_spec.size(0), text_spec.size(1), pad_size, text_spec.size(3),
                device=text_spec.device, dtype=text_spec.real.dtype
            )
            imag_pad = torch.zeros_like(real_pad)
            text_spec = torch.complex(
                torch.cat([text_spec.real, real_pad], dim=2),
                torch.cat([text_spec.imag, imag_pad], dim=2),
            )
        elif text_spec.size(2) > freq_len:
            text_spec = text_spec[:, :, :freq_len, :]
        return text_spec

    def seas_plain_event_tokens(self, text_spec, channels):
        if text_spec is None:
            return torch.zeros(
                1, channels, self.seas_num_bands, self.embed_size,
                device=self.embedding_real.device
            )
        event = 0.5 * (text_spec.real.mean(dim=2) + text_spec.imag.mean(dim=2))
        return event.unsqueeze(2).expand(-1, -1, self.seas_num_bands, -1)
        
    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    def realEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embedding_real
        return x * y

    def imagEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embedding_imag
        return x * y

    def realEmb_dec(self, x):
        # x: [Batch, Input length, Channel]
        # x = x.permute(0, 2, 1)
        # x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embedding_real_dec.unsqueeze(0).unsqueeze(0)
        return torch.matmul(x,y)

    def imagEmb_dec(self, x):
        # x: [Batch, Input length, Channel]
        # x = x.permute(0, 2, 1)
        # x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embedding_imag_dec.unsqueeze(0).unsqueeze(0)
        return torch.matmul(x,y)

    def realEmb_ffd(self, x):
        # x: [Batch, Input length, Channel]
        # x = x.permute(0, 2, 1)
        # x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embedding_real_ffd.unsqueeze(0).unsqueeze(0)
        return torch.matmul(x,y)

    def imagEmb_ffd(self, x):
        # x: [Batch, Input length, Channel]
        # x = x.permute(0, 2, 1)
        # x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embedding_imag_ffd.unsqueeze(0).unsqueeze(0)
        return torch.matmul(x,y)
        

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
                 - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, text_embeddings=None):
        # x: [Batch, Input length, Channel]
        # B, T, N_c = x_dec.shape
        # embedding x: [B, N, T, D]
        
        # normalize
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x = torch.fft.rfft(x_enc, dim=1, norm='ortho')  # B x N x N_c

        # x = self.tokenEmb(x)
        x_real = self.realEmb(x.real) - self.imagEmb(x.imag)
        x_imag = self.imagEmb(x.real) + self.realEmb(x.imag)

        sample = x_real[:,0,:,:]
        position_emb = self.position_emb(sample).unsqueeze(1)
        
        x_real += position_emb
        x_imag += position_emb
        
        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)

        B, N_c, T, D = x_real.shape
        bias = x

        text = text_embeddings

        used_seas_fusion = False
        text_spec_seas = self.seas_text_spec(text_embeddings, N_c, x.size(2)) if self.use_seas else None
        if self.use_seas and text_spec_seas is not None:
            if self.seas_disable_asd:
                band_tokens, band_weights = self.seas_plain_band_tokens(x)
            else:
                x, band_tokens, band_weights = self.seas_decomposer(x)
            if self.seas_disable_sed:
                event_tokens = self.seas_plain_event_tokens(text_spec_seas, channels=N_c)
            else:
                event_tokens = self.seas_event_distiller(text_spec_seas, channels=N_c)
            event_mask = getattr(self, "_seas_event_mask", None)
            if event_mask is not None:
                event_mask = event_mask.to(event_tokens.device, dtype=event_tokens.dtype).view(1, 1, -1, 1)
                if event_mask.size(2) == event_tokens.size(2):
                    event_tokens = event_tokens * event_mask
            if not self.seas_disable_sfm:
                x = self.seas_modulator(x, band_tokens, event_tokens, band_weights)
            real, imag = x.real, x.imag
            used_seas_fusion = True
        elif self.use_text_s4d_spectral and text_embeddings is not None and text_embeddings.dim() == 4:
            if self.configs.only_text_input:
                real, imag = self.spectral_text_fusion(torch.zeros_like(x), text_embeddings)
            else:
                real, imag = self.spectral_text_fusion(x, text_embeddings)
            used_seas_fusion = True
        elif self.configs.fuse_history:
            attn_mask = ConstantMask(x.real.shape[0], x.real.shape[1], text_embeddings.shape[1], device=x_enc.device)
            B, N_c, H_f, D = x.shape
            B, N_c, H, H_f, D = text.shape
            x = x.reshape(B*N_c, H_f, D).contiguous()
            text = text[:,:,:,0,:].reshape(B*N_c, H, D)

            # print('text0: ', text.shape)
            # print('x0: ', x.shape)
            
            output, attn_score = self.freq_attn_layer(x, text, text, attn_mask)
            
            output = output.reshape(B, N_c, H_f, D).contiguous()
            x = x.reshape(B, N_c, H_f, D).contiguous()
            # text = text.reshape(B, N_c, H, H_f, D)
            if self.dominance_freq != self.H_f:
                one_mask = torch.ones(B, N_c, self.dominance_freq, self.embed_size).to(output.device)
                zero_mask = torch.ones(B, N_c, self.H_f - self.dominance_freq, self.embed_size).to(output.device)
                mask = torch.cat([one_mask, zero_mask], dim=2)
                output = output*mask
            # real = output.real + bias.real
            # imag = output.imag + bias.imag

            # real = self.real_norm_1(real)
            # imag = self.imag_norm_1(imag)
            # text += x
            if self.configs.use_product:
                real = torch.mul(output.real, x.real) - torch.mul(output.imag, x.imag)
                imag = torch.mul(output.real, x.imag) + torch.mul(output.imag, x.real)
            else:
                real = output.real
                imag = output.imag
            # real = F.relu(real)
            # imag = F.relu(imag)

            real += x.real
            imag += x.imag
            # real += self.dropout(x.real)
            # imag += self.dropout(x.imag)

            real = self.real_norm_2(real)
            imag = self.imag_norm_2(imag)
        elif self.configs.sum_fusion:
            real_text = torch.sum(text_embeddings.real, dim=2)
            imag_text = torch.sum(text_embeddings.imag, dim=2)
            real = x.real + real_text
            imag = x.imag + imag_text

        else:
            real = x.real
            imag = x.imag
        if self.configs.only_text_input and not used_seas_fusion:
            real = text_embeddings.real[:,:,:,0,:]
            imag = text_embeddings.imag[:,:,:,0,:]
        # print('real: ', real.shape)
        # print('imag: ', imag.shape)
        if self.use_seas and not self.seas_disable_hsg:
            low_specxy_real, low_specxy_imag = self.seas_generator(real, imag, self.pred_len)
        else:
            low_specxy_real = self.freq_upsampler_real(real.permute(0,1,3,2)).permute(0,1,3,2)-self.freq_upsampler_imag(imag.permute(0,1,3,2)).permute(0,1,3,2)
            low_specxy_imag = self.freq_upsampler_real(imag.permute(0,1,3,2)).permute(0,1,3,2)+self.freq_upsampler_imag(real.permute(0,1,3,2)).permute(0,1,3,2)  # B x T_f x emb_size
        
        # low_specxy_real = F.relu(low_specxy_real)
        # low_specxy_imag = F.relu(low_specxy_imag)
        # low_specxy_real = self.real_norm_3(low_specxy_real.permute(0,1,3,2)).permute(0,1,3,2)
        # low_specxy_imag = self.imag_norm_3(low_specxy_imag.permute(0,1,3,2)).permute(0,1,3,2)
        
        x = torch.stack([low_specxy_real, low_specxy_imag], dim=-1)
        # text = torch.stack([real, imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)

        
        real, imag = x.real, x.imag

        
        x_real = self.realEmb_dec(real) - self.imagEmb_dec(imag)
        x_imag = self.imagEmb_dec(real) + self.realEmb_dec(imag)
        
        # x_real = self.realEmb_dec(x.real) - self.imagEmb_dec(x.imag)
        # x_imag = self.imagEmb_dec(x.real) + self.realEmb_dec(x.imag)

        text = torch.stack([x_real, x_imag], dim=-1)
        # text = torch.stack([real, imag], dim=-1)
        text = F.softshrink(text, lambd=self.sparsity_threshold)
        text = torch.view_as_complex(text)
        x = torch.fft.irfft(text, n=self.pred_len, dim=2, norm="ortho")
        # print('x: ', x.shape)
        # x = x + bias
        # if not self.configs.proj_per_freq:
        #     x = self.fc(x.reshape(B, N_c, -1)).permute(0, 2, 1)
        # else:
        #     x = self.fc(x)
        #     x = x.squeeze(-1).transpose(1,2)
        # # x += x_dec
        x = x * stdev.unsqueeze(-1) + means.unsqueeze(-1)
        x = x.squeeze(1)
        return x

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, text_embeddings=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text_embeddings)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
