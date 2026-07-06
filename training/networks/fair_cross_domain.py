# networks/fair_cross_domain
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import BACKBONE
from networks.xception import Xception 
import torch_dct as dct
import math
import os

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

class ShallowFreqCNN(nn.Module):
    def __init__(self, in_channels=9):
        super(ShallowFreqCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x.view(x.size(0), -1)

class AudioCNN(nn.Module):
    def __init__(self, in_channels=1):
        super(AudioCNN, self).__init__()
        # CNN ringan untuk memproses gambar spectrogram
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)  # Output: [B, 128]

class GatedFusion(nn.Module):
    """
    Menggabungkan 3 stream (spatial, freq, temporal) dengan gate yang
    belajar sendiri seberapa besar tiap stream harus dipercaya.
 
    Input : s, f, t masing-masing [B, proj_dim]
    Output: fused [B, proj_dim]
    """
    def __init__(self, proj_dim: int):
        super().__init__()
        # Gate: input = concat 3 stream → output = bobot per stream (softmax)
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 3),   # 3 bobot untuk s, f, t
            nn.Softmax(dim=-1),       # jumlahnya selalu 1
        )
        self.norm = nn.LayerNorm(proj_dim)
 
    def forward(self, s, f, t):
        # s, f, t: [B, proj_dim]
        concat = torch.cat([s, f, t], dim=-1)          # [B, 3*proj_dim]
        weights = self.gate(concat)                    # [B, 3]
        w_s = weights[:, 0:1]                          # [B, 1]
        w_f = weights[:, 1:2]
        w_t = weights[:, 2:3]
        fused = w_s * s + w_f * f + w_t * t           # [B, proj_dim]
        return self.norm(fused)
 

class AudioGate(nn.Module):
    """
    Menghitung scalar gate [0,1] dari fitur audio + konteks visual.
    Jika audio tidak informatif (Indo-FS / audio asli) gate → 0.
    Jika audio informatif (Indo-VC / MM) gate → mendekati 1.
 
    Input : audio_feat [B, 128], visual_ctx [B, proj_dim]
    Output: gate_weight [B, 1]  (scalar per sample)
    """
    def __init__(self, audio_dim: int, visual_dim: int, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold
        self.gate = nn.Sequential(
            nn.Linear(audio_dim + visual_dim, 128),
            # nn.ReLU(),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
 
    def forward(self, audio_feat, visual_ctx):
        combined = torch.cat([audio_feat, visual_ctx], dim=-1)
        gate_weight = self.gate(combined)
        gate_weight = torch.where(gate_weight < self.threshold, torch.zeros_like(gate_weight), gate_weight)
        return gate_weight

@BACKBONE.register_module(module_name='fair_cross_domain')
class FairCrossDomainNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.num_classes = model_config.get('num_classes', 2)
        self.num_demographics = model_config.get('num_demographics', 3) 
        
        # 1. Spatial Branch: XceptionNet
        self.spatial_branch = Xception(model_config) 
        spatial_dim_raw = 2048 # Gunakan dimensi murni Xception

        pretrained_path = './training/pretrained/xception-b5690688.pth'
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model_dict = self.spatial_branch.state_dict()
            
            # Perbaiki dimensi 2D -> 4D untuk pointwise convolution
            new_state_dict = {}
            for k, v in state_dict.items():
                if k in model_dict:
                    # Jika file .pth bentuknya 2D tapi model minta 4D, tambahkan dimensi (1, 1)
                    if len(v.shape) == 2 and len(model_dict[k].shape) == 4:
                        new_state_dict[k] = v.unsqueeze(-1).unsqueeze(-1)
                    else:
                        new_state_dict[k] = v
                else:
                    new_state_dict[k] = v
                    
            self.spatial_branch.load_state_dict(new_state_dict, strict=False)
            print(f"✅ BERHASIL: Otak Pretrained Xception (dengan perbaikan shape) dimuat dari {pretrained_path}!")
        else:
            print(f"❌ PERINGATAN FATAL: File pretrained tidak ditemukan di {pretrained_path}. Model akan belajar dari nol!")
            
        # 2. Frequency Branch
        self.freq_branch = ShallowFreqCNN(in_channels=9)
        freq_dim = 64
        
        # 3. Temporal Branch: Bi-LSTM
        self.hidden_dim = 128
        self.temporal_branch = nn.LSTM(input_size=spatial_dim_raw, hidden_size=self.hidden_dim, 
                                       num_layers=1, batch_first=True, bidirectional=True)
        temporal_dim = self.hidden_dim * 2 # 256
        
        self.proj_dim = 512  
        self.spatial_proj = nn.Linear(spatial_dim_raw, self.proj_dim)
        self.freq_proj = nn.Linear(freq_dim, self.proj_dim)
        self.temporal_proj = nn.Linear(temporal_dim, self.proj_dim)

        self.audio_branch = AudioCNN(in_channels=1)
        self.audio_proj   = nn.Linear(128, self.proj_dim)
        audio_cfg = model_config.get('audio_config', {})
        gate_threshold = audio_cfg.get('gate_threshold', 0.2)
        self.audio_gate = AudioGate(audio_dim=128, visual_dim=self.proj_dim, threshold=gate_threshold)

        # 4. Cross-Domain Fusion Module
        self.fusion = GatedFusion(self.proj_dim)

        # 5. Multi-Task Heads
        self.primary_head = nn.Sequential(
            nn.Linear(self.proj_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, self.num_classes)
        )

        self.visual_classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, self.num_classes)
        )

        self.auxiliary_head = nn.Sequential(
            nn.Linear(self.proj_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_demographics)
        )

        self.freq_classifier = nn.Sequential(
            nn.Linear(freq_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, self.num_classes)
        )

    def extract_dct(self, x):
        """
        Multi-scale DCT dengan resolusi dikurangi untuk efisiensi memory.
        DCT tidak butuh resolusi penuh — informasi frekuensi tetap tertangkap
        di resolusi yang lebih kecil.
        """
        results = []
        # Ukuran DCT yang lebih kecil dari resolusi penuh
        dct_sizes = [64, 32, 16]  # bukan 256, 128, 64
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False): 
                # Paksa input menjadi float32
                x_float = x.float() 
                
                for size in dct_sizes:
                    x_scaled = F.interpolate(
                        x_float, size=(size, size),
                        mode='bilinear', align_corners=False
                    )
                    dct_result = dct.dct_2d(x_scaled, norm='ortho')
                    magnitude = torch.abs(dct_result)
                    
                    magnitude = torch.clamp(magnitude, min=1e-4)
                    magnitude = torch.log(magnitude)
                    
                    if size != 64:
                        magnitude = F.interpolate(
                            magnitude, size=(64, 64),
                            mode='bilinear', align_corners=False
                        )
                    results.append(magnitude)
                
                out_tensor = torch.cat(results, dim=1)
                mean = out_tensor.mean(dim=[2, 3], keepdim=True)
                std  = out_tensor.std(dim=[2, 3], keepdim=True).clamp(min=1e-6)
                out_tensor = (out_tensor - mean) / std
                
        # Kembalikan tensor ke tipe asalnya (float16) agar model bisa lanjut jalan
        return out_tensor.to(x.dtype)


    def features(self, x, audios=None):

        current_epoch  = getattr(self, '_current_epoch',  0)
        training_stage = getattr(self, 'training_stage',  2)

        B, T, C, H, W = x.size()
        x_flat = x.view(B * T, C, H, W)
        
        if self.training:
            rand_val = torch.rand(1).item()
            if rand_val < 0.4:  # 40% chance simulasikan resolusi rendah
                scale = torch.FloatTensor(1).uniform_(0.25, 0.6).item()
                small_h = max(32, int(H * scale))
                small_w = max(32, int(W * scale))
                x_flat = F.interpolate(x_flat, size=(small_h, small_w), 
                                      mode='bilinear', align_corners=False)
                x_flat = F.interpolate(x_flat, size=(H, W), 
                                      mode='bilinear', align_corners=False)

        # --- Spatial Branch ---
        spatial_feats_raw = self.spatial_branch.features(x_flat) 
        spatial_feats_raw = F.relu(spatial_feats_raw)
        spatial_feats_pooled = F.adaptive_avg_pool2d(spatial_feats_raw, (1, 1)) 
        spatial_feats_flat = spatial_feats_pooled.view(B * T, -1) # 2048 murni
        spatial_feats = spatial_feats_flat.view(B, T, -1) 
        
        # --- Frequency Branch ---
        freq_input = self.extract_dct(x_flat)
        # print(f"[DEBUG] freq_input shape: {freq_input.shape}")
        freq_feats_flat = self.freq_branch(freq_input) 
        # print(f"[DEBUG] freq_feats_flat shape: {freq_feats_flat.shape}")
        freq_feats = freq_feats_flat.view(B, T, -1) 
        
        # ─────────────────────────────────────────────────────────
        # STAGE 1: tanpa LSTM, tanpa audio
        # ─────────────────────────────────────────────────────────
        if training_stage == 1:
            s_mean = self.spatial_proj(spatial_feats.mean(dim=1))   # [B, 512]
            f_mean = self.freq_proj(freq_feats.mean(dim=1))          # [B, 512]
 
            # Fused: rata-rata spatial + freq tanpa gate
            # (LSTM dan audio tidak ikut di stage 1)
            final_fused = self.fusion.norm(
                (s_mean + f_mean) / 2.0
            )
 
            spatial_residual    = spatial_feats.mean(dim=1)          # [B, 2048]
            freq_feats_pooled   = freq_feats.mean(dim=1)              # [B, 64]
            return final_fused, spatial_residual, freq_feats_pooled
 
        # ─────────────────────────────────────────────────────────
        # STAGE 2: full pipeline — LSTM + GatedFusion + AudioGate
        # ─────────────────────────────────────────────────────────
 
        # ── Temporal Branch (Bi-LSTM) ─────────────────────────
        temporal_output, _ = self.temporal_branch(spatial_feats)   # [B, T, 256]
 
        # Proyeksi ke proj_dim
        s = self.spatial_proj(spatial_feats)    # [B, T, 512]
        f = self.freq_proj(freq_feats)           # [B, T, 512]
        t = self.temporal_proj(temporal_output)  # [B, T, 512]
 
        # Pool ke [B, 512] untuk fusion
        s_mean = s.mean(dim=1)
        f_mean = f.mean(dim=1)
        t_mean = t.mean(dim=1)

        # ── GatedFusion ──
        fused = self.fusion(s_mean, f_mean, t_mean)   # [B, 512]
        
        # ── Audio Branch + Gate (Pendekatan A) ──────────────────
        # Deteksi apakah audio tersedia (file .wav ada)
        if audios is None:
            audios = torch.zeros((B, 1, 128, 256), device=x.device)
 
        is_audio_empty = (audios.abs().sum(dim=(1, 2, 3)) < 1e-6)  # [B] bool
 
 
        # Training dropout: 30% audio di-mute agar model tidak overfit audio
        # (tidak berlaku saat inference)
        if self.training and not is_audio_empty.all():
            if torch.rand(1).item() < 0.3:
                audios         = torch.zeros_like(audios)
                is_audio_empty = torch.ones(B, dtype=torch.bool, device=x.device)
 
        # Proses audio
        a_feats_raw = self.audio_branch(audios)          # [B, 128]
 
        # ── AudioGate: belajar sendiri seberapa percaya audio ───
        # visual_ctx = representasi visual saat ini
        visual_ctx       = fused.detach()                # detach agar gate tidak ganggu gradient fusion
        gate_weight      = self.audio_gate(a_feats_raw, visual_ctx)   # [B, 1]
 
        # Zero-kan gate untuk audio yang memang kosong (file tidak ada)
        # Indo-FS: gate belajar sendiri → output → mendekati 0
        valid_audio_mask = (~is_audio_empty).float().unsqueeze(-1)     # [B, 1]
        gate_weight      = gate_weight * valid_audio_mask
 
        # Proyeksi audio lalu scale dengan gate
        a_proj           = self.audio_proj(a_feats_raw)               # [B, 512]
        audio_contribution = a_proj * gate_weight                     # [B, 512]
 
        # Tambahkan kontribusi audio ke fused (residual style)
        # Bobot 0.2 agar audio tidak dominasi di awal training
        final_fused = fused + 0.2 * audio_contribution                # [B, 512]
 
        spatial_residual  = spatial_feats.mean(dim=1)   # [B, 2048]
        freq_feats_pooled = freq_feats.mean(dim=1)       # [B, 64]
 
        return final_fused, spatial_residual, freq_feats_pooled

    def forward(self, x, audios=None, alpha=1.0):
        fused_feat, spatial_residual, freq_feats_pooled = self.features(x, audios)
 
        fusion_logits = self.primary_head(fused_feat)
        visual_logits = self.visual_classifier(spatial_residual)
 
        cls_out       = self.primary_head(fused_feat)

        freq_logits   = self.freq_classifier(freq_feats_pooled)
 
        rev_feat       = grad_reverse(fused_feat, alpha)
        demographic_out = self.auxiliary_head(rev_feat)
 
        return cls_out, demographic_out, fused_feat, freq_logits