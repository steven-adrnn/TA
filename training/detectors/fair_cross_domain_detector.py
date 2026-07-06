# fair_cross_domain_detector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train

@DETECTOR.register_module(module_name='fair_cross_domain_detector')
class FairCrossDomainDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        
        # Definisi Loss untuk Fairness
        self.demographic_loss = nn.CrossEntropyLoss() 
        
        # Hyperparameters untuk combined loss
        self.alpha_adv = config.get('weight_adv', 1.0)
        self.beta_daw = config.get('weight_daw', 0.5)
        self.gamma_cvar = config.get('weight_cvar', 0.5)
        self.weight_freq = config.get('weight_freq', 0.3)

        self.stage1_epochs = config.get('two_stage', {}).get('stage1_epochs', 15)

        num_demo = config['backbone_config']['num_demographics'] + 1
        self.register_buffer('group_loss_accum', torch.zeros(num_demo))
        self.register_buffer('group_loss_count', torch.zeros(num_demo))
        self._epoch_group_losses = {}  # snapshot dari epoch sebelumnya

    def update_group_loss_buffer(self, per_sample_loss, demo_label):
        """Dipanggil setiap forward pass untuk akumulasi statistik."""
        for g in torch.unique(demo_label):
            g_int = g.item()
            if g_int >= 13:  # ← skip NON_INDO
                continue
            mask = (demo_label == g)
            self.group_loss_accum[g_int] += per_sample_loss[mask].sum().detach()
            self.group_loss_count[g_int] += mask.sum().detach()

    def compute_epoch_daw_loss(self, per_sample_loss, demo_label):
        indo_mask = demo_label < 13
        if indo_mask.sum() < 2:
            return torch.tensor(0.0, device=per_sample_loss.device)
        
        per_sample_loss_indo = per_sample_loss[indo_mask]
        demo_label_indo = demo_label[indo_mask]

        # SELALU hitung dari batch saat ini (ada gradient)
        unique_demos = torch.unique(demo_label_indo)
        if len(unique_demos) < 2:
            return torch.tensor(0.0, device=per_sample_loss.device)
        
        group_losses = []
        for g in unique_demos:
            mask = (demo_label_indo == g)
            if mask.sum() > 0:
                group_losses.append(per_sample_loss_indo[mask].mean())
        
        if len(group_losses) < 2:
            return torch.tensor(0.0, device=per_sample_loss.device)
        
        # torch.stack mempertahankan gradient
        stacked = torch.stack(group_losses)
        
        # Gunakan epoch stats sebagai target anchor (detached), bukan sebagai pengganti
        if self._epoch_group_losses and len(self._epoch_group_losses) >= 2:
            epoch_means = [v for v in self._epoch_group_losses.values()]
            global_mean = sum(epoch_means) / len(epoch_means)
            # Dorong setiap group mendekati global mean (anchored)
            anchor = torch.tensor(global_mean, device=per_sample_loss.device)
            loss = torch.mean((stacked - anchor) ** 2)
            return torch.clamp(loss, max=1.0)
        
        loss = torch.var(stacked)
        return torch.clamp(loss, max=1.0)

    
    def reset_epoch_buffer(self):
        """
        Dipanggil di akhir setiap epoch (dari trainer).
        Simpan rata-rata sebagai referensi epoch berikutnya.
        """
        self._epoch_group_losses = {}
        for g in range(len(self.group_loss_accum)):
            count = self.group_loss_count[g].item()
            if count > 0:
                self._epoch_group_losses[g] = (
                    self.group_loss_accum[g] / count
                ).item()
        
        # Reset buffer
        self.group_loss_accum.zero_()
        self.group_loss_count.zero_()

    def build_backbone(self, config):
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config'].copy()
        model_config['audio_config'] = config.get('audio_config', {})
        backbone = backbone_class(model_config)
        return backbone
    
    def build_loss(self, config):
        # Mengambil loss utama (Cross Entropy) dari registry DeepfakeBench
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        images = data_dict['image']
        audios = data_dict.get('audio', None)
        if len(images.shape) == 4:
            images = images.unsqueeze(1)
        # Panggil fungsi features dari backbone
        return self.backbone.features(images, audios)

    def classifier(self, features: torch.tensor) -> torch.tensor:
        # Panggil klasifikasi utama (Real/Fake) dari backbone
        return self.backbone.primary_head(features)
        
    def calc_cvar_loss(self, losses, alpha=0.5):
        # Menghitung Conditional Value-at-Risk (Fokus pada kasus terburuk)
        if len(losses) == 0:
            return torch.tensor(0.0).to(losses.device)
        var_threshold = torch.quantile(losses, alpha)
        tail_losses = losses[losses >= var_threshold]
        return tail_losses.mean() if len(tail_losses) > 0 else losses.mean()


    def calc_equalized_odds_loss(self, pred_cls, label, demo_label):
        """
        Menghitung loss EO dengan L1 Distance terhadap Global Mean yang di-detach.
        Mencegah model collapse (menebak 0.99 untuk semua sampel).
        """
        indo_mask = demo_label < 13
        if indo_mask.sum() < 4:
            return torch.tensor(0.0, device=pred_cls.device, requires_grad=False)
        
        pred_cls = pred_cls[indo_mask]
        label = label[indo_mask]
        demo_label = demo_label[indo_mask]

        prob = torch.softmax(pred_cls, dim=1)[:, 1]
        unique_demos = torch.unique(demo_label)
        
        if len(unique_demos) < 2:
            return torch.tensor(0.0, device=pred_cls.device, requires_grad=True)
        
        # 1. Hitung Global Soft FPR & TPR sebagai Anchor
        global_real_mask = (label == 0)
        global_fake_mask = (label == 1)
        
        # Gunakan .detach() agar model tidak mengakali rata-rata global
        global_fpr = prob[global_real_mask].mean().detach() if global_real_mask.sum() > 0 else torch.tensor(0.0, device=pred_cls.device)
        global_tpr = prob[global_fake_mask].mean().detach() if global_fake_mask.sum() > 0 else torch.tensor(0.0, device=pred_cls.device)

        eo_loss = torch.tensor(0.0, device=pred_cls.device)
        valid_groups = 0
        
        for g in unique_demos:
            mask = (demo_label == g)
            g_label = label[mask].float()
            g_prob = prob[mask]
            
            # 2. Penalti absolut (L1) grup terhadap Global Anchor
            real_mask = (g_label == 0)
            if real_mask.sum() >= 2:
                soft_fpr = g_prob[real_mask].mean()
                eo_loss = eo_loss + torch.abs(soft_fpr - global_fpr)
                valid_groups += 1
            
            fake_mask = (g_label == 1)
            if fake_mask.sum() >= 2:
                soft_tpr = g_prob[fake_mask].mean()
                eo_loss = eo_loss + torch.abs(soft_tpr - global_tpr)
                valid_groups += 1
        
        # Rata-ratakan loss agar tidak meledak jika sukunya banyak
        if valid_groups > 0:
            eo_loss = eo_loss / valid_groups
        else:
            # Jika sampel terlalu tersebar, matikan EO loss untuk batch ini
            return torch.tensor(0.0, device=pred_cls.device, requires_grad=True)
            
        return eo_loss

    def get_losses(self, data_dict, pred_dict):
        label         = data_dict['label']
        pred_cls      = pred_dict['cls']
        freq_logits   = pred_dict.get('freq_logits', None)
        device        = pred_cls.device
        current_epoch = getattr(self, 'epoch', 0)
 
        # ── Loss utama selalu dihitung ───────────────────────────
        loss_bce  = self.loss_func(pred_cls, label)
        loss_freq = torch.tensor(0.0, device=device)
        if freq_logits is not None:
            loss_freq = F.cross_entropy(freq_logits, label)
 
        zero = lambda: torch.tensor(0.0, device=device)
 
        # ════════════════════════════════════════════════════════
        # FASE 1: epoch < stage1_epochs
        # Hanya bce + freq — biarkan Xception + FreqCNN belajar dulu
        # TANPA gangguan fairness loss sama sekali
        # ════════════════════════════════════════════════════════
        if current_epoch < self.stage1_epochs:
            overall_loss = loss_bce + self.weight_freq * loss_freq
            return {
                'overall':   overall_loss,
                'cls':       loss_bce,
                'freq_loss': loss_freq,
                'adv_loss':  zero(),
                'cvar_loss': zero(),
                'daw_loss':  zero(),
            }
 
        # ════════════════════════════════════════════════════════
        # FASE 2: epoch >= stage1_epochs
        # Tambahkan fairness loss dengan ramp-up linear 5 epoch
        # ════════════════════════════════════════════════════════
        demographic_pred = pred_dict.get('demographic', None)
        if 'demographic_label' not in data_dict or demographic_pred is None:
            # Tidak ada label demografi → hanya bce + freq
            overall_loss = loss_bce + self.weight_freq * loss_freq
            return {
                'overall':   overall_loss,
                'cls':       loss_bce,
                'freq_loss': loss_freq,
                'adv_loss':  zero(),
                'cvar_loss': zero(),
                'daw_loss':  zero(),
            }
 
        demo_label = data_dict['demographic_label']
 
        # Ramp up: epoch stage1 → 0.0, epoch stage1+5 → 1.0
        ramp = min(1.0, (current_epoch - self.stage1_epochs) / 5.0)

        # ── Adversarial demographic loss ──────────────────────
        indo_mask = demo_label < 13
        loss_adv  = zero()
        if indo_mask.sum() > 0:
            loss_adv = self.demographic_loss(
                demographic_pred[indo_mask], demo_label[indo_mask]
            )
 
        # ── DAW dan CVaR pakai detached per_sample_loss ───────
        # PENTING: detach agar fairness loss tidak merusak gradient
        # dari loss_bce melalui shared computation graph
        per_sample_loss_with_grad = F.cross_entropy(
            pred_cls, label, reduction='none'
        )

        loss_daw  = self.compute_epoch_daw_loss(per_sample_loss_with_grad, demo_label)
        loss_cvar = self.calc_cvar_loss(per_sample_loss_with_grad, alpha=0.90)

        # ── Update Buffer pakai yang di-detach ───────
        # Buffer untuk epoch berikutnya baru pakai versi yang di-detach agar memory tidak bocor
        self.update_group_loss_buffer(per_sample_loss_with_grad.detach(), demo_label)
 
        # ── EO loss pakai pred_cls (butuh gradient) ───────────
        loss_eo = self.calc_equalized_odds_loss(pred_cls, label, demo_label)
 
        loss_adv  = torch.clamp(loss_adv,  max=5.0)
        loss_daw  = torch.clamp(loss_daw,  max=1.0)
        loss_cvar = torch.clamp(loss_cvar, max=5.0)
        loss_eo   = torch.clamp(loss_eo,   max=2.0)
        
        overall_loss = (
            loss_bce
            + self.weight_freq * loss_freq
            + ramp * self.alpha_adv  * loss_adv
            + ramp * self.beta_daw   * loss_daw
            + ramp * self.gamma_cvar * loss_cvar
            + ramp * 0.005           * loss_eo
        )
 
        # Update buffer untuk epoch berikutnya
        # self.update_group_loss_buffer(per_sample_loss_with_grad.detach(), demo_label)
 
        return {
            'overall':   overall_loss,
            'cls':       loss_bce,
            'freq_loss': loss_freq,
            'adv_loss':  loss_adv,
            'cvar_loss': loss_cvar,
            'daw_loss':  loss_daw,
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap, f1 = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'f1': f1}

    def forward(self, data_dict, inference=False):
        current_epoch = getattr(self, 'epoch', 0)
        max_epoch     = self.config.get('nEpochs', 55)
        p             = current_epoch / max_epoch
 
        # Jadwal GRL alpha
        scheduled_alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
 
        training_stage = 1 if current_epoch < self.stage1_epochs else 2
        self.backbone._current_epoch  = current_epoch
        self.backbone.training_stage  = training_stage
 
        images = data_dict['image']
        audios = data_dict.get('audio', None)
 
        if len(images.shape) == 4:
            images = images.unsqueeze(1)
 
        cls_out, demographic_out, features, freq_logits = self.backbone(
            images, audios, alpha=scheduled_alpha
        )
 
        prob = torch.softmax(cls_out, dim=1)[:, 1]
 
        return {
            'cls':         cls_out,
            'demographic': demographic_out,
            'prob':        prob,
            'feat':        features,
            'freq_logits': freq_logits,
        }
