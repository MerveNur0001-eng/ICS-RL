# src/modules/policy_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Piksel bazlı embedding seçimi için 6 katmanlı konvolüsyonel politika ağı.

    Girdi  : (batch, 3, H, W)  VEYA  (3, H, W)  — her ikisi kabul edilir
    Çıktı  : per-piksel aksiyon logitleri, shape (batch, H, W)

    DÜZELTİLMİŞ:
    - forward(): (3,H,W) gelirse otomatik (1,3,H,W) yapılır
      → "too many indices for tensor of dimension 3" hatası çözüldü
    - conv6 bias: -3.0 → başlangıçta %5 piksel seçilir, ajan öğrenince artar
    """

    def __init__(self,
                 input_channels: int = 3,
                 hidden_dim:     int = 64):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim,     hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim,     hidden_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim,     hidden_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(hidden_dim,     hidden_dim, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(hidden_dim,     1,          kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.bn4 = nn.BatchNorm2d(hidden_dim)
        self.bn5 = nn.BatchNorm2d(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
       
        nn.init.constant_(self.conv6.bias, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)   
        # ─────────────────────────────────────────────────────────────
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return self.conv6(x).squeeze(1)   

    def select_actions(self,
                       logits:        torch.Tensor,
                       mask:          torch.Tensor,
                       deterministic: bool = True,
                       n_bits:        int  = None) -> torch.Tensor:
        masked_logits = logits + (mask - 1.0) * 1e4
        probs = torch.sigmoid(masked_logits)

        if not deterministic:
            actions = torch.bernoulli(probs)

        elif n_bits is not None:
            actions = torch.zeros_like(probs)
            for b in range(probs.shape[0]):
                candidate_probs = probs[b][mask[b] == 1.0]
                n_candidates    = len(candidate_probs)

                if n_candidates < n_bits:
                    raise RuntimeError(
                        f"Görüntü {b}: candidate sayısı ({n_candidates}) "
                        f"n_bits ({n_bits})'ten az."
                    )

                sorted_probs, _ = torch.sort(candidate_probs, descending=True)
                tau              = sorted_probs[n_bits - 1].item()

                selected   = (probs[b] >= tau).float() * mask[b]
                n_selected = int(selected.sum().item())

                if n_selected > n_bits:
                    excess_indices = torch.where(
                        (probs[b] == tau) & (mask[b] == 1.0)
                    )
                    excess_rows = excess_indices[0]
                    excess_cols = excess_indices[1]
                    n_remove    = n_selected - n_bits
                    perm        = torch.randperm(len(excess_rows))[:n_remove]
                    selected[excess_rows[perm], excess_cols[perm]] = 0.0

                actions[b] = selected
        else:
            actions = (probs > 0.5).float()

        return actions * mask