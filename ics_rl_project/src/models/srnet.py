import torch
import torch.nn as nn
import torch.nn.functional as F


class SRNet(nn.Module):
    """
    SRNet v7 — DENGELI VERSİYON

    v5 SORUNU: çok kolay → val_acc=%99 ezberleme
    v6 SORUNU: çok zor   → val_acc=%50 hiç öğrenmeme
    
    v7 HEDEF: val_acc = 0.63-0.72 (stabil, gerçek öğrenme)

    DEĞİŞİKLİKLER:
    - noise_std KALDIRILDI: SRM sinyalini bozuyordu, model öğrenemedi
    - dropout: 0.55 → 0.35 (öğrenmeye izin ver)
    - spatial_drop: 0.30 → 0.15 (gradient akışı düzeldi)
    - BN momentum: 0.05 → 0.10 (normal hız)
    - Kanal sayısı: 8 → 12 (biraz daha kapasite)
    - res blok: 2 adet (v6 gibi, v5'teki 3'ten az)
    
    PAYLOAD: 0.25 (v6'daki 0.15 çok azdı, v5'teki 0.40 çok fazlaydı)
    """

    _SRM_KERNELS = torch.tensor([
        [[-1,  2, -1],
         [ 2, -4,  2],
         [-1,  2, -1]],
        [[ 0, -1,  0],
         [ 0,  2,  0],
         [ 0, -1,  0]],
        [[-1,  0,  1],
         [ 0,  0,  0],
         [ 1,  0, -1]],
    ], dtype=torch.float32).unsqueeze(1) / 4.0

    def __init__(self):
        super().__init__()

        self.srm = nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.srm.weight.copy_(self._SRM_KERNELS)
        self.srm.weight.requires_grad = False

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.10),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(12, momentum=0.10),
            nn.ReLU(inplace=True),
        )

        self.res1 = self._make_res_block(12, 12)
        self.res2 = self._make_res_block(12, 12)

        self.spatial_drop = nn.Dropout2d(p=0.15)

        self.pool1 = self._make_pool_block(12,  32)
        self.pool2 = self._make_pool_block(32,  64)
        self.pool3 = self._make_pool_block(64, 128)
        self.pool4 = self._make_pool_block(128, 256)

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout    = nn.Dropout(p=0.35)
        self.fc         = nn.Linear(256, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _make_res_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.10),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.10),
        )

    @staticmethod
    def _make_pool_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.10),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.srm(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = F.relu(self.res1(x) + x)
        x = self.spatial_drop(x)

        x = F.relu(self.res2(x) + x)
        x = self.spatial_drop(x)

        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.pool4(x)

        x = self.global_avg(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)