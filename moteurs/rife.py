"""RIFE 4.26 (interpolation d'images vidéo) : architecture IFNet adaptée de hzwer/Practical-RIFE
(train_log/IFNet_HDv3.py et model/warplayer.py), exécutée dans l'environnement de diffusion (torch seulement).
Sortie identique au fichier officiel (écart maximal 0,0 sur des images aléatoires). Poids officiels
flownet_v4.26.pkl (même empreinte que l'archive de hzwer), publiés par HolyWu/vs-rife.

MIT License

Copyright (c) 2021 hzwer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_grilles = {}
def warp(img, flow):
    k = (str(flow.device), str(flow.size()))
    if k not in _grilles:
        n, _, h, w = flow.shape
        gx = torch.linspace(-1.0, 1.0, w, device=flow.device).view(1, 1, 1, w).expand(n, -1, h, -1)
        gy = torch.linspace(-1.0, 1.0, h, device=flow.device).view(1, 1, h, 1).expand(n, -1, -1, w)
        _grilles[k] = torch.cat([gx, gy], 1)
    flow = torch.cat([flow[:, 0:1] / ((img.shape[3] - 1.0) / 2.0), flow[:, 1:2] / ((img.shape[2] - 1.0) / 2.0)], 1)
    g = (_grilles[k] + flow).permute(0, 2, 3, 1)
    return F.grid_sample(img, g.to(img.dtype), mode="bilinear", padding_mode="border", align_corners=True)

def conv(i, o, k=3, s=1, p=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, dilation=d, bias=True), nn.LeakyReLU(0.2, True))

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1); self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1); self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        x = self.relu(self.cnn0(x)); x = self.relu(self.cnn1(x)); x = self.relu(self.cnn2(x))
        return self.cnn3(x)

class ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

class IFBlock(nn.Module):
    def __init__(self, inp, c):
        super().__init__()
        self.conv0 = nn.Sequential(conv(inp, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))
    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        tmp = self.lastconv(self.convblock(self.conv0(x)))
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]

class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, 192); self.block1 = IFBlock(8 + 4 + 8 + 8, 128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, 96); self.block3 = IFBlock(8 + 4 + 8 + 8, 64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, 32); self.encode = Head()
    def forward(self, img0, img1, timestep=0.5, scale=1.0):
        t = img0[:, :1] * 0 + timestep
        f0, f1 = self.encode(img0), self.encode(img1)
        flow = mask = feat = None
        w0, w1 = img0, img1
        for i, b in enumerate([self.block0, self.block1, self.block2, self.block3, self.block4]):
            s = 2 ** (4 - i) / scale  # 16,8,4,2,1 (/scale)
            if flow is None:
                flow, mask, feat = b(torch.cat((img0, img1, f0, f1, t), 1), None, scale=s)
            else:
                wf0, wf1 = warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4])
                fd, mask, feat = b(torch.cat((w0, w1, wf0, wf1, t, mask, feat), 1), flow, scale=s)
                flow = flow + fd
            w0, w1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        return w0 * m + w1 * (1 - m)

def charger(chemin, device="cpu"):
    sd = torch.load(chemin, map_location="cpu", weights_only=True)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith(("teacher.", "caltime."))}  # training only
    net = IFNet()
    net.load_state_dict(sd, strict=True)
    return net.eval().to(device)

@torch.inference_mode()
def intercaler(net, a, b, t=0.5, scale=1.0):
    """a, b : tenseurs (1,3,H,W) float dans [0,1]. Renvoie l'image au temps t."""
    h, w = a.shape[2:]
    m = max(64, int(64 / scale))  # Practical-RIFE uses 128; vs-rife uses 64 for 4.25/4.26
    ph, pw = math.ceil(h / m) * m, math.ceil(w / m) * m
    pad = (0, pw - w, 0, ph - h)
    out = net(F.pad(a, pad), F.pad(b, pad), t, scale)
    return out[:, :, :h, :w].clamp(0, 1)

