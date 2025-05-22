import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.models as models
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange

class GazeEncoder(nn.Module):
    def __init__(self, input_dim, dim_feat, dropout=0.3):
        super(GazeEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, dim_feat, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(dim_feat, dim_feat, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(dim_feat, dim_feat, kernel_size=9, padding=4)
        self.drop = nn.Dropout1d(dropout)
        self.pool = nn.AvgPool1d(3)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.drop(F.relu(self.conv1(x)))
        x = self.drop(F.relu(self.conv2(x)))
        x = self.drop(F.relu(self.conv3(x)))
        x = self.pool(x)
        return x.permute(0, 2, 1)

class IMUEncoder(nn.Module):
    def __init__(self, input_dim, dim_feat, dropout=0.3):
        super(IMUEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, dim_feat, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(dim_feat, dim_feat, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(dim_feat, dim_feat, kernel_size=9, padding=4)
        self.drop = nn.Dropout1d(dropout)
        self.pool = nn.AvgPool1d(3)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.drop(F.relu(self.conv1(x)))
        x = self.drop(F.relu(self.conv2(x)))
        x = self.drop(F.relu(self.conv3(x)))
        x = self.pool(x)
        return x.permute(0, 2, 1)

class RGBEncoder(nn.Module):
    def __init__(self, input_dim, dim_feat, dropout=0.3, out_size=7):
        super(RGBEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, dim_feat, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(dim_feat, dim_feat, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(dim_feat, dim_feat, kernel_size=5, padding=2)
        self.drop = nn.Dropout2d(dropout)
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((out_size, out_size))
        self.input_dim = input_dim
        self.dim_feat = dim_feat
        self.out_size = out_size
    def forward(self, x):
        x = self.drop(F.relu(self.conv1(x)))
        x = self.maxpool(x)
        x = self.drop(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.drop(F.relu(self.conv3(x)))
        x = self.avgpool(x)
        x = x.view(-1, self.dim_feat, self.out_size*self.out_size).permute(0, 2, 1)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MultimodalTransformer(nn.Module):
    def __init__(self, num_classes = 2, dim_feat = 32, input_dim = [3,6,3], sequence_length = 120, dropout = 0.3, out_size=7):
        super(MultimodalTransformer, self).__init__()
        if input_dim[0] > 0:
            self.gaze_encoder = GazeEncoder(input_dim[0], dim_feat, dropout)
            self.gaze_pe = nn.Parameter(torch.randn(1, sequence_length, dim_feat))
        if input_dim[1] > 0:
            self.imu_encoder = IMUEncoder(input_dim[1], dim_feat, dropout)
            self.imu_pe = nn.Parameter(torch.randn(1, sequence_length, dim_feat))
        if input_dim[2] > 0:
            self.rgb_encoder = RGBEncoder(input_dim[2], dim_feat, dropout, out_size)
            self.rgb_pe = nn.Parameter(torch.randn(1, out_size*out_size, dim_feat))

        self.cls = nn.Parameter(torch.randn(1, 1, dim_feat))
        self.transformer = Transformer(dim=dim_feat, depth=3, heads=dim_feat//16, dim_head=dim_feat, mlp_dim=dim_feat, dropout=dropout)
        self.fc = nn.Linear(dim_feat, num_classes)
        
    def forward(self, x):
        gaze, imu, rgb = x.get('gaze'), x.get('odom'), x.get('rgb')
        bsz = gaze.size(0) if gaze is not None else imu.size(0) if imu is not None else rgb.size(0)
        tokens = [self.cls.repeat(bsz,1,1)]

        if gaze is not None: #(b n d -> b n d)
            gaze_tokens = self.gaze_encoder(gaze) 
            gaze_pe = self.gaze_pe.repeat(bsz,1,1)[:,:gaze_tokens.size(1)]
            tokens.append(gaze_tokens + gaze_pe)

        if imu is not None: #(b n d -> b n d)
            imu_tokens = self.imu_encoder(imu) 
            imu_pe = self.imu_pe.repeat(bsz,1,1)[:,:imu_tokens.size(1)]
            tokens.append(imu_tokens + imu_pe)

        if rgb is not None: #(b c H W -> b hw d)
            rgb_tokens = self.rgb_encoder(rgb) 
            rgb_pe = self.rgb_pe.repeat(bsz,1,1)[:,:rgb_tokens.size(1)]
            tokens.append(rgb_tokens + rgb_pe)

        # Merge tokens, transformer, select [CLS], then classification head
        x = torch.cat(tokens, 1)
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.fc(x)
        return x
