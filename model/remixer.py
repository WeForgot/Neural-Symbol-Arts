import torch
import torch.nn as nn
import torch.nn.functional as F

class RemixerBlock(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        self.pre_glu = nn.Linear(dim, dim*2)
        self.gelu = nn.GELU()

        self.memory = nn.parameter.Parameter(torch.zeros(seq_len, dim))
        self.alpha = nn.parameter.Parameter(torch.zeros(1))

        nn.init.xavier_uniform_(self.memory)
        nn.init.uniform_(self.alpha)
        
    def forward(self, x):
        x = self.pre_glu(x)
        x1, x2 = x.chunk(2, dim=-1)
        x1 = self.gelu(x1)
        x = torch.cat([x1, x2], dim=-1)
        x = F.glu(x)
        gr = self.memory.softmax(dim=-1)
        gr = x * gr
        dt = x * gr
        sb = x - gr
        x = self.alpha * dt + (1 - self.alpha) * sb
        return x

class RemixerEncoderLayer(nn.Module):
    def __init__(self, seq_len, dim, heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.remixer = RemixerBlock(seq_len, dim)
    
    def forward(self, x):
        x1, _ = self.mha(x, x, x)
        x = self.norm(x1 + x)
        x = self.remixer(x)
        return x

class RemixerDecoderLayer(nn.Module):
    def __init__(self, seq_len, dim, heads):
        super().__init__()
        self.mha1 = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mha2 = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.remixer = RemixerBlock(seq_len, dim)
    
    def forward(self, x):
        x, context = x
        x1, _ = self.mha1(x, x, x)
        x = self.norm1(x1 + x)

        x2, _ = self.mha2(x, context, context)
        x = self.norm2(x2 + x)

        x = self.remixer(x)
        return x, context

class RemixerEncoder(nn.Module):
    def __init__(self, seq_len, dim, heads, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(RemixerEncoderLayer(seq_len, dim, heads))
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)

class RemixerDecoder(nn.Module):
    def __init__(self, seq_len, dim, heads, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(RemixerDecoderLayer(seq_len, dim, heads))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x, context):
        return self.decoder([x, context])[0]

class RemixerTransformer(nn.Module):
    def __init__(self, dim, enc_seq_len, enc_depth, enc_heads, dec_seq_len, dec_depth, dec_heads):
        super().__init__()
        self.encoder = RemixerEncoder(enc_seq_len, dim, enc_heads, enc_depth)
        self.decoder = RemixerDecoder(dec_seq_len, dim, dec_heads, dec_depth)

        self.enc_embs = nn.Embedding(enc_seq_len, dim)
        self.dec_embs = nn.Embedding(dec_seq_len, dim)
    
    def forward(self, x):
        b = x.shape[0]
        x = x + self.enc_embs.weight.repeat(b, 1, 1)
        x = self.encoder(x)

        dec_embs = self.dec_embs.weight.repeat(b, 1, 1)
        x = self.decoder(dec_embs, x)
        return x