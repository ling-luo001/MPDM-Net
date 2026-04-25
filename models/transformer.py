import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, GRU, Linear, LayerNorm, Dropout


class FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0):
        super(FFN, self).__init__()
        self.gru = GRU(d_model, d_model * 2, 1, bidirectional=bidirectional)
        if bidirectional:
            self.linear = Linear(d_model * 2 * 2, d_model)
        else:
            self.linear = Linear(d_model * 2, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)  # GRU 输出
        x = F.leaky_relu(x)  # 激活函数
        x = self.dropout(x)  # Dropout
        x = self.linear(x)  # 全连接层

        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, bidirectional=True, dropout=0):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(d_model)
        self.attention = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = Dropout(dropout)

        self.norm2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, bidirectional=bidirectional)
        self.dropout2 = Dropout(dropout)

        self.norm3 = LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        xt = self.norm1(x)
        xt, _ = self.attention(xt, xt, xt,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask)
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

        x = self.norm3(x)

        return x

class TSTransformerBlock2(nn.Module):
    def __init__(self, d_model , n_heads):
        super(TSTransformerBlock2, self).__init__()
        self.time_transformer = TransformerBlock(d_model=d_model, n_heads=n_heads)
        self.freq_transformer = TransformerBlock(d_model=d_model, n_heads=n_heads)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x

class TSTransformerBlock3(nn.Module):
    def __init__(self, d_model, n_heads):
        super(TSTransformerBlock3, self).__init__()
        self.time_transformer = TransformerBlock(d_model=d_model, n_heads=n_heads)
        self.freq_transformer = TransformerBlock(d_model=d_model, n_heads=n_heads)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x

def main():
    x = torch.randn(4, 256, 50, 40)
    b, c, t, f = x.size()
    x = x.permute(0, 3, 2, 1).contiguous().view(b, f*t, c)
    transformer = TransformerBlock(d_model=256, n_heads=4)
    x = transformer(x)
    x =  x.view(b, f, t, c).permute(0, 3, 2, 1)
    print(x.size())


if __name__ == '__main__':
    main()