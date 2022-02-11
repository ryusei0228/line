import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class FFN(nn.Module):
    def __init__(self, d_model=768, d_ff=2048, drop_rate=0.1, activation=gelu):
        super(FFN, self).__init__()
        self.drop_rate = drop_rate
        self.l1 = nn.Conv1d(d_model, d_ff, 1)
        self.l2 = nn.Conv1d(d_ff, d_model, 1)

        self.activation = activation

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        out = self.norm(x)
        out = self.activation(self.l1(out.transpose(1, 2)))
        out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.l2(out)
        out = F.dropout(out.transpose(1, 2), p=self.drop_rate, training=self.training)
        return x + out