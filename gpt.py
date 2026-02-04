import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embed size should be divisible by number of heads"

        self.SEQ = nn.Linear(embed_size, embed_size * 3, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, seq, mask):
        N, seq_len, _ = seq.shape
        
        QKV = self.SEQ(seq)
        QKV = QKV.reshape(N, seq_len, 3, self.heads, self.head_dim)
        QKV = QKV.permute(2, 0, 3, 1, 4)

        Q, K, V = QKV[0], QKV[1], QKV[2]

        energy = torch.einsum("nhqd,nhkd->nhqk", [Q, K])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** (1/2)), dim=3)

        out = torch.einsum("nhqk,nhkd->nhqd", [attention, V])
        out = out.permute(0, 2, 1, 3).reshape(N, seq_len, self.embed_size)

        out = self.fc_out(out)
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_length=5000):
        super().__init__()

        pe = torch.zeros(max_length, embed_size)
        position = torch.arange(0, max_length).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_size, 2) *
            (-math.log(10000.0) / embed_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x, mask)
        Z = self.dropout(self.norm1(attention + x))

        forward = self.feed_forward(Z)
        X = self.dropout(self.norm2(forward + Z))

        return X

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)

        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, target_mask):

        out = self.transformer_block(x, target_mask)

        return out
    
class Decoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            target_vocab_size,
            embed_size,
            num_layers,
            heads, 
            forward_expansion,
            dropout,
            device,
            max_length
    ):
        super(Decoder, self).__init__()

        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)


        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.fc_out.weight = self.word_embedding.weight


    def forward(self, x, target_mask):
        N, seq_len = x.shape

        pos = torch.arange(0, seq_len, device=self.device).unsqueeze(0)
        pos = pos.expand(N, seq_len)
        
        x = self.dropout(
            self.word_embedding(x) + self.position_embedding(pos)
        )

        for layer in self.layers:
            x = layer(x, target_mask)

        out = self.fc_out(x)

        return out

class GPT(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            target_vocab_size,
            src_pad_index,
            target_pad_index,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cuda",
            max_length=100
    ):
        super(GPT, self).__init__()

        self.decoder = Decoder(
            src_vocab_size,
            target_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_index = src_pad_index
        self.target_pad_index = target_pad_index
        self.device = device

    def make_padding_mask(self, target):
        # target: (N, T)
        return (target != self.target_pad_index).unsqueeze(1).unsqueeze(2)
        # shape: (N, 1, 1, T)


    def make_target_mask(self, target):
        N, T = target.shape

        causal_mask = torch.tril(torch.ones((T, T), device=target.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        # (1, 1, T, T)

        padding_mask = (target != self.target_pad_index).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, T)

        return causal_mask * padding_mask

    
    def forward(self, target):
        target_mask = self.make_target_mask(target)

        out = self.decoder(target, target_mask)

        return out
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BOS_IDX = 1 

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    target = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    bos = torch.full((target.size(0), 1), BOS_IDX, device=target.device)
    input_tokens = torch.cat([bos, target[:, :-1]], dim=1)


    src_pad_index = 0
    target_pad_index = 0
    src_vocab_size = 10
    target_vocab_size = 10

    model = GPT(src_vocab_size, target_vocab_size, src_pad_index, target_pad_index, device=device).to(device)

    out = model(input_tokens)

    print(out.shape)