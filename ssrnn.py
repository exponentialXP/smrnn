import torch
import torch.nn as nn
import torch.nn.functional as F

n_embd = 384
n_layers = 6
batch_size = 16
block_size = 8

class Layer(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.scale = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.LayerNorm(n_embd),
        )

        self.proj = nn.Sequential(
            nn.Linear(n_embd*2, n_embd),
            nn.LayerNorm(n_embd),
        )

    def forward(self, stuff, mode='timestep'):
        if mode=='timestep':
            scale = stuff['scale']
            token = stuff['token']
            h = stuff['h']

            return self.proj(torch.cat((token, h), dim=1)) * scale
        
        elif mode=='preload':
            seq = stuff['seq']

            return self.scale(seq)

class Model(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)

        self.layers = nn.ModuleList([Layer(n_embd) for _ in range(n_layers)])

        self.head = nn.Linear(n_embd, vocab_size)

        self.embedding.weight = self.head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / (2 * int(n_layers**0.5)))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        emb = self.embedding(idx)

        B, T, C = emb.shape

        h = torch.zeros((B, C)).to(emb.device)

        logits = []
        scales = []

        seq = emb.view(B * T, C)

        for layer in self.layers:
            output = layer({'seq': seq}, mode='preload')
            scale = output.view(B, T, C)
            scales.append(scale)
            
        for t in range(T):
            token = emb[:, t, :]

            for num, layer in enumerate(self.layers):
                h = F.silu(h + layer({'scale': scales[num][:, t, :], 'token': token, 'h': h}, mode='timestep'))

            logits.append(self.head(h))

        logits = torch.stack(logits, dim=1)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad==True)
