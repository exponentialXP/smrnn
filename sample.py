import torch
import torch.nn as nn
from torch.nn import functional as F

from ssrnn import Model

batch_size = 32
block_size = 8
max_iters = 4000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50

n_embd = 384
n_layers = 6

torch.manual_seed(1337)

import tiktoken

enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab

model = Model(vocab_size, n_embd, n_layers).to(device)

ckpt = torch.load('ckpt.pt')
model.load_state_dict(ckpt['model_dict'])

ckpt = None

prompt = """\n"""
ctx = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

whereToOutput = 'write'

print(prompt, flush=True, end="")

torch.set_float32_matmul_precision('medium')

while True:
    ctx = model.generate(ctx, max_new_tokens=1)
    print(enc.decode([ctx[0][-1].item()]), flush=True, end="")
