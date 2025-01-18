import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import os
from ssrnn import Model

batch_size = 32
accum_steps = 1
block_size = 128
max_iters = 4000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50

n_embd = 256
n_layers = 4

torch.manual_seed(1337)

import tiktoken

enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab

import numpy as np

def get_batch(split):
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('val.bin', dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = Model(vocab_size, n_embd, n_layers).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if os.path.exists('ckpt.pt'):
    ckpt = torch.load('ckpt.pt')

    model.load_state_dict(ckpt['model_dict'])
    optimizer.load_state_dict(ckpt['optim_dict'])

    ckpt = None

print(f'Params: {model.count_params():,}')

xb, yb = get_batch('train')

for iter in tqdm(range(max_iters)):
    for accum_step in range(accum_steps):
        logits, loss = model(xb, yb)
        loss = loss / accum_steps

        xb, yb = get_batch('train')

        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

    optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        torch.save(
            {
                'model_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict()
            }, 'ckpt.pt'
        )

context = torch.tensor(enc.encode("""\n"""), dtype=torch.long, device=device).unsqueeze(0)
print(enc.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
