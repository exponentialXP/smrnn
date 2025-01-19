This new architecture is based off the RNN architecture, with almost the same speed and memory usage, now has identical loss to the transformer architecture, but O(n^2)->O(n) time complexity!

In almost all modern language models, the transformer architecture is used and is the beating heart of the model. However, the transformer architecture is extremely inefficient when used for long sequences. I will introduce a new RNN-like architecture which I have called matmul and scale RNNs, which has linear time-complexity, opposite to transformer’s quadratic time-complexity. 
<br> Like the traditional RNNs, the first step is to run the tokens through an embedding layer. Then for each timestep, we update our hidden state using a Scale(token) layer and Matmul(h, token) layer, initializing h at 0. The result is 
H[t] = For each layer: h’ = Silu(h + MatmulLayer(concat(h, scale(token) * H[t-1])))
Then, like any other network we convert them to logits (like a reverse embedding) and apply softmax.

***Scale Layer: <br>
Linear(in_dim=emb dim, out_dim=emb dim), <br>
LayerNorm(shape=emb dim)*** <br>
***Matmul Layer: <br>
Linear(in_dim=emb dim*2, out_dim=emb dim), <br>
LayerNorm(shape=emb dim)*** <br>
<p>
WHY THIS ARCHITECTURE? <br>
First of all, transformer’s memory requirements and computational time dramatically increase depending on context length O(n^2) while RNNs have a time complexity of O(n). Secondly, this architecture’s performance is in a different ballpark to RNNs and LSTMs and even comparable to transformers while keeping half or more of the efficiency as vanilla RNNs.

![image](https://github.com/user-attachments/assets/ae905e0e-7615-422b-abf6-2acc356bc1c0)

**Hyperparameters:
Learning rate = 3e-4,
Batch Size = 32,
Sequence Length = 128,
Optimizer = AdamW,
Emb Dim / State Dim = 256,
Layers = 4,
Heads (transformer only) = 4,
Tokenizer = GPT2 (tiktoken),
Gradient Clipping (On) Max Norm = 1**

| Architecture  | Validation Loss (FineWeb-Edu) |
| ------------- | ------------- |
| SMRNN | 5.5 |
| Transformer  | 5.4 |
| Vanilla RNN | 7.7 |
