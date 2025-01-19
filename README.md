This new architecture is based off the RNN architecture, with almost the same speed and memory usage, now has identical loss to the transformer architecture, but O(n^2)->O(n) time complexity!

| Architecture  | Validation Loss (FineWeb-Edu) |
| ------------- | ------------- |
| SMRNN | 5.5 |
| Transformer  | 5.4 |
| Vanilla RNN | 7.7 |
