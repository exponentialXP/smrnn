from datasets import load_dataset
from tqdm import tqdm

max_examples = 720_000 # 1 example = 1,000 tokens 
test_size = 0.01
dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split=f'train[:{max_examples}]', cache_dir='./cache', trust_remote_code=False)

split_dataset = dataset.train_test_split(test_size=test_size, seed=42, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')

def process(example):
    tokens = tokenizer.encode_ordinary(example['text'])
    tokens.append(tokenizer.eot_token)

    return {'tokens': tokens, 'len': len(tokens)}

tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc='Tokenizing splits',
)

import numpy as np

for split, example in tokenized.items():
    length = np.sum(example['len'], dtype=np.int64)
    filename = f"{split}.bin"
    map = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(length,))

    total_batches = int(min(max_examples*test_size, 512))
    if total_batches < 1:
        exit("!!<<Number of batches too small>>!!")

    start_idx = 0

    for ix in tqdm(range(total_batches), desc=f'Writing {filename}'):
        batch = example.shard(num_shards=total_batches, index=ix, contiguous=True).with_format('numpy')
        map_batch = np.concatenate(batch['tokens'])
        map[start_idx: start_idx + len(map_batch)] = map_batch
        start_idx += len(map_batch)

    map.flush()
