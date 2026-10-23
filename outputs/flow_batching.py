"""Length-bucketed shard iterator and padded batch collation for Flow training."""
import random
from concurrent.futures import ThreadPoolExecutor

import torch


def collate(records, pin_memory=False):
    if not records:
        raise ValueError("empty batch")
    batch = len(records)
    length = max(len(record["aa"]) for record in records)
    embedding_dim = records[0]["embed"].shape[-1]
    result = dict(
        embed=torch.zeros(batch, length, embedding_dim, dtype=records[0]["embed"].dtype),
        aa=torch.zeros(batch, length, dtype=torch.long),
        ca=torch.zeros(batch, length, 3),
        rot=torch.eye(3).expand(batch, length, 3, 3).clone(),
        atoms=torch.zeros(batch, length, 4, 3),
        atom_mask=torch.zeros(batch, length, 4, dtype=torch.bool),
        segments=torch.full((batch, length), -1, dtype=torch.long),
        mask=torch.zeros(batch, length, dtype=torch.bool),
    )
    for i, record in enumerate(records):
        n = len(record["aa"])
        for key in ("embed", "aa", "ca", "rot", "atoms", "atom_mask", "segments"):
            result[key][i, :n] = record[key]
        result["mask"][i, :n] = True
    if pin_memory:
        result = {key: value.pin_memory() for key, value in result.items()}
    return result


def load_shard(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def train_batches(cache_dir, index, batch_size, seed, epoch, pin_memory=False):
    """Prefetch one shard; bucket by length within each shard to limit padding."""
    order = [shard for shard in index["shards"] if shard["splits"].get("train", 0)]
    rng = random.Random(seed + epoch)
    rng.shuffle(order)
    with ThreadPoolExecutor(max_workers=1) as pool:
        next_load = pool.submit(load_shard, cache_dir / order[0]["file"])
        for position, shard in enumerate(order):
            records = next_load.result()
            if position + 1 < len(order):
                next_load = pool.submit(load_shard, cache_dir / order[position + 1]["file"])
            records = [record for record in records if record["split"] == "train"]
            rng.shuffle(records)
            batches = []
            for start in range(0, len(records), batch_size * 16):
                window = sorted(records[start:start + batch_size * 16],
                                key=lambda record: len(record["aa"]))
                batches.extend(window[offset:offset + batch_size]
                               for offset in range(0, len(window), batch_size))
            rng.shuffle(batches)
            for group in batches:
                yield collate(group, pin_memory=pin_memory)
