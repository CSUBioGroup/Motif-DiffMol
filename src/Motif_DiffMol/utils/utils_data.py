import os
import torch
import datasets
from rdkit import RDLogger
from safe.tokenizer import SAFETokenizer
from src.Motif_DiffMol.utils.bracket_safe_converter import safe2bracketsafe

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
RDLogger.DisableLog('rdApp.*')

def get_last_checkpoint(save_dir):
    if os.path.exists(save_dir):
        filenames = os.listdir(save_dir)
        if filenames:
            last_filename = sorted(
                filenames,
                key=lambda x: int(x.replace(".ckpt", "").split("-")[-1])
            )[-1]
            return os.path.join(save_dir, last_filename)

def get_tokenizer():
    tk = SAFETokenizer.from_pretrained("safe-gpt").get_pretrained()
    tk.add_tokens(['<', '>'])
    return tk

class MoleculeCollator:

    def __init__(self, config):
        self.tokenizer = get_tokenizer()
        self.max_length = config.model.max_position_embeddings

    def __call__(self, examples):
        for example in examples: example['input'] = safe2bracketsafe(example['input'])
        batch = self.tokenizer([example['input'] for example in examples],
                               return_tensors='pt',
                               padding=True,
                               truncation=True,
                               max_length=self.max_length)
        del batch['token_type_ids']
        return batch

def get_dataloader(config, global_step=0, rank=0, world_size=1):
    dataset = datasets.load_dataset('datamol-io/safe-gpt', streaming=True, split='train')
    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank)

    if global_step > 0:
        skip_count = global_step * config.loader.batch_size
        dataset = dataset.skip(skip_count)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.loader.batch_size,
        collate_fn=MoleculeCollator(config),
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=False,
        persistent_workers=config.loader.persistent_workers
    )

    return loader

