"""Gene-grouped batch sampler for DeSCORT training."""

import math
from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import Sampler
from transformers import Trainer


class GeneGroupedBatchSampler(Sampler):
    """Batch sampler where each batch contains samples from the same gene."""

    def __init__(self, gene_indices: List, batch_size: int, shuffle: bool = True,
                 seed: int = 42, drop_last: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)

        self.gene_to_indices: dict = {}
        for sample_idx, gene_label in enumerate(gene_indices):
            self.gene_to_indices.setdefault(gene_label, []).append(sample_idx)
        for gene in self.gene_to_indices:
            self.gene_to_indices[gene] = [int(i) for i in self.gene_to_indices[gene]]

        self.gene_list = sorted(self.gene_to_indices.keys())
        self.num_genes = len(self.gene_list)

    def _build_batches(self) -> List[List[int]]:
        all_batches = []
        for gene in self.gene_list:
            pool = self.gene_to_indices[gene][:]
            if self.shuffle:
                self.rng.shuffle(pool)
            for start in range(0, len(pool), self.batch_size):
                batch = pool[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                if len(batch) < self.batch_size and not self.drop_last:
                    extra = self.rng.choice(pool, size=self.batch_size - len(batch), replace=True)
                    batch = batch + [int(e) for e in extra]
                all_batches.append(batch)
        if self.shuffle:
            self.rng.shuffle(all_batches)
        return all_batches

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._build_batches():
            yield batch

    def __len__(self) -> int:
        total = 0
        for gene in self.gene_list:
            n = len(self.gene_to_indices[gene])
            total += n // self.batch_size if self.drop_last else max(1, math.ceil(n / self.batch_size))
        return total


class DistributedGeneGroupedBatchSampler(Sampler):
    """Distributed version: all GPUs share the same seed, each takes its shard."""

    def __init__(self, gene_indices: List, batch_size: int, num_replicas: int = 1,
                 rank: int = 0, shuffle: bool = True, seed: int = 42, drop_last: bool = False):
        self.base_seed = seed
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.gene_to_indices: dict = {}
        for sample_idx, gene_label in enumerate(gene_indices):
            self.gene_to_indices.setdefault(gene_label, []).append(sample_idx)
        for gene in self.gene_to_indices:
            self.gene_to_indices[gene] = [int(i) for i in self.gene_to_indices[gene]]
        self.gene_list = sorted(self.gene_to_indices.keys())
        self.num_genes = len(self.gene_list)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _build_all_batches_with_seed(self, seed: int) -> List[List[int]]:
        rng = np.random.RandomState(seed)
        all_batches = []
        for gene in self.gene_list:
            pool = self.gene_to_indices[gene][:]
            if self.shuffle:
                rng.shuffle(pool)
            for start in range(0, len(pool), self.batch_size):
                batch = pool[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                if len(batch) < self.batch_size and not self.drop_last:
                    extra = rng.choice(pool, size=self.batch_size - len(batch), replace=True)
                    batch = batch + [int(e) for e in extra]
                all_batches.append(batch)
        if self.shuffle:
            rng.shuffle(all_batches)
        return all_batches

    def __iter__(self) -> Iterator[List[int]]:
        seed = self.base_seed + self.epoch
        all_batches = self._build_all_batches_with_seed(seed)
        total_size = len(all_batches) // self.num_replicas * self.num_replicas
        all_batches = all_batches[:total_size]
        for i in range(self.rank, total_size, self.num_replicas):
            yield all_batches[i]

    def __len__(self) -> int:
        total = sum(len(self.gene_to_indices[g]) // self.batch_size if self.drop_last
                    else math.ceil(len(self.gene_to_indices[g]) / self.batch_size)
                    for g in self.gene_list)
        total = total // self.num_replicas * self.num_replicas
        return total // self.num_replicas


class BalancedMultiGeneBatchSampler(Sampler):
    """Equal gene exposure, multiple populations per batch, reproducible epochs.

    Each round visits every gene exactly once. A gene contributes a fixed number
    of independently drawn wells; replacement is used only for small groups.
    """
    def __init__(self, gene_indices, genes_per_batch=4, wells_per_gene=8,
                 rounds=4, seed=42):
        if min(genes_per_batch, wells_per_gene, rounds) < 1:
            raise ValueError('Batch dimensions and rounds must be positive')
        self.groups = {}
        for i, gene in enumerate(gene_indices):
            self.groups.setdefault(gene, []).append(i)
        self.genes = sorted(self.groups)
        self.genes_per_batch, self.wells_per_gene = genes_per_batch, wells_per_gene
        self.rounds, self.seed, self.epoch = rounds, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.rounds * math.ceil(len(self.genes) / self.genes_per_batch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.rounds):
            order = rng.permutation(len(self.genes))
            for start in range(0, len(order), self.genes_per_batch):
                batch = []
                for i in order[start:start + self.genes_per_batch]:
                    pool = self.groups[self.genes[i]]
                    batch.extend(rng.choice(pool, self.wells_per_gene,
                                            replace=len(pool) < self.wells_per_gene).tolist())
                yield batch


class _EpochDataLoader(torch.utils.data.DataLoader):
    def set_epoch(self, epoch):
        if hasattr(self.batch_sampler, 'set_epoch'):
            self.batch_sampler.set_epoch(epoch)


class GeneGroupedTrainer(Trainer):
    """Trainer using gene-grouped batch sampler. Multi-GPU aware."""

    def __init__(self, *args, gene_column: str = "Perturb_Gene",
                 genes_per_batch: int = 1, wells_per_gene: int = 8,
                 gene_sampling_rounds: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self._gene_column = gene_column
        self._genes_per_batch = genes_per_batch
        self._wells_per_gene = wells_per_gene
        self._gene_sampling_rounds = gene_sampling_rounds

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        gene_indices = self.train_dataset[self._gene_column]

        if self._genes_per_batch > 1:
            if self.args.world_size > 1:
                raise ValueError('Balanced multi-gene training currently requires one GPU')
            batch_sampler = BalancedMultiGeneBatchSampler(
                gene_indices, genes_per_batch=self._genes_per_batch,
                wells_per_gene=self._wells_per_gene, rounds=self._gene_sampling_rounds,
                seed=self.args.seed,
            )
        elif self.args.world_size > 1:
            batch_sampler = DistributedGeneGroupedBatchSampler(
                gene_indices=gene_indices,
                batch_size=self.args.per_device_train_batch_size,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                shuffle=True, seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last,
            )
        else:
            batch_sampler = GeneGroupedBatchSampler(
                gene_indices=gene_indices,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True, seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last,
            )

        worker_options = {}
        if self.args.dataloader_num_workers > 0:
            worker_options.update(prefetch_factor=self.args.dataloader_prefetch_factor,
                                  persistent_workers=self.args.dataloader_persistent_workers)
        return _EpochDataLoader(
            self.train_dataset, batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            **worker_options,
        )
