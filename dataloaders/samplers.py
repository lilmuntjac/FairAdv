import time
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.indices = list(range(len(dataset)))
        self.subgroup_to_indices = defaultdict(list)

        # Assuming the dataset has a 'subgroup' attribute for each instance
        for idx in self.indices:
            *_, subgroup = dataset[idx]  # Unpack all items, subgroup will be the last item
            # Ensure that subgroup is indeed a tuple (subgroup info), not an attribute or image
            if not isinstance(subgroup, tuple):
                raise ValueError("Expected the last return value to be a subgroup tuple.")
            self.subgroup_to_indices[subgroup].append(idx)
        
        self.batches = self.make_batches()

    def make_batches(self):
        """
        Creates batches in such a way that each batch is balanced across subgroups.
        Instances from the subgroup with the most instances are prioritized for inclusion,
        and instances from smaller subgroups can be selected more than once to fill a batch.
        """
        # Shuffle indices within each subgroup for randomness
        subgroup_indices = defaultdict(list)
        for subgroup, indices in self.subgroup_to_indices.items():
            np.random.shuffle(indices)
            subgroup_indices[subgroup] = indices

        # Determine the maximum number of full batches possible
        largest_subgroup_size = max(len(indices) for indices in subgroup_indices.values())
        num_subgroups = len(subgroup_indices)
        instances_per_subgroup_per_batch = max(1, self.batch_size // num_subgroups)
        max_batches = largest_subgroup_size // instances_per_subgroup_per_batch

        # Extend indices of smaller subgroups to ensure they can contribute to all batches
        for subgroup, indices in subgroup_indices.items():
            if len(indices) < max_batches * instances_per_subgroup_per_batch:
                repeats = (max_batches * instances_per_subgroup_per_batch // len(indices)) + 1
                extended_indices = indices * repeats
                subgroup_indices[subgroup] = extended_indices[:max_batches * instances_per_subgroup_per_batch]

        batches = []
        for batch_idx in range(max_batches):
            batch = []
            # Collect instances from each subgroup for the current batch
            for subgroup in subgroup_indices:
                start_idx = batch_idx * instances_per_subgroup_per_batch
                end_idx = start_idx + instances_per_subgroup_per_batch
                batch.extend(subgroup_indices[subgroup][start_idx:end_idx])

            # If the batch is underfilled, top it off with extra instances from any subgroup
            while len(batch) < self.batch_size:
                extra_indices = np.random.choice(self.indices, self.batch_size - len(batch), replace=True).tolist()
                batch.extend(extra_indices)

            # Shuffle the batch to mix instances from different subgroups
            np.random.shuffle(batch)
            batches.append(batch)

        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)