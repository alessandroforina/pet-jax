import numpy as np

from collections import namedtuple

from jaxpme.batched_mixed.batching import get_batch as jaxpme_batcher
from jaxpme.batched_mixed.batching import prepare as jaxpme_prepare
from marathon.data.batching import batch_labels
from marathon.data.properties import DEFAULT_PROPERTIES
from marathon.data.sample import to_labels
from marathon.extra.edge_to_edge import get_neighborlist
from marathon.utils import next_size

Batch = namedtuple(
    "Batch",
    (
        "atomic_numbers",
        "reverse",
        "sr",
        "nopbc",
        "pbc",
        "labels",
    ),
)

Sample = namedtuple(
    "Sample",
    (
        "structure",
        "labels",
    ),
)


def to_batch(
    samples,
    keys,
    batch_size=None,
    strategies=None,
    shapes=None,
    properties=DEFAULT_PROPERTIES,
):
    if strategies is None:
        strategies = {"default": "powers_of_2"}

    if batch_size is not None:
        assert batch_size > len(samples)
    else:
        batch_size = next_size(len(samples) + 1, strategy="powers_of_2")

    labels, structures = [], []

    num_atoms = 0
    max_neighbors = 0
    for sample in samples:
        labels.append(sample.labels)
        structures.append(sample.structure)
        num_atoms += sample.structure["positions"].shape[0]
        max_neighbors = max(max_neighbors, sample.structure["max_neighbors"])

    if shapes is None:
        default = strategies.pop("default", "powers_of_2")

        num_neighbors = next_size(max_neighbors + 1, strategy="multiples")
        num_atoms = next_size(num_atoms + 1, strategy=strategies.get("fine", default))
        num_pairs = num_atoms * num_neighbors

        _, sr, nopbc, pbc = jaxpme_batcher(
            structures,
            strategy=default,
            num_structures_pbc=strategies.get("fine", default),
            num_pairs_nonpbc=strategies.get("coarse", default),
            num_pairs=num_pairs,
            num_structures=batch_size,
        )
    else:
        num_neighbors = shapes["neighbors"]
        assert num_neighbors > max_neighbors
        kwargs = {
            "num_structures": batch_size,
            "num_structures_pbc": shapes["pbc"],
            "num_atoms": shapes["atoms"],
            "num_atoms_pbc": shapes["atoms_pbc"],
            "num_pairs": shapes["atoms"] * num_neighbors,
            "num_pairs_nonpbc": shapes["pairs_nonpbc"],
            "num_k": shapes["k"],
            "strategy": "multiples",
        }
        _, sr, nopbc, pbc = jaxpme_batcher(structures, **kwargs)

    num_structures = sr.cell.shape[0]
    num_atoms = sr.positions.shape[0]

    atomic_numbers = np.zeros(num_atoms, dtype=int)
    Z = np.concatenate([sample.structure["atomic_numbers"] for sample in samples])
    atomic_numbers[: len(Z)] = Z

    labels = batch_labels(labels, num_structures, num_atoms, keys, properties=properties)

    centers, others, reverse, pair_mask = get_neighborlist(
        sr.centers,
        sr.others,
        sr.pair_mask,
        num_atoms,
        num_neighbors,
        cell_shifts=sr.cell_shifts,
    )
    cell_shifts = np.zeros((centers.shape[0], 3), dtype=int)
    cell_shifts[pair_mask] = sr.cell_shifts[sr.pair_mask]

    new_pair_to_structure = np.ones(
        pair_mask.shape[0], dtype=sr.pair_to_structure.dtype
    ) * len(samples)
    new_pair_to_structure[pair_mask] = sr.pair_to_structure[sr.pair_mask]

    sr = sr._replace(
        centers=centers,
        others=others,
        cell_shifts=cell_shifts,
        pair_mask=pair_mask,
        pair_to_structure=new_pair_to_structure,
    )

    return Batch(atomic_numbers, reverse, sr, nopbc, pbc, labels)


def to_sample(
    atoms,
    cutoff,
    keys=("energy", "forces"),
    energy=True,
    forces=True,
    stress=False,
    lr_wavelength=None,
    smearing=None,
    properties=DEFAULT_PROPERTIES,
):
    structure = jaxpme_prepare(
        atoms, cutoff, lr_wavelength=lr_wavelength, smearing=smearing, dtype=np.float32
    )
    labels = to_labels(
        atoms,
        keys=keys,
        energy=energy,
        forces=forces,
        stress=stress,
        properties=properties,
    )

    if len(structure["centers"]) > 0:
        structure["max_neighbors"] = int(
            np.unique(structure["centers"], return_counts=True)[1].max()
        )
    else:
        structure["max_neighbors"] = 0

    return Sample(structure, labels)
