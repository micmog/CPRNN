import pickle
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset

import matflow as mf

from cprnn.utils import qu2eu


class StressDataset(Dataset):
    in_size = 10
    out_size = 7

    def __init__(self, data: np.ndarray, rtn_scaled: bool = True):
        self.n_batch = data.shape[0]
        self.n_steps = data.shape[1]

        self.mindata = np.min(data, axis=(0, 1))
        self.maxdata = np.max(data, axis=(0, 1))
        self._data = data
        # self._data = self._data[np.random.permutation(self.n_batch)]
        self.rtn_scaled = rtn_scaled

        assert self._data.dtype == np.float32

    # @classmethod
    # def concatenate(cls, datasets: Tuple["MyDataset"]) -> "MyDataset":
    #     for ds in datasets[1:]:
    #         assert ds.n_steps == datasets[0].n_steps

    #     data = np.concatenate([ds._data for ds in datasets], axis=0)
    #     return cls(data)

    @property
    def scaled_data(self) -> np.ndarray:
        return self.scale_data(self._data)

    @property
    def data(self) -> np.ndarray:
        return self.scaled_data if self.rtn_scaled else self._data

    def scale_data(self, data, idx=None):
        if idx is None:
            idx = slice(idx)
        scaled_data = np.copy(data)
        scaled_data -= self.mindata[idx]
        scaled_data /= self.maxdata[idx] - self.mindata[idx]
        return scaled_data

    def scale_inputs(self, data):
        return self.scale_data(data, idx=slice(None, self.in_size))

    def scale_outputs(self, data):
        return self.scale_data(data, idx=slice(self.in_size, None))

    def recover_data(self, scaled_data, idx=None):
        if idx is None:
            idx = slice(idx)
        data = np.copy(scaled_data)
        data *= self.maxdata[idx] - self.mindata[idx]
        data += self.mindata[idx]
        return data

    def recover_inputs(self, scaled_data):
        return self.recover_data(scaled_data, idx=slice(None, self.in_size))

    def recover_outputs(self, scaled_data):
        return self.recover_data(scaled_data, idx=slice(self.in_size, None))

    def __getitem__(self, index):
        val = self._data[index]
        return self.scale_data(val) if self.rtn_scaled else val

    # def __getitems__(self, index):

    def __len__(self):
        return self.n_batch


class StateDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.n_batch = data.shape[0]
        self.state_size = data.shape[1]

        self.data = data

        assert self.data.dtype == np.float32

    def __getitem__(self, index):
        return self.data[index]

    # def __getitems__(self, index):

    def __len__(self):
        return self.n_batch


def load_dataset(filename: str) -> tuple[StressDataset, StateDataset]:
    scale_seeds = 1.0
    scale_eulers = 1 / (2 * np.pi)

    with open(filename, "rb") as handle:
        datasets = pickle.load(handle)

    n_batch_all = len(datasets)
    n_steps = next(iter(datasets.values()))["F"].shape[0]
    # filter out incomplete datasets
    datasets = {
        k: v
        for k, v in datasets.items()
        if "eulers" in v and "P" in v and v["P"].size == n_steps
    }
    n_batch = len(datasets)
    print(f"n_batch {n_batch}, n_steps {n_steps}, num removed {n_batch_all - n_batch}")

    data = np.zeros(
        (n_batch, n_steps, StressDataset.in_size + StressDataset.out_size),
        dtype=np.float32,
    )
    state = np.zeros((n_batch, StateDataset.state_size), dtype=np.float32)

    for i, dataset in enumerate(datasets.values()):
        state[i] = (
            np.vstack(
                (dataset["seeds"] * scale_seeds, dataset["eulers"] * scale_eulers)
            )
            .flatten()
            .astype(np.float32)
        )
        data[i] = np.hstack(
            (
                dataset["F"].reshape((-1, 9)),
                np.linalg.det(dataset["F"])[:, np.newaxis],
                dataset["S"][:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]],
                dataset["P"][:, np.newaxis],
            )
        ).astype(np.float32)

    # shuffle data here
    return StressDataset(data), StateDataset(state)


def load_dataset_matflow(
    filename: str, use_cache: bool = True, **kwargs
) -> tuple[StressDataset, StateDataset]:
    filepath_cache = cache_path(filename)
    if use_cache and filepath_cache.exists():
        print("Loading cached file")
        data = load_data_cache(filepath_cache)
    else:
        data = load_data_matflow(filename)
        if use_cache:
            print("Saving cached file")
            save_data_cache(filepath_cache, data)

    return StressDataset(data[0], **kwargs), StateDataset(data[1])


def cache_path(filename: str):
    return Path(filename).with_suffix(".npz")


def load_data_cache(filepath: Path) -> tuple[np.ndarray]:
    data = np.load(filepath)
    return data["data"], data["state"]


def save_data_cache(filepath: Path, data: tuple[np.ndarray]) -> None:
    np.savez_compressed(filepath, data=data[0], state=data[1])


def load_data_matflow(filename: str, state_name: str) -> tuple[np.ndarray]:
    wk = mf.Workflow(filename)

    sim_elmts = wk.tasks.simulate_VE_loading_damask_HC.elements

    state_loader = MatflowStateLoader.get_loader(state_name)(wk)
    loop_elmts = (sim_elmts,) + state_loader.loop_elmts
    n_batch = len(sim_elmts)

    sim_is = []
    state = None
    data = None
    for i, loop_elmt in enumerate(zip(*loop_elmts)):
        if i % 10 == 0:
            print(i)

        sim_elmt, state_elmt = loop_elmt[0], loop_elmt[1:]

        try:
            VE_response = sim_elmt.outputs.VE_response.value
        except TypeError:
            print(f"Missing `VE_response` for sim {i}")
            continue
        if data is None:
            n_steps = len(VE_response["volume_data"]["F"]["meta"]["increments"])
            data = np.zeros(
                (n_batch, n_steps, StressDataset.in_size + StressDataset.out_size),
                dtype=np.float32,
            )

        n_steps_i = len(VE_response["volume_data"]["F"]["meta"]["increments"])
        if n_steps != n_steps_i:
            print(f"Steps missing from sim {i}. Expected {n_steps}, got {n_steps_i}")
            continue
        # TODO: check all sims are same length
        # TODO: check all sims have the right data

        F = VE_response["volume_data"]["F"]["data"][:]
        s_sigma = VE_response["volume_data"]["s_sigma"]["data"][:]
        p_sigma = VE_response["volume_data"]["p_sigma"]["data"][:]
        data[i] = np.hstack(
            (
                F.reshape((-1, 9)),
                np.linalg.det(F)[:, np.newaxis],
                s_sigma[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]],
                p_sigma[:, np.newaxis],
            )
        ).astype(np.float32)

        try:
            state_i = state.get_state(state_elmt)
        except TypeError:
            print(f"Missing workflow output for sim {i}")
            continue
        if state is None:
            state = np.zeros((n_batch, len(state)), dtype=np.float32)
        state[i] = state_i

        sim_is.append(i)

    # shuffle data here
    return data[sim_is], state[sim_is]


class MatflowStateLoader(ABC):
    def __init__(self, wk) -> None:
        self.wk = wk

    @abstractmethod
    def loop_elmts(self):
        pass

    @abstractmethod
    def get_state(self, state_elmt):
        pass

    @staticmethod
    def get_loader(cls, state_name: str):
        state_loader = {
            "voronoi": MatflowStateLoaderVor,
            "fingerprint": MatflowStateLoaderFP,
        }.get(state_name.lower())
        if state_loader is None:
            raise ValueError(f"Unknown state type `{state_name}`")
        return state_loader


class MatflowStateLoaderVor(MatflowStateLoader):
    scale_seeds = 1e5
    scale_eulers = 1 / (2 * np.pi)

    def __init__(self, wk) -> None:
        super().__init__(wk)

    @property
    def loop_elmts(self):
        return (self.wk.tasks.generate_microstructure_seeds_from_random.elements,)

    def get_state(self, state_elmt):
        seeds_elmt = state_elmt[0]
        try:
            seeds = seeds_elmt.outputs.microstructure_seeds.value
        except TypeError:
            print("Missing `seeds`")
            raise TypeError()

        eulers = qu2eu(seeds.orientations.data[:], P=-1)
        state = np.vstack(
            (
                seeds.position * self.scale_seeds,
                eulers * self.scale_eulers,
            )
        )
        return state.flatten().astype(np.float32)


class MatflowStateLoaderFP(MatflowStateLoader):
    def __init__(self, wk) -> None:
        super().__init__(wk)

    @property
    def loop_elmts(self):
        return (self.wk.tasks.get_3dvae_fingerprint.elements,)

    def get_state(self, state_elmt):
        fp_elmt = state_elmt[0]
        try:
            fingerprint = fp_elmt.outputs.fingerprint.value[:]
        except TypeError:
            print("Missing `fingerprint`")
            raise TypeError()
        return fingerprint.flatten().astype(np.float32)
