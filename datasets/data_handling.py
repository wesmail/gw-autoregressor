import os
import math
import glob
import h5py
import numpy as np

# Torch imports
import torch
from torch.utils.data import Dataset, get_worker_info

# PyTorch Lightning imports
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split


def compute_theta(m1, m2, s1z, s2z):
    """Convert raw params → conditioning vector θ."""
    M = m1 + m2
    eta = (m1 * m2) / (M * M)
    chirp_mass = (m1 * m2) ** (3 / 5) / (M ** (1 / 5))
    return np.array([np.log(chirp_mass), eta, s1z, s2z], dtype=np.float32)


class MergerWindowDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for merger-centered waveform windows.

    Strategy
    ────────
    • HDF5 files are opened **lazily** and kept open for the lifetime of
      each DataLoader worker (or the main process if num_workers=0).
      This avoids the two extremes: loading everything into RAM *and*
      opening/closing a file on every __getitem__ call.
    • HDF5 is NOT fork-safe, so handles are created per-worker via
      worker_init_fn (see _worker_init below).
    • Only the small parameter arrays (mass, spin → theta) are preloaded
      into RAM since they're tiny (~600 KB for 100k samples).

    Returns dict:
        x:      [T, C, K]   time tokens
        y:      [T*K, C]    targets (next-token waveform)
        x_fft:  [T, C, F]   per-token FFT magnitude features
        theta:  [4]          conditioning parameters
    """

    def __init__(
        self,
        data_dir: str,
        kernel_size: int = 64,
        stride: int = 64,
        n_files: int | None = None,
        n_samples_per_file: int = 10000,
        pattern: str = "merger_windows_*.h5",
        freq_keep_bins: int = 8,
        freq_log1p: bool = True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.scale = np.float32(1e24)

        # STFT params (kept for compatibility)
        self.stft_n_fft = 2 * kernel_size
        self.stft_hop = stride
        self.stft_win_length = self.stft_n_fft

        # Frequency features
        self.freq_keep_bins = int(freq_keep_bins)
        self.freq_log1p = bool(freq_log1p)

        # ── discover files ─────────────────────────────────────────── #
        self.files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".h5")
        )
        if n_files is not None:
            self.files = self.files[:n_files]

        self.n_files = len(self.files)
        self.n_samples_per_file = n_samples_per_file
        self.total_len = self.n_files * self.n_samples_per_file
        print(f"Found {self.n_files} HDF5 file(s) in {data_dir}  "
              f"({self.total_len} samples)")

        # ── preload ONLY the tiny param arrays → theta ─────────────── #
        # ~16 bytes × 4 params × N_total ≈ negligible
        self._theta = self._preload_theta()

        # ── lazy HDF5 handle cache (populated per-worker) ──────────── #
        self._h5_handles: dict[int, h5py.File] = {}

    # ------------------------------------------------------------------ #
    #  Lightweight param preload
    # ------------------------------------------------------------------ #
    def _preload_theta(self) -> torch.Tensor:
        """Load only mass/spin params from every file (a few KB each)."""
        thetas = []
        for fpath in self.files:
            with h5py.File(fpath, "r") as f:
                m1  = f["params"]["mass1"][:]
                m2  = f["params"]["mass2"][:]
                s1z = f["params"]["spin1z"][:]
                s2z = f["params"]["spin2z"][:]
            M = m1 + m2
            eta = (m1 * m2) / (M * M)
            chirp = (m1 * m2) ** (3 / 5) / (M ** (1 / 5))
            theta = np.stack([np.log(chirp), eta, s1z, s2z], axis=-1)
            thetas.append(torch.from_numpy(theta.astype(np.float32)))
        return torch.cat(thetas, dim=0)  # [N_total, 4]

    # ------------------------------------------------------------------ #
    #  Per-worker HDF5 handle management
    # ------------------------------------------------------------------ #
    def _get_h5(self, file_idx: int) -> h5py.File:
        """Return a persistent handle, opening it on first access."""
        if file_idx not in self._h5_handles:
            self._h5_handles[file_idx] = h5py.File(
                self.files[file_idx], "r", swmr=True
            )
        return self._h5_handles[file_idx]

    def close_handles(self):
        """Explicitly close all open HDF5 handles."""
        for h in self._h5_handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._h5_handles.clear()

    def __del__(self):
        self.close_handles()

    # ------------------------------------------------------------------ #
    #  Frequency helper
    # ------------------------------------------------------------------ #
    def _freq_features(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token FFT magnitude: [T, C, K] → [T, C, F]."""
        mag = torch.fft.rfft(x, dim=-1).abs()
        if self.freq_log1p:
            mag = torch.log1p(mag)
        Fkeep = min(self.freq_keep_bins, mag.shape[-1])
        return mag[..., :Fkeep]

    # ------------------------------------------------------------------ #
    #  Dataset interface
    # ------------------------------------------------------------------ #
    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        file_idx = item // self.n_samples_per_file
        sample_idx = item % self.n_samples_per_file  # ← FIXED BUG

        h5 = self._get_h5(file_idx)
        hp = h5["waveforms"]["h_plus"][sample_idx].astype(np.float32)
        hc = h5["waveforms"]["h_cross"][sample_idx].astype(np.float32)

        # [L, 2] float32, scaled in-place
        waveform = np.stack([hp, hc], axis=-1)
        waveform *= self.scale
        waveform = torch.from_numpy(waveform)  # [L, 2]

        # Time tokens: [L, C] → [T+1, C, K]
        tokens = waveform.unfold(0, self.kernel_size, self.stride)
        x = tokens[:-1]   # [T, C, K]
        y = tokens[1:]     # [T, C, K]

        # Frequency features (no data leakage)
        x_freq = self._freq_features(x)  # [T, C, F]

        # Reshape y → [T*K, C]
        T, C, K = y.shape
        y = y.permute(0, 2, 1).contiguous().view(T * K, C)

        return {
            "x": x,
            "y": y,
            "x_freq": x_freq,
            "theta": self._theta[item],
        }


# ======================================================================= #
#  Worker init — ensures each DataLoader worker gets fresh HDF5 handles
# ======================================================================= #
def _worker_init(worker_id: int):
    """
    Called once per DataLoader worker.  Clears any inherited (and now
    invalid) HDF5 handles so they're re-opened cleanly in this process.
    """
    worker_info = get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # random_split wraps in a Subset — unwrap to get the real dataset
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        if hasattr(dataset, "close_handles"):
            dataset.close_handles()


# ======================================================================= #
#  LightningDataModule
# ======================================================================= #
class GWDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "/mnt/d/waleed/G2Net/ET/PureWaveforms",
        n_files: int | None = None,
        n_samples_per_file: int = 10000,
        kernel_size: int = 64,
        stride: int = 64,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.n_files = n_files
        self.n_samples_per_file = n_samples_per_file
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

    def setup(self, stage=None):
        full_dataset = MergerWindowDataset(
            data_dir=self.data_dir,
            kernel_size=self.kernel_size,
            stride=self.stride,
            n_files=self.n_files,
            n_samples_per_file=self.n_samples_per_file,
        )
        train_len = int(0.8 * len(full_dataset))
        val_len = int(0.1 * len(full_dataset))
        test_len = len(full_dataset) - train_len - val_len
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_len, val_len, test_len]
        )

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_worker_init,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False)