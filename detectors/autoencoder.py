"""
detectors/autoencoder.py — PyTorch autoencoder anomaly detector.

Trained on the FFT magnitude spectrum of healthy windows.
Anomaly score = mean squared reconstruction error.
Unusual spectra (faulty motors) produce higher reconstruction error.

Architecture
------------
    Encoder: 512 → 128 → 64 → 16  (bottleneck)
    Decoder: 16  → 64  → 128 → 512
    Activation: ReLU (hidden), Sigmoid (output for normalised input)
    Loss: MSELoss

The detector works directly on raw FFT spectra — the feature dict produced
by FeatureExtractor must include a 'fft_spectrum' key holding the
spectrum as a list/array.  This key is added by AutoencoderDetector.prepare_input()
so callers can still use the standard FeatureVector pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import BaseDetector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyTorch model definition (lazy-imported so the class is importable even
# when torch is not installed — it fails only at instantiation time).
# ---------------------------------------------------------------------------

def _build_model(input_dim: int):
    """Return a fresh autoencoder nn.Module."""
    import torch
    import torch.nn as nn

    class _Autoencoder(nn.Module):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(),
                nn.Linear(128, 64),    nn.ReLU(),
                nn.Linear(64, 16),     nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 64),      nn.ReLU(),
                nn.Linear(64, 128),     nn.ReLU(),
                nn.Linear(128, in_dim), nn.Sigmoid(),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return _Autoencoder(input_dim)


class AutoencoderDetector(BaseDetector):
    """
    Reconstruction-error anomaly detector using a fully-connected autoencoder.

    Parameters
    ----------
    n_fft_bins : int
        Number of FFT magnitude bins fed to the autoencoder (input dimension).
        Default 512 covers 0–500 Hz at 1 kHz sample rate.
    epochs : int
        Training epochs.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size for training.
    threshold : float, optional
        Reconstruction error above which a window is flagged anomalous.
    """

    def __init__(
        self,
        n_fft_bins: int = 512,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        threshold: Optional[float] = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self.n_fft_bins  = n_fft_bins
        self.epochs      = epochs
        self.lr          = lr
        self.batch_size  = batch_size

        self._model      = None   # nn.Module, built in fit()
        self._device     = None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_fft_spectrum(window: np.ndarray, n_bins: int = 512) -> np.ndarray:
        """
        Compute normalised FFT magnitude spectrum from a raw window.

        Parameters
        ----------
        window : np.ndarray
            1-D raw acceleration samples.
        n_bins : int
            Number of output bins (first half of FFT).

        Returns
        -------
        np.ndarray  shape (n_bins,)  values in [0, 1]
        """
        fft_mag = np.abs(np.fft.rfft(window, n=n_bins * 2))[:n_bins]
        max_val = fft_mag.max()
        return (fft_mag / max_val).astype(np.float32) if max_val > 0 else fft_mag

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def fit(
        self,
        features: List[Dict[str, float]],
        raw_windows: Optional[List[np.ndarray]] = None,
    ) -> "AutoencoderDetector":
        """
        Train the autoencoder on healthy spectra.

        Parameters
        ----------
        features : list of dict
            Standard feature dicts (used only for threshold calibration).
        raw_windows : list of np.ndarray, optional
            Raw 10 s windows to build spectra from.
            If None, features must contain 'fft_spectrum' as a list.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._device = torch.device("cpu")  # Pi is CPU-only
        self._model  = _build_model(self.n_fft_bins).to(self._device)

        # Build spectrum matrix
        if raw_windows is not None:
            spectra = np.stack([
                self.compute_fft_spectrum(w, self.n_fft_bins) for w in raw_windows
            ])
        elif "fft_spectrum" in features[0]:
            spectra = np.array([f["fft_spectrum"] for f in features], dtype=np.float32)
        else:
            raise ValueError(
                "Provide either raw_windows or feature dicts with 'fft_spectrum'."
            )

        X = torch.tensor(spectra, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X), batch_size=self.batch_size, shuffle=True
        )
        criterion = nn.MSELoss()
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        self._model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self._device)
                recon = self._model(batch)
                loss  = criterion(recon, batch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(batch)
            avg_loss = epoch_loss / len(X)
            if epoch % 10 == 0 or epoch == 1:
                logger.info("Autoencoder epoch %d/%d — loss=%.6f", epoch, self.epochs, avg_loss)

        self._is_fitted = True

        # Calibrate threshold from healthy reconstruction errors
        if self.threshold is None:
            errors = [self._reconstruction_error(s) for s in spectra]
            mu, std = np.mean(errors), np.std(errors)
            self.threshold = float(mu + 3 * std)
            logger.info(
                "Autoencoder threshold set to %.6f (mu=%.6f, std=%.6f)",
                self.threshold, mu, std,
            )

        return self

    def _reconstruction_error(self, spectrum: np.ndarray) -> float:
        """MSE between input spectrum and autoencoder reconstruction."""
        import torch
        self._model.eval()
        with torch.no_grad():
            x     = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0)
            recon = self._model(x)
            error = float(torch.nn.functional.mse_loss(recon, x).item())
        return error

    def score(self, feature: Dict[str, float]) -> float:
        """
        Score using reconstruction error of the FFT spectrum.

        The feature dict must contain 'fft_spectrum' (a list/array).
        Use `prepare_feature(window)` to build a compatible dict.
        """
        self._require_fitted()
        if "fft_spectrum" not in feature:
            raise KeyError(
                "'fft_spectrum' missing from feature dict.  "
                "Build it with AutoencoderDetector.prepare_feature(window)."
            )
        spectrum = np.array(feature["fft_spectrum"], dtype=np.float32)
        return self._reconstruction_error(spectrum)

    def prepare_feature(self, window: np.ndarray) -> Dict:
        """
        Helper: compute FFT spectrum from a raw window and return a dict
        that can be passed directly to score() or predict().
        """
        spectrum = self.compute_fft_spectrum(window, self.n_fft_bins)
        return {"fft_spectrum": spectrum.tolist()}

    def save(self, path: str) -> None:
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self._model.state_dict(),
            "n_fft_bins":  self.n_fft_bins,
            "threshold":   self.threshold,
            "epochs":      self.epochs,
            "lr":          self.lr,
            "batch_size":  self.batch_size,
        }, path)
        logger.info("AutoencoderDetector saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "AutoencoderDetector":
        import torch
        payload = torch.load(path, map_location="cpu")
        obj = cls(
            n_fft_bins=payload["n_fft_bins"],
            epochs=payload["epochs"],
            lr=payload["lr"],
            batch_size=payload["batch_size"],
            threshold=payload["threshold"],
        )
        obj._device = torch.device("cpu")
        obj._model  = _build_model(obj.n_fft_bins)
        obj._model.load_state_dict(payload["model_state"])
        obj._model.eval()
        obj._is_fitted = True
        logger.info("AutoencoderDetector loaded from %s", path)
        return obj
