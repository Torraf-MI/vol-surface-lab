"""Volatilità locale di Dupire ricavata dalla superficie SSVI e simulazione Monte Carlo.

Nella forma di Gatheral, con la varianza totale implicita w(k, T)::

    σ²_loc(k, T) = (∂w/∂T) / g(k, T)

dove g è la stessa funzione di Durrleman dei controlli di arbitraggio: una superficie senza arbitraggio (∂w/∂T ≥ 0
e g ≥ 0) dà sempre una varianza locale positiva. Il modello a volatilità locale riprezza per costruzione tutte le
europee della superficie, ma implica una dinamica dello smile diversa da quella della volatilità stocastica: è qui che
nascono le differenze sui prodotti esotici.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .svi import SSVI, g


@dataclass
class VolatilitaLocale:
    T: np.ndarray       # griglia delle scadenze
    k: np.ndarray       # griglia della log-moneyness rispetto al forward
    sigma: np.ndarray   # volatilità locale, forma (len(T), len(k))

    def __post_init__(self):
        self._f = RegularGridInterpolator((self.T, self.k), self.sigma, bounds_error=False, fill_value=None)

    def __call__(self, k, t):
        k, t = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(t, dtype=float))
        pts = np.column_stack([np.clip(t.ravel(), self.T[0], self.T[-1]), np.clip(k.ravel(), self.k[0], self.k[-1])])
        return self._f(pts).reshape(k.shape)


def da_ssvi(s: SSVI, T_max: float = 2.0, k_min: float = -1.2, k_max: float = 0.8, nT: int = 60, nk: int = 121,
            cap: float = 1.5) -> VolatilitaLocale:
    """Griglia della volatilità locale di Dupire dalla superficie SSVI (derivata in T alle differenze finite)."""
    Tg = np.linspace(0.01, T_max, nT)
    kg = np.linspace(k_min, k_max, nk)
    TT, KK = np.meshgrid(Tg, kg, indexing="ij")
    h = 1e-4
    dwdT = (s.w(KK, TT + h) - s.w(KK, np.maximum(TT - h, 1e-6))) / (TT + h - np.maximum(TT - h, 1e-6))
    w, dw, d2w = s.derivate_k(KK, TT)
    den = np.maximum(g(KK, w, dw, d2w), 1e-6)
    var = np.maximum(dwdT, 1e-8) / den
    return VolatilitaLocale(Tg, kg, np.minimum(np.sqrt(var), cap))


def simula(S0: float, r: float, q: float, lv: VolatilitaLocale, T: float, passi: int, percorsi: int,
           seed: int = 0) -> np.ndarray:
    """Percorsi con dlnS = (r − q − σ²/2) dt + σ_loc(ln(S/F_t), t) dW, Eulero sul logaritmo, antitetiche."""
    rng = np.random.default_rng(seed)
    meta = (percorsi + 1) // 2
    dt = T / passi
    x = np.zeros(2 * meta)      # ln(S_t / F_t)
    out = np.empty((2 * meta, passi + 1))
    out[:, 0] = S0
    for i in range(1, passi + 1):
        t = (i - 0.5) * dt
        z = rng.standard_normal(meta)
        z = np.concatenate([z, -z])
        sig = lv(x, np.full_like(x, t))
        x = x - 0.5 * sig**2 * dt + sig * np.sqrt(dt) * z
        out[:, i] = S0 * np.exp((r - q) * i * dt + x)
    return out[:percorsi]
