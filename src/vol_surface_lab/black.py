"""Formula di Black sul forward, greche e volatilità implicita.

Tutte le funzioni lavorano sul prezzo forward ``F`` e sul fattore di sconto ``D``: il prezzo di una call è
``D · [F·N(d1) − K·N(d2)]``. Così dividendi e tassi entrano solo attraverso ``F`` e ``D``, che si ricavano dalla
parità put-call (vedi :mod:`vol_surface_lab.mercato`).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1d2(F, K, T, sigma):
    F, K, T, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, sigma))
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return d1, d1 - sd


def prezzo(F, K, T, sigma, D=1.0, call=True):
    """Prezzo di Black di una call (``call=True``) o di una put europea."""
    d1, d2 = _d1d2(F, K, T, sigma)
    c = np.asarray(D) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    p = c - np.asarray(D) * (np.asarray(F) - np.asarray(K))
    out = np.where(call, c, p)
    return float(out) if out.ndim == 0 else out


def delta(F, K, T, sigma, D=1.0, call=True):
    """Delta rispetto al forward (moltiplicato per D)."""
    d1, _ = _d1d2(F, K, T, sigma)
    out = np.asarray(D) * np.where(call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    return float(out) if out.ndim == 0 else out


def vega(F, K, T, sigma, D=1.0):
    """Derivata del prezzo rispetto alla volatilità (per 1,00 di volatilità, non per un punto)."""
    d1, _ = _d1d2(F, K, T, sigma)
    out = np.asarray(D) * np.asarray(F) * norm.pdf(d1) * np.sqrt(T)
    return float(out) if out.ndim == 0 else out


def gamma(F, K, T, sigma, D=1.0):
    d1, _ = _d1d2(F, K, T, sigma)
    out = np.asarray(D) * norm.pdf(d1) / (np.asarray(F) * sigma * np.sqrt(T))
    return float(out) if out.ndim == 0 else out


def vol_implicita(p: float, F: float, K: float, T: float, D: float = 1.0, call: bool = True,
                  vmin: float = 1e-4, vmax: float = 5.0) -> float:
    """Volatilità implicita di Black con il metodo di Brent; ``nan`` se il prezzo è fuori dai limiti di arbitraggio."""
    intrinseco = D * max(F - K, 0.0) if call else D * max(K - F, 0.0)
    massimo = D * F if call else D * K
    if not np.isfinite(p) or p <= intrinseco + 1e-12 or p >= massimo:
        return float("nan")
    f = lambda s: prezzo(F, K, T, s, D, call) - p  # noqa: E731
    if f(vmin) > 0 or f(vmax) < 0:
        return float("nan")
    return brentq(f, vmin, vmax, xtol=1e-10, maxiter=200)


def vol_implicite(prezzi, F, K, T, D=1.0, call=True) -> np.ndarray:
    """Versione vettoriale di :func:`vol_implicita` (gli argomenti si allineano con il broadcasting)."""
    b = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (prezzi, F, K, T, D)), np.asarray(call))
    out = np.empty(b[0].shape)
    for i in np.ndindex(out.shape):
        out[i] = vol_implicita(b[0][i], b[1][i], b[2][i], b[3][i], b[4][i], bool(b[5][i]))
    return out
