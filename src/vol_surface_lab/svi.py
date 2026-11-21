"""SVI e SSVI: parametrizzazioni dello smile e controlli di assenza di arbitraggio.

Notazione di Gatheral: ``k = ln(K/F)`` è la log-moneyness, ``w(k) = σ²(k)·T`` la varianza totale implicita.

* **SVI raw** (Gatheral, 2004), una scadenza alla volta: ``w(k) = a + b·[ρ(k − m) + √((k − m)² + σ²)]``.
* **SSVI** (Gatheral e Jacquier, 2014), tutta la superficie con tre parametri più la curva della varianza ATM θ(T):
  ``w(k, θ) = θ/2 · [1 + ρφk + √((φk + ρ)² + 1 − ρ²)]`` con ``φ(θ) = η / (θ^γ (1 + θ)^(1−γ))``.
* **Butterfly**: la densità neutrale al rischio è positiva se ``g(k) ≥ 0`` (condizione di Durrleman).
* **Calendar**: la varianza totale non deve diminuire con la scadenza a parità di k.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


# ----------------------------------------------------------------------------- SVI raw
@dataclass
class SVI:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def w(self, k):
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2))

    def dw(self, k):
        k = np.asarray(k, dtype=float)
        return self.b * (self.rho + (k - self.m) / np.sqrt((k - self.m) ** 2 + self.sigma**2))

    def d2w(self, k):
        k = np.asarray(k, dtype=float)
        return self.b * self.sigma**2 / ((k - self.m) ** 2 + self.sigma**2) ** 1.5

    def vol(self, k, T: float):
        return np.sqrt(np.maximum(self.w(k), 1e-12) / T)

    def dict(self) -> dict:
        return asdict(self)


def g(k, w, dw, d2w):
    """g(k) = (1 − k w'/(2w))² − (w'²/4)(1/w + 1/4) + w''/2."""
    k = np.asarray(k, dtype=float)
    w = np.maximum(np.asarray(w, dtype=float), 1e-12)
    return (1 - k * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 0.25) + d2w / 2


def densita(k, w, dw, d2w):
    """Densità neutrale al rischio di k = ln(K/F) implicita nello smile: g(k)/√(2πw) · exp(−d₂²/2)."""
    k = np.asarray(k, dtype=float)
    w = np.maximum(np.asarray(w, dtype=float), 1e-12)
    d2 = -k / np.sqrt(w) - np.sqrt(w) / 2
    return g(k, w, dw, d2w) / np.sqrt(2 * np.pi * w) * np.exp(-0.5 * d2**2)


def calibra_svi(k, w_mkt, pesi=None, tentativi: int = 6, seed: int = 0) -> SVI:
    """SVI raw su una scadenza con i minimi quadrati sulla varianza totale e più punti di partenza."""
    k, w_mkt = np.asarray(k, dtype=float), np.asarray(w_mkt, dtype=float)
    pesi = np.ones_like(k) if pesi is None else np.asarray(pesi, dtype=float)
    rng = np.random.default_rng(seed)
    wmin, wmax = float(w_mkt.min()), float(w_mkt.max())
    span = max(k.max() - k.min(), 0.05)

    def res(x):
        a, b, rho, m, s = x
        r = pesi * (SVI(a, b, rho, m, s).w(k) - w_mkt)
        vincolo = min(a + b * s * np.sqrt(1 - rho**2), 0.0)  # varianza minima non negativa
        return np.append(r, 10.0 * vincolo)

    lo = [-wmax, 1e-6, -0.999, k.min() - span, 1e-4]
    hi = [wmax, 10.0, 0.999, k.max() + span, 2.0 * span]
    migliore = None
    for t in range(tentativi):
        x0 = [0.5 * wmin, (wmax - wmin) / span + 1e-3, -0.5, 0.0, 0.1] if t == 0 else [
            rng.uniform(0, wmin), rng.uniform(0.01, 2 * (wmax - wmin) / span + 0.05), rng.uniform(-0.9, 0.3),
            rng.uniform(-0.3 * span, 0.3 * span), rng.uniform(0.02, 0.5 * span)]
        x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        r = least_squares(res, x0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, max_nfev=2000)
        if migliore is None or r.cost < migliore.cost:
            migliore = r
    return SVI(*migliore.x)


# ----------------------------------------------------------------------------- SSVI
@dataclass
class SSVI:
    rho: float
    eta: float
    gamma: float
    scadenze: np.ndarray        # scadenze dei nodi della curva θ
    theta_nodi: np.ndarray      # varianza totale ATM ai nodi (non decrescente)

    def __post_init__(self):
        T = np.concatenate([[0.0], np.asarray(self.scadenze, dtype=float)])
        th = np.concatenate([[0.0], np.asarray(self.theta_nodi, dtype=float)])
        self._interp = PchipInterpolator(T, th, extrapolate=True)
        self._pendenza_finale = (th[-1] - th[-2]) / (T[-1] - T[-2])

    def theta(self, T):
        T = np.asarray(T, dtype=float)
        tmax = float(self.scadenze[-1])
        oltre = self.theta_nodi[-1] + self._pendenza_finale * (T - tmax)
        return np.where(T <= tmax, self._interp(np.minimum(T, tmax)), oltre)

    def phi(self, th):
        th = np.maximum(np.asarray(th, dtype=float), 1e-10)
        return self.eta / (th**self.gamma * (1 + th) ** (1 - self.gamma))

    def w(self, k, T):
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        th = self.theta(T)
        p = self.phi(th)
        return th / 2 * (1 + self.rho * p * k + np.sqrt((p * k + self.rho) ** 2 + 1 - self.rho**2))

    def vol(self, k, T):
        T = np.asarray(T, dtype=float)
        return np.sqrt(np.maximum(self.w(k, T), 1e-14) / np.maximum(T, 1e-10))

    def derivate_k(self, k, T):
        """w, ∂w/∂k e ∂²w/∂k² in forma chiusa."""
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        th = self.theta(T)
        p = self.phi(th)
        r = np.sqrt((p * k + self.rho) ** 2 + 1 - self.rho**2)
        w = th / 2 * (1 + self.rho * p * k + r)
        dw = th / 2 * p * (self.rho + (p * k + self.rho) / r)
        d2w = th / 2 * p**2 * (1 - self.rho**2) / r**3
        return w, dw, d2w

    def condizione_butterfly(self) -> float:
        """η(1 + |ρ|): con γ ≤ 1/2 la superficie è libera da arbitraggio butterfly se non supera 2."""
        return self.eta * (1 + abs(self.rho))

    def dict(self) -> dict:
        return {"rho": self.rho, "eta": self.eta, "gamma": self.gamma}


def calibra_ssvi(k, T, w_mkt, theta_atm: dict[float, float], pesi=None) -> SSVI:
    """SSVI su tutta la superficie. La curva θ(T) viene dagli smile per scadenza resa non decrescente; ρ, η e γ si
    stimano con il vincolo η(1 + |ρ|) ≤ 2 e γ ∈ [0,05; 0,5], che garantisce l'assenza di arbitraggio statico."""
    k, T, w_mkt = (np.asarray(x, dtype=float) for x in (k, T, w_mkt))
    pesi = np.ones_like(k) if pesi is None else np.asarray(pesi, dtype=float)
    sc = np.array(sorted(theta_atm))
    th = np.maximum.accumulate(np.array([theta_atm[t] for t in sc]))
    th = th + 1e-6 * np.arange(len(th))   # strettamente crescente

    def costruisci(x):
        rho, u, gam = x
        eta = 2.0 / (1 + abs(rho)) * u    # u ∈ (0, 1]: il vincolo è soddisfatto per costruzione
        return SSVI(rho, eta, gam, sc, th)

    def res(x):
        return pesi * (costruisci(x).w(k, T) - w_mkt)

    r = least_squares(res, [-0.6, 0.5, 0.4], bounds=([-0.999, 1e-3, 0.05], [0.999, 1.0, 0.5]), xtol=1e-12,
                      ftol=1e-12)
    return costruisci(r.x)


# ----------------------------------------------------------------------------- controlli
def controllo_butterfly(fn_derivate, k_griglia) -> dict:
    """Minimo di g(k) sulla griglia per uno smile (``fn_derivate(k)`` restituisce w, w', w'')."""
    w, dw, d2w = fn_derivate(k_griglia)
    gg = g(k_griglia, w, dw, d2w)
    i = int(np.argmin(gg))
    return {"g_min": float(gg[i]), "k_min": float(k_griglia[i]), "violazione": bool(gg[i] < -1e-9),
            "zona": k_griglia[gg < -1e-9]}


def controllo_calendar(fn_w, scadenze, k_griglia) -> list[dict]:
    """Coppie di scadenze consecutive in cui la varianza totale scende: arbitraggio calendar."""
    out = []
    sc = sorted(scadenze)
    for t1, t2 in zip(sc[:-1], sc[1:]):
        diff = fn_w(k_griglia, t2) - fn_w(k_griglia, t1)
        i = int(np.argmin(diff))
        out.append({"T1": t1, "T2": t2, "diff_min": float(diff[i]), "k_min": float(k_griglia[i]),
                    "violazione": bool(diff[i] < -1e-9)})
    return out


def skew_atm(fn_vol, T: float, h: float = 1e-3) -> float:
    """Pendenza dello smile al forward, ∂σ/∂k a k = 0."""
    return float((fn_vol(h, T) - fn_vol(-h, T)) / (2 * h))
