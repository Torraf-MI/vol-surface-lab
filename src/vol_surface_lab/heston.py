"""Modello di Heston (volatilità stocastica) e di Bates (Heston con salti).

* funzione caratteristica nella forma «little trap» di Albrecher et al. (2007), stabile anche per scadenze lunghe;
* prezzi delle europee con la formula di Lewis (2001) e quadratura di Gauss-Legendre, vettoriale sugli strike;
* calibrazione ai prezzi di mercato con i minimi quadrati sugli errori in volatilità (prezzo / vega);
* simulazione Monte Carlo con lo schema di Eulero «full truncation» (Lord, Koekkoek e van Dijk, 2010).

Dinamica sotto la misura neutrale al rischio::

    dS/S = (r − q) dt + √v dW₁ (+ salti lognormali compensati in Bates)
    dv   = κ (θ − v) dt + ξ √v dW₂,     d⟨W₁, W₂⟩ = ρ dt
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

from . import black

_NODI, _PESI = np.polynomial.legendre.leggauss(192)
_U_MAX = 250.0
_U = 0.5 * _U_MAX * (_NODI + 1.0)
_W = 0.5 * _U_MAX * _PESI


@dataclass
class Heston:
    kappa: float = 2.0   # velocità di ritorno della varianza
    theta: float = 0.04  # varianza di lungo periodo
    xi: float = 0.5      # volatilità della varianza
    rho: float = -0.7    # correlazione tra prezzo e varianza
    v0: float = 0.04     # varianza iniziale

    def feller(self) -> float:
        """Rapporto di Feller 2κθ/ξ²: sopra 1 la varianza non tocca mai lo zero."""
        return 2 * self.kappa * self.theta / self.xi**2

    def vettore(self) -> np.ndarray:
        return np.array([self.kappa, self.theta, self.xi, self.rho, self.v0])

    def dict(self) -> dict:
        return asdict(self)


@dataclass
class Salti:
    """Salti lognormali di Merton: intensità annua, media e deviazione standard del logaritmo del salto."""

    intensita: float = 0.1
    media: float = -0.10
    dev: float = 0.10

    def compensatore(self) -> float:
        return np.exp(self.media + 0.5 * self.dev**2) - 1.0

    def dict(self) -> dict:
        return asdict(self)


def funzione_caratteristica(u, T: float, h: Heston, salti: Salti | None = None):
    """E[exp(iu·ln(S_T/F_T))] per Heston o, con ``salti``, per Bates (u può essere complesso)."""
    u = np.asarray(u, dtype=complex)
    k, th, xi, rho, v0 = h.kappa, h.theta, h.xi, h.rho, h.v0
    beta = k - 1j * rho * xi * u
    d = np.sqrt(beta**2 + xi**2 * (1j * u + u**2))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * T)
    C = k * th / xi**2 * ((beta - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    Dv = (beta - d) / xi**2 * (1.0 - e) / (1.0 - g * e)
    ln_phi = C + Dv * v0
    if salti is not None and salti.intensita > 0:
        lam, mj, dj = salti.intensita, salti.media, salti.dev
        ln_phi = ln_phi + lam * T * (np.exp(1j * u * mj - 0.5 * dj**2 * u**2) - 1.0 - 1j * u * salti.compensatore())
    return np.exp(ln_phi)


def prezzo_europeo(F: float, K, T: float, h: Heston, D: float = 1.0, call=True, salti: Salti | None = None):
    """Prezzo di call e put europee con la formula di Lewis; ``K`` e ``call`` possono essere vettori."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    x = np.log(F / K)
    phi = funzione_caratteristica(_U - 0.5j, T, h, salti)
    integrando = np.real(np.exp(1j * np.outer(x, _U)) * phi) / (_U**2 + 0.25)
    c = F - np.sqrt(F * K) / np.pi * (integrando @ _W)
    c = np.maximum(c, np.maximum(F - K, 0.0))
    out = D * np.where(call, c, c - (F - K))
    return out


def vol_implicite_modello(F: float, K, T: float, h: Heston, salti: Salti | None = None) -> np.ndarray:
    """Smile implicito del modello su una scadenza (opzioni OTM: put sotto il forward, call sopra)."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    call = K >= F
    p = prezzo_europeo(F, K, T, h, 1.0, call, salti)
    return black.vol_implicite(p, F, K, T, 1.0, call)


# ----------------------------------------------------------------------------- calibrazione
LIMITI = (np.array([0.05, 0.002, 0.05, -0.99, 0.002]), np.array([12.0, 1.0, 3.0, 0.5, 1.0]))


@dataclass
class Calibrazione:
    modello: Heston
    rmse_vol: float                 # errore quadratico medio in volatilità (in unità di volatilità)
    errori: np.ndarray              # vol del modello − vol di mercato, per opzione
    iterazioni: int


def calibra(quotazioni, iniziale: Heston | None = None, pesi=None) -> Calibrazione:
    """Calibra Heston a un insieme di quotazioni pulite.

    ``quotazioni`` è un DataFrame con le colonne ``T``, ``K``, ``F``, ``D``, ``call``, ``prezzo`` e ``vol``
    (volatilità implicita di mercato). Si minimizzano gli errori di prezzo divisi per la vega, che approssimano
    gli errori in volatilità e pesano allo stesso modo le opzioni lontane e vicine al denaro.
    """
    q = quotazioni.reset_index(drop=True)
    gruppi = [(T, g.index.to_numpy()) for T, g in q.groupby("T")]
    vg = np.maximum(black.vega(q["F"].to_numpy(), q["K"].to_numpy(), q["T"].to_numpy(), q["vol"].to_numpy(),
                               q["D"].to_numpy()), 1e-3 * q["F"].to_numpy())
    w = np.ones(len(q)) if pesi is None else np.asarray(pesi, dtype=float)
    p_mkt = q["prezzo"].to_numpy()

    def residui(x):
        h = Heston(*x)
        r = np.empty(len(q))
        for T, idx in gruppi:
            r[idx] = prezzo_europeo(q.loc[idx[0], "F"], q.loc[idx, "K"].to_numpy(), T, h, q.loc[idx[0], "D"],
                                    q.loc[idx, "call"].to_numpy())
        return w * (r - p_mkt) / vg

    if iniziale is None:
        breve = q.loc[q["T"] == q["T"].min()]
        lungo = q.loc[q["T"] == q["T"].max()]
        atm_b = breve.iloc[(np.log(breve["K"] / breve["F"])).abs().argmin()]["vol"]
        atm_l = lungo.iloc[(np.log(lungo["K"] / lungo["F"])).abs().argmin()]["vol"]
        iniziale = Heston(kappa=1.5, theta=atm_l**2, xi=0.6, rho=-0.6, v0=atm_b**2)
    x0 = np.clip(iniziale.vettore(), LIMITI[0] + 1e-6, LIMITI[1] - 1e-6)
    ris = least_squares(residui, x0, bounds=LIMITI, x_scale=np.array([1.0, 0.02, 0.3, 0.3, 0.02]),
                        xtol=1e-10, ftol=1e-10, max_nfev=400)
    h = Heston(*ris.x)
    errori = ris.fun / w
    return Calibrazione(h, float(np.sqrt(np.mean(errori**2))), errori, int(ris.nfev))


# ----------------------------------------------------------------------------- Monte Carlo
def simula(S0: float, r: float, q: float, h: Heston, T: float, passi: int, percorsi: int, seed: int = 0,
           salti: Salti | None = None) -> np.ndarray:
    """Percorsi del sottostante (``percorsi × (passi + 1)``) con Eulero full truncation e variabili antitetiche."""
    rng = np.random.default_rng(seed)
    meta = (percorsi + 1) // 2
    dt = T / passi
    lnS = np.full(2 * meta, np.log(S0))
    v = np.full(2 * meta, h.v0)
    out = np.empty((2 * meta, passi + 1))
    out[:, 0] = S0
    comp = salti.compensatore() if salti is not None else 0.0
    lam = salti.intensita if salti is not None else 0.0
    for t in range(1, passi + 1):
        z1 = rng.standard_normal(meta)
        z2 = h.rho * z1 + np.sqrt(1 - h.rho**2) * rng.standard_normal(meta)
        z1, z2 = np.concatenate([z1, -z1]), np.concatenate([z2, -z2])
        vp = np.maximum(v, 0.0)
        sv = np.sqrt(vp * dt)
        lnS = lnS + (r - q - lam * comp - 0.5 * vp) * dt + sv * z1
        if lam > 0:
            n = rng.poisson(lam * dt, 2 * meta)
            lnS = lnS + n * salti.media + np.sqrt(n) * salti.dev * rng.standard_normal(2 * meta)
        v = v + h.kappa * (h.theta - vp) * dt + h.xi * sv * z2
        out[:, t] = np.exp(lnS)
    return out[:percorsi]
