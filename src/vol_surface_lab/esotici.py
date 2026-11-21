"""Prodotti esotici e rischio di modello: lo stesso prezzo delle europee, prezzi diversi per gli esotici.

I payoff sono quelli che si trovano dentro i certificati di investimento: la put con barriera down-and-in di un
Bonus o di un Cash Collect, la digitale con barriera di un autocallable, l'opzione forward start di un cliquet.
I prezzi sono in percentuale del nozionale (lo spot iniziale).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Prodotto:
    nome: str
    descrizione: str
    tipo: str               # "put", "put_di", "one_touch", "forward_start"
    scadenza: float = 1.0
    strike: float = 1.0     # in percentuale dello spot iniziale
    barriera: float = 0.0
    inizio: float = 0.0     # per la forward start

    def payoff(self, percorsi: np.ndarray, tempi: np.ndarray) -> np.ndarray:
        S0 = percorsi[:, 0:1]
        x = percorsi / S0
        fine = x[:, -1]
        if self.tipo == "put":
            return np.maximum(self.strike - fine, 0.0)
        if self.tipo == "put_di":
            toccata = x.min(axis=1) <= self.barriera
            return np.where(toccata, np.maximum(self.strike - fine, 0.0), 0.0)
        if self.tipo == "one_touch":
            return (x.min(axis=1) <= self.barriera).astype(float)
        if self.tipo == "forward_start":
            i = int(np.argmin(np.abs(tempi - self.inizio)))
            return np.maximum(fine / x[:, i] - self.strike, 0.0)
        raise ValueError(f"tipo sconosciuto: {self.tipo}")


PRODOTTI = (
    Prodotto("Put europea 90%", "Put con strike al 90% dello spot: un'opzione semplice, che tutti i modelli devono "
             "prezzare allo stesso modo perché sono calibrati sulle europee.", "put", 1.0, 0.90),
    Prodotto("Put down-and-in 100% / barriera 70%", "La put che l'investitore vende dentro un certificato Bonus o "
             "Cash Collect: si attiva se l'indice tocca il 70% in un qualunque giorno dell'anno.", "put_di", 1.0, 1.0,
             0.70),
    Prodotto("Digitale one-touch 85%", "Paga il 100% del nozionale a scadenza se l'indice tocca l'85% almeno una "
             "volta: è la probabilità neutrale al rischio di toccare la barriera, scontata.", "one_touch", 1.0, 0.0,
             0.85),
    Prodotto("Forward start ATM 6m → 12m", "Call at-the-money che parte tra 6 mesi e scade tra un anno: dipende dallo "
             "smile futuro, che i modelli prevedono in modo diverso.", "forward_start", 1.0, 1.0, 0.0, 0.5),
)


def prezza(prodotto: Prodotto, percorsi: np.ndarray, T: float, sconto: float) -> tuple[float, float]:
    """Prezzo Monte Carlo e suo errore standard, in percentuale del nozionale."""
    tempi = np.linspace(0.0, T, percorsi.shape[1])
    pay = prodotto.payoff(percorsi, tempi) * sconto
    return float(pay.mean()), float(pay.std(ddof=1) / np.sqrt(len(pay)))


def simula_black_scholes(S0: float, r: float, q: float, sigma: float, T: float, passi: int, percorsi: int,
                         seed: int = 0) -> np.ndarray:
    """Moto browniano geometrico con volatilità costante (il modello «di riferimento» più semplice)."""
    rng = np.random.default_rng(seed)
    meta = (percorsi + 1) // 2
    dt = T / passi
    z = rng.standard_normal((meta, passi))
    z = np.vstack([z, -z])
    incr = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    lnS = np.log(S0) + np.concatenate([np.zeros((2 * meta, 1)), np.cumsum(incr, axis=1)], axis=1)
    return np.exp(lnS)[:percorsi]
