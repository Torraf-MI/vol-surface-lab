"""Catene di opzioni: generatore sintetico, lettura da CSV, pulizia, forward e sconto dalla parità put-call.

Il mercato dimostrativo è un indice azionario europeo sintetico a 5.000 punti. I prezzi «veri» vengono da un
modello di Bates (Heston con salti), così il progetto può misurare quanto ogni modello si avvicina alla realtà. Sopra
i prezzi veri si costruiscono bid e ask realistici e si aggiungono gli errori tipici dei dati di fine giornata:
quotazioni vecchie, bid a zero sulle ali, prezzi incrociati e fuori dai limiti di arbitraggio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import black
from .heston import Heston, Salti, prezzo_europeo

SCADENZE_STANDARD = (1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5, 2.0)
ETICHETTE = {1 / 12: "1 mese", 2 / 12: "2 mesi", 3 / 12: "3 mesi", 6 / 12: "6 mesi", 9 / 12: "9 mesi",
             1.0: "1 anno", 1.5: "18 mesi", 2.0: "2 anni"}


def etichetta(T: float) -> str:
    for t, e in ETICHETTE.items():
        if abs(t - T) < 1e-6:
            return e
    return f"{T * 12:.0f} mesi" if T < 1.5 else f"{T:.1f} anni".replace(".", ",")


@dataclass
class Mercato:
    """Dati del sottostante: livello, tasso privo di rischio e rendimento da dividendi (continui)."""

    nome: str = "Indice azionario europeo (sintetico)"
    spot: float = 5000.0
    tasso: float = 0.025
    dividendi: float = 0.03
    tick: float = 0.1

    def forward(self, T):
        return self.spot * np.exp((self.tasso - self.dividendi) * np.asarray(T, dtype=float))

    def sconto(self, T):
        return np.exp(-self.tasso * np.asarray(T, dtype=float))


@dataclass
class ModelloVero:
    """Il modello che genera il mercato sintetico (Bates)."""

    heston: Heston = field(default_factory=lambda: Heston(kappa=1.6, theta=0.04, xi=0.6, rho=-0.7, v0=0.025))
    salti: Salti = field(default_factory=lambda: Salti(intensita=0.5, media=-0.10, dev=0.08))


def genera_catena(m: Mercato, vero: ModelloVero, scadenze=SCADENZE_STANDARD, seed: int = 7,
                  errori: bool = True) -> pd.DataFrame:
    """Catena sintetica di call e put con bid, ask ed errori di dati realistici."""
    rng = np.random.default_rng(seed)
    righe = []
    for T in scadenze:
        F, D = float(m.forward(T)), float(m.sconto(T))
        sd = 0.2 * np.sqrt(T)
        passo = 25.0 if T <= 0.25 else 50.0 if T <= 1 else 100.0
        kmin, kmax = F * np.exp(-3.2 * sd - 0.03), F * np.exp(2.0 * sd + 0.02)
        strikes = np.arange(np.ceil(kmin / passo) * passo, kmax, passo)
        if len(strikes) > 45:   # le scadenze lunghe hanno strike più radi
            strikes = strikes[:: int(np.ceil(len(strikes) / 45))]
        for call in (True, False):
            vero_p = prezzo_europeo(F, strikes, T, vero.heston, D, call, vero.salti)
            iv = black.vol_implicite(vero_p, F, strikes, T, D, call)
            iv = np.where(np.isfinite(iv), iv, 0.2)
            vg = black.vega(F, strikes, T, iv, D)
            k = np.log(strikes / F)
            meta_spread = np.maximum(vg * (0.003 + 0.004 * np.abs(k) / np.sqrt(T)), m.tick)
            rumore = rng.normal(0, 0.25, len(strikes)) * meta_spread
            mid = np.maximum(vero_p + rumore, 0.0)
            bid = np.floor((mid - meta_spread) / m.tick) * m.tick
            ask = np.ceil((mid + meta_spread) / m.tick) * m.tick
            bid = np.where(bid < m.tick, 0.0, bid)
            for K, b, a, v in zip(strikes, bid, ask, vero_p):
                righe.append({"T": T, "K": float(K), "call": call, "bid": round(b, 2), "ask": round(a, 2),
                              "vero": float(v), "errore": ""})
    df = pd.DataFrame(righe)
    if errori:
        df = _aggiungi_errori(df, m, rng)
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    return df.sort_values(["T", "call", "K"]).reset_index(drop=True)


def _aggiungi_errori(df: pd.DataFrame, m: Mercato, rng) -> pd.DataFrame:
    df = df.copy()
    F = m.forward(df["T"])
    otm = (df["call"] & (df["K"] >= F)) | (~df["call"] & (df["K"] < F))
    validi = df.index[otm & (df["bid"] > 0) & (df["vero"] > 20 * m.tick)]
    scelte = rng.choice(validi, size=14, replace=False)
    for i, j in enumerate(scelte):
        if i < 8:     # quotazione vecchia: prezzo spostato del 20-40%
            f = rng.choice([-1, 1]) * rng.uniform(0.2, 0.4)
            df.loc[j, ["bid", "ask"]] = (df.loc[j, ["bid", "ask"]] * (1 + f)).round(1).to_numpy()
            df.loc[j, "errore"] = "quotazione vecchia"
        elif i < 11:  # bid e ask incrociati
            b, a = df.loc[j, "bid"], df.loc[j, "ask"]
            df.loc[j, ["bid", "ask"]] = [a + m.tick, b]
            df.loc[j, "errore"] = "bid sopra l'ask"
        else:         # spread anomalo
            df.loc[j, "ask"] = round(df.loc[j, "ask"] * 2.5 + 10 * m.tick, 1)
            df.loc[j, "errore"] = "spread anomalo"
    return df


def carica_csv(percorso, spot: float, data_valutazione: date | None = None) -> pd.DataFrame:
    """Legge una catena di opzioni da CSV.

    Colonne richieste: ``scadenza`` (data AAAA-MM-GG o anni), ``strike``, ``tipo`` (C o P), ``bid``, ``ask``.
    Il formato è quello degli export di fine giornata delle borse e dei principali fornitori di dati.
    """
    df = pd.read_csv(percorso)
    df.columns = [c.strip().lower() for c in df.columns]
    if pd.api.types.is_numeric_dtype(df["scadenza"]):
        T = df["scadenza"].astype(float)
    else:
        d0 = pd.Timestamp(data_valutazione or date.today())
        T = (pd.to_datetime(df["scadenza"]) - d0).dt.days / 365.0
    out = pd.DataFrame({"T": T.round(6), "K": df["strike"].astype(float),
                        "call": df["tipo"].astype(str).str.upper().str.startswith("C"),
                        "bid": df["bid"].astype(float), "ask": df["ask"].astype(float)})
    out["vero"], out["errore"] = np.nan, ""
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out = out[out["T"] > 1 / 365]
    return out.sort_values(["T", "call", "K"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- forward e sconto
def parita_put_call(df: pd.DataFrame, spot: float, banda: float = 0.25) -> pd.DataFrame:
    """Forward e fattore di sconto per scadenza dalla regressione C − P = D·F − D·K sugli strike vicini allo spot.

    È il metodo dei desk: non servono ipotesi su dividendi e tassi, che il mercato incorpora già nei prezzi.
    """
    out = []
    buone = df[(df["bid"] > 0) & (df["ask"] > df["bid"]) & ((df["ask"] - df["bid"]) < 0.3 * df["mid"])]
    for T, g in buone.groupby("T"):
        c = g[g["call"]].set_index("K")["mid"]
        p = g[~g["call"]].set_index("K")["mid"]
        comuni = c.index.intersection(p.index)
        comuni = comuni[np.abs(comuni / spot - 1) <= banda]
        y = (c.loc[comuni] - p.loc[comuni]).to_numpy()
        X = np.column_stack([np.ones(len(comuni)), comuni.to_numpy()])
        # regressione robusta: si tolgono uno alla volta i residui oltre 3 deviazioni assolute mediane
        if len(y) < 2:
            continue
        ok = np.ones(len(y), dtype=bool)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        for _ in range(len(y) - 3):
            res = np.abs(y - X @ beta)
            scala = 1.4826 * np.median(res[ok]) + 1e-4 * spot
            peggiore = np.argmax(np.where(ok, res, -1))
            if res[peggiore] <= 3 * scala:
                break
            ok[peggiore] = False
            beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        D = -beta[1]
        out.append({"T": T, "F": beta[0] / D, "D": D, "punti": int(ok.sum()),
                    "tasso_implicito": -np.log(D) / T, "carry_implicito": np.log(beta[0] / D / spot) / T})
    return pd.DataFrame(out).set_index("T")


# ----------------------------------------------------------------------------- pulizia
def _violazione_convessita(K, C, mezzo) -> float:
    """Somma dei butterfly negativi oltre lo spread: λ·C(K₀) + (1 − λ)·C(K₂) − C(K₁) deve essere ≥ 0."""
    if len(K) < 3:
        return 0.0
    lam = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
    farfalla = lam * C[:-2] + (1 - lam) * C[2:] - C[1:-1]
    tolleranza = lam * mezzo[:-2] + (1 - lam) * mezzo[2:] + mezzo[1:-1]
    return float(np.sum(np.maximum(-(farfalla + tolleranza), 0.0)))


MOTIVI = ("bid nullo", "bid sopra l'ask", "spread troppo largo", "fuori dai limiti di arbitraggio",
          "convessità violata", "fuori dallo smile")


def pulisci(df: pd.DataFrame, fwd: pd.DataFrame, spread_max: float = 0.6, soglia_smile: float = 2.0,
            tick: float = 0.1) -> pd.DataFrame:
    """Pulizia in più passaggi. Restituisce tutte le quotazioni OTM con la colonna ``scarto`` (vuota se valida).

    1. bid nullo; 2. bid sopra l'ask; 3. spread relativo oltre ``spread_max``; 4. prezzo fuori dai limiti di
    arbitraggio; 5. convessità del prezzo rispetto allo strike violata (butterfly negativo); 6. volatilità
    implicita lontana dallo smile stimato, oltre ``soglia_smile`` volte il mezzo spread in volatilità.
    """
    from .svi import calibra_svi

    q = df.join(fwd[["F", "D"]], on="T")
    q = q[(q["call"] & (q["K"] >= q["F"])) | (~q["call"] & (q["K"] < q["F"]))].copy()
    q["k"] = np.log(q["K"] / q["F"])
    q["scarto"] = ""
    q.loc[q["bid"] <= 0, "scarto"] = MOTIVI[0]
    q.loc[(q["scarto"] == "") & (q["bid"] >= q["ask"]), "scarto"] = MOTIVI[1]
    rel = (q["ask"] - q["bid"]) / q["mid"].clip(lower=1e-9)
    q.loc[(q["scarto"] == "") & (rel > spread_max), "scarto"] = MOTIVI[2]
    intr = np.where(q["call"], q["D"] * (q["F"] - q["K"]).clip(lower=0), q["D"] * (q["K"] - q["F"]).clip(lower=0))
    q.loc[(q["scarto"] == "") & (q["mid"] <= intr), "scarto"] = MOTIVI[3]

    # convessità: prezzo della call equivalente (dalla parità) convesso e decrescente nello strike
    q["call_eq"] = np.where(q["call"], q["mid"], q["mid"] + q["D"] * (q["F"] - q["K"]))
    q["mezzo"] = 0.5 * (q["ask"] - q["bid"])
    for _, g in q[q["scarto"] == ""].groupby("T"):
        idx = list(g.sort_values("K").index)
        while len(idx) >= 4:
            eccesso = _violazione_convessita(q.loc[idx, "K"].to_numpy(), q.loc[idx, "call_eq"].to_numpy(),
                                             q.loc[idx, "mezzo"].to_numpy())
            if eccesso <= 0:
                break
            # si toglie il punto la cui eliminazione riduce di più la violazione complessiva
            colonne = [q.loc[idx, c].to_numpy() for c in ("K", "call_eq", "mezzo")]
            prove = [_violazione_convessita(*(np.delete(x, j) for x in colonne)) for j in range(len(idx))]
            j = int(np.argmin(prove))
            q.loc[idx[j], "scarto"] = MOTIVI[4]
            idx.pop(j)

    vol = black.vol_implicite(q["mid"], q["F"], q["K"], q["T"], q["D"], q["call"].to_numpy())
    vb = black.vol_implicite(q["bid"], q["F"], q["K"], q["T"], q["D"], q["call"].to_numpy())
    va = black.vol_implicite(q["ask"], q["F"], q["K"], q["T"], q["D"], q["call"].to_numpy())
    q["vol"], q["vol_bid"], q["vol_ask"] = vol, vb, va
    q.loc[(q["scarto"] == "") & ~np.isfinite(q["vol"]), "scarto"] = MOTIVI[3]
    mezzo = 0.5 * (q["vol_ask"] - q["vol_bid"].fillna(0))
    q["mezzo_spread_vol"] = mezzo.where(np.isfinite(mezzo), np.nan)

    # smile: SVI ripetuto, eliminando ogni volta il punto più lontano con eliminazione dei punti lontani
    for T, g in q[q["scarto"] == ""].groupby("T"):
        idx = g.index
        for _ in range(10):
            ok = idx[q.loc[idx, "scarto"] == ""]
            if len(ok) < 6:
                break
            s = calibra_svi(q.loc[ok, "k"], q.loc[ok, "vol"] ** 2 * T, tentativi=3)
            dist = np.abs(q.loc[ok, "vol"] - s.vol(q.loc[ok, "k"], T))
            tol = np.maximum(soglia_smile * q.loc[ok, "mezzo_spread_vol"].fillna(0.01), 0.006)
            fuori = ok[(dist > tol).to_numpy()]
            if len(fuori) == 0:
                break
            peggiore = fuori[np.argmax((dist / tol).loc[fuori].to_numpy())]
            q.loc[peggiore, "scarto"] = MOTIVI[5]
    q["prezzo"] = q["mid"]
    return q.drop(columns=["call_eq", "mezzo"]).reset_index(drop=True)
