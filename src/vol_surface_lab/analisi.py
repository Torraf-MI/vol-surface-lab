"""Pipeline completa: dalla catena di opzioni grezza ai prezzi degli esotici sotto modelli diversi."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import black, esotici, heston, mercato, svi, volatilita_locale
from .configurazione import Configurazione


@dataclass
class Analisi:
    cfg: Configurazione
    catena: pd.DataFrame                 # quotazioni grezze
    forward: pd.DataFrame                # forward e sconto dalla parità put-call
    quotazioni: pd.DataFrame             # OTM con la colonna «scarto»
    smile: dict[float, svi.SVI]          # SVI raw per scadenza
    ssvi: svi.SSVI
    calibrazione: heston.Calibrazione
    lv: volatilita_locale.VolatilitaLocale
    errori: pd.DataFrame                 # errore medio per scadenza e modello, in punti di volatilità
    controlli: dict
    struttura: pd.DataFrame              # vol ATM e skew per scadenza
    esotici: pd.DataFrame                # prezzi per prodotto e modello
    esotici_es: pd.DataFrame             # errori standard Monte Carlo
    leggi_skew: dict = field(default_factory=dict)

    @property
    def valide(self) -> pd.DataFrame:
        return self.quotazioni[self.quotazioni["scarto"] == ""]

    @property
    def scadenze(self) -> list[float]:
        return sorted(self.smile)

    def vol_ssvi(self, k, T):
        return self.ssvi.vol(k, T)

    def vol_heston(self, K, T):
        F = float(self.forward.loc[T, "F"]) if T in self.forward.index else float(self.cfg.mercato.forward(T))
        return heston.vol_implicite_modello(F, K, T, self.calibrazione.modello)

    def vol_vera(self, K, T):
        if self.cfg.vero is None:
            return np.full(np.shape(K), np.nan)
        F = float(self.cfg.mercato.forward(T))
        return heston.vol_implicite_modello(F, K, T, self.cfg.vero.heston, self.cfg.vero.salti)

    def sintesi_pulizia(self) -> pd.DataFrame:
        q = self.quotazioni
        t = q.groupby("T").agg(quotazioni=("K", "size"), valide=("scarto", lambda s: int((s == "").sum())))
        t["scartate"] = t["quotazioni"] - t["valide"]
        return t


def _pesi(q: pd.DataFrame) -> np.ndarray:
    """Più peso alle quotazioni strette: 1 / mezzo spread in volatilità, con un minimo di 0,2 punti."""
    hs = q["mezzo_spread_vol"].fillna(0.01).clip(lower=0.002).to_numpy()
    return 1.0 / hs / np.mean(1.0 / hs)


def analizza(cfg: Configurazione, simula: bool = True) -> Analisi:
    m = cfg.mercato
    if cfg.catena_csv:
        catena = mercato.carica_csv(cfg.catena_csv, m.spot, cfg.data_valutazione)
    else:
        catena = mercato.genera_catena(m, cfg.vero, cfg.scadenze, seed=cfg.seed)
    fwd = mercato.parita_put_call(catena, m.spot)
    q = mercato.pulisci(catena, fwd, tick=m.tick)
    v = q[q["scarto"] == ""]

    # 1. SVI raw per scadenza
    smile = {}
    for T, g in v.groupby("T"):
        w = g["vol"] ** 2 * T
        pesi = _pesi(g) / (2 * g["vol"].to_numpy() * T)
        smile[T] = svi.calibra_svi(g["k"], w, pesi / pesi.mean())
    theta = {T: float(s.w(0.0)) for T, s in smile.items()}

    # 2. SSVI su tutta la superficie
    pesi = _pesi(v) / (2 * v["vol"].to_numpy() * v["T"].to_numpy())
    sur = svi.calibra_ssvi(v["k"], v["T"], v["vol"] ** 2 * v["T"], theta, pesi / pesi.mean())

    # 3. Heston
    cal = heston.calibra(v, pesi=np.sqrt(_pesi(v)))

    # 4. volatilità locale
    lv = volatilita_locale.da_ssvi(sur, T_max=max(2.0, max(smile)))

    # errori per scadenza (punti di volatilità, media assoluta)
    righe = []
    for T, g in v.groupby("T"):
        e_svi = smile[T].vol(g["k"], T) - g["vol"]
        e_ssvi = sur.vol(g["k"].to_numpy(), T) - g["vol"]
        e_h = heston.vol_implicite_modello(float(g["F"].iloc[0]), g["K"].to_numpy(), T, cal.modello) - g["vol"]
        dentro = lambda e: float(np.mean(np.abs(e) <= g["mezzo_spread_vol"].fillna(0.01)))  # noqa: E731
        righe.append({"T": T, "SVI per scadenza": 100 * np.mean(np.abs(e_svi)), "SSVI": 100 * np.mean(np.abs(e_ssvi)),
                      "Heston": 100 * np.nanmean(np.abs(e_h)), "dentro_spread_svi": dentro(e_svi),
                      "dentro_spread_ssvi": dentro(e_ssvi), "dentro_spread_heston": dentro(e_h),
                      "mezzo_spread": 100 * float(g["mezzo_spread_vol"].median())})
    errori = pd.DataFrame(righe).set_index("T")

    # controlli di arbitraggio
    kg = np.linspace(-1.5, 1.0, 801)
    butterfly = {T: svi.controllo_butterfly(lambda k, s=s: (s.w(k), s.dw(k), s.d2w(k)), kg) for T, s in smile.items()}
    calendar = svi.controllo_calendar(lambda k, T: smile[T].w(k), list(smile), kg)
    butterfly_ssvi = {T: svi.controllo_butterfly(lambda k, T=T: sur.derivate_k(k, np.full_like(k, T)), kg)
                      for T in smile}
    calendar_ssvi = svi.controllo_calendar(lambda k, T: sur.w(k, np.full_like(k, T)), list(smile), kg)
    controlli = {"butterfly": butterfly, "calendar": calendar, "butterfly_ssvi": butterfly_ssvi,
                 "calendar_ssvi": calendar_ssvi, "condizione_ssvi": sur.condizione_butterfly()}

    # struttura a termine di vol ATM e skew
    righe = []
    for T in sorted(smile):
        F = float(fwd.loc[T, "F"])
        Kh = F * np.exp(np.array([-1e-3, 0.0, 1e-3]))
        vh = heston.vol_implicite_modello(F, Kh, T, cal.modello)
        r = {"T": T, "atm_mercato": float(smile[T].vol(0.0, T)), "skew_mercato": svi.skew_atm(smile[T].vol, T),
             "atm_ssvi": float(sur.vol(0.0, T)), "skew_ssvi": svi.skew_atm(lambda k, t: sur.vol(k, t), T),
             "atm_heston": float(vh[1]), "skew_heston": float((vh[2] - vh[0]) / 2e-3)}
        if cfg.vero is not None:
            Fv = float(m.forward(T))
            vv = heston.vol_implicite_modello(Fv, Fv * np.exp(np.array([-1e-3, 0.0, 1e-3])), T, cfg.vero.heston,
                                              cfg.vero.salti)
            r |= {"atm_vero": float(vv[1]), "skew_vero": float((vv[2] - vv[0]) / 2e-3)}
        righe.append(r)
    struttura = pd.DataFrame(righe).set_index("T")
    leggi = {}
    for col in ("skew_mercato", "skew_heston"):
        y = np.abs(struttura[col].to_numpy())
        b = np.polyfit(np.log(struttura.index.to_numpy()), np.log(y), 1)
        leggi[col] = float(b[0])

    # 5. esotici
    es, es_se = _prezza_esotici(cfg, sur, cal.modello, lv, fwd) if simula else (pd.DataFrame(), pd.DataFrame())
    return Analisi(cfg, catena, fwd, q, smile, sur, cal, lv, errori, controlli, struttura, es, es_se, leggi)


def _prezza_esotici(cfg: Configurazione, sur: svi.SSVI, h: heston.Heston, lv, fwd: pd.DataFrame):
    m = cfg.mercato
    T = max(p.scadenza for p in cfg.prodotti)
    passi = int(round(cfg.passi_anno * T))
    # tasso e carry implicito dalle opzioni alla scadenza più vicina a T
    Tn = min(fwd.index, key=lambda t: abs(t - T))
    r = float(fwd.loc[Tn, "tasso_implicito"])
    qd = r - float(fwd.loc[Tn, "carry_implicito"])
    atm = float(sur.vol(0.0, T))
    modelli = {
        "Black-Scholes (vol ATM)": esotici.simula_black_scholes(m.spot, r, qd, atm, T, passi, cfg.percorsi, cfg.seed),
        "Volatilità locale": volatilita_locale.simula(m.spot, r, qd, lv, T, passi, cfg.percorsi, cfg.seed),
        "Heston": heston.simula(m.spot, r, qd, h, T, passi, cfg.percorsi, cfg.seed),
    }
    if cfg.vero is not None:
        modelli["Bates (modello vero)"] = heston.simula(m.spot, m.tasso, m.dividendi, cfg.vero.heston, T, passi,
                                                       cfg.percorsi, cfg.seed, cfg.vero.salti)
    prezzi, se = {}, {}
    for nome, P in modelli.items():
        prezzi[nome], se[nome] = {}, {}
        for p in cfg.prodotti:
            idx = int(round(p.scadenza / T * passi))
            pr, e = esotici.prezza(p, P[:, : idx + 1], p.scadenza, float(np.exp(-r * p.scadenza)))
            prezzi[nome][p.nome], se[nome][p.nome] = 100 * pr, 100 * e
    df, dse = pd.DataFrame(prezzi), pd.DataFrame(se)
    # prezzo di mercato delle europee dalla superficie (riferimento)
    mk = {}
    for p in cfg.prodotti:
        if p.tipo == "put":
            F = m.spot * np.exp((r - qd) * p.scadenza)
            K = p.strike * m.spot
            vol = float(sur.vol(np.log(K / F), p.scadenza))
            mk[p.nome] = 100 * black.prezzo(F, K, p.scadenza, vol, np.exp(-r * p.scadenza), call=False) / m.spot
        else:
            mk[p.nome] = np.nan
    df.insert(0, "Superficie di mercato", pd.Series(mk))
    return df, dse
