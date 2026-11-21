import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from vol_surface_lab import black
from vol_surface_lab.esotici import Prodotto, prezza, simula_black_scholes
from vol_surface_lab.mercato import Mercato, ModelloVero, carica_csv, genera_catena, parita_put_call, pulisci


@pytest.fixture(scope="module")
def catena():
    m = Mercato()
    return m, genera_catena(m, ModelloVero(), seed=7)


def test_catena_ha_bid_e_ask(catena):
    _, df = catena
    assert {"T", "K", "call", "bid", "ask", "mid"} <= set(df.columns)
    assert (df.loc[df["errore"] == "", "ask"] >= df.loc[df["errore"] == "", "bid"]).all()


def test_parita_ritrova_forward_e_sconto(catena):
    m, df = catena
    f = parita_put_call(df, m.spot)
    T = f.index.to_numpy()
    assert np.allclose(f["F"], m.forward(T), rtol=5e-4)
    assert np.allclose(f["D"], m.sconto(T), atol=2e-3)


def test_pulizia_scarta_gli_errori_evidenti(catena):
    m, df = catena
    q = pulisci(df, parita_put_call(df, m.spot))
    assert (q.loc[q["errore"] == "bid sopra l'ask", "scarto"] != "").all()
    assert (q.loc[q["errore"] == "spread anomalo", "scarto"] != "").all()
    assert (q.loc[q["bid"] <= 0, "scarto"] != "").all()
    vecchie = q[q["errore"] == "quotazione vecchia"]
    assert (vecchie["scarto"] != "").mean() >= 0.5
    # la gran parte delle quotazioni buone resta
    buone = q[(q["errore"] == "") & (q["bid"] > 0)]
    assert (buone["scarto"] == "").mean() > 0.85


def test_pulizia_tiene_solo_le_otm(catena):
    m, df = catena
    q = pulisci(df, parita_put_call(df, m.spot))
    assert ((q["call"] & (q["K"] >= q["F"])) | (~q["call"] & (q["K"] < q["F"]))).all()


def test_csv(tmp_path):
    righe = []
    for T in (0.25, 1.0):
        for K in (90.0, 100.0, 110.0):
            for tipo in ("C", "P"):
                p = black.prezzo(100.0 * np.exp(0.01 * T), K, T, 0.2, np.exp(-0.03 * T), tipo == "C")
                righe.append({"scadenza": T, "strike": K, "tipo": tipo, "bid": round(p - 0.05, 2),
                              "ask": round(p + 0.05, 2)})
    pd.DataFrame(righe).to_csv(tmp_path / "c.csv", index=False)
    df = carica_csv(tmp_path / "c.csv", 100.0)
    assert len(df) == 12 and df["call"].sum() == 6
    f = parita_put_call(df, 100.0)
    assert f.loc[1.0, "D"] == pytest.approx(np.exp(-0.03), abs=2e-3)


def test_barriere_in_e_out_sommano_alla_put():
    P = simula_black_scholes(100.0, 0.0, 0.0, 0.25, 1.0, 252, 20_000, seed=4)
    put = prezza(Prodotto("p", "", "put", 1.0, 1.0), P, 1.0, 1.0)[0]
    di = prezza(Prodotto("di", "", "put_di", 1.0, 1.0, 0.7), P, 1.0, 1.0)[0]
    x = P / P[:, :1]
    do = np.where(x.min(axis=1) > 0.7, np.maximum(1.0 - x[:, -1], 0), 0).mean()
    assert di + do == pytest.approx(put, rel=1e-12)
    assert put == pytest.approx(black.prezzo(1.0, 1.0, 1.0, 0.25, call=False), rel=0.02)


def test_one_touch_formula_chiusa():
    s, T, B, passi = 0.2, 1.0, 0.85, 252
    P = simula_black_scholes(1.0, 0.0, 0.0, s, T, passi, 60_000, seed=5)
    mc = prezza(Prodotto("ot", "", "one_touch", T, 0.0, B), P, T, 1.0)[0]
    # monitoraggio discreto: barriera spostata di exp(−0,5826 σ √dt) (Broadie, Glasserman e Kou)
    Bc = B * np.exp(-0.5826 * s * np.sqrt(T / passi))
    mu = -0.5 * s**2
    a = (np.log(Bc) - mu * T) / (s * np.sqrt(T))
    b = (np.log(Bc) + mu * T) / (s * np.sqrt(T))
    an = norm.cdf(a) + Bc ** (2 * mu / s**2) * norm.cdf(b)
    assert mc == pytest.approx(an, abs=0.01)


def test_forward_start_in_black_scholes():
    P = simula_black_scholes(1.0, 0.0, 0.0, 0.2, 1.0, 252, 40_000, seed=6)
    mc = prezza(Prodotto("fs", "", "forward_start", 1.0, 1.0, 0.0, 0.5), P, 1.0, 1.0)[0]
    assert mc == pytest.approx(black.prezzo(1.0, 1.0, 0.5, 0.2), rel=0.02)
