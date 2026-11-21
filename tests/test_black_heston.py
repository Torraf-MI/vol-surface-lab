import numpy as np
import pytest

from vol_surface_lab import black
from vol_surface_lab.heston import Heston, Salti, calibra, funzione_caratteristica, prezzo_europeo, simula


def test_parita_put_call_black():
    F, K, T, s, D = 100.0, np.array([80.0, 100.0, 125.0]), 0.7, 0.23, 0.97
    c = black.prezzo(F, K, T, s, D, call=True)
    p = black.prezzo(F, K, T, s, D, call=False)
    assert np.allclose(c - p, D * (F - K))


@pytest.mark.parametrize("K,call", [(70.0, False), (100.0, True), (140.0, True)])
def test_vol_implicita_inverte_il_prezzo(K, call):
    p = black.prezzo(100.0, K, 0.5, 0.31, 0.99, call)
    assert black.vol_implicita(p, 100.0, K, 0.5, 0.99, call) == pytest.approx(0.31, abs=1e-8)


def test_vol_implicita_fuori_dai_limiti():
    assert np.isnan(black.vol_implicita(0.5, 100.0, 50.0, 1.0, 1.0, True))   # sotto l'intrinseco
    assert np.isnan(black.vol_implicita(101.0, 100.0, 50.0, 1.0, 1.0, True))  # sopra il forward


def test_vega_differenze_finite():
    h = 1e-5
    num = (black.prezzo(100, 95, 1.0, 0.2 + h) - black.prezzo(100, 95, 1.0, 0.2 - h)) / (2 * h)
    assert black.vega(100, 95, 1.0, 0.2) == pytest.approx(num, rel=1e-6)


def test_funzione_caratteristica_martingala():
    h = Heston(2.0, 0.05, 0.8, -0.7, 0.03)
    assert funzione_caratteristica(-1j, 2.0, h) == pytest.approx(1.0)
    assert funzione_caratteristica(-1j, 2.0, h, Salti(0.5, -0.1, 0.1)) == pytest.approx(1.0)
    assert funzione_caratteristica(0.0, 2.0, h) == pytest.approx(1.0)


def test_heston_senza_vol_della_varianza_e_black():
    h = Heston(kappa=1.0, theta=0.04, xi=1e-4, rho=0.0, v0=0.04)
    K = np.array([70.0, 90.0, 100.0, 115.0, 140.0])
    assert np.allclose(prezzo_europeo(100.0, K, 1.0, h), black.prezzo(100.0, K, 1.0, 0.2), atol=1e-5)


def test_bates_senza_salti_e_heston():
    h = Heston(1.5, 0.04, 0.6, -0.7, 0.03)
    K = np.array([80.0, 100.0, 120.0])
    assert np.allclose(prezzo_europeo(100, K, 1.0, h, salti=Salti(0.0)), prezzo_europeo(100, K, 1.0, h))


def test_parita_put_call_heston():
    h = Heston(1.5, 0.04, 0.6, -0.7, 0.03)
    K = np.array([80.0, 100.0, 120.0])
    c = prezzo_europeo(100, K, 0.8, h, 0.98, True)
    p = prezzo_europeo(100, K, 0.8, h, 0.98, False)
    assert np.allclose(c - p, 0.98 * (100 - K))


def test_salti_negativi_aumentano_le_put_lontane():
    h = Heston(1.5, 0.04, 0.6, -0.7, 0.03)
    senza = prezzo_europeo(100, 75.0, 0.1, h, call=False)[0]
    con = prezzo_europeo(100, 75.0, 0.1, h, call=False, salti=Salti(0.5, -0.12, 0.08))[0]
    assert con > 1.5 * senza


@pytest.mark.parametrize("salti", [None, Salti(0.4, -0.1, 0.08)])
def test_monte_carlo_coerente_con_la_formula(salti):
    h = Heston(1.5, 0.04, 0.6, -0.7, 0.03)
    P = simula(100.0, 0.0, 0.0, h, 1.0, 200, 40_000, seed=3, salti=salti)
    K = np.array([85.0, 100.0, 115.0])
    mc = np.maximum(P[:, -1][:, None] - K, 0).mean(axis=0)
    an = prezzo_europeo(100.0, K, 1.0, h, salti=salti)
    assert np.allclose(mc, an, rtol=0.03, atol=0.05)
    assert P[:, -1].mean() == pytest.approx(100.0, rel=0.005)


def test_calibrazione_ritrova_i_parametri():
    import pandas as pd

    vero = Heston(2.0, 0.05, 0.7, -0.65, 0.03)
    righe = []
    for T in (0.25, 0.5, 1.0, 2.0):
        K = 100 * np.exp(np.linspace(-0.35, 0.2, 9) * np.sqrt(T))
        call = K >= 100
        p = prezzo_europeo(100.0, K, T, vero, 1.0, call)
        for k, c, pp in zip(K, call, p):
            righe.append({"T": T, "K": k, "F": 100.0, "D": 1.0, "call": c, "prezzo": pp,
                          "vol": black.vol_implicita(pp, 100.0, k, T, 1.0, c)})
    cal = calibra(pd.DataFrame(righe))
    assert cal.rmse_vol < 1e-4
    assert cal.modello.rho == pytest.approx(-0.65, abs=0.02)
    assert cal.modello.v0 == pytest.approx(0.03, abs=0.002)
    assert cal.modello.xi == pytest.approx(0.7, abs=0.05)
