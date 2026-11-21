import numpy as np
import pytest

from vol_surface_lab import black
from vol_surface_lab.esotici import simula_black_scholes
from vol_surface_lab.svi import (
    SSVI,
    SVI,
    calibra_ssvi,
    calibra_svi,
    controllo_butterfly,
    controllo_calendar,
    densita,
    g,
    skew_atm,
)
from vol_surface_lab.volatilita_locale import da_ssvi
from vol_surface_lab.volatilita_locale import simula as simula_lv


def test_svi_ritrova_i_parametri():
    vero = SVI(a=0.02, b=0.12, rho=-0.6, m=0.05, sigma=0.15)
    k = np.linspace(-0.6, 0.4, 25)
    s = calibra_svi(k, vero.w(k))
    assert np.allclose(s.w(k), vero.w(k), atol=1e-7)
    assert s.rho == pytest.approx(-0.6, abs=1e-3)


def test_g_vale_uno_con_volatilita_piatta():
    k = np.linspace(-1, 1, 11)
    assert np.allclose(g(k, np.full_like(k, 0.04), 0 * k, 0 * k), 1.0)


def test_densita_integra_a_uno():
    s = SVI(a=0.02, b=0.12, rho=-0.6, m=0.05, sigma=0.15)
    k = np.linspace(-4, 3, 20_001)
    dens = densita(k, s.w(k), s.dw(k), s.d2w(k))
    assert np.trapezoid(dens, k) == pytest.approx(1.0, abs=2e-3)
    # e il forward è una martingala: E[e^k] = 1
    assert np.trapezoid(dens * np.exp(k), k) == pytest.approx(1.0, abs=3e-3)


def test_butterfly_individua_una_svi_con_arbitraggio():
    # esempio di Axel Vogt riportato da Gatheral e Jacquier (2014): g(k) negativa
    s = SVI(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    k = np.linspace(-1.5, 1.5, 601)
    c = controllo_butterfly(lambda x: (s.w(x), s.dw(x), s.d2w(x)), k)
    assert c["violazione"]


def test_calendar_individua_varianze_che_si_incrociano():
    s1 = SVI(0.02, 0.2, -0.5, 0.0, 0.1)
    s2 = SVI(0.03, 0.05, -0.5, 0.0, 0.1)   # ali più piatte a scadenza più lunga
    sm = {0.5: s1, 1.0: s2}
    out = controllo_calendar(lambda k, T: sm[T].w(k), [0.5, 1.0], np.linspace(-1, 1, 201))
    assert out[0]["violazione"]


def _ssvi(eta=1.0, rho=-0.6, gamma=0.45):
    T = np.array([0.25, 0.5, 1.0, 2.0])
    return SSVI(rho, eta, gamma, T, 0.04 * T)


def test_ssvi_con_il_vincolo_non_ha_arbitraggio():
    s = _ssvi(eta=2.0 / 1.6 * 0.99, rho=-0.6)
    assert s.condizione_butterfly() <= 2.0
    k = np.linspace(-1.5, 1.0, 501)
    for T in (0.1, 0.5, 1.0, 2.0, 3.0):
        assert not controllo_butterfly(lambda x, T=T: s.derivate_k(x, np.full_like(x, T)), k)["violazione"]
    assert not any(c["violazione"] for c in controllo_calendar(lambda x, T: s.w(x, np.full_like(x, T)),
                                                                 [0.1, 0.5, 1.0, 2.0, 3.0], k))


def test_ssvi_vol_atm_e_skew():
    s = _ssvi()
    assert float(s.vol(0.0, 1.0)) == pytest.approx(0.2, abs=1e-6)
    assert skew_atm(lambda k, t: s.vol(k, t), 1.0) < 0


def test_calibra_ssvi_ritrova_la_superficie():
    s = _ssvi(eta=0.9, rho=-0.55, gamma=0.4)
    T = np.repeat([0.25, 0.5, 1.0, 2.0], 15)
    k = np.tile(np.linspace(-0.4, 0.25, 15), 4)
    stima = calibra_ssvi(k, T, s.w(k, T), {t: 0.04 * t for t in (0.25, 0.5, 1.0, 2.0)})
    assert stima.rho == pytest.approx(-0.55, abs=0.01)
    assert stima.eta == pytest.approx(0.9, abs=0.02)


def test_volatilita_locale_piatta():
    s = _ssvi(eta=1e-6, rho=0.0)
    lv = da_ssvi(s, T_max=2.0)
    assert np.allclose(lv.sigma[5:, 20:100], 0.2, atol=2e-3)


def test_volatilita_locale_riprezza_le_europee():
    s = _ssvi()
    lv = da_ssvi(s, T_max=1.0)
    P = simula_lv(100.0, 0.0, 0.0, lv, 1.0, 100, 40_000, seed=1)
    for K in (80.0, 100.0, 115.0):
        call = K >= 100
        mc = np.maximum((P[:, -1] - K) if call else (K - P[:, -1]), 0).mean()
        an = black.prezzo(100.0, K, 1.0, float(s.vol(np.log(K / 100), 1.0)), 1.0, call)
        assert mc == pytest.approx(an, rel=0.03, abs=0.05)


def test_volatilita_locale_piu_ripida_della_implicita():
    s = _ssvi()
    lv = da_ssvi(s, T_max=1.0)
    k = np.array([-0.05, 0.05])
    pend_imp = np.diff(s.vol(k, 0.5))[0]
    pend_loc = np.diff(lv(k, np.full(2, 0.5)))[0]
    assert pend_loc < 1.5 * pend_imp < 0


def test_monte_carlo_black_scholes_martingala():
    P = simula_black_scholes(100.0, 0.03, 0.01, 0.2, 1.0, 50, 50_000, seed=2)
    assert P[:, -1].mean() == pytest.approx(100 * np.exp(0.02), rel=0.004)
