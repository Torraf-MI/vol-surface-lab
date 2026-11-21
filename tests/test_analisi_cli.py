import json

import numpy as np
import pandas as pd

from vol_surface_lab import black, cli, configurazione


def test_svi_per_scadenza_dentro_lo_spread(analisi_demo):
    e = analisi_demo.errori
    assert (e["SVI per scadenza"] < e["mezzo_spread"]).all()
    assert e["dentro_spread_svi"].mean() > 0.9


def test_heston_fatica_sulla_scadenza_breve(analisi_demo):
    e = analisi_demo.errori["Heston"]
    assert e.iloc[0] > 2 * e.iloc[1:].median()


def test_ssvi_senza_arbitraggio(analisi_demo):
    c = analisi_demo.controlli
    assert c["condizione_ssvi"] <= 2 + 1e-9
    assert not any(b["violazione"] for b in c["butterfly_ssvi"].values())
    assert not any(x["violazione"] for x in c["calendar_ssvi"])


def test_skew_decrescente_con_la_scadenza(analisi_demo):
    s = analisi_demo.struttura
    assert (np.diff(np.abs(s["skew_mercato"])) < 0).all()
    assert -0.8 < analisi_demo.leggi_skew["skew_mercato"] < -0.3


def test_esotici_rischio_di_modello(analisi_demo):
    e = analisi_demo.esotici
    put = "Put europea 90%"
    di = "Put down-and-in 100% / barriera 70%"
    # le europee: volatilità locale e Heston vicine al mercato, Black-Scholes con la vol ATM no
    assert abs(e.loc[put, "Volatilità locale"] / e.loc[put, "Superficie di mercato"] - 1) < 0.06
    assert abs(e.loc[put, "Heston"] / e.loc[put, "Superficie di mercato"] - 1) < 0.06
    assert e.loc[put, "Black-Scholes (vol ATM)"] < 0.85 * e.loc[put, "Superficie di mercato"]
    # la put con barriera: Black-Scholes la sottovaluta di molto
    assert e.loc[di, "Black-Scholes (vol ATM)"] < 0.5 * e.loc[di, "Bates (modello vero)"]


def test_configurazione_json(tmp_path):
    cfg = configurazione.esempio()
    p = configurazione.salva(cfg, tmp_path / "c.json")
    cfg2 = configurazione.carica(p)
    assert cfg2.vero.salti.intensita == cfg.vero.salti.intensita
    assert [x.nome for x in cfg2.prodotti] == [x.nome for x in cfg.prodotti]
    assert json.loads(p.read_text(encoding="utf-8"))["mercato"]["spot"] == 5000.0


def test_cli_demo(tmp_path, capsys):
    out = tmp_path / "r.html"
    assert cli.main(["--demo", "--percorsi", "2000", "--output", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    assert "Superficie di volatilità" in html and "Heston" in html
    assert "Report salvato" in capsys.readouterr().out


def test_cli_csv(tmp_path):
    righe = []
    for T in (0.25, 0.5, 1.0):
        F, D = 100 * np.exp(0.01 * T), np.exp(-0.03 * T)
        for K in np.arange(70.0, 131.0, 5.0):
            k = np.log(K / F)
            vol = 0.2 - 0.25 * k + 0.3 * k * k
            for tipo in ("C", "P"):
                p = black.prezzo(F, K, T, vol, D, tipo == "C")
                righe.append({"scadenza": T, "strike": K, "tipo": tipo, "bid": round(max(p - 0.03, 0), 2),
                              "ask": round(p + 0.03, 2)})
    pd.DataFrame(righe).to_csv(tmp_path / "c.csv", index=False)
    out = tmp_path / "r.html"
    assert cli.main(["--csv", str(tmp_path / "c.csv"), "--spot", "100", "--veloce", "--output", str(out)]) == 0
    assert out.exists()
