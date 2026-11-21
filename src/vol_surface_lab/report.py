"""Grafici interattivi (plotly) e report HTML della superficie di volatilità."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape
from plotly.offline import get_plotlyjs, get_plotlyjs_version
from plotly.subplots import make_subplots

from .analisi import Analisi
from .mercato import MOTIVI, etichetta

BLU, BLU_SCURO, ACQUA, ARANCIO = "#2a78d6", "#1c5cab", "#1baf7a", "#eb6834"
ROSSO, GIALLO, VIOLA, GRIGIO = "#e34948", "#e8a317", "#8a5cd1", "#8c8b86"
TESTO, TESTO_SECONDARIO, GRIGLIA = "#222222", "#52514e", "#e6e5e1"
COLORI_MODELLI = {"SVI per scadenza": BLU, "SSVI": ACQUA, "Heston": ARANCIO, "Bates (modello vero)": GRIGIO,
                  "Black-Scholes (vol ATM)": GIALLO, "Volatilità locale": VIOLA}
COLORI_SCARTI = dict(zip(MOTIVI, [GRIGIO, ROSSO, GIALLO, VIOLA, ARANCIO, BLU_SCURO]))
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"


def _num(x: float, dec: int = 0) -> str:
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def _pct(x: float, dec: int = 1, segno: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    s = f"{x * 100:+.{dec}f}%" if segno else f"{x * 100:.{dec}f}%"
    return s.replace(".", ",").replace("-", "−")


def _layout(fig: go.Figure, titolo: str, altezza: int = 430) -> go.Figure:
    fig.update_layout(
        title={"text": titolo, "x": 0, "font": {"size": 15, "color": TESTO}},
        height=altezza, template="plotly_white",
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "size": 12, "color": TESTO_SECONDARIO},
        margin={"l": 65, "r": 20, "t": 70, "b": 55}, separators=",.",
        hoverlabel={"bgcolor": "white", "font_color": TESTO}, legend={"orientation": "h", "y": -0.14, "x": 0},
    )
    fig.update_xaxes(gridcolor=GRIGLIA, zeroline=False)
    fig.update_yaxes(gridcolor=GRIGLIA, zerolinecolor="#bdbcb6")
    return fig


# ----------------------------------------------------------------------------- grafici
def grafico_smile(a: Analisi) -> go.Figure:
    """Smile per scadenza: quotazioni bid-ask, SVI, SSVI e Heston."""
    sc = a.scadenze
    ncol = 4
    nrow = int(np.ceil(len(sc) / ncol))
    fig = make_subplots(nrow, ncol, subplot_titles=[etichetta(T) for T in sc], horizontal_spacing=0.05,
                        vertical_spacing=0.14)
    for i, T in enumerate(sc):
        r, c = i // ncol + 1, i % ncol + 1
        g = a.valide[a.valide["T"] == T]
        F = float(g["F"].iloc[0])
        m = g["K"] / F
        fig.add_scatter(x=m, y=100 * g["vol"], mode="markers", marker={"color": TESTO, "size": 5},
                        error_y={"type": "data", "symmetric": False, "array": 100 * (g["vol_ask"] - g["vol"]),
                                 "arrayminus": 100 * (g["vol"] - g["vol_bid"].fillna(g["vol"])), "color": "#b5b4ae",
                                 "thickness": 1, "width": 0},
                        name="Mercato (bid-ask)", legendgroup="m", showlegend=i == 0, row=r, col=c,
                        hovertemplate="K/F %{x:.3f}<br>vol %{y:.2f}%<extra></extra>")
        kk = np.linspace(g["k"].min() - 0.02, g["k"].max() + 0.02, 120)
        linee = {"SVI per scadenza": a.smile[T].vol(kk, T), "SSVI": a.ssvi.vol(kk, T),
                 "Heston": a.vol_heston(F * np.exp(kk), T)}
        for nome, y in linee.items():
            fig.add_scatter(x=np.exp(kk), y=100 * y, mode="lines", line={"color": COLORI_MODELLI[nome], "width": 2,
                            "dash": "dot" if nome == "Heston" else None}, name=nome, legendgroup=nome,
                            showlegend=i == 0, row=r, col=c, hovertemplate=f"{nome}<br>%{{y:.2f}}%<extra></extra>")
    fig.update_yaxes(ticksuffix="%")
    return _layout(fig, "Smile di volatilità implicita per scadenza (asse x: strike / forward)", 330 * nrow + 80)


def grafico_superficie(a: Analisi) -> go.Figure:
    T = np.linspace(min(a.scadenze), max(a.scadenze), 40)
    m = np.linspace(0.6, 1.3, 50)
    TT, MM = np.meshgrid(T, m, indexing="ij")
    Z = 100 * a.ssvi.vol(np.log(MM), TT)
    fig = go.Figure(go.Surface(x=MM, y=TT, z=Z, colorscale=[[0, "#dbe9f8"], [0.5, BLU], [1, "#0f2f5c"]],
                               colorbar={"title": "vol %", "len": 0.7},
                               hovertemplate="K/F %{x:.2f}<br>T %{y:.2f} anni<br>vol %{z:.2f}%<extra></extra>"))
    fig.update_layout(scene={"xaxis_title": "strike / forward", "yaxis_title": "scadenza (anni)",
                             "zaxis_title": "vol implicita %", "camera": {"eye": {"x": -1.6, "y": -1.5, "z": 0.7}}})
    _layout(fig, "Superficie di volatilità implicita SSVI, senza arbitraggio statico", 560)
    fig.update_layout(margin={"l": 10, "r": 10, "t": 60, "b": 10})
    return fig


def grafico_pulizia(a: Analisi) -> go.Figure:
    q = a.quotazioni
    sc = sorted({a.scadenze[0], a.scadenze[len(a.scadenze) // 2], a.scadenze[-1]})
    fig = make_subplots(1, len(sc), subplot_titles=[etichetta(T) for T in sc], horizontal_spacing=0.06)
    for i, T in enumerate(sc):
        g = q[q["T"] == T]
        ok = g[g["scarto"] == ""]
        fig.add_scatter(x=ok["K"] / ok["F"], y=100 * ok["vol"], mode="markers", name="Quotazione valida",
                        marker={"color": "#b5b4ae", "size": 6}, legendgroup="ok", showlegend=i == 0, row=1, col=i + 1)
        for motivo in MOTIVI:
            s = g[g["scarto"] == motivo]
            if len(s) == 0:
                continue
            y = 100 * s["vol"].where(np.isfinite(s["vol"]), np.nan)
            fig.add_scatter(x=s["K"] / s["F"], y=y, mode="markers", name=motivo.capitalize(), legendgroup=motivo,
                            marker={"color": COLORI_SCARTI[motivo], "size": 10, "symbol": "x"}, row=1, col=i + 1,
                            showlegend=True, hovertemplate=f"{motivo}<br>K/F %{{x:.3f}}<br>%{{y:.2f}}%<extra></extra>")
    # una sola voce di legenda per motivo
    visti = set()
    for tr in fig.data:
        if tr.name in visti:
            tr.showlegend = False
        visti.add(tr.name)
    fig.update_yaxes(ticksuffix="%")
    return _layout(fig, "Pulizia dei dati: le quotazioni scartate e la loro volatilità implicita", 420)


def grafico_errori(a: Analisi) -> go.Figure:
    e = a.errori
    x = [etichetta(T) for T in e.index]
    fig = go.Figure()
    for nome in ("SVI per scadenza", "SSVI", "Heston"):
        fig.add_bar(x=x, y=e[nome], name=nome, marker_color=COLORI_MODELLI[nome],
                    hovertemplate=f"{nome}<br>%{{y:.2f}} punti<extra></extra>")
    fig.add_scatter(x=x, y=e["mezzo_spread"], mode="lines+markers", name="Mezzo spread bid-ask (mediana)",
                    line={"color": TESTO, "dash": "dash"})
    fig.update_yaxes(title="errore medio assoluto (punti di vol)")
    return _layout(fig, "Quanto bene ogni modello riproduce il mercato", 420)


def grafico_struttura(a: Analisi) -> go.Figure:
    s = a.struttura
    T = s.index.to_numpy()
    fig = make_subplots(1, 2, subplot_titles=["Volatilità ATM", "Skew ATM |∂σ/∂k| (scala logaritmica)"],
                        horizontal_spacing=0.1)
    serie = [("mercato", "Mercato (SVI)", BLU), ("heston", "Heston", ARANCIO)]
    if "atm_vero" in s:
        serie.append(("vero", "Bates (modello vero)", GRIGIO))
    for suf, nome, col in serie:
        fig.add_scatter(x=T, y=100 * s[f"atm_{suf}"], mode="lines+markers", name=nome, line={"color": col},
                        legendgroup=nome, row=1, col=1)
        fig.add_scatter(x=T, y=np.abs(s[f"skew_{suf}"]), mode="lines+markers", name=nome, line={"color": col},
                        legendgroup=nome, showlegend=False, row=1, col=2)
    b = a.leggi_skew["skew_mercato"]
    c = np.exp(np.polyfit(np.log(T), np.log(np.abs(s["skew_mercato"])), 1)[1])
    tt = np.linspace(T.min(), T.max(), 60)
    fig.add_scatter(x=tt, y=c * tt**b, mode="lines", line={"color": BLU, "dash": "dot"},
                    name=f"Legge di potenza T^{b:.2f}".replace(".", ","), row=1, col=2)
    fig.update_xaxes(title="scadenza (anni)", type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(title="scadenza (anni)", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    return _layout(fig, "Struttura a termine: livello e pendenza dello smile", 420)


def grafico_locale(a: Analisi) -> go.Figure:
    fig = go.Figure()
    for T, col in ((0.25, ACQUA), (1.0, BLU)):
        k = np.linspace(-0.35, 0.2, 100)
        fig.add_scatter(x=np.exp(k), y=100 * a.ssvi.vol(k, T), mode="lines", line={"color": col},
                        name=f"Implicita {etichetta(T)}")
        fig.add_scatter(x=np.exp(k), y=100 * a.lv(k, np.full_like(k, T)), mode="lines",
                        line={"color": col, "dash": "dash"}, name=f"Locale {etichetta(T)}")
    fig.update_xaxes(title="strike / forward")
    fig.update_yaxes(ticksuffix="%")
    return _layout(fig, "Volatilità locale di Dupire e volatilità implicita", 430)


def grafico_calendar(a: Analisi) -> go.Figure | None:
    c = min(a.controlli["calendar"], key=lambda c: c["diff_min"])   # la violazione peggiore
    k = np.linspace(-1.5, 1.0, 400)
    fig = go.Figure()
    for T, col in ((c["T1"], ACQUA), (c["T2"], BLU)):
        fig.add_scatter(x=np.exp(k), y=a.smile[T].w(k), mode="lines", line={"color": col},
                        name=f"SVI per scadenza, {etichetta(T)}")
        fig.add_scatter(x=np.exp(k), y=a.ssvi.w(k, np.full_like(k, T)), mode="lines",
                        line={"color": col, "dash": "dot"}, name=f"SSVI, {etichetta(T)}")
    zona = k[a.smile[c["T2"]].w(k) < a.smile[c["T1"]].w(k)]
    if len(zona):
        fig.add_vrect(x0=float(np.exp(zona.min())), x1=float(np.exp(zona.max())), fillcolor=ROSSO, opacity=0.1,
                      line_width=0, annotation_text="arbitraggio calendar", annotation_position="top left")
    fig.update_xaxes(title="strike / forward")
    fig.update_yaxes(title="varianza totale w = σ²T")
    return _layout(fig, "Arbitraggio calendar: la varianza totale non può scendere con la scadenza", 420)


def grafico_densita(a: Analisi) -> go.Figure:
    from .svi import densita

    fig = go.Figure()
    k = np.linspace(-0.8, 0.5, 400)
    palette = [ACQUA, BLU, VIOLA, ARANCIO]
    sc = a.scadenze
    scelte = sorted({sc[int(round(i))] for i in np.linspace(0, len(sc) - 1, min(4, len(sc)))})
    for T, col in zip(scelte, palette):
        w, dw, d2w = a.ssvi.derivate_k(k, np.full_like(k, T))
        fig.add_scatter(x=np.exp(k), y=densita(k, w, dw, d2w), mode="lines", line={"color": col},
                        name=etichetta(T), fill="tozeroy", opacity=0.5)
    fig.update_xaxes(title="livello dell'indice a scadenza / forward")
    fig.update_yaxes(title="densità neutrale al rischio")
    return _layout(fig, "Le distribuzioni neutrali al rischio implicite nella superficie (coda sinistra spessa)", 420)


def grafico_esotici(a: Analisi) -> go.Figure:
    e, se = a.esotici, a.esotici_es
    rif = "Bates (modello vero)" if "Bates (modello vero)" in e else "Heston"
    fig = go.Figure()
    for nome in [c for c in e.columns if c not in ("Superficie di mercato", rif)]:
        d = (e[nome] / e[rif] - 1) * 100
        err = se[nome] / e[rif] * 100
        fig.add_bar(x=e.index, y=d, name=nome, marker_color=COLORI_MODELLI.get(nome, GRIGIO),
                    error_y={"type": "data", "array": 2 * err, "color": "#8c8b86", "thickness": 1},
                    hovertemplate=f"{nome}<br>%{{y:+.1f}}%<extra></extra>")
    breve = rif.split(" (")[0]
    fig.update_yaxes(title=f"differenza rispetto a {breve}", ticksuffix="%")
    return _layout(fig, "Rischio di modello: stesse europee, prezzi diversi per gli esotici", 440)


def _grafici(a: Analisi) -> dict[str, go.Figure]:
    g = {"smile": grafico_smile(a), "superficie": grafico_superficie(a), "pulizia": grafico_pulizia(a),
         "errori": grafico_errori(a), "struttura": grafico_struttura(a), "locale": grafico_locale(a),
         "calendar": grafico_calendar(a), "densita": grafico_densita(a)}
    if len(a.esotici):
        g["esotici"] = grafico_esotici(a)
    return g


def salva_grafici(a: Analisi, cartella: str | Path) -> list[Path]:
    cartella = Path(cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    out = []
    for nome, fig in _grafici(a).items():
        p = cartella / f"{nome}.png"
        fig.write_image(p, width=1100, height=fig.layout.height or 450, scale=1.5)
        out.append(p)
    return out


# ----------------------------------------------------------------------------- contesto
def contesto(a: Analisi) -> dict:
    m = a.cfg.mercato
    pul = a.sintesi_pulizia()
    scarti = a.quotazioni["scarto"].value_counts()
    h = a.calibrazione.modello
    e = a.errori
    viol_cal = [c for c in a.controlli["calendar"] if c["violazione"]]
    viol_bf = [T for T, b in a.controlli["butterfly"].items() if b["violazione"]]
    ctx = {
        "nome": m.nome, "spot": _num(m.spot), "oggi": date.today().strftime("%d/%m/%Y"),
        "n_quot": _num(len(a.catena)), "n_otm": _num(len(a.quotazioni)), "n_valide": _num(len(a.valide)),
        "n_scadenze": len(a.scadenze),
        "pulizia": [{"T": etichetta(T), "quot": int(r["quotazioni"]), "valide": int(r["valide"]),
                     "scartate": int(r["scartate"])} for T, r in pul.iterrows()],
        "scarti": [{"motivo": k.capitalize(), "n": int(v)} for k, v in scarti.items() if k],
        "forward": [{"T": etichetta(T), "F": _num(r["F"], 1), "D": _num(r["D"], 4),
                     "r": _pct(r["tasso_implicito"], 2), "carry": _pct(r["carry_implicito"], 2),
                     "punti": int(r["punti"])} for T, r in a.forward.iterrows()],
        "svi": [{"T": etichetta(T), **{k: _num(v, 4) for k, v in s.dict().items()},
                 "err": _num(e.loc[T, "SVI per scadenza"], 2), "g": _num(a.controlli["butterfly"][T]["g_min"], 3)}
                for T, s in a.smile.items()],
        "ssvi": {k: _num(float(v), 3) for k, v in a.ssvi.dict().items()},
        "cond_ssvi": _num(a.controlli["condizione_ssvi"], 3),
        "viol_cal": [{"T1": etichetta(c["T1"]), "T2": etichetta(c["T2"]), "k": _num(float(np.exp(c["k_min"])), 2),
                      "diff": _num(c["diff_min"], 4)} for c in viol_cal],
        "viol_bf": [etichetta(T) for T in viol_bf],
        "heston": {"kappa": _num(h.kappa, 2), "theta": _num(h.theta, 4), "vol_lungo": _pct(np.sqrt(h.theta), 1),
                   "xi": _num(h.xi, 2), "rho": _num(h.rho, 2), "v0": _num(h.v0, 4), "vol0": _pct(np.sqrt(h.v0), 1),
                   "feller": _num(h.feller(), 2), "rmse": _num(100 * a.calibrazione.rmse_vol, 2)},
        "errori": [{"T": etichetta(T), "svi": _num(r["SVI per scadenza"], 2), "ssvi": _num(r["SSVI"], 2),
                    "heston": _num(r["Heston"], 2), "spread": _num(r["mezzo_spread"], 2),
                    "d_svi": _pct(r["dentro_spread_svi"], 0), "d_ssvi": _pct(r["dentro_spread_ssvi"], 0),
                    "d_heston": _pct(r["dentro_spread_heston"], 0)} for T, r in e.iterrows()],
        "err_medio": {c: _num(float(e[c].mean()), 2) for c in ("SVI per scadenza", "SSVI", "Heston")},
        "struttura": [{"T": etichetta(T), **{c: (_pct(v, 2) if c.startswith("atm") else _num(v, 3))
                                             for c, v in r.items()}} for T, r in a.struttura.iterrows()],
        "ha_vero": a.cfg.vero is not None,
        "esp_skew": _num(a.leggi_skew["skew_mercato"], 2), "esp_skew_h": _num(a.leggi_skew["skew_heston"], 2),
        "dimostrativo": a.cfg.dimostrativo,
    }
    if a.cfg.vero is not None:
        v = a.cfg.vero
        ctx["vero"] = {"kappa": _num(v.heston.kappa, 2), "theta": _num(v.heston.theta, 3), "xi": _num(v.heston.xi, 2),
                       "rho": _num(v.heston.rho, 2), "v0": _num(v.heston.v0, 3),
                       "lam": _num(v.salti.intensita, 2), "mu": _pct(v.salti.media, 0), "dev": _pct(v.salti.dev, 0)}
    if len(a.esotici):
        cols = list(a.esotici.columns)
        ctx["esotici_cols"] = cols
        ctx["esotici"] = [{"nome": n, "valori": [("–" if not np.isfinite(r[c]) else _num(r[c], 2) + "%") for c in cols],
                           "descrizione": next(p.descrizione for p in a.cfg.prodotti if p.nome == n)}
                          for n, r in a.esotici.iterrows()]
        ctx["percorsi"] = _num(a.cfg.percorsi)
    kpi = [("Quotazioni valide", f"{len(a.valide)} su {len(a.quotazioni)}", "opzioni OTM dopo la pulizia"),
           ("Errore SVI per scadenza", f"{ctx['err_medio']['SVI per scadenza']} punti", "di volatilità, in media"),
           ("Errore Heston", f"{ctx['err_medio']['Heston']} punti", "cinque parametri per tutta la superficie"),
           ("Skew ATM", f"∝ T^{ctx['esp_skew']}", "pendenza dello smile per scadenza")]
    if len(a.esotici):
        e2 = a.esotici
        nome = "Put down-and-in 100% / barriera 70%"
        if nome in e2.index:
            r = e2.loc[nome]
            kpi.append(("Put con barriera 70%", f"{_num(r['Black-Scholes (vol ATM)'], 2)}% → {_num(r['Heston'], 2)}%",
                        "Black-Scholes contro Heston"))
    ctx["kpi"] = kpi
    return ctx


def build_report(a: Analisi, output: str | Path, data_note: str = "", offline: bool = False) -> Path:
    env = Environment(loader=PackageLoader("vol_surface_lab", "templates"), autoescape=select_autoescape())
    g = {k: fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})
         for k, fig in _grafici(a).items()}
    html = env.get_template("report.html.j2").render(
        **contesto(a), grafici=g, data_note=data_note, plotly_cdn=PLOTLY_CDN,
        plotly_inline=get_plotlyjs() if offline else None)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
