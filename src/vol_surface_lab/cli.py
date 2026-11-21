"""Riga di comando: dalla catena di opzioni alla superficie di volatilità e ai prezzi degli esotici.

Esempi::

    vol-surface-lab --demo
    vol-surface-lab --config esempi/indice_sintetico.json --grafici docs/img
    vol-surface-lab --csv catena.csv --spot 5000 --data 2026-11-20
"""

from __future__ import annotations

import argparse
import sys

from . import configurazione
from .analisi import analizza
from .mercato import etichetta
from .report import _num, build_report, salva_grafici

NOTA_DEMO = ("Caso dimostrativo: il mercato è sintetico, generato da un modello di Bates con spread bid-ask ed errori "
             "di dati realistici. Nessun dato reale di mercato.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vol-surface-lab", description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true", help="Usa il mercato sintetico dimostrativo")
    src.add_argument("--config", help="File JSON con mercato, modello o file CSV, simulazione e prodotti")
    src.add_argument("--csv", help="Catena di opzioni in CSV (scadenza, strike, tipo, bid, ask)")
    p.add_argument("--spot", type=float, help="Livello del sottostante (con --csv)")
    p.add_argument("--data", help="Data di valutazione AAAA-MM-GG (con --csv)")
    p.add_argument("--output", default="output/report.html", help="Percorso del report HTML")
    p.add_argument("--grafici", help="Cartella dove salvare anche i grafici PNG (richiede kaleido)")
    p.add_argument("--offline", action="store_true", help="Include plotly.js nel report (funziona senza rete)")
    p.add_argument("--veloce", action="store_true", help="Salta il Monte Carlo degli esotici")
    p.add_argument("--percorsi", type=int, help="Percorsi Monte Carlo")
    p.add_argument("--seed", type=int, help="Seme del generatore casuale")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.demo:
        cfg = configurazione.esempio()
    elif args.config:
        cfg = configurazione.carica(args.config)
    else:
        if args.spot is None:
            print("--csv richiede --spot", file=sys.stderr)
            return 2
        cfg = configurazione.Configurazione(dimostrativo=False, vero=None, catena_csv=args.csv,
                                            data_valutazione=args.data)
        cfg.mercato.spot = args.spot
        cfg.mercato.nome = "Catena da file"
    if args.percorsi:
        cfg.percorsi = args.percorsi
    if args.seed is not None:
        cfg.seed = args.seed
    a = analizza(cfg, simula=not args.veloce)
    path = build_report(a, args.output, data_note=NOTA_DEMO if cfg.dimostrativo else "", offline=args.offline)
    if args.grafici:
        salva_grafici(a, args.grafici)

    print(f"{cfg.mercato.nome}: {len(a.valide)} quotazioni valide su {len(a.quotazioni)} opzioni OTM")
    print("\nErrore medio per scadenza (punti di volatilità)")
    print(f"  {'scadenza':<10}{'SVI':>7}{'SSVI':>7}{'Heston':>8}{'spread':>8}")
    for T, r in a.errori.iterrows():
        print(f"  {etichetta(T):<10}{_num(r['SVI per scadenza'], 2):>7}{_num(r['SSVI'], 2):>7}"
              f"{_num(r['Heston'], 2):>8}{_num(r['mezzo_spread'], 2):>8}")
    h = a.calibrazione.modello
    print(f"\nHeston: κ={_num(h.kappa, 2)} θ={_num(h.theta, 4)} ξ={_num(h.xi, 2)} ρ={_num(h.rho, 2)} "
          f"v0={_num(h.v0, 4)} (Feller {_num(h.feller(), 2)})")
    viol = [c for c in a.controlli["calendar"] if c["violazione"]]
    print(f"Arbitraggi calendar nelle SVI per scadenza: {len(viol)}; SSVI: η(1+|ρ|) = "
          f"{_num(a.controlli['condizione_ssvi'], 3)} ≤ 2")
    if len(a.esotici):
        print("\nPrezzi degli esotici (% del nozionale)")
        cols = [c for c in a.esotici.columns if c != "Superficie di mercato"]
        print("  " + " " * 38 + "".join(f"{c.split(' (')[0][:14]:>16}" for c in cols))
        for n, r in a.esotici.iterrows():
            print(f"  {n:<38}" + "".join(f"{_num(r[c], 2):>16}" for c in cols))
    print(f"\nReport salvato in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
