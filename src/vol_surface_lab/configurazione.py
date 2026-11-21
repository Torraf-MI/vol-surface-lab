"""Configurazione dell'analisi: mercato, modello che genera i dati sintetici o file CSV, simulazione e prodotti."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .esotici import PRODOTTI, Prodotto
from .heston import Heston, Salti
from .mercato import SCADENZE_STANDARD, Mercato, ModelloVero


@dataclass
class Configurazione:
    mercato: Mercato = field(default_factory=Mercato)
    vero: ModelloVero | None = field(default_factory=ModelloVero)
    catena_csv: str | None = None          # se indicato, i dati vengono dal file e non dal generatore
    data_valutazione: str | None = None
    scadenze: tuple = SCADENZE_STANDARD
    seed: int = 7
    percorsi: int = 40_000
    passi_anno: int = 252
    prodotti: tuple[Prodotto, ...] = PRODOTTI
    dimostrativo: bool = True


def esempio() -> Configurazione:
    """Il caso dimostrativo: indice europeo sintetico a 5.000 punti, mercato generato da un modello di Bates."""
    return Configurazione()


def carica(percorso: str | Path) -> Configurazione:
    d = json.loads(Path(percorso).read_text(encoding="utf-8"))
    cfg = Configurazione(dimostrativo=False)
    if "mercato" in d:
        cfg.mercato = Mercato(**d["mercato"])
    if "catena_csv" in d:
        base = Path(percorso).parent
        p = Path(d["catena_csv"])
        cfg.catena_csv = str(p if p.is_absolute() else base / p)
        cfg.vero = None
        cfg.data_valutazione = d.get("data_valutazione")
    elif "modello_vero" in d:
        mv = d["modello_vero"]
        cfg.vero = ModelloVero(Heston(**mv["heston"]), Salti(**mv.get("salti", {"intensita": 0.0})))
    for chiave in ("seed", "percorsi", "passi_anno"):
        if chiave in d:
            setattr(cfg, chiave, d[chiave])
    if "scadenze" in d:
        cfg.scadenze = tuple(float(x) for x in d["scadenze"])
    if "prodotti" in d:
        cfg.prodotti = tuple(Prodotto(**p) for p in d["prodotti"])
    return cfg


def salva(cfg: Configurazione, percorso: str | Path) -> Path:
    d: dict = {"mercato": cfg.mercato.__dict__, "seed": cfg.seed, "percorsi": cfg.percorsi,
               "passi_anno": cfg.passi_anno, "scadenze": list(cfg.scadenze),
               "prodotti": [p.__dict__ for p in cfg.prodotti]}
    if cfg.catena_csv:
        d["catena_csv"] = cfg.catena_csv
        d["data_valutazione"] = cfg.data_valutazione
    elif cfg.vero is not None:
        d["modello_vero"] = {"heston": cfg.vero.heston.dict(), "salti": cfg.vero.salti.dict()}
    p = Path(percorso)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
