"""Uso come libreria: volatilità implicita, Heston, SVI e superficie completa sul mercato sintetico."""

import numpy as np

from vol_surface_lab import Heston, Salti, analizza, esempio, prezzo_black, prezzo_europeo, vol_implicita

# Black e volatilità implicita
p = prezzo_black(F=5000, K=4500, T=0.5, sigma=0.22, D=0.99, call=False)
iv = vol_implicita(p, 5000, 4500, 0.5, 0.99, call=False)
print(f"Put 4500 a sei mesi con vol 22%: {p:.2f}; vol implicita ritrovata {iv:.4f}")

# Heston e Bates: lo stesso modello con e senza salti
h = Heston(kappa=1.6, theta=0.04, xi=0.6, rho=-0.7, v0=0.025)
K = np.array([4000.0, 4500.0, 5000.0])
print("Put a un mese, Heston:", prezzo_europeo(5000, K, 1 / 12, h, call=False).round(2))
print("Put a un mese, Bates: ", prezzo_europeo(5000, K, 1 / 12, h, call=False, salti=Salti(0.5, -0.1, 0.08)).round(2))

# Tutta la pipeline sul mercato sintetico
cfg = esempio()
cfg.percorsi = 10_000
a = analizza(cfg)
print(a.sintesi_pulizia())
print(a.errori[["SVI per scadenza", "SSVI", "Heston", "mezzo_spread"]].round(2))
print(a.calibrazione.modello)
print(a.esotici.round(2))
