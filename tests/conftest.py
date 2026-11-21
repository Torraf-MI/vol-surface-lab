import pytest

from vol_surface_lab import analizza, esempio


@pytest.fixture(scope="session")
def analisi_demo():
    cfg = esempio()
    cfg.percorsi = 6_000
    return analizza(cfg)
