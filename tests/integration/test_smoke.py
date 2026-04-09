from scripts.smoke import run_smoke


def test_phase1_smoke_runs() -> None:
    run_smoke("configs/default.yaml")
