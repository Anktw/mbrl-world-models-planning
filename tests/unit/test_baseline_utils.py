from mbrl.evaluation.baselines import MethodRun, parse_seed_list


def test_parse_seed_list() -> None:
    seeds = parse_seed_list("7, 13,23")
    assert seeds == [7, 13, 23]


def test_parse_seed_list_rejects_empty() -> None:
    try:
        parse_seed_list("   ")
        raise AssertionError("Expected ValueError for empty seed list")
    except ValueError:
        pass


def test_method_run_dataclass_fields() -> None:
    run = MethodRun(
        method="CEM",
        seed=7,
        cumulative_reward=-10.0,
        mean_reward=-3.33,
        reward_per_step=-0.01,
    )
    assert run.method == "CEM"
    assert run.seed == 7
