from model import get_moe_layers, is_moe_layer


def test_edge_moe_placement_returns_front_back_layers():
    assert get_moe_layers("edge_moe", 24, front_moe_layers=4, back_moe_layers=6) == (
        list(range(4)) + list(range(18, 24))
    )
    assert is_moe_layer(0, 24, 4, 6)
    assert not is_moe_layer(4, 24, 4, 6)
    assert is_moe_layer(23, 24, 4, 6)


def test_baseline_placements():
    assert get_moe_layers("dense", 6, n_moe_layers=2) == []
    assert get_moe_layers("full_moe", 6) == [0, 1, 2, 3, 4, 5]
    assert get_moe_layers("early_moe", 6, n_moe_layers=2) == [0, 1]
    assert get_moe_layers("late_moe", 6, n_moe_layers=2) == [4, 5]
    assert get_moe_layers("middle_moe", 6, n_moe_layers=2) == [2, 3]
    assert get_moe_layers("edge_moe", 6, n_moe_layers=3) == [0, 4, 5]
    assert get_moe_layers("random_moe", 6, n_moe_layers=3, seed=7) == get_moe_layers(
        "random_moe", 6, n_moe_layers=3, seed=7
    )
