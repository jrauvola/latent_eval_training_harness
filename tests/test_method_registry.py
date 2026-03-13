from latent_harness.training.methods import get_method_recipe, list_method_recipes


def test_method_registry_lists_expected_methods() -> None:
    keys = [recipe.key for recipe in list_method_recipes()]
    assert keys == ["coconut", "codi", "colar", "cot_sft", "no_cot_sft", "sim_cot"]


def test_only_codi_is_implemented_today() -> None:
    assert get_method_recipe("codi").implemented is True
    assert get_method_recipe("cot_sft").implemented is True
    assert get_method_recipe("no_cot_sft").implemented is True
    assert get_method_recipe("coconut").implemented is False
    assert get_method_recipe("sim_cot").implemented is False
    assert get_method_recipe("colar").implemented is False
