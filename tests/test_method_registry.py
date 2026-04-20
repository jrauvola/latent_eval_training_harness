from latent_harness.training.methods import get_method_recipe, list_method_recipes


def test_method_registry_lists_expected_methods() -> None:
    keys = [recipe.key for recipe in list_method_recipes()]
    assert keys == ["coconut", "codi", "colar", "cot_sft", "no_cot_sft", "sim_cot"]


def test_implemented_methods_today() -> None:
    assert get_method_recipe("codi").implemented is True
    assert get_method_recipe("cot_sft").implemented is True
    assert get_method_recipe("no_cot_sft").implemented is True
    assert get_method_recipe("sim_cot").implemented is True
    assert get_method_recipe("coconut").implemented is False
    assert get_method_recipe("colar").implemented is False


def test_sim_cot_recipe_has_runtime_and_data_builders() -> None:
    recipe = get_method_recipe("sim_cot")
    assert recipe.runtime_builder is not None
    assert recipe.data_module_builder is not None
    # Callable without raising assert_implemented.
    recipe.assert_implemented()
