import pytest
import numpy as np
import inspect

from formula_finder.variables import (
    ALL_INDICIES,
    ASTRO_CONSTANT_INDICIES,
    ASTRO_CONSTANT_NODES,
    CUSTOM_VARIABLE_NODES,
    CUSTOM_VARIABLES_INDICIES,
    MATH_CONSTANT_INDICIES,
    MATH_CONSTANT_NODES,
    MATH_FUNCTION_INDICIES,
    MATH_FUNCTION_NODES,
    add_x_and_custom_variables_to_ending_indicies,
    convert_array_to_binary_tree,
    get_ending_indicies,
    get_node_from_index,
    update_indicies,
)

# pytest tests for get_node_from_index
def test_get_node_from_index_symbol():
    # add variables m1 and m2 to CUSTOM_VARIABLE_NODES
    CUSTOM_VARIABLE_NODES["m1"] = "m1"
    CUSTOM_VARIABLE_NODES["m2"] = "m2"
    update_indicies()

    # assert CUSTOM_VARIABLES_INDICIES == 1, CUSTOM_VARIABLES_INDICIES
    # assert all_indicies == 1, all_indicies[0]

    # test get_node_from_index with symbol_indicies
    assert get_node_from_index(CUSTOM_VARIABLES_INDICIES[0]) == ("custom_var", "m1")
    assert get_node_from_index(CUSTOM_VARIABLES_INDICIES[1]) == ("custom_var", "m2")

    # test x, a, b and c get returned from get_node_from_index
    assert get_node_from_index(0) == ("symbol", "x")
    assert get_node_from_index(1) == ("symbol", "a")
    assert get_node_from_index(2) == ("symbol", "b")
    assert get_node_from_index(3) == ("symbol", "c")

    # test get_node_from_index with MATH_FUNCTION_INDICIES
    assert get_node_from_index(MATH_FUNCTION_INDICIES[0]) == (
        "math_func",
        MATH_FUNCTION_NODES["add"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[1]) == (
        "math_func",
        MATH_FUNCTION_NODES["divide"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[2]) == (
        "math_func",
        MATH_FUNCTION_NODES["multiply"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[3]) == (
        "math_func",
        MATH_FUNCTION_NODES["subtract"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[4]) == (
        "math_func",
        MATH_FUNCTION_NODES["mod"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[5]) == (
        "math_func",
        MATH_FUNCTION_NODES["exponent"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[6]) == (
        "math_func",
        MATH_FUNCTION_NODES["log"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[7]) == (
        "math_func",
        MATH_FUNCTION_NODES["log10"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[8]) == (
        "math_func",
        MATH_FUNCTION_NODES["sin"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[9]) == (
        "math_func",
        MATH_FUNCTION_NODES["cos"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[10]) == (
        "math_func",
        MATH_FUNCTION_NODES["tan"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[11]) == (
        "math_func",
        MATH_FUNCTION_NODES["asin"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[12]) == (
        "math_func",
        MATH_FUNCTION_NODES["acos"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[13]) == (
        "math_func",
        MATH_FUNCTION_NODES["atan"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[14]) == (
        "math_func",
        MATH_FUNCTION_NODES["sqrt"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[15]) == (
        "math_func",
        MATH_FUNCTION_NODES["abs"],
    )
    assert get_node_from_index(MATH_FUNCTION_INDICIES[16]) == (
        "math_func",
        MATH_FUNCTION_NODES["neg"],
    )

    # test get_node_from_index with MATH_CONSTANT_INDICIES
    assert get_node_from_index(MATH_CONSTANT_INDICIES[0]) == (
        "math_const",
        MATH_CONSTANT_NODES["pi"],
    )
    assert get_node_from_index(MATH_CONSTANT_INDICIES[1]) == (
        "math_const",
        MATH_CONSTANT_NODES["e"],
    )

    # test get_node_from_index with ASTRO_CONSTANT_INDICIES
    assert get_node_from_index(ASTRO_CONSTANT_INDICIES[0]) == (
        "astro_const",
        ASTRO_CONSTANT_NODES["G"],
    )
    assert get_node_from_index(ASTRO_CONSTANT_INDICIES[1]) == (
        "astro_const",
        ASTRO_CONSTANT_NODES["c"],
    )


def test_convert_array_to_binary_tree():
    # tests for convert_array_to_binary_tree function

    # generate example binary tree array
    assert convert_array_to_binary_tree(np.array([4, 1, 3]), 1, 2, 3, 4, [29, 30]) == 6


def test_get_ending_indicies():
    np.testing.assert_array_equal(get_ending_indicies([4, 1, 3]), [1, 2])


# def test_add_x_and_custom_variables_to_ending_indicies():
#     print(
#         add_x_and_custom_variables_to_ending_indicies(
#             np.array([5, 6, 2, 25, 0, 23, 11, 27, 0, 3, 26, 21, 2, 26, 2])
#         )
#     )
#     np.testing.assert_array_equal(
#         add_x_and_custom_variables_to_ending_indicies(
#             np.array([5, 6, 2, 25, 0, 23, 11, 27, 0, 3, 26, 21, 2, 26, 2])
#         ),
#         [5, 6, 30, 0, 29, 23, 11, 27, 0, 3, 26, 21, 2, 26, 2],
#     )
