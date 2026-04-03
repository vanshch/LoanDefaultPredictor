from model_utils import load_test_data, expected_columns
from predict import x_TEST_PATH, Y_TEST_PATH


def test_expected_columns_load_data():
    x_test, _ = load_test_data(x_TEST_PATH, Y_TEST_PATH)
    actual = set(expected_columns)
    expected = set(x_test.columns.to_list())

    assert (
        actual == expected
    ), f"Missing {actual - expected}, and extra {expected-actual}"
