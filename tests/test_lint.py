import pytest
from mutobj.lint import check


def test_lint() -> None:
    results = check(["mutagent.*"])
    if results:
        pytest.fail(results.format())
