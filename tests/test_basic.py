"""Basic tests - verify package can be imported."""

import mutagent
import mutobj


def test_import_mutagent():
    assert hasattr(mutagent, "__version__")


def test_version():
    assert mutagent.__version__  # 只验证版本号存在，不硬编码具体值
