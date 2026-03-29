"""测试 core/stock.py 通用股票 API"""

import pytest
import pandas as pd
from unittest.mock import patch


class TestCodeConversion:
    """代码格式转换"""

    # -- to_tushare --
    def test_tushare_from_tushare(self):
        from core.stock import to_tushare
        assert to_tushare("000001.SZ") == "000001.SZ"
        assert to_tushare("600000.sh") == "600000.SH"

    def test_tushare_from_qlib(self):
        from core.stock import to_tushare
        assert to_tushare("SZ000001") == "000001.SZ"
        assert to_tushare("SH600000") == "600000.SH"

    def test_tushare_from_internal(self):
        from core.stock import to_tushare
        assert to_tushare("sz000001") == "000001.SZ"
        assert to_tushare("sh600000") == "600000.SH"

    def test_tushare_from_digits(self):
        from core.stock import to_tushare
        assert to_tushare("600000") == "600000.SH"
        assert to_tushare("000001") == "000001.SZ"
        assert to_tushare("300750") == "300750.SZ"

    # -- to_qlib --
    def test_qlib_from_tushare(self):
        from core.stock import to_qlib
        assert to_qlib("000001.SZ") == "SZ000001"
        assert to_qlib("600000.SH") == "SH600000"

    def test_qlib_from_internal(self):
        from core.stock import to_qlib
        assert to_qlib("sz000001") == "SZ000001"

    def test_qlib_from_qlib(self):
        from core.stock import to_qlib
        assert to_qlib("SZ000001") == "SZ000001"

    def test_qlib_from_digits(self):
        from core.stock import to_qlib
        assert to_qlib("600000") == "SH600000"

    # -- to_internal --
    def test_internal_from_tushare(self):
        from core.stock import to_internal
        assert to_internal("000001.SZ") == "sz000001"

    def test_internal_from_qlib(self):
        from core.stock import to_internal
        assert to_internal("SZ000001") == "sz000001"

    def test_internal_from_internal(self):
        from core.stock import to_internal
        assert to_internal("sz000001") == "sz000001"

    # -- 北交所 --
    def test_bj_exchange(self):
        from core.stock import to_tushare, to_qlib, to_internal
        assert to_tushare("BJ430047") == "430047.BJ"
        assert to_qlib("430047.BJ") == "BJ430047"
        assert to_internal("430047.BJ") == "bj430047"
        assert to_tushare("830946") == "830946.BJ"


MOCK_DF = pd.DataFrame({
    "ts_code": ["000001.SZ", "600000.SH", "300750.SZ"],
    "name": ["平安银行", "浦发银行", "宁德时代"],
    "industry": ["银行", "银行", "电池"],
})


class TestStockInfo:
    """名称/行业查询"""

    def _patch_load(self):
        return patch("core.stock._load_stock_basic", return_value=MOCK_DF)

    def test_get_name(self):
        from core.stock import get_name
        with self._patch_load():
            assert get_name("000001.SZ") == "平安银行"
            assert get_name("SZ000001") == "平安银行"
            assert get_name("sz000001") == "平安银行"
            assert get_name("000001") == "平安银行"

    def test_get_name_not_found(self):
        from core.stock import get_name
        with self._patch_load():
            assert get_name("999999.SZ") is None

    def test_get_industry(self):
        from core.stock import get_industry
        with self._patch_load():
            assert get_industry("300750.SZ") == "电池"

    def test_search_by_name(self):
        from core.stock import search
        with self._patch_load():
            results = search("银行")
            assert len(results) == 2

    def test_search_by_code(self):
        from core.stock import search
        with self._patch_load():
            results = search("300750")
            assert len(results) == 1
            assert results[0]["name"] == "宁德时代"

    def test_search_by_industry(self):
        from core.stock import search
        with self._patch_load():
            results = search("电池")
            assert len(results) == 1


class TestDisplay:
    """可读输出"""

    def _patch_load(self):
        return patch("core.stock._load_stock_basic", return_value=MOCK_DF)

    def test_display(self):
        from core.stock import display
        with self._patch_load():
            assert display("SZ000001") == "平安银行(000001.SZ)"

    def test_display_not_found(self):
        from core.stock import display
        with self._patch_load():
            assert display("999999.SZ") == "999999.SZ"

    def test_display_list(self):
        from core.stock import display_list
        with self._patch_load():
            result = display_list(["sz000001", "600000.SH"])
            assert result == ["平安银行(000001.SZ)", "浦发银行(600000.SH)"]


class TestGetAllCodes:
    """全量代码列表"""

    def _patch_load(self):
        return patch("core.stock._load_stock_basic", return_value=MOCK_DF)

    def test_all_tushare(self):
        from core.stock import get_all_codes
        with self._patch_load():
            codes = get_all_codes("tushare")
            assert "000001.SZ" in codes

    def test_all_qlib(self):
        from core.stock import get_all_codes
        with self._patch_load():
            codes = get_all_codes("qlib")
            assert "SZ000001" in codes

    def test_all_internal(self):
        from core.stock import get_all_codes
        with self._patch_load():
            codes = get_all_codes("internal")
            assert "sz000001" in codes
