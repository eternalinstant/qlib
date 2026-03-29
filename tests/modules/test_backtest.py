"""
回测引擎模块测试
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.backtest.base import BacktestEngine, BacktestResult
from modules.backtest.qlib_engine import (
    _raw_data_root,
    _raw_data_path_for_instrument,
    _round_limit_price,
    _get_price_limit_pct,
    _get_limit_prices,
    _can_buy_at_open,
    _can_sell_at_open,
    _ensure_tradability_constraints_supported,
    _collect_required_instruments,
    _quote_row,
    _sum_symbol_returns,
    _compute_rebalance_day,
    _load_ranked_selection_orders,
    _load_raw_trade_quotes,
    main as qlib_main,
    QlibBacktestEngine,
)
from config.config import CONFIG


class MockBacktestEngine(BacktestEngine):
    """用于测试的模拟回测引擎"""
    
    def run(self, strategy=None):
        return BacktestResult(
            daily_returns=pd.Series([0.01, 0.02, -0.01]),
            portfolio_value=pd.Series([1.0, 1.01, 1.0]),
        )


class TestBacktestResult:
    """测试 BacktestResult 数据类"""

    @pytest.fixture
    def sample_result(self):
        dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01] * 2, index=dates)
        value = (1 + returns).cumprod()
        return BacktestResult(
            daily_returns=returns,
            portfolio_value=value,
        )

    def test_total_return(self, sample_result):
        assert abs(sample_result.total_return - (sample_result.portfolio_value.iloc[-1] - 1)) < 1e-10

    def test_annual_return(self, sample_result):
        annual = sample_result.annual_return
        assert annual != 0  # 应该有年化收益

    def test_sharpe_ratio(self, sample_result):
        sharpe = sample_result.sharpe_ratio
        assert isinstance(sharpe, float)

    def test_max_drawdown(self, sample_result):
        dd = sample_result.max_drawdown
        assert dd <= 0  # 回撤应为负数

    def test_print_summary(self, sample_result, capsys):
        sample_result.print_summary(100000)
        captured = capsys.readouterr()
        assert "资金损益" in captured.out or "初始资金" in captured.out

    def test_empty_result(self):
        result = BacktestResult(
            daily_returns=pd.Series(dtype=float),
            portfolio_value=pd.Series(dtype=float),
        )
        assert result.total_return == 0.0
        assert result.annual_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown == 0.0

    def test_negative_terminal_value_clamps_annual_return(self, capsys):
        dates = pd.date_range("2024-01-01", "2024-01-05", freq="D")
        returns = pd.Series([0.0, -0.5, -0.8, 0.0, 0.0], index=dates)
        portfolio_value = pd.Series([1.0, 0.5, -0.1, -0.1, -0.1], index=dates)
        result = BacktestResult(daily_returns=returns, portfolio_value=portfolio_value)

        assert result.total_return < -1.0
        assert result.annual_return == -1.0

        result.print_summary(100000)
        captured = capsys.readouterr()
        assert "-100.00%" in captured.out


class TestBacktestEngine:
    """测试 BacktestEngine 基类"""

    def test_prepare_without_strategy(self):
        """测试无策略时的 _prepare 方法"""
        engine = MockBacktestEngine()
        
        # 模拟 load_selections
        with patch('modules.backtest.base.load_selections') as mock_load:
            with patch('modules.backtest.base.MarketPositionController') as mock_controller:
                mock_load.return_value = (
                    {pd.Timestamp("2024-01-01"): {"SZ000001"}},
                    {pd.Timestamp("2024-01-01")}
                )
                mock_controller_instance = Mock()
                mock_controller_instance.load_market_data = Mock()
                mock_controller.return_value = mock_controller_instance
                
                result = engine._prepare(strategy=None)
                
        # 验证返回结果
        assert len(result) == 4  # date_to_symbols, rebalance_dates, controller, topk
        date_to_symbols, rebalance_dates, controller, topk = result
        assert isinstance(date_to_symbols, dict)
        assert isinstance(rebalance_dates, set)
        assert controller is not None
        assert topk == CONFIG.get("topk", 20)

    def test_prepare_with_strategy(self):
        """测试有策略时的 _prepare 方法"""
        engine = MockBacktestEngine()
        
        # 创建模拟策略
        mock_strategy = Mock()
        mock_strategy.load_selections.return_value = (
            {pd.Timestamp("2024-01-01"): {"SZ000001"}},
            {pd.Timestamp("2024-01-01")}
        )
        mock_strategy.build_position_controller.return_value = Mock()
        mock_strategy.topk = 30
        
        with patch('modules.backtest.base.load_selections') as mock_load:
            result = engine._prepare(strategy=mock_strategy)
        
        # 验证策略的方法被调用
        mock_strategy.load_selections.assert_called_once()
        mock_strategy.build_position_controller.assert_called_once()
        
        # 验证返回值
        date_to_symbols, rebalance_dates, controller, topk = result
        assert topk == 30  # 使用策略指定的 topk

    def test_prepare_controller_load_market_data(self):
        """测试仓位控制器加载市场数据"""
        engine = MockBacktestEngine()
        
        with patch('modules.backtest.base.load_selections') as mock_load:
            with patch('modules.backtest.base.MarketPositionController') as mock_ctrl_class:
                mock_load.return_value = (
                    {pd.Timestamp("2024-01-01"): {"SZ000001"}},
                    {pd.Timestamp("2024-01-01")}
                )
                
                mock_controller = Mock()
                mock_controller.load_market_data = Mock()
                mock_ctrl_class.return_value = mock_controller
                
                engine._prepare(strategy=None)
                
        # 验证 load_market_data 被调用
        mock_controller.load_market_data.assert_called_once()

    def test_prepare_controller_no_market_data(self):
        """测试无 load_market_data 方法的控制器"""
        engine = MockBacktestEngine()
        
        with patch('modules.backtest.base.load_selections') as mock_load:
            with patch('modules.backtest.base.MarketPositionController') as mock_ctrl_class:
                mock_load.return_value = (
                    {pd.Timestamp("2024-01-01"): {"SZ000001"}},
                    {pd.Timestamp("2024-01-01")}
                )
                
                # 创建一个没有 load_market_data 方法的控制器
                mock_controller = Mock(spec=[])  # 空spec
                mock_ctrl_class.return_value = mock_controller
                
                # 不应抛出异常
                engine._prepare(strategy=None)


class TestQlibBacktestHelpers:
    """测试 Qlib 回测辅助逻辑"""

    def test_collect_required_instruments(self):
        date_to_symbols = {
            pd.Timestamp("2024-01-02"): {"SZ000002", "SZ000001"},
            pd.Timestamp("2024-01-03"): {"SZ000003", "SZ000002"},
        }

        assert _collect_required_instruments(date_to_symbols) == [
            "SZ000001",
            "SZ000002",
            "SZ000003",
        ]

    def test_raw_data_helpers(self):
        fake_config = Mock()
        fake_config.get.return_value = "~/repo/data/qlib_data/cn_data"

        with patch("modules.backtest.qlib_engine.CONFIG", fake_config):
            root = _raw_data_root()

        assert str(root).endswith("data/qlib_data/raw_data")
        assert _raw_data_path_for_instrument("SZ000001").name == "sz000001.parquet"
        assert pd.isna(_round_limit_price(np.nan))

    def test_load_ranked_selection_orders(self, tmp_path):
        csv_path = tmp_path / "selection.csv"
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
                "rank": [1, 2, 1],
                "symbol": ["SZ000003", "SZ000002", "SZ000001"],
            }
        ).to_csv(csv_path, index=False)

        strategy = Mock()
        strategy.selections_path.return_value = csv_path

        result = _load_ranked_selection_orders(strategy=strategy)

        assert result[pd.Timestamp("2024-01-02")] == ["SZ000001", "SZ000002"]
        assert result[pd.Timestamp("2024-01-03")] == ["SZ000003"]

    def test_load_raw_trade_quotes_builds_prev_close(self, tmp_path):
        raw_file = tmp_path / "sz000001.parquet"
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "open": [10.0, 10.2, 10.3],
                "close": [10.1, 10.4, 10.5],
            }
        ).to_parquet(raw_file, index=False)

        with patch("modules.backtest.qlib_engine._raw_data_root", return_value=tmp_path):
            quotes = _load_raw_trade_quotes(["SZ000001"], "2024-01-02", "2024-01-03")

        assert list(quotes.index.get_level_values("instrument").unique()) == ["SZ000001"]
        assert quotes.loc[(pd.Timestamp("2024-01-02"), "SZ000001"), "prev_close"] == 10.1
        assert quotes.loc[(pd.Timestamp("2024-01-03"), "SZ000001"), "prev_close"] == 10.4

    def test_load_raw_trade_quotes_warns_and_skips_missing_files(self, tmp_path, capsys):
        with patch("modules.backtest.qlib_engine._raw_data_root", return_value=tmp_path):
            quotes = _load_raw_trade_quotes(["SZ000001"], "2024-01-02", "2024-01-03")

        captured = capsys.readouterr()
        assert quotes.empty
        assert "raw_data 缺少 1 个标的文件" in captured.out

    def test_price_limit_pct(self):
        assert _get_price_limit_pct("SH600000", "2024-01-01") == 0.10
        assert _get_price_limit_pct("SZ300001", "2020-08-21") == 0.10
        assert _get_price_limit_pct("SZ300001", "2020-08-24") == 0.20
        assert _get_price_limit_pct("SH688001", "2024-01-01") == 0.20
        assert _get_price_limit_pct("BJ430001", "2024-01-01") == 0.30
        assert _get_price_limit_pct("SH600000", "2024-01-01", is_st=True) == 0.05

    def test_open_limit_checks(self):
        up_limit, down_limit = _get_limit_prices("SH600000", "2024-01-01", 10.00)

        assert up_limit == 11.00
        assert down_limit == 9.00
        assert _can_buy_at_open("SH600000", "2024-01-01", 10.99, 10.00) is True
        assert _can_buy_at_open("SH600000", "2024-01-01", 11.00, 10.00) is False
        assert _can_sell_at_open("SH600000", "2024-01-01", 9.01, 10.00) is True
        assert _can_sell_at_open("SH600000", "2024-01-01", 9.00, 10.00) is False

    def test_limit_helpers_handle_invalid_inputs(self):
        up_limit, down_limit = _get_limit_prices("SH600000", "2024-01-01", np.nan)

        assert pd.isna(up_limit)
        assert pd.isna(down_limit)
        assert _can_buy_at_open("SH600000", "2024-01-01", np.nan, 10.0) is False
        assert _can_buy_at_open("SH600000", "2024-01-01", 10.0, np.nan) is False
        assert _can_sell_at_open("SH600000", "2024-01-01", 0.0, 10.0) is False
        assert _can_sell_at_open("SH600000", "2024-01-01", 10.0, 0.0) is False
        with patch("modules.backtest.qlib_engine._get_limit_prices", return_value=(np.nan, 9.0)):
            assert _can_buy_at_open("SH600000", "2024-01-01", 10.0, 10.0) is False
        with patch("modules.backtest.qlib_engine._get_limit_prices", return_value=(11.0, np.nan)):
            assert _can_sell_at_open("SH600000", "2024-01-01", 10.0, 10.0) is False

    def test_tradability_constraint_requires_explicit_support(self):
        with patch("modules.backtest.qlib_engine._raw_data_root", return_value=Path("/tmp/nonexistent_raw_data")):
            with pytest.raises(FileNotFoundError):
                _ensure_tradability_constraints_supported(True, False)

            with pytest.raises(FileNotFoundError):
                _ensure_tradability_constraints_supported(False, True)

        with patch("modules.backtest.qlib_engine._raw_data_root", return_value=Path("/tmp")):
            _ensure_tradability_constraints_supported(True, False)
            _ensure_tradability_constraints_supported(False, True)
            _ensure_tradability_constraints_supported(False, False)

    def test_load_ranked_selection_orders_handles_missing_and_empty_csv(self, tmp_path):
        strategy = Mock()
        strategy.selections_path.return_value = tmp_path / "missing.csv"

        with pytest.raises(FileNotFoundError):
            _load_ranked_selection_orders(strategy=strategy)

        empty_csv = tmp_path / "empty.csv"
        pd.DataFrame(columns=["date", "rank", "symbol"]).to_csv(empty_csv, index=False)
        strategy.selections_path.return_value = empty_csv

        assert _load_ranked_selection_orders(strategy=strategy) == {}

    def test_load_raw_trade_quotes_handles_empty_inputs_and_empty_windows(self, tmp_path):
        assert _load_raw_trade_quotes([], "2024-01-01", "2024-01-02").empty

        raw_file = tmp_path / "sz000001.parquet"
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01"]),
                "open": [10.0],
                "close": [10.1],
            }
        ).to_parquet(raw_file, index=False)

        with patch("modules.backtest.qlib_engine._raw_data_root", return_value=tmp_path):
            quotes = _load_raw_trade_quotes(["SZ000001"], "2024-01-02", "2024-01-03")

        assert quotes.empty

        empty_raw_file = tmp_path / "sz000002.parquet"
        pd.DataFrame(columns=["date", "open", "close"]).to_parquet(empty_raw_file, index=False)
        with patch("modules.backtest.qlib_engine._raw_data_root", return_value=tmp_path):
            quotes = _load_raw_trade_quotes(["SZ000002"], "2024-01-02", "2024-01-03")
        assert quotes.empty

    def test_quote_row_and_sum_symbol_returns_edge_cases(self):
        assert _quote_row(pd.DataFrame(), "AAA") is None
        assert _quote_row(None, "AAA") is None

        duplicate_quotes = pd.DataFrame(
            {"open": [10.0, 10.2], "prev_close": [9.9, 10.0]},
            index=["AAA", "AAA"],
        )
        row = _quote_row(duplicate_quotes, "AAA")
        assert row["open"] == pytest.approx(10.2)

        total, available, missing = _sum_symbol_returns(
            pd.DataFrame({"daily_ret": [0.01]}, index=["AAA"]),
            set(),
            "daily_ret",
        )
        assert total == 0.0
        assert available == set()
        assert missing == set()

        penalized_missing = set()
        total, available, missing = _sum_symbol_returns(
            pd.DataFrame({"daily_ret": [0.01]}, index=["AAA"]),
            {"BBB"},
            "daily_ret",
            penalized_missing=penalized_missing,
            apply_penalty=False,
        )
        assert total == 0.0
        assert available == set()
        assert missing == {"BBB"}
        assert penalized_missing == set()

    def test_rebalance_day_uses_close_to_close_returns(self):
        day_px = pd.DataFrame(
            {
                "daily_ret": [0.02, -0.01],
                "open_to_close_ret": [0.50, 0.40],
            },
            index=["AAA", "BBB"],
        )

        result = _compute_rebalance_day(
            day_px,
            selected={"AAA", "BBB"},
            prev_selected={"OLD"},
            topk=2,
            penalized_missing=set(),
        )

        assert result["stock_slot_return"] == pytest.approx(0.005)
        assert result["buy_count"] == 2
        assert result["sell_count"] == 1
        assert result["held_symbols"] == {"AAA", "BBB"}

    def test_rebalance_day_blocks_limit_down_sell_and_reduces_buy_slots(self):
        day_px = pd.DataFrame(
            {"daily_ret": [0.01, -0.02]},
            index=["KEEP", "OLD"],
        )
        raw_day_quotes = pd.DataFrame(
            {"open": [9.0], "close": [8.8], "prev_close": [10.0]},
            index=["OLD"],
        )

        with patch("modules.backtest.qlib_engine.is_st_on_date", return_value=False):
            result = _compute_rebalance_day(
                day_px,
                selected={"KEEP", "NEW"},
                prev_selected={"KEEP", "OLD"},
                topk=2,
                penalized_missing=set(),
                ranked_selected=["KEEP", "NEW"],
                raw_day_quotes=raw_day_quotes,
                trade_date="2024-01-02",
                block_limit_down_sell=True,
            )

        assert result["sell_count"] == 0
        assert result["blocked_sell_count"] == 1
        assert result["buy_count"] == 0
        assert result["held_symbols"] == {"KEEP", "OLD"}

    def test_rebalance_day_blocks_limit_up_buy(self):
        day_px = pd.DataFrame(
            {"daily_ret": [0.01]},
            index=["KEEP"],
        )
        raw_day_quotes = pd.DataFrame(
            {"open": [11.0], "close": [11.2], "prev_close": [10.0]},
            index=["NEW"],
        )

        with patch("modules.backtest.qlib_engine.is_st_on_date", return_value=False):
            result = _compute_rebalance_day(
                day_px,
                selected={"KEEP", "NEW"},
                prev_selected={"KEEP"},
                topk=2,
                penalized_missing=set(),
                ranked_selected=["KEEP", "NEW"],
                raw_day_quotes=raw_day_quotes,
                trade_date="2024-01-02",
                block_limit_up_buy=True,
            )

        assert result["buy_count"] == 0
        assert result["blocked_buy_count"] == 1
        assert result["held_symbols"] == {"KEEP"}
        assert result["cash_slot_count"] == 1

    def test_rebalance_day_handles_zero_topk(self):
        result = _compute_rebalance_day(
            pd.DataFrame({"daily_ret": [0.01]}, index=["AAA"]),
            selected={"AAA"},
            prev_selected=set(),
            topk=0,
            penalized_missing=set(),
        )

        assert result["stock_slot_return"] == 0.0
        assert result["cash_slot_count"] == 0

    def test_rebalance_day_skips_ranked_symbols_already_held(self):
        result = _compute_rebalance_day(
            pd.DataFrame({"daily_ret": [0.01, 0.02]}, index=["KEEP", "NEW"]),
            selected={"KEEP", "NEW"},
            prev_selected={"KEEP"},
            topk=2,
            penalized_missing=set(),
            ranked_selected=["KEEP", "NEW"],
        )

        assert result["buy_count"] == 1
        assert result["held_symbols"] == {"KEEP", "NEW"}

    def test_rebalance_day_all_ranked_symbols_already_held(self):
        result = _compute_rebalance_day(
            pd.DataFrame({"daily_ret": [0.01]}, index=["KEEP"]),
            selected={"KEEP"},
            prev_selected={"KEEP"},
            topk=1,
            penalized_missing=set(),
            ranked_selected=["KEEP"],
        )

        assert result["buy_count"] == 0
        assert result["held_symbols"] == {"KEEP"}

    def test_qlib_backtest_loads_only_selected_instruments(self):
        engine = QlibBacktestEngine()
        strategy = Mock()
        strategy.name = "test_strategy"
        strategy.trading_cost = {}
        strategy.universe = "all"
        strategy.position_model = "full"

        rebal_dates = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-04"),
        ]
        date_to_symbols = {
            rebal_dates[0]: {"SZ000001", "SZ000002"},
            rebal_dates[1]: {"SZ000002", "SZ000003"},
        }
        px_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-02"), "SZ000002"),
                (pd.Timestamp("2024-01-02"), "SZ000003"),
                (pd.Timestamp("2024-01-03"), "SZ000001"),
                (pd.Timestamp("2024-01-03"), "SZ000002"),
                (pd.Timestamp("2024-01-03"), "SZ000003"),
                (pd.Timestamp("2024-01-04"), "SZ000001"),
                (pd.Timestamp("2024-01-04"), "SZ000002"),
                (pd.Timestamp("2024-01-04"), "SZ000003"),
            ],
            names=["datetime", "instrument"],
        )
        df_px = pd.DataFrame({"close": [10.0, 11.0, 12.0, 10.2, 11.1, 12.3, 10.3, 11.4, 12.2]}, index=px_index)

        with patch.object(engine, "_prepare", return_value=(date_to_symbols, set(rebal_dates), None, 2)):
            with patch("modules.backtest.qlib_engine.init_qlib"):
                with patch("modules.backtest.qlib_engine.setup_logger", return_value=Mock()):
                    with patch("modules.backtest.qlib_engine.TradeLogger"):
                        with patch("modules.backtest.qlib_engine.filter_instruments", side_effect=lambda instruments, exclude_st=False: instruments):
                            with patch("modules.backtest.qlib_engine.load_features_safe", return_value=df_px) as mock_load_features:
                                with patch("pandas.DataFrame.to_csv"):
                                    result = engine.run(strategy=strategy)

        assert not result.portfolio_value.empty
        mock_load_features.assert_called_once()
        assert mock_load_features.call_args.args[0] == ["SZ000001", "SZ000002", "SZ000003"]

    def test_qlib_backtest_returns_empty_when_no_selected_instruments(self):
        engine = QlibBacktestEngine()
        strategy = Mock()
        strategy.name = "empty_strategy"
        strategy.trading_cost = {}
        strategy.universe = "all"
        strategy.position_model = "full"

        logger = Mock()
        with patch.object(engine, "_prepare", return_value=({}, set(), None, 2)):
            with patch("modules.backtest.qlib_engine.init_qlib"):
                with patch("modules.backtest.qlib_engine.setup_logger", return_value=logger):
                    with patch("modules.backtest.qlib_engine.TradeLogger"):
                        result = engine.run(strategy=strategy)

        assert result.daily_returns.empty
        assert result.portfolio_value.empty
        logger.error.assert_called_once()

    def test_qlib_backtest_fixed_position_strategy_runs(self):
        engine = QlibBacktestEngine()
        strategy = Mock()
        strategy.name = "fixed_strategy"
        strategy.trading_cost = {}
        strategy.universe = "all"
        strategy.position_model = "fixed"
        strategy.position_params = {"stock_pct": 0.75}

        rebal_dates = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-04"),
        ]
        date_to_symbols = {
            rebal_dates[0]: {"SZ000001", "SZ000002"},
            rebal_dates[1]: {"SZ000001", "SZ000002"},
        }
        px_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-02"), "SZ000002"),
                (pd.Timestamp("2024-01-03"), "SZ000001"),
                (pd.Timestamp("2024-01-03"), "SZ000002"),
                (pd.Timestamp("2024-01-04"), "SZ000001"),
                (pd.Timestamp("2024-01-04"), "SZ000002"),
            ],
            names=["datetime", "instrument"],
        )
        df_px = pd.DataFrame({"close": [10.0, 11.0, 10.2, 11.1, 10.3, 11.2]}, index=px_index)

        with patch.object(engine, "_prepare", return_value=(date_to_symbols, set(rebal_dates), None, 2)):
            with patch("modules.backtest.qlib_engine.init_qlib"):
                with patch("modules.backtest.qlib_engine.setup_logger", return_value=Mock()):
                    with patch("modules.backtest.qlib_engine.TradeLogger"):
                        with patch("modules.backtest.qlib_engine.filter_instruments", side_effect=lambda instruments, exclude_st=False: instruments):
                            with patch("modules.backtest.qlib_engine.load_features_safe", return_value=df_px):
                                with patch("pandas.DataFrame.to_csv"):
                                    result = engine.run(strategy=strategy)

        assert not result.portfolio_value.empty
        assert result.daily_returns.iloc[0] != 0

    def test_qlib_backtest_enables_tradability_data_path(self):
        engine = QlibBacktestEngine()
        strategy = Mock()
        strategy.name = "tradability_strategy"
        strategy.trading_cost = {"block_limit_up_buy": True}
        strategy.universe = "all"
        strategy.position_model = "full"

        rebal_dates = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-04"),
        ]
        date_to_symbols = {
            rebal_dates[0]: {"SZ000001"},
            rebal_dates[1]: {"SZ000001"},
        }
        px_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-03"), "SZ000001"),
                (pd.Timestamp("2024-01-04"), "SZ000001"),
            ],
            names=["datetime", "instrument"],
        )
        df_px = pd.DataFrame({"close": [10.0, 10.2, 10.3]}, index=px_index)
        raw_quotes = pd.DataFrame(
            {
                "open": [10.1],
                "close": [10.2],
                "prev_close": [10.0],
            },
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2024-01-03"), "SZ000001")],
                names=["datetime", "instrument"],
            ),
        )

        with patch.object(engine, "_prepare", return_value=(date_to_symbols, set(rebal_dates), None, 1)):
            with patch("modules.backtest.qlib_engine.init_qlib"):
                with patch("modules.backtest.qlib_engine.setup_logger", return_value=Mock()):
                    with patch("modules.backtest.qlib_engine.TradeLogger"):
                        with patch("modules.backtest.qlib_engine.filter_instruments", side_effect=lambda instruments, exclude_st=False: instruments):
                            with patch("modules.backtest.qlib_engine.load_features_safe", return_value=df_px):
                                with patch("modules.backtest.qlib_engine._ensure_tradability_constraints_supported"):
                                    with patch("modules.backtest.qlib_engine._load_ranked_selection_orders", return_value={rebal_dates[0]: ["SZ000001"]}) as mock_ranked:
                                        with patch("modules.backtest.qlib_engine._load_raw_trade_quotes", return_value=raw_quotes) as mock_raw:
                                            with patch("pandas.DataFrame.to_csv"):
                                                result = engine.run(strategy=strategy)

        assert not result.portfolio_value.empty
        mock_ranked.assert_called_once_with(strategy)
        mock_raw.assert_called_once_with(
            {"SZ000001"},
            CONFIG.get("start_date", "2019-01-01"),
            CONFIG.get("end_date", "2026-02-26"),
        )

    def test_qlib_backtest_controller_allocation_branch(self):
        engine = QlibBacktestEngine()
        strategy = Mock()
        strategy.name = "controller_strategy"
        strategy.trading_cost = {
            "open_cost": 0.0,
            "close_cost": 0.0,
            "sell_stamp_tax_rate": 0.0,
            "min_buy_commission": 0.0,
            "min_sell_commission": 0.0,
        }
        strategy.universe = "all"
        strategy.position_model = "dynamic"

        controller = Mock()
        controller.get_bond_daily_return.return_value = 0.001
        controller.get_allocation.return_value = Mock(
            stock_pct=0.5,
            regime="risk_off",
            opportunity_level="low",
            market_drawdown=-0.12,
        )

        rebal_dates = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-04"),
        ]
        date_to_symbols = {
            rebal_dates[0]: {"SZ000001", "SZ000002"},
            rebal_dates[1]: {"SZ000001", "SZ000002"},
        }
        px_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-02"), "SZ000002"),
                (pd.Timestamp("2024-01-03"), "SZ000001"),
                (pd.Timestamp("2024-01-03"), "SZ000002"),
                (pd.Timestamp("2024-01-04"), "SZ000001"),
                (pd.Timestamp("2024-01-04"), "SZ000002"),
            ],
            names=["datetime", "instrument"],
        )
        df_px = pd.DataFrame(
            {"close": [10.0, 20.0, 10.2, 20.4, 10.3, 20.5]},
            index=px_index,
        )

        with patch.object(engine, "_prepare", return_value=(date_to_symbols, set(rebal_dates), controller, 2)):
            with patch("modules.backtest.qlib_engine.init_qlib"):
                with patch("modules.backtest.qlib_engine.setup_logger", return_value=Mock()):
                    with patch("modules.backtest.qlib_engine.TradeLogger"):
                        with patch("modules.backtest.qlib_engine.filter_instruments", side_effect=lambda instruments, exclude_st=False: instruments):
                            with patch("modules.backtest.qlib_engine.load_features_safe", return_value=df_px):
                                with patch("modules.backtest.qlib_engine._load_bond_etf_returns", return_value=None):
                                    with patch("pandas.DataFrame.to_csv"):
                                        result = engine.run(strategy=strategy)

        assert not result.portfolio_value.empty
        assert controller.get_allocation.call_count == 1
        assert result.daily_returns.iloc[0] == pytest.approx(0.001)
        assert result.daily_returns.iloc[1] > result.daily_returns.iloc[0]

    def test_qlib_backtest_returns_empty_when_rebalance_selection_under_topk(self):
        engine = QlibBacktestEngine()
        strategy = Mock()
        strategy.name = "underfilled_strategy"
        strategy.trading_cost = {}
        strategy.universe = "all"
        strategy.position_model = "full"

        rebal_dates = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-04"),
        ]
        date_to_symbols = {
            rebal_dates[0]: {"SZ000001"},
            rebal_dates[1]: {"SZ000001"},
        }
        px_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-03"), "SZ000001"),
                (pd.Timestamp("2024-01-04"), "SZ000001"),
            ],
            names=["datetime", "instrument"],
        )
        df_px = pd.DataFrame({"close": [10.0, 10.2, 10.3]}, index=px_index)
        logger = Mock()

        with patch.object(engine, "_prepare", return_value=(date_to_symbols, set(rebal_dates), None, 2)):
            with patch("modules.backtest.qlib_engine.init_qlib"):
                with patch("modules.backtest.qlib_engine.setup_logger", return_value=logger):
                    with patch("modules.backtest.qlib_engine.TradeLogger"):
                        with patch("modules.backtest.qlib_engine.filter_instruments", side_effect=lambda instruments, exclude_st=False: instruments):
                            with patch("modules.backtest.qlib_engine.load_features_safe", return_value=df_px):
                                result = engine.run(strategy=strategy)

        assert result.daily_returns.empty
        assert result.portfolio_value.empty
        logger.error.assert_called_with("无有效回测数据")

    def test_qlib_backtest_handles_missing_raw_quotes_on_rebalance_day(self):
        engine = QlibBacktestEngine()
        strategy = Mock()
        strategy.name = "raw_quote_gap_strategy"
        strategy.trading_cost = {"block_limit_up_buy": True}
        strategy.universe = "all"
        strategy.position_model = "full"

        rebal_dates = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-04"),
        ]
        date_to_symbols = {
            rebal_dates[0]: {"SZ000001"},
            rebal_dates[1]: {"SZ000001"},
        }
        px_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-03"), "SZ000001"),
                (pd.Timestamp("2024-01-04"), "SZ000001"),
            ],
            names=["datetime", "instrument"],
        )
        df_px = pd.DataFrame({"close": [10.0, 10.2, 10.3]}, index=px_index)
        empty_raw_quotes = pd.DataFrame(
            columns=["open", "close", "prev_close"],
            index=pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"]),
        )

        with patch.object(engine, "_prepare", return_value=(date_to_symbols, set(rebal_dates), None, 1)):
            with patch("modules.backtest.qlib_engine.init_qlib"):
                with patch("modules.backtest.qlib_engine.setup_logger", return_value=Mock()):
                    with patch("modules.backtest.qlib_engine.TradeLogger"):
                        with patch("modules.backtest.qlib_engine.filter_instruments", side_effect=lambda instruments, exclude_st=False: instruments):
                            with patch("modules.backtest.qlib_engine.load_features_safe", return_value=df_px):
                                with patch("modules.backtest.qlib_engine._load_bond_etf_returns", return_value=None):
                                    with patch("modules.backtest.qlib_engine._ensure_tradability_constraints_supported"):
                                        with patch("modules.backtest.qlib_engine._load_ranked_selection_orders", return_value={rebal_dates[0]: ["SZ000001"]}):
                                            with patch("modules.backtest.qlib_engine._load_raw_trade_quotes", return_value=empty_raw_quotes):
                                                with patch("pandas.DataFrame.to_csv"):
                                                    result = engine.run(strategy=strategy)

        assert not result.daily_returns.empty
        assert result.daily_returns.iloc[0] == 0.0

    def test_qlib_backtest_skips_missing_day_slice(self):
        engine = QlibBacktestEngine()
        strategy = Mock()
        strategy.name = "missing_day_slice_strategy"
        strategy.trading_cost = {}
        strategy.universe = "all"
        strategy.position_model = "full"

        rebal_dates = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-05"),
        ]
        date_to_symbols = {
            rebal_dates[0]: {"SZ000001"},
            rebal_dates[1]: {"SZ000001"},
        }
        px_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-04"), "SZ000001"),
                (pd.Timestamp("2024-01-05"), "SZ000001"),
            ],
            names=["datetime", "instrument"],
        )
        df_px = pd.DataFrame({"close": [10.0, 10.4, 10.5]}, index=px_index)

        with patch.object(engine, "_prepare", return_value=(date_to_symbols, set(rebal_dates), None, 1)):
            with patch("modules.backtest.qlib_engine.init_qlib"):
                with patch("modules.backtest.qlib_engine.setup_logger", return_value=Mock()):
                    with patch("modules.backtest.qlib_engine.TradeLogger"):
                        with patch("modules.backtest.qlib_engine.filter_instruments", side_effect=lambda instruments, exclude_st=False: instruments):
                            with patch("modules.backtest.qlib_engine.load_features_safe", return_value=df_px):
                                with patch("pandas.DataFrame.to_csv"):
                                    result = engine.run(strategy=strategy)

        assert len(result.daily_returns) == 2
        assert pd.Timestamp("2024-01-03") not in result.daily_returns.index

    def test_qlib_main_prints_summary_and_saved_path(self, capsys):
        strategy = Mock()
        result = Mock()
        result.metadata = {"results_file": "/tmp/backtest.csv"}

        with patch("modules.backtest.qlib_engine.QlibBacktestEngine") as mock_engine_cls:
            mock_engine = mock_engine_cls.return_value
            mock_engine.run.return_value = result
            returned = qlib_main(strategy=strategy)

        result.print_summary.assert_called_once_with(CONFIG.get("initial_capital", 500000))
        captured = capsys.readouterr()
        assert "结果已保存" in captured.out
        assert returned is result
