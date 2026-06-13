#!/usr/bin/env python3
"""
基于 QLib 的因子挖掘系统
=====================================
功能：
1. 因子表达式库（技术指标、基本面指标等）
2. 因子 IC 分析（信息系数）
3. 因子有效性测试（IR、IC衰减、换手率等）
4. 因子相关性分析
5. 因子组合优化
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from config.config import CONFIG


class FactorMining:
    """因子挖掘器"""
    
    def __init__(self):
        """初始化因子挖掘器"""
        self._init_qlib()
        self.factor_library = self._build_factor_library()
    
    def _init_qlib(self):
        """初始化 Qlib"""
        import qlib
        from qlib.config import REG_CN
        
        os.environ["JOBLIB_START_METHOD"] = "fork"
        provider_uri = Path(CONFIG.get("paths.qlib_data", "~/code/qlib/data/qlib_data/cn_data")).expanduser()
        
        if not provider_uri.exists():
            print(f"[ERROR] Qlib 数据目录不存在: {provider_uri}")
            sys.exit(1)
        
        qlib.init(provider_uri=str(provider_uri), region=REG_CN)
        
        try:
            from qlib.config import C
            C.n_jobs = 1
        except Exception:
            pass
        
        print(f"[OK] Qlib 初始化成功")
    
    def _build_factor_library(self) -> Dict[str, Dict[str, Any]]:
        """
        构建因子表达式库
        
        包括：
        1. 技术指标因子
        2. 基本面因子
        3. 量价关系因子
        4. 风险因子
        5. 质量因子
        """
        factor_library = {
            # ===== 技术指标因子 =====
            "technical": {
                # 动量类
                "momentum_5d": "Ref($close, 5) / $close - 1",
                "momentum_10d": "Ref($close, 10) / $close - 1",
                "momentum_20d": "Ref($close, 20) / $close - 1",
                "momentum_60d": "Ref($close, 60) / $close - 1",
                "momentum_120d": "Ref($close, 120) / $close - 1",
                "momentum_250d": "Ref($close, 250) / $close - 1",
                
                # 反转类
                "reversal_5d": "-1 * (Ref($close, 5) / $close - 1)",
                "reversal_10d": "-1 * (Ref($close, 10) / $close - 1)",
                "reversal_20d": "-1 * (Ref($close, 20) / $close - 1)",
                
                # 波动率类
                "volatility_5d": "Std(($close - Ref($close, 1)) / Ref($close, 1), 5)",
                "volatility_10d": "Std(($close - Ref($close, 1)) / Ref($close, 1), 10)",
                "volatility_20d": "Std(($close - Ref($close, 1)) / Ref($close, 1), 20)",
                "volatility_60d": "Std(($close - Ref($close, 1)) / Ref($close, 1), 60)",
                
                # 量价关系
                "volume_price_5d": "Mean($volume, 5) / Mean($close, 5)",
                "volume_price_10d": "Mean($volume, 10) / Mean($close, 10)",
                "volume_price_20d": "Mean($volume, 20) / Mean($close, 20)",
                
                # 均线类
                "ma_ratio_5_20": "Mean($close, 5) / Mean($close, 20) - 1",
                "ma_ratio_10_60": "Mean($close, 10) / Mean($close, 60) - 1",
                "ma_ratio_20_120": "Mean($close, 20) / Mean($close, 120) - 1",
                
                # 价格位置
                "price_position_20d": "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 1e-8)",
                "price_position_60d": "($close - Min($close, 60)) / (Max($close, 60) - Min($close, 60) + 1e-8)",
                
                # RSI类
                "rsi_14d": """
                    100 - 100 / (1 +
                    Mean(Greater($close - Ref($close, 1), 0), 14) /
                    (Mean(Abs($close - Ref($close, 1)), 14) + 1e-8)
                    )
                """,

                # ===== 迅投技术因子（基于现有数据）=====
                # 乖离率系列
                "bias_5d": "($close / Mean($close, 5) - 1)",  # 5日乖离率
                "bias_10d": "($close / Mean($close, 10) - 1)",  # 10日乖离率
                "bias_20d": "($close / Mean($close, 20) - 1)",  # 20日乖离率
                "bias_60d": "($close / Mean($close, 60) - 1)",  # 60日乖离率

                # 变动速率(ROC)系列
                "roc_6d": "($close / Ref($close, 6) - 1)",  # 6日变动速率
                "roc_12d": "($close / Ref($close, 12) - 1)",  # 12日变动速率
                "roc_60d": "($close / Ref($close, 60) - 1)",  # 60日变动速率
                "roc_120d": "($close / Ref($close, 120) - 1)",  # 120日变动速率

                # 线性回归斜率
                "linreg_slope_6d": "Slope($close, 6)",  # 6日线性回归斜率
                "linreg_slope_12d": "Slope($close, 12)",  # 12日线性回归斜率
                "linreg_slope_24d": "Slope($close, 24)",  # 24日线性回归斜率

                # 换手率比率
                "turnover_ratio_5_120": "Mean($turnover_rate, 5) / Mean($turnover_rate, 120)",  # 5日/120日换手率比率
                "turnover_ratio_10_120": "Mean($turnover_rate, 10) / Mean($turnover_rate, 120)",  # 10日/120日换手率比率
                "turnover_ratio_20_120": "Mean($turnover_rate, 20) / Mean($turnover_rate, 120)",  # 20日/120日换手率比率
            },
            
            # ===== 基本面因子 =====
            "fundamental": {
                # ===== 基于实际可用字段的财务因子 =====
                # 估值指标（日频，数据完整）
                "pe_ratio": "$pe",  # 市盈率
                "pb_ratio": "$pb",  # 市净率
                "ps_ratio": "$ps",  # 市销率
                "total_mv": "$total_mv",  # 总市值
                "circ_mv": "$circ_mv",  # 流通市值
                
                # 盈利能力（基于利润表字段计算）
                "roe_calc": "$n_income / $total_hldr_eqy_exc_min_int",  # 净资产收益率（净利润/股东权益）
                "roa_calc": "$n_income / $total_assets",  # 总资产收益率（净利润/总资产）
                "net_margin_calc": "$n_income / $total_revenue",  # 净利率（净利润/总收入）
                "gross_margin_calc": "($total_revenue - $total_cogs) / $total_revenue",  # 毛利率（毛利/总收入）
                "operating_margin": "$operate_profit / $total_revenue",  # 营业利润率
                
                # 成长性（基于季度数据计算变化率）
                "revenue_growth_qoq": "$total_revenue / Ref($total_revenue, 1) - 1",  # 营收环比增长率
                "profit_growth_qoq": "$total_profit / Ref($total_profit, 1) - 1",  # 利润环比增长率
                "asset_growth_qoq": "$total_assets / Ref($total_assets, 1) - 1",  # 资产环比增长率
                
                # 财务健康与偿债能力
                "debt_to_assets_calc": "$total_liab / $total_assets",  # 资产负债率（负债/资产）
                "debt_to_equity_calc": "$total_liab / $total_hldr_eqy_exc_min_int",  # 负债权益比
                "current_ratio_calc": "$total_cur_assets / $total_cur_liab",  # 流动比率（流动资产/流动负债）
                
                # 营运效率
                "asset_turnover_calc": "$total_revenue / $total_assets",  # 资产周转率（收入/资产）
                # 注：ar_turn, inv_turn, ca_turn等字段数据为空
                
                # 现金流指标
                "operating_cash_flow": "$n_cashflow_act",  # 经营活动现金流净额
                "cash_flow_to_sales": "$n_cashflow_act / $total_revenue",  # 现金流营收比
                "cash_flow_to_assets": "$n_cashflow_act / $total_assets",  # 现金流资产比
                "free_cash_flow": "$free_cashflow",  # 自由现金流

                # ===== 直接使用原始财务数据（基于已有字段）=====
                # 原始利润表数据
                "net_income": "$n_income",  # 净利润（原始值）
                "total_revenue": "$total_revenue",  # 总收入（原始值）
                "total_profit": "$total_profit",  # 利润总额（原始值）
                "operate_profit": "$operate_profit",  # 营业利润（原始值）
                
                # 原始资产负债表数据
                "total_assets_raw": "$total_assets",  # 总资产（原始值）
                "total_liab_raw": "$total_liab",  # 总负债（原始值）
                "shareholders_equity": "$total_hldr_eqy_exc_min_int",  # 股东权益（原始值）
                
                # 简单规模调整（对数化处理）
                "log_total_assets": "Log($total_assets)",  # 总资产对数
                "log_total_revenue": "Log($total_revenue)",  # 总收入对数
                "log_net_income": "Log($n_income)",  # 净利润对数
                
                # 规模指标（常用于因子标准化）
                "asset_size": "$total_assets / Mean($total_assets, 20)",  # 资产规模相对值
                "revenue_size": "$total_revenue / Mean($total_revenue, 20)",  # 收入规模相对值
                
                # ===== 原QLib字段（可能不存在或数据为空）=====
                # 以下字段在QLib中可能不存在或数据为空，保留但注释
                # "roe": "$ROE",  # 可能不存在
                # "roa": "$ROA",  # 可能不存在
                # "net_margin": "$Net_Margin",  # 可能不存在
                # "gross_margin": "$Gross_Margin",  # 可能不存在
                # "revenue_growth": "$Revenue_Growth",  # 可能不存在
                # "profit_growth": "$Profit_Growth",  # 可能不存在
                # "pcf_ratio": "$PCF",  # 市现率（可能不存在）
                # "current_ratio": "$Current_Ratio",  # 可能不存在
                # "debt_to_equity": "$Debt_to_Equity",  # 可能不存在
                # "asset_turnover": "$Asset_Turnover",  # 可能不存在
            },
            
            # ===== 风险因子 =====
            "risk": {
                "beta_20d": """
                    Cov(($close - Ref($close, 1)) / Ref($close, 1), 
                        ($market_return), 20) / Var($market_return, 20)
                """,
                "beta_60d": """
                    Cov(($close - Ref($close, 1)) / Ref($close, 1), 
                        ($market_return), 60) / Var($market_return, 60)
                """,
                "idiosyncratic_vol_20d": """
                    Std(($close - Ref($close, 1)) / Ref($close, 1) - 
                        beta_20d * $market_return, 20)
                """,
                "max_drawdown_20d": """
                    Min(($close / CumMax($close, 20) - 1))
                """,
            },
            
            # ===== 质量因子 =====
            "quality": {
                "accruals": """
                    ($operating_cash_flow - $net_income) / $total_assets
                """,
                "profitability_score": """
                    ($roe + $roa + $net_margin) / 3
                """,
                "stability_score": """
                    (1 / (Std($roe, 8) + 1e-8) + 1 / (Std($net_margin, 8) + 1e-8)) / 2
                """,
            },
            
            # ===== 情绪因子 =====
            "sentiment": {
                "turnover_5d": "Mean($turnover_rate, 5)",
                "turnover_20d": "Mean($turnover_rate, 5)",
                "turnover_std_20d": "Std($turnover_rate, 20)",
                "abnormal_volume_5d": """
                    $volume / Mean($volume, 20) - 1
                """,
            },
        }
        
        return factor_library
    
    def load_features_safe(self, instruments, fields, start_time, end_time, freq="day"):
        """安全加载因子数据"""
        from qlib.data import D
        
        if not isinstance(instruments, list):
            inst_list = D.list_instruments(instruments, start_time=start_time, end_time=end_time)
            inst_list = list(inst_list.keys())
        else:
            inst_list = list(instruments)
        
        return D.features(inst_list, fields, start_time, end_time, freq)
    
    def get_valid_instruments(self, start_date="2019-01-01", end_date="2026-02-26"):
        """获取有效股票列表"""
        from qlib.data import D
        
        instruments = D.instruments(market="all")
        df_close = self.load_features_safe(
            instruments, ["$close"], start_date, end_date
        )
        
        valid_instruments = [
            i for i in df_close.index.get_level_values("instrument").unique()
            if not i.startswith("BJ")  # 排除北交所
            and not any(i.startswith(p) for p in ["SH43","SZ43","SH83","SZ83","SH87","SZ87"])
        ]
        
        return valid_instruments
    
    def test_factor_ic(self, factor_expr: str, factor_name: str, 
                      start_date="2019-01-01", end_date="2026-02-26",
                      freq="day") -> Dict[str, Any]:
        """
        测试因子 IC（信息系数）
        
        Parameters
        ----------
        factor_expr : str
            因子表达式
        factor_name : str
            因子名称
        start_date : str
            开始日期
        end_date : str
            结束日期
        freq : str
            频率
            
        Returns
        -------
        Dict[str, Any]
            IC 分析结果
        """
        from qlib.data import D
        import pandas as pd
        
        print(f"\n[测试因子] {factor_name}")
        print(f"  表达式: {factor_expr}")
        
        # 获取有效股票
        instruments = self.get_valid_instruments(start_date, end_date)
        print(f"  股票数量: {len(instruments)}")
        
        try:
            # 加载因子数据
            df_factor = self.load_features_safe(
                instruments, [factor_expr], start_date, end_date, freq
            )
            df_factor.columns = [factor_name]
            
            # 加载收益率数据（未来一期）
            ret_expr = "Ref($close, -1) / $close - 1"  # 未来一期收益率
            df_return = self.load_features_safe(
                instruments, [ret_expr], start_date, end_date, freq
            )
            df_return.columns = ["next_return"]
            
            # 合并数据
            df_merged = pd.concat([df_factor, df_return], axis=1).dropna()
            
            if df_merged.empty:
                print(f"  [WARN] 无有效数据")
                return {"factor_name": factor_name, "error": "无有效数据"}
            
            # 计算 IC
            ic_results = []
            dates = df_merged.index.get_level_values("datetime").unique()
            
            for dt in dates:
                try:
                    cross_section = df_merged.xs(dt, level="datetime")
                    if len(cross_section) < 20:  # 至少20只股票
                        continue
                    
                    # 计算截面相关性（IC）
                    ic = cross_section[factor_name].corr(cross_section["next_return"])
                    ic_results.append({"date": dt, "ic": ic})
                except:
                    continue
            
            if not ic_results:
                print(f"  [WARN] 无法计算 IC")
                return {"factor_name": factor_name, "error": "无法计算 IC"}
            
            # 分析 IC
            df_ic = pd.DataFrame(ic_results).set_index("date")
            mean_ic = df_ic["ic"].mean()
            std_ic = df_ic["ic"].std()
            ir = mean_ic / std_ic if std_ic > 0 else 0
            ic_positive_ratio = (df_ic["ic"] > 0).mean()
            
            print(f"  IC 均值: {mean_ic:.6f}")
            print(f"  IC 标准差: {std_ic:.6f}")
            print(f"  IR: {ir:.4f}")
            print(f"  IC>0 比例: {ic_positive_ratio:.2%}")
            print(f"  有效截面数: {len(df_ic)}")
            
            return {
                "factor_name": factor_name,
                "expression": factor_expr,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "ir": ir,
                "ic_positive_ratio": ic_positive_ratio,
                "n_cross_sections": len(df_ic),
                "ic_series": df_ic["ic"],
            }
            
        except Exception as e:
            print(f"  [ERROR] 因子测试失败: {e}")
            return {"factor_name": factor_name, "error": str(e)}
    
    def mine_factors(self, category="technical", 
                    start_date="2019-01-01", end_date="2026-02-26",
                    top_n=10):
        """
        挖掘指定类别的最优因子
        
        Parameters
        ----------
        category : str
            因子类别（technical, fundamental, risk, quality, sentiment）
        start_date : str
            开始日期
        end_date : str
            结束日期
        top_n : int
            返回最优因子的数量
            
        Returns
        -------
        pd.DataFrame
            因子挖掘结果
        """
        print(f"\n{'='*60}")
        print(f"  因子挖掘 - {category} 类别")
        print(f"{'='*60}")
        
        if category not in self.factor_library:
            print(f"[ERROR] 无效的因子类别: {category}")
            print(f"可用类别: {list(self.factor_library.keys())}")
            return None
        
        factors = self.factor_library[category]
        print(f"待测试因子数量: {len(factors)}")
        
        results = []
        for factor_name, factor_expr in factors.items():
            result = self.test_factor_ic(
                factor_expr, factor_name, start_date, end_date
            )
            
            if "error" not in result:
                results.append({
                    "factor_name": factor_name,
                    "expression": factor_expr,
                    "mean_ic": result["mean_ic"],
                    "ir": result["ir"],
                    "ic_positive_ratio": result["ic_positive_ratio"],
                    "n_cross_sections": result["n_cross_sections"],
                    "category": category,
                })
        
        if not results:
            print("[WARN] 无有效测试结果")
            return None
        
        # 排序并选择最优因子
        df_results = pd.DataFrame(results)
        df_results["abs_mean_ic"] = df_results["mean_ic"].abs()
        df_results = df_results.sort_values("abs_mean_ic", ascending=False)
        
        print(f"\n{'='*60}")
        print(f"  挖掘结果 - Top {top_n}")
        print(f"{'='*60}")
        
        print(f"\n{'排名':<4} {'因子名称':<20} {'IC均值':<10} {'IR':<8} {'IC>0比例':<10} {'类别':<10}")
        print("-" * 70)
        
        for i, (_, row) in enumerate(df_results.head(top_n).iterrows(), 1):
            print(f"{i:<4} {row['factor_name']:<20} {row['mean_ic']:<10.6f} "
                  f"{row['ir']:<8.4f} {row['ic_positive_ratio']:<10.2%} {row['category']:<10}")
        
        return df_results
    
    def analyze_factor_correlation(self, factor_names: List[str], 
                                  factor_exprs: List[str],
                                  start_date="2019-01-01", end_date="2026-02-26"):
        """
        分析因子相关性
        
        Parameters
        ----------
        factor_names : List[str]
            因子名称列表
        factor_exprs : List[str]
            因子表达式列表
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        pd.DataFrame
            因子相关性矩阵
        """
        print(f"\n{'='*60}")
        print(f"  因子相关性分析")
        print(f"{'='*60}")
        
        # 加载因子数据
        instruments = self.get_valid_instruments(start_date, end_date)
        print(f"股票数量: {len(instruments)}")
        
        try:
            # 加载所有因子数据
            df_factors = self.load_features_safe(
                instruments, factor_exprs, start_date, end_date
            )
            df_factors.columns = factor_names
            
            # 计算相关性（使用最近截面）
            latest_date = df_factors.index.get_level_values("datetime").max()
            try:
                latest_cross = df_factors.xs(latest_date, level="datetime")
            except KeyError:
                # 如果最新日期无数据，使用最近有数据的日期
                dates = df_factors.index.get_level_values("datetime").unique()
                latest_date = dates[-1]
                latest_cross = df_factors.xs(latest_date, level="datetime")
            
            # 计算相关性矩阵
            corr_matrix = latest_cross.corr()
            
            print(f"\n分析日期: {latest_date.date()}")
            print(f"有效股票数: {len(latest_cross)}")
            
            print(f"\n相关性矩阵:")
            print(corr_matrix.round(3))
            
            # 识别高度相关的因子对
            high_corr_pairs = []
            for i in range(len(factor_names)):
                for j in range(i+1, len(factor_names)):
                    corr = abs(corr_matrix.iloc[i, j])
                    if corr > 0.7:
                        high_corr_pairs.append({
                            "factor1": factor_names[i],
                            "factor2": factor_names[j],
                            "correlation": corr_matrix.iloc[i, j]
                        })
            
            if high_corr_pairs:
                print(f"\n高度相关因子对 (|corr| > 0.7):")
                for pair in high_corr_pairs:
                    print(f"  {pair['factor1']} - {pair['factor2']}: {pair['correlation']:.3f}")
            else:
                print(f"\n无高度相关因子对")
            
            return corr_matrix
            
        except Exception as e:
            print(f"[ERROR] 相关性分析失败: {e}")
            return None
    
    def generate_factor_report(self, output_dir="./results/factor_reports"):
        """
        生成因子挖掘报告
        
        Parameters
        ----------
        output_dir : str
            输出目录
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"  生成因子挖掘报告")
        print(f"{'='*60}")
        
        # 挖掘各类别最优因子
        all_results = []
        
        for category in self.factor_library.keys():
            print(f"\n>>> 挖掘类别: {category}")
            results = self.mine_factors(category, top_n=5)
            
            if results is not None:
                results["category"] = category
                all_results.append(results.head(5))
        
        if not all_results:
            print("[WARN] 无挖掘结果")
            return
        
        # 合并结果
        df_all = pd.concat(all_results, ignore_index=True)
        
        # 保存结果
        report_file = output_path / f"factor_mining_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_all.to_csv(report_file, index=False, encoding='utf-8-sig')
        print(f"\n[OK] 报告已保存: {report_file}")
        
        # 可视化
        self._plot_factor_results(df_all, output_path)
        
        return df_all
    
    def _plot_factor_results(self, df_results, output_path):
        """绘制因子挖掘结果图"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. IC均值分布
            ax1 = axes[0, 0]
            colors = plt.cm.Set3(np.linspace(0, 1, len(df_results)))
            bars = ax1.bar(range(len(df_results)), df_results["mean_ic"].abs(), color=colors)
            ax1.set_title("因子 IC 绝对值分布", fontsize=14, fontweight="bold")
            ax1.set_xlabel("因子")
            ax1.set_ylabel("|IC均值|")
            ax1.set_xticks(range(len(df_results)))
            ax1.set_xticklabels(df_results["factor_name"], rotation=45, ha="right")
            
            # 2. IR分布
            ax2 = axes[0, 1]
            ax2.bar(range(len(df_results)), df_results["ir"].abs(), color=colors)
            ax2.set_title("因子 IR 绝对值分布", fontsize=14, fontweight="bold")
            ax2.set_xlabel("因子")
            ax2.set_ylabel("|IR|")
            ax2.set_xticks(range(len(df_results)))
            ax2.set_xticklabels(df_results["factor_name"], rotation=45, ha="right")
            
            # 3. 按类别分组
            ax3 = axes[1, 0]
            category_stats = df_results.groupby("category")["mean_ic"].mean()
            category_stats.plot(kind="bar", ax=ax3, color=plt.cm.Set3(np.linspace(0, 1, len(category_stats))))
            ax3.set_title("各因子类别平均 IC", fontsize=14, fontweight="bold")
            ax3.set_xlabel("类别")
            ax3.set_ylabel("平均 IC")
            ax3.tick_params(axis='x', rotation=45)
            
            # 4. IC>0比例
            ax4 = axes[1, 1]
            ax4.bar(range(len(df_results)), df_results["ic_positive_ratio"], color=colors)
            ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            ax4.set_title("因子 IC>0 比例", fontsize=14, fontweight="bold")
            ax4.set_xlabel("因子")
            ax4.set_ylabel("IC>0 比例")
            ax4.set_xticks(range(len(df_results)))
            ax4.set_xticklabels(df_results["factor_name"], rotation=45, ha="right")
            ax4.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = output_path / "factor_mining_charts.png"
            plt.savefig(chart_file, dpi=300, bbox_inches="tight")
            plt.close()
            
            print(f"[OK] 图表已保存: {chart_file}")
            
        except Exception as e:
            print(f"[WARN] 可视化失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="基于 QLib 的因子挖掘系统")
    parser.add_argument("--category", "-c", default="technical",
                       choices=["technical", "fundamental", "risk", "quality", "sentiment", "all"],
                       help="因子类别")
    parser.add_argument("--start_date", "-s", default="2019-01-01",
                       help="开始日期")
    parser.add_argument("--end_date", "-e", default="2026-02-26",
                       help="结束日期")
    parser.add_argument("--top_n", "-n", type=int, default=10,
                       help="显示最优因子数量")
    parser.add_argument("--report", "-r", action="store_true",
                       help="生成完整报告")
    
    args = parser.parse_args()
    
    # 创建因子挖掘器
    print("\n" + "="*60)
    print("  基于 QLib 的因子挖掘系统")
    print("="*60)
    
    miner = FactorMining()
    
    if args.report:
        # 生成完整报告
        miner.generate_factor_report()
    elif args.category == "all":
        # 测试所有类别
        for category in ["technical", "fundamental", "risk", "quality", "sentiment"]:
            miner.mine_factors(category, args.start_date, args.end_date, args.top_n)
    else:
        # 测试指定类别
        miner.mine_factors(args.category, args.start_date, args.end_date, args.top_n)
    
    print("\n" + "="*60)
    print("  因子挖掘完成")
    print("="*60)


if __name__ == "__main__":
    main()