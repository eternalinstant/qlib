#!/usr/bin/env python3
"""
数据质量检查与自动修复主控脚本

核心流程（最多 5 轮）:
  1. run_full_check()     — 复用 DataValidator 全量检查
  2. diagnose()           — 根据检查结果诊断根因
  3. execute_fix()        — 调用修复函数
  4. 回到步骤 1 重验
  5. 全部通过 → 退出
  6. 5 轮未通过 → 写 Q&A 文件 → 退出

用法:
  python scripts/data_quality_guard.py
  python scripts/data_quality_guard.py --max-retries 3
"""

import contextlib
import gc
import io
import os
import subprocess
import sys
import time
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Reuse ANSI codes and status constants from validate_data
from scripts.validate_data import (
    C_RESET, C_RED, C_GREEN, C_YELLOW, C_CYAN, C_BOLD,
    STATUS_PASS, STATUS_FAIL, STATUS_WARN, STATUS_SKIP,
)


# ── FixAction types ─────────────────────────────────────────────────────

class FixType(Enum):
    FIX_DOWNLOAD = "download"             # 重新下载缺失 tushare 文件
    FIX_UPDATE = "update"                 # 增量更新 tushare 数据
    FIX_REBUILD_RAW = "rebuild_raw"       # 重建 raw_data
    FIX_REBUILD_BINS = "rebuild_bins"     # 重建前复权 bin 文件
    FIX_REBUILD_CALENDAR = "rebuild_cal"  # 重建日历
    FIX_REPAIR_PROVIDER = "repair_prov"   # 修复 price provider
    FIX_REGENERATE_FACTOR = "regen_factor"  # 重建 factor_data.parquet
    FIX_REBUILD_INSTRUMENTS = "rebuild_inst"  # 重建 instruments/all.txt


@dataclass
class FixAction:
    fix_type: FixType
    description: str
    targets: list = field(default_factory=list)


@dataclass
class CheckReport:
    results: list = field(default_factory=list)

    @property
    def _counts(self) -> Counter:
        return Counter(r["status"] for r in self.results)

    @property
    def fail_count(self) -> int:
        return self._counts[STATUS_FAIL]

    @property
    def warn_count(self) -> int:
        return self._counts[STATUS_WARN]

    @property
    def pass_count(self) -> int:
        return self._counts[STATUS_PASS]

    @property
    def skip_count(self) -> int:
        return self._counts[STATUS_SKIP]

    def all_pass(self) -> bool:
        return self.fail_count == 0

    def failed_checks(self) -> list:
        return [r for r in self.results if r["status"] == STATUS_FAIL]

    def warning_checks(self) -> list:
        return [r for r in self.results if r["status"] == STATUS_WARN]


# ── DataQualityGuard ───────────────────────────────────────────────────

class DataQualityGuard:
    def __init__(self, max_retries: int = 5):
        self.max_retries = max_retries
        self.tushare_dir = PROJECT_ROOT / "data" / "tushare"
        self.qlib_dir = PROJECT_ROOT / "data" / "qlib_data" / "cn_data"
        self.raw_dir = PROJECT_ROOT / "data" / "qlib_data" / "raw_data"
        self.qa_dir = PROJECT_ROOT / "Q&A"
        self.qa_dir.mkdir(parents=True, exist_ok=True)
        self._fix_history: list = []

    # ── Phase 1: Full check (reuse DataValidator) ──────────────────────

    def run_full_check(self) -> CheckReport:
        """全量数据检查，复用 validate_data.py 的 DataValidator"""
        from scripts.validate_data import DataValidator

        logger.info("=" * 60)
        logger.info("阶段一：全量数据检查")
        logger.info("=" * 60)

        validator = DataValidator(data_root=str(self.qlib_dir.parent))

        with contextlib.redirect_stdout(io.StringIO()):
            results = validator.run_all()

        validator.print_report()
        validator.save_json()

        report = CheckReport(results=results)

        logger.info(
            f"检查结果: PASS={report.pass_count}, FAIL={report.fail_count}, "
            f"WARN={report.warn_count}, SKIP={report.skip_count}"
        )

        return report

    # ── Phase 2: Diagnose ──────────────────────────────────────────────

    def diagnose(self, report: CheckReport) -> list:
        """根据检查结果诊断根因，返回修复动作列表"""
        logger.info("=" * 60)
        logger.info("阶段二：诊断根因")
        logger.info("=" * 60)

        actions = []
        for check in report.failed_checks():
            actions.extend(self._diagnose_check(check, is_fail=True))
        for check in report.warning_checks():
            actions.extend(self._diagnose_check(check, is_fail=False))

        # 去重
        seen = set()
        unique_actions = []
        for a in actions:
            key = (a.fix_type, frozenset(a.targets) if a.targets else None)
            if key not in seen:
                seen.add(key)
                unique_actions.append(a)

        if unique_actions:
            logger.info(f"诊断完成，发现 {len(unique_actions)} 个修复动作:")
            for i, action in enumerate(unique_actions, 1):
                logger.info(f"  {i}. [{action.fix_type.value}] {action.description}")
        else:
            logger.info("诊断完成，无自动修复动作")

        return unique_actions

    def _diagnose_check(self, check: dict, is_fail: bool) -> list:
        """分析单条检查结果"""
        actions = []
        name = check["check"]
        status = check["status"]
        data = check.get("data", {})

        # Tushare 源文件缺失
        if "Tushare 源文件完整性" in name:
            missing = data.get("missing", [])
            if missing:
                actions.append(FixAction(
                    FixType.FIX_DOWNLOAD,
                    f"下载缺失的 Tushare 文件: {missing}",
                    targets=missing,
                ))

        # Daily_basic 覆盖不足
        if "Daily_basic" in name and "股票/日期覆盖" in name:
            low_days = data.get("low_coverage_days", 0)
            if low_days > 0:
                actions.append(FixAction(
                    FixType.FIX_UPDATE,
                    f"增量更新 daily_basic ({low_days} 天覆盖不足)",
                    targets=["daily_basic.parquet"],
                ))

        # Adj_factor 覆盖不足
        if "Adj_factor 覆盖率" in name:
            low = data.get("low_coverage_stocks", 0)
            if low > 0:
                actions.append(FixAction(
                    FixType.FIX_UPDATE,
                    f"增量更新 adj_factor ({low} 只股票覆盖不足)",
                    targets=["adj_factor.parquet"],
                ))

        # 财务数据缺失
        if "财务数据完整性" in name:
            files = data.get("files", {})
            missing = [k for k, v in files.items() if v.get("status") != "ok"]
            if missing:
                actions.append(FixAction(
                    FixType.FIX_UPDATE,
                    f"更新财务数据: {missing}",
                    targets=missing,
                ))

        # instruments 日期不一致
        if "instruments 日期" in name and is_fail:
            actions.append(FixAction(
                FixType.FIX_REBUILD_INSTRUMENTS,
                "重建 instruments/all.txt（日期与 raw_data 不一致）",
            ))

        # Bin 文件完整性问题
        if "Bin 文件完整性" in name and is_fail:
            truncated = data.get("truncated_bins", 0)
            missing_fields = data.get("missing_field_dirs", 0)
            if truncated > 0:
                actions.append(FixAction(
                    FixType.FIX_REPAIR_PROVIDER,
                    f"修复 {truncated} 个超范围 bin 文件",
                ))
            if missing_fields > 0:
                actions.append(FixAction(
                    FixType.FIX_REBUILD_BINS,
                    f"重建 {missing_fields} 个缺字段股票的 bin 文件",
                ))

        # 前复权不一致
        if "复权一致性" in name and status != STATUS_PASS:
            mismatches = data.get("total_mismatches", 0)
            if mismatches > 0:
                actions.append(FixAction(
                    FixType.FIX_REBUILD_BINS,
                    f"重建前复权 bin（{mismatches} 条偏差 >5%）",
                ))

        # 复权后 OHLC 违反
        if "复权后 OHLC" in name and is_fail:
            actions.append(FixAction(
                FixType.FIX_REBUILD_BINS,
                "重建前复权 bin（复权后 OHLC 逻辑违反）",
            ))

        # Factor 空值率过高
        if "Factor 空值率" in name and status == STATUS_FAIL:
            all_null = data.get("all_null_factors", [])
            if all_null:
                actions.append(FixAction(
                    FixType.FIX_REGENERATE_FACTOR,
                    f"重建 factor_data.parquet（{len(all_null)} 个因子全空）",
                ))

        # 日历连续性问题
        if "日历连续性" in name and is_fail:
            actions.append(FixAction(
                FixType.FIX_REBUILD_CALENDAR,
                "重建交易日历（连续性问题）",
            ))

        # 数据新鲜度
        if "数据新鲜度" in name:
            days = data.get("days_since_calendar", 999)
            if days > 7:
                actions.append(FixAction(
                    FixType.FIX_UPDATE,
                    f"增量更新数据（数据已过期 {days} 天）",
                    targets=["daily_basic.parquet", "adj_factor.parquet"],
                ))

        return actions

    # ── Phase 3: Execute fixes ─────────────────────────────────────────

    def execute_fix(self, actions: list) -> bool:
        """执行修复动作列表"""
        logger.info("=" * 60)
        logger.info("阶段三：执行修复")
        logger.info("=" * 60)

        all_ok = True
        for action in actions:
            logger.info(f"执行: [{action.fix_type.value}] {action.description}")
            self._fix_history.append({
                "time": datetime.now().isoformat(),
                "type": action.fix_type.value,
                "description": action.description,
            })

            try:
                handler = self._dispatch_table.get(action.fix_type)
                if handler is None:
                    logger.warning(f"  -> 未知修复类型: {action.fix_type}")
                    all_ok = False
                    continue
                ok = handler(action.targets) if action.targets else handler()
                if ok:
                    logger.info("  -> 成功")
                else:
                    logger.warning("  -> 失败")
                    all_ok = False
            except Exception as e:
                logger.error(f"  -> 异常: {e}")
                all_ok = False

            gc.collect()

        return all_ok

    # Dict dispatch for fix types
    _dispatch_table: dict = {}  # populated after method definitions

    def _fix_download(self, targets: list) -> bool:
        """重新下载缺失的 tushare 文件"""
        self._ensure_tushare_token()

        from modules.data.tushare_downloader import TushareDownloader

        dl = TushareDownloader()
        dl.MAX_WORKERS = 4

        file_to_method = {
            "daily_basic.parquet": dl.download_daily_basic,
            "adj_factor.parquet": dl.download_adj_factor,
            "income.parquet": dl.download_income,
            "balancesheet.parquet": dl.download_balancesheet,
            "cashflow.parquet": dl.download_cashflow,
            "fina_indicator.parquet": dl.download_financial_indicator,
            "index_daily.parquet": dl.download_index_daily,
            "index_weight.parquet": dl.download_index_weight,
            "namechange.parquet": dl.download_namechange,
        }

        return self._run_target_map(targets, file_to_method, "下载")

    def _fix_update(self, targets: list) -> bool:
        """增量更新 tushare 数据"""
        self._ensure_tushare_token()

        from scripts.data_update import (
            update_daily_basic,
            update_stock_basic,
            update_fina_indicator,
            update_income,
            update_cashflow,
            update_balancesheet,
        )

        update_map = {
            "daily_basic.parquet": update_daily_basic,
            "stock_basic.csv": update_stock_basic,
            "fina_indicator.parquet": update_fina_indicator,
            "income.parquet": update_income,
            "cashflow.parquet": update_cashflow,
            "balancesheet.parquet": update_balancesheet,
        }

        return self._run_target_map(targets, update_map, "更新")

    @staticmethod
    def _run_target_map(targets: list, mapping: dict, label: str) -> bool:
        """Generic: run each target's callable, log failures."""
        all_ok = True
        for target in targets:
            fn = mapping.get(target)
            if fn:
                try:
                    fn()
                except Exception as e:
                    logger.error(f"{label} {target} 失败: {e}")
                    all_ok = False
            else:
                logger.warning(f"未知文件: {target}")
                all_ok = False
        return all_ok

    def _make_converter(self):
        """Create a TushareToQlibConverter with standard paths."""
        from modules.data.tushare_to_qlib import TushareToQlibConverter
        return TushareToQlibConverter(
            tushare_dir=str(self.tushare_dir),
            qlib_dir=str(self.qlib_dir),
        )

    def _fix_rebuild_raw(self) -> bool:
        """重建 raw_data"""
        logger.info("重建 raw_data（调用 build_qlib_data.py step 2）...")
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_qlib_data.py"),
                "build",
                "--skip-download",
                "--workers", "4",
            ],
            capture_output=False,
        )
        return result.returncode == 0

    def _fix_rebuild_bins(self, _targets=None) -> bool:
        """重建前复权 bin 文件"""
        logger.info("重建前复权 bin 文件...")
        converter = self._make_converter()
        written = converter.build_adjusted_bins_batched(batch_size=1000)
        logger.info(f"写入 {written} 只股票的 bin 文件")
        return written > 0

    def _fix_rebuild_calendar(self, _targets=None) -> bool:
        """重建交易日历"""
        db_path = self.tushare_dir / "daily_basic.parquet"
        if not db_path.exists():
            logger.error("缺少 daily_basic.parquet，无法重建日历")
            return False

        db = pd.read_parquet(db_path, columns=["trade_date"])
        dates = pd.to_datetime(db["trade_date"].unique(), format="%Y%m%d").sort_values()

        cal_file = self.qlib_dir / "calendars" / "day.txt"
        cal_file.parent.mkdir(parents=True, exist_ok=True)
        dates.strftime("%Y-%m-%d").to_series().to_csv(
            cal_file, index=False, header=False
        )
        logger.info(f"日历重建: {len(dates)} 个交易日 ({dates.min().date()} ~ {dates.max().date()})")
        return True

    def _fix_repair_provider(self, _targets=None) -> bool:
        """修复 price provider"""
        logger.info("修复 price provider...")
        converter = self._make_converter()
        stats = converter.repair_price_provider()
        logger.info(f"修复统计: {stats}")
        return True

    def _fix_regenerate_factor(self, _targets=None) -> bool:
        """重建 factor_data.parquet"""
        logger.info("重建 factor_data.parquet...")
        converter = self._make_converter()
        df = converter.convert()
        if df is not None:
            converter.save(df)
            del df
            gc.collect()
            return True
        return False

    def _fix_rebuild_instruments(self, _targets=None) -> bool:
        """重建 instruments/all.txt"""
        logger.info("重建 instruments/all.txt...")
        inst_file = self.qlib_dir / "instruments" / "all.txt"
        inst_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.raw_dir.exists():
            logger.error("raw_data 目录不存在")
            return False

        lines = []
        for raw_path in sorted(self.raw_dir.glob("*.parquet")):
            if raw_path.name.startswith("."):
                continue
            try:
                df = pd.read_parquet(raw_path, columns=["date"])
                if df.empty:
                    continue
                dates = pd.to_datetime(df["date"], errors="coerce").dropna()
                if dates.empty:
                    continue
                lines.append(f"{raw_path.stem}\t{dates.min():%Y-%m-%d}\t{dates.max():%Y-%m-%d}")
            except Exception:
                continue

        inst_file.write_text("\n".join(lines) + "\n")
        logger.info(f"instruments/all.txt: {len(lines)} 只股票")
        return True

    # Wire dispatch table after all methods are defined
    _dispatch_table = {
        FixType.FIX_DOWNLOAD: _fix_download,
        FixType.FIX_UPDATE: _fix_update,
        FixType.FIX_REBUILD_RAW: _fix_rebuild_raw,
        FixType.FIX_REBUILD_BINS: _fix_rebuild_bins,
        FixType.FIX_REBUILD_CALENDAR: _fix_rebuild_calendar,
        FixType.FIX_REPAIR_PROVIDER: _fix_repair_provider,
        FixType.FIX_REGENERATE_FACTOR: _fix_regenerate_factor,
        FixType.FIX_REBUILD_INSTRUMENTS: _fix_rebuild_instruments,
    }

    # ── Q&A report ─────────────────────────────────────────────────────

    def _write_qa(self, attempt: int, report: CheckReport, reason: str):
        """写入 Q&A 人工介入报告"""
        today = datetime.now()
        qa_path = self.qa_dir / f"{today:%Y%m%d}.md"

        lines = [
            f"\n# 数据质量检查问题报告 - {today:%Y-%m-%d %H:%M:%S}\n",
            f"\n## 运行情况\n",
            f"- 当前轮次: 第 {attempt} / {self.max_retries} 轮\n",
            f"- 终止原因: {reason}\n",
            f"\n## 检查结果汇总\n",
            f"- PASS: {report.pass_count}\n",
            f"- FAIL: {report.fail_count}\n",
            f"- WARN: {report.warn_count}\n",
            f"- SKIP: {report.skip_count}\n",
            f"\n## 发现的问题\n",
            f"\n### FAIL 项\n",
        ]
        for check in report.failed_checks():
            lines.append(f"- **{check['check']}**: {check['detail']}\n")

        lines.append("\n### WARN 项\n")
        for check in report.warning_checks():
            lines.append(f"- **{check['check']}**: {check['detail']}\n")

        if self._fix_history:
            lines.append("\n## 尝试过的修复\n")
            for fix in self._fix_history:
                lines.append(f"- [{fix['time']}] [{fix['type']}] {fix['description']}\n")

        lines.extend([
            "\n## 建议\n",
            "\n1. 检查 Tushare Token 是否有效: `echo $TUSHARE_TOKEN`\n",
            "2. 检查网络连接和 Tushare API 状态\n",
            "3. 查看上方 FAIL 项的详细信息，手动修复\n",
            "4. 修复后重新运行: `python scripts/data_quality_guard.py`\n",
            "\n---\n\n",
        ])

        with open(qa_path, "a") as f:
            f.writelines(lines)

        logger.info(f"Q&A 报告已写入: {qa_path}")

    # ── Helpers ────────────────────────────────────────────────────────

    def _ensure_tushare_token(self):
        """确保 Tushare Token 已设置"""
        token = os.environ.get("TUSHARE_TOKEN")
        if not token:
            env_sh = PROJECT_ROOT / "env.sh"
            if env_sh.exists():
                with open(env_sh) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("export TUSHARE_TOKEN="):
                            token = line.split("=", 1)[1].strip().strip('"').strip("'")
                            os.environ["TUSHARE_TOKEN"] = token
                            break

        if not token:
            raise EnvironmentError(
                "缺少 TUSHARE_TOKEN 环境变量。请运行: source env.sh"
            )

    # ── Main loop ──────────────────────────────────────────────────────

    def run(self) -> bool:
        """主循环：检查 -> 诊断 -> 修复 -> 重验"""
        start_time = time.time()

        print(f"\n{C_BOLD}{'='*70}{C_RESET}")
        print(f"{C_BOLD}  数据质量检查与自动修复{C_RESET}")
        print(f"{C_BOLD}  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C_RESET}")
        print(f"{C_BOLD}  最大重试: {self.max_retries} 轮{C_RESET}")
        print(f"{C_BOLD}{'='*70}{C_RESET}\n")

        for attempt in range(1, self.max_retries + 1):
            print(f"\n{C_CYAN}{'─'*70}{C_RESET}")
            print(f"{C_CYAN}  第 {attempt}/{self.max_retries} 轮{C_RESET}")
            print(f"{C_CYAN}{'─'*70}{C_RESET}\n")

            # Phase 1: Check
            report = self.run_full_check()

            if report.all_pass():
                elapsed = time.time() - start_time
                print(f"\n{C_GREEN}{C_BOLD}{'='*70}{C_RESET}")
                print(f"{C_GREEN}{C_BOLD}  全部检查通过！{C_RESET}")
                print(f"{C_GREEN}{C_BOLD}  PASS={report.pass_count}, WARN={report.warn_count}, "
                      f"SKIP={report.skip_count}{C_RESET}")
                print(f"{C_GREEN}{C_BOLD}  总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min){C_RESET}")
                print(f"{C_GREEN}{C_BOLD}{'='*70}{C_RESET}\n")
                return True

            # Phase 2: Diagnose
            actions = self.diagnose(report)
            if not actions:
                self._write_qa(attempt, report, "无法自动诊断根因")
                elapsed = time.time() - start_time
                print(f"\n{C_YELLOW}无法自动诊断，Q&A 报告已生成。{C_RESET}")
                print(f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")
                return False

            # Phase 3: Fix
            success = self.execute_fix(actions)
            if not success:
                logger.warning("部分修复动作失败，将进入下一轮检查")

            gc.collect()

        # 超过最大重试次数
        self._write_qa(self.max_retries, report, "达到最大重试次数")
        elapsed = time.time() - start_time
        print(f"\n{C_RED}{C_BOLD}达到最大重试次数 ({self.max_retries})，Q&A 报告已生成。{C_RESET}")
        print(f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")
        return False


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="数据质量检查与自动修复主控脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="最大重试轮数 (默认 5)",
    )
    args = parser.parse_args()

    guard = DataQualityGuard(max_retries=args.max_retries)
    ok = guard.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
