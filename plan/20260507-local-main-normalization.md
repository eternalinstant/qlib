# 本地与 main 归一计划

日期：2026-05-07

## 执行结果

执行日期：2026-05-07

- 本地备份分支：`backup_local_main_before_normalization_20260507`
- GitHub 备份分支：`origin/backup_local_main_before_normalization_20260507`
- 归一集成分支：`integrate/main-normalization-20260507`
- GitHub 集成分支：`origin/integrate/main-normalization-20260507`
- 归一基线：`origin/main` at `db08695`
- 备份快照提交：`f041dc3 backup: local state before main normalization 2026-05-07`

已合入集成分支的主题提交：

1. `c1b1486 fix: normalize data updater idempotency`
2. `bff0326 feat: add csi800 universe support and deterministic selection`
3. `c3975f1 feat: add cq10 model signal pipeline`
4. `03b8f59 fix: normalize backtest data and platform paths`

已验证：

- `python3 -m pytest tests/modules/test_precheck.py tests/modules/test_data_processing.py tests/modules/test_updater.py tests/test_migration_linux.py` → `76 passed`
- `python3 -m pytest tests/core/test_universe.py tests/core/test_selection.py` → `34 passed`
- `python3 -m pytest tests/modules/test_feature_pruning.py tests/modules/test_portfolio_overlay.py tests/modules/test_predictive_signal.py` → `38 passed`
- `python3 -m pytest tests/modules/test_backtest.py tests/modules/test_composite.py tests/modules/test_compare.py` → `60 passed`
- `PYTHONPYCACHEPREFIX=/tmp/qlib_pycache python3 -m py_compile ...` → 通过

暂缓合入项保持不变：`csi800_opt` 批量扫参、`push25_cq10_v3_vol_norm`、`push25_cq10_k8d2_csi800`、一次性分析脚本和结果目录。

## 执行前状态

- 执行前本地分支：`main`
- 执行前远端基线：`origin/main`
- 执行前分支关系：本地 `main` 相对 `origin/main` 为 `ahead 1, behind 14`
- 已有云端备份分支：`backup_pre_sync_20260506_010757`
- 当前工作区仍有未提交改动，不能直接把本地 `main` 推到远端 `main`

结论：不能直接做快进合并，也不建议把本地整包覆盖到远端。归一应以 `origin/main` 为干净基线，按主题回放本地能力。

## 归一目标

1. 远端 `main` 保持可运行、可迁移、可测试。
2. 本地已经验证过的数据更新链路进入主线。
3. 本地研究型策略和模型信号链路以实验模块方式进入，不覆盖远端 Alpha158/LGBM 主线。
4. 代码自动识别 macOS/Linux 差异，并通过统一路径、环境、进程和数据目录策略屏蔽差异。
5. 每一批合入后都能通过最小验证集，不把不可复现的本地状态带到云端。

## 分支策略

先保留当前本地 `main` 不再继续堆改动，新建集成分支：

```bash
git fetch origin
git switch -c integrate/main-normalization-20260507 origin/main
```

然后分主题提交：

1. `topic/data-pipeline-sync`
2. `topic/platform-compat`
3. `topic/selection-csi800`
4. `topic/model-signal-cq10`
5. `topic/backtest-vol-norm`

每个 topic 合入前都先跑对应测试，最终再开 PR 或推送集成分支。

## 文件处理原则

### 远端优先

- Linux/bootstrap 相关脚本
- 数据质量守卫脚本
- Alpha158/LGBM 主线配置
- 远端新增测试和迁移检查

### 本地优先

- 已修复的数据更新幂等性逻辑
- `csi800` 股票池支持
- 选股确定性排序
- `modules/modeling/*` 模型信号链路
- `cq10` 的 `csi300 + fixed` 普通变体

### 暂缓合入

- `config/models/csi800_opt/*` 大批量扫参配置
- `push25_cq10_v3_vol_norm`，等 `vol_norm` 回测引擎单独合入后再启用
- `push25_cq10_k8d2_csi800`，等 `csi800` 主线支持合入后再启用
- 一次性分析脚本和结果目录

## 自动识别并屏蔽 macOS/Linux 差异

### 统一平台识别

新增或整理统一平台能力，建议集中在 `utils/platform.py` 或 `core/platform.py`：

- 使用 `platform.system()` 判断 `Darwin` / `Linux`
- 使用 `platform.machine()` 判断 `arm64` / `x86_64`
- 提供 `is_macos()`、`is_linux()`、`project_root()`、`runtime_profile()` 等函数
- 不允许业务代码散落 `sys.platform == "darwin"` 判断

### 统一路径解析

所有路径通过 `pathlib.Path` 和统一路径层推导：

- 继续收敛到 `modules/data/paths.py`
- 禁止硬编码 `/Users/sxt`、`/home/...`、`/opt/...`
- 配置文件只存相对项目根或环境变量路径
- 数据目录从 `config/paths.yaml`、`QLIB_*` 环境变量和项目根推导
- `~` 和 `$VAR` 展开只在配置层做一次

验收要求：

```bash
rg "/Users/sxt|/home/|\\\\Users\\\\" --glob "*.py" --glob "*.sh" --glob "*.yaml"
python3 -m pytest tests/test_migration_linux.py
```

### 屏蔽 Shell 差异

Shell 脚本必须：

- 用脚本自身位置解析 `PROJECT_ROOT`
- 不依赖当前工作目录
- 不依赖 macOS 专有命令作为唯一实现
- Linux/macOS 分支放在同一个 helper 函数里

示例规范：

```bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
```

### 屏蔽代理和网络差异

本地 macOS 常见 `127.0.0.1:7890` 代理不能污染云端或 Linux CI：

- 网络访问统一读取环境变量
- 默认不强制设置代理
- 如果发现代理指向 `127.0.0.1`，只在本机交互环境使用
- CI/Linux 环境下应允许禁用代理
- 数据更新脚本应输出当前是否启用代理，但不打印 token

### 屏蔽文件系统差异

需要防止 macOS 默认大小写不敏感、Linux 大小写敏感带来的问题：

- instrument 文件名统一小写，例如 `sh600000.parquet`
- 股票代码内部格式统一通过 helper 转换
- 所有 parquet/bin 路径通过函数生成，不手拼路径
- 测试中覆盖 `SH600000`、`sh600000`、`600000.SH` 三种输入

### 屏蔽系统能力差异

数据和测试流程不要依赖 macOS 专有能力：

- 不依赖 `sysctl` 结果作为逻辑条件
- 不依赖 Keychain、LaunchAgent、Finder、osascript
- CPU/线程数通过 `os.cpu_count()` 或配置上限控制
- 临时目录通过 `tempfile` 或项目 `tmp` 目录获取

## 实施阶段

### 阶段 0：冻结当前状态

目标：确认所有本地改动都有恢复点。

动作：

```bash
git status --short --branch
git branch backup/local-main-before-normalization-20260507
```

若需要保存当前未提交状态，单独提交到备份分支，不直接污染集成分支。

### 阶段 1：同步数据链路

合入范围：

- `modules/data/*`
- `scripts/build_qlib_data.py`
- `scripts/data_quality_guard.py`
- `scripts/validate_data.py`
- `scripts/verify_alpha158_data.py`
- `tests/modules/test_precheck.py`
- `tests/modules/test_data_processing.py`
- `tests/modules/test_updater.py`
- `tests/test_migration_linux.py`

必须保留的本地修复：

- `.download_state.json` 缺失时从 `raw_data` 自动重建
- `index_daily` 先于 `daily_basic / adj_factor` 更新
- `convert_to_qlib()` 末尾最终 provider repair
- 当天交易日早于数据发布时间时不误触发市场更新
- `csi800` 数据预检支持

验证：

```bash
python3 -m pytest tests/modules/test_precheck.py tests/modules/test_data_processing.py tests/modules/test_updater.py tests/test_migration_linux.py
```

### 阶段 2：平台兼容层

合入范围：

- 统一路径解析
- 统一平台识别
- Shell 脚本动态项目根
- 迁移兼容测试

验收标准：

- macOS 本地可跑
- Linux 环境不出现硬编码本机路径
- 测试不依赖 macOS 专有命令
- 所有数据路径可由配置或环境变量覆盖

### 阶段 3：选股与股票池归一

合入范围：

- `core/selection.py`
- `core/universe.py`
- `core/strategy.py`
- 对应测试

目标：

- 保留远端已有 `csi300` 行为
- 增加本地 `csi800` 支持
- 保留选股确定性排序，避免同分结果漂移
- 不改变默认策略输出

验证：

```bash
python3 -m pytest tests/core/test_universe.py tests/core/test_selection.py
```

### 阶段 4：cq10 普通版模型信号链路

先只合入 `csi300 + fixed` 的普通变体。

合入范围：

- `modules/modeling/*`
- `scripts/generate_model_scores.py`
- `scripts/backtest_model_signal.py`
- `config/models/push25_cq10_k8d2_very_tight.yaml`
- 必要的 README 说明

暂不启用：

- `push25_cq10_v3_vol_norm`
- `push25_cq10_k8d2_csi800`
- `config/models/csi800_opt/*`

验证：

```bash
python3 scripts/generate_model_scores.py --config config/models/push25_cq10_k8d2_very_tight.yaml
python3 scripts/backtest_model_signal.py --config config/models/push25_cq10_k8d2_very_tight.yaml --engine qlib
```

### 阶段 5：回测引擎差异归一

合入范围：

- `modules/backtest/qlib_engine.py`
- `modules/backtest/pybroker_engine.py`
- `modules/backtest/compare.py`
- `modules/backtest/composite.py`
- 回测相关测试

拆分原则：

- 先合收益率口径和 provider/raw_data 双源修复
- 再合权重级手续费
- 最后合 `vol_norm`

`vol_norm` 进入后，才允许启用 `push25_cq10_v3_vol_norm`。

## 最终验收

基础验证：

```bash
python3 -m pytest tests/modules/test_precheck.py tests/modules/test_data_processing.py tests/modules/test_updater.py tests/test_migration_linux.py
python3 -m pytest tests/core/test_universe.py tests/core/test_selection.py
python3 -m pytest tests/modules/test_backtest.py
```

数据验证：

```bash
python3 scripts/validate_data.py
python3 scripts/verify_alpha158_data.py
```

策略验证：

```bash
python3 scripts/generate_model_scores.py --config config/models/push25_cq10_k8d2_very_tight.yaml
python3 scripts/backtest_model_signal.py --config config/models/push25_cq10_k8d2_very_tight.yaml --engine qlib
```

跨平台验证：

```bash
python3 -m pytest tests/test_migration_linux.py
rg "/Users/sxt|/home/|\\\\Users\\\\" --glob "*.py" --glob "*.sh" --glob "*.yaml"
```

## 风险与处理

- 风险：本地数据文件很大，不适合进入 Git。
  处理：只提交脚本、测试和配置，不提交 `data/`、`results/`、`backups/`。

- 风险：`cq10` 配置上云后缺少模型链路。
  处理：`cq10` 普通版必须和 `modules/modeling/*` 及两个入口脚本一起合入。

- 风险：`csi800` 配置先于股票池支持进入主线。
  处理：`push25_cq10_k8d2_csi800.yaml` 暂缓，等 `core/universe.py` 和数据预检支持合入后再启用。

- 风险：macOS 本地代理影响 Linux/CI。
  处理：网络代理只通过环境变量显式启用，并在 CI/Linux 默认禁用本机回环代理。

## 完成定义

满足以下条件才认为归一完成：

1. 集成分支从 `origin/main` 创建，不依赖当前脏工作区。
2. 数据链路、平台兼容、选股股票池、模型信号、回测引擎分主题提交。
3. macOS 和 Linux 路径差异由代码自动识别或统一路径层屏蔽。
4. `tests/test_migration_linux.py` 和数据更新相关测试通过。
5. `cq10` 普通版能在集成分支生成分数并完成回测。
6. 远端 `main` 不接收未验证的扫参配置和一次性实验脚本。
