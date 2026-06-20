# 跨平台运行（Windows / Linux / macOS）

`qlib_quant` 的同一份 `src/` layout 代码树可在 **Windows、Linux、macOS** 上直接运行，无需逐机器改配置。
路径、平台差异、临时目录、文件编码都已收敛到统一层。

## 工作原理（无需手动改路径）

- 所有数据/输出路径都**相对项目根**解析（见 `src/common/paths.py` 与 `src/common/config.py`），
  不依赖当前工作目录，也不写死某台机器的 home。`src/common/configs/paths.yaml`、`src/common/configs/trading.yaml` 只存相对路径。
- 平台识别统一走 `src/common/platform.py`（`is_windows() / is_macos() / is_linux()`），业务代码不散落 `sys.platform` 判断。
- qlib 初始化在 `src/data/qlib_init.py`，Windows 侧避免依赖 POSIX-only 的进程模型。
- 数据/结果/日志目录都按需 `mkdir(parents=True, exist_ok=True)` 自动创建，**新机器无需手动建目录**。
- 临时文件统一走 `utils/platform.temp_dir()`（系统临时目录），不写死 `/tmp`。

### 数据放在仓库外（可选）

默认数据目录是 `<仓库根>/data/qlib_data/cn_data`。若要放到别处（如 Linux 独立数据盘），
设环境变量即可，无需改代码或配置：

- `QLIB_PROJECT_ROOT`：把“项目根”指向别的目录（影响所有相对路径推导）。
- 或在 `src/common/configs/paths.yaml` 的 `data.qlib_data` 填绝对路径。

## Windows 安装

需要 Python 3.9+。`pyqlib` 含 C/Cython 扩展，安装前请准备好编译环境：
**装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)（勾选“使用 C++ 的桌面开发”）**，
或改用 conda 的预编译 `pyqlib`。

```powershell
# 1. 建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装项目依赖（PowerShell 下 .[full] 要加引号，否则被当通配符）
pip install -e ".[full]"

# 3. 设 Tushare token（三选一）
$env:TUSHARE_TOKEN = "your_token_here"        # 仅当前会话
setx TUSHARE_TOKEN "your_token_here"          # 持久化，重开终端生效
# 或写到仓库根的 .env 文件：TUSHARE_TOKEN=your_token_here

# 4. 首次初始化 + 日常更新（自动建目录、拉历史数据、生成 provider）
python main.py update
```

## Windows 每日运行

跨平台入口统一是 `python main.py ...`，不依赖 `.sh`：

```powershell
python main.py run                 # 全流程：更新数据 + 回测验证
# 或分步：
python main.py update              # 只更新数据
python main.py select -s <策略>    # 只选股
python main.py backtest -s <策略> -e qlib
```

用**任务计划程序（Task Scheduler）**做每日自动跑：新建任务，操作设为
`程序: <仓库>\.venv\Scripts\python.exe`、`参数: main.py run`、`起始于: <仓库根>`，
触发器设在交易日收盘后即可。

### 中文输出乱码

`main.py` 已强制 UTF-8 输出。若**直接**运行其它脚本（`scripts/*.py`）时中文乱码，
设 `$env:PYTHONUTF8 = "1"`（让整个 Python 进程用 UTF-8），或用支持 UTF-8 的 Windows Terminal。

## Linux / macOS 安装

```bash
# 一键脚本（Ubuntu/Debian）
bash scripts/setup_new_env.sh

# 或手动
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[full]
export TUSHARE_TOKEN=your_token_here
python main.py update
```

Linux/macOS 仍可用 `scripts/daily_run.sh`（update → select → backtest）；Windows 用上面的 `python main.py run`。

## 跨平台自检

```bash
python -m pytest tests/test_migration_linux.py -q
```

这是 OS 无关的跨平台门：校验无写死的主机路径（`/Users`、`/home`、`\Users\`、盘符、`/tmp`）、
平台识别恰有一个为真、核心模块可导入。Windows 与 Linux 上结果一致。
