"""QMT 环境配置"""

# MiniQMT 交易路径：必须指向 userdata_mini 目录（不是安装根目录），
# 否则 XtQuantTrader.connect() 返回 -1、查不到账户
QMT_PATH = r"C:\Users\Administrator\国金QMT交易端模拟\userdata_mini"

# 数据服务地址
MINI_QMT_IP = "127.0.0.1"
MINI_QMT_PORT = 58610

# 交易会话 ID（多进程时需递增）
SESSION_ID = 123456

# 账户 ID（留空则运行时自动检测）
ACCOUNT_ID = ""

# 默认复权方式: "none", "front", "back", "front_ratio", "back_ratio"
DEFAULT_DIVIDEND_TYPE = "front"

# Python 解释器路径
PYTHON_EXE = r"C:\Python312\python.exe"
