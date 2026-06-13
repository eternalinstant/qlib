"""数据层（data）——行情/股票池/因子源 + 数据更新管道。

五层架构第 4 层（app → strategy → engine → data → common）。
注：本目录同时是数据存储根（qlib_data/ tushare/ selections/ 等子目录被 .gitignore），
Python 代码模块与数据子目录共存；import 仅解析 .py 模块，不触及数据子目录。
"""
