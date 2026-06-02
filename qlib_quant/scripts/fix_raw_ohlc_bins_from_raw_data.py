#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qlib provider 一致性修复脚本

修复两类问题：
1. close.day.bin 超过日历长度 → 戽断到到日历末尾
2 OHLCVA.bin 超过日历长度 → 截断到到日历末尾
3. close 与 OHLCVA end_idx 不一致 → 从 raw_data 全量重建
 4. OHLCV 尾部大量 0 值 → 从 raw_data 全量重建
 5. 字段间 start_idx 不对齐 → 统一所有字段
 6. splice-point 比例导致价格偏移

 7. 0 值破坏 OHLC 约束性
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

