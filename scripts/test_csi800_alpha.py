"""测试 csi800 instrument list 能否加载 alpha158 特征"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.qlib_init import init_qlib
init_qlib()
from qlib.data import D
from core.universe import get_universe_instruments

# 获取 csi800 全期股票列表
insts = get_universe_instruments(start_date='2019-01-01', end_date='2026-04-15', universe='csi800')
print(f"csi800 instruments: {len(insts)}")

# 用训练配置中的实际 alpha158 表达式
fields = ["Ref($close, 20)/$close", "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 20)"]

# 分批测试：每次100只
batch_size = 100
for i in range(0, len(insts), batch_size):
    batch = insts[i:i+batch_size]
    try:
        df = D.features(batch, fields, start_time='2019-01-01', end_time='2026-04-15', freq='day')
        print(f"Batch {i//batch_size + 1} ({i}-{i+len(batch)-1}): OK, shape={df.shape}")
    except Exception as e:
        print(f"Batch {i//batch_size + 1} ({i}-{i+len(batch)-1}): FAILED - {e}")
        # 二分查找出问题的股票
        lo, hi = 0, len(batch)
        while lo < hi - 1:
            mid = (lo + hi) // 2
            try:
                D.features(batch[lo:mid], fields, start_time='2019-01-01', end_time='2026-04-15', freq='day')
                lo = mid
            except:
                hi = mid
        print(f"  Problem stock: {batch[lo]}")
        # 验证
        try:
            D.features([batch[lo]], fields, start_time='2019-01-01', end_time='2026-04-15', freq='day')
            print(f"  Single stock OK - issue is cross-stock")
        except Exception as e2:
            print(f"  Single stock FAIL: {e2}")
        break
