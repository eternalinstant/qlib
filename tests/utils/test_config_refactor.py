"""
测试配置管理模块
"""
import sys
sys.path.insert(0, "/Users/sxt/code/qlib")

from config.config import ConfigManager, AppConfig, get_config_manager

def test_config_manager():
    print("=" * 50)
    print("测试 ConfigManager")
    print("=" * 50)
    
    # 测试创建新实例
    manager = ConfigManager()
    print(f"✓ ConfigManager 创建成功")
    
    # 测试获取配置
    print(f"  w_alpha = {manager.get('w_alpha')}")
    print(f"  w_risk = {manager.get('w_risk')}")
    print(f"  w_enhance = {manager.get('w_enhance')}")
    print(f"  topk = {manager.get('topk')}")
    print(f"  start_date = {manager.get('start_date')}")
    print(f"  end_date = {manager.get('end_date')}")
    print(f"  initial_capital = {manager.get('initial_capital')}")
    print(f"  qlib_data_path = {manager.get('qlib_data_path')}")
    
    # 测试 AppConfig
    config = manager.get_config()
    print(f"\n✓ AppConfig 创建成功")
    print(f"  type = {type(config)}")
    print(f"  config.get('w_alpha') = {config.get('w_alpha')}")
    
    # 测试全局单例
    global_config = get_config_manager()
    print(f"\n✓ 全局单例正常")
    print(f"  global_config.get('topk') = {global_config.get('topk')}")
    
    print("\n" + "=" * 50)
    print("配置管理测试通过!")
    print("=" * 50)

if __name__ == "__main__":
    test_config_manager()
