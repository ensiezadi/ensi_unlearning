#!/usr/bin/env python3
"""
PLT_NET_Mini 数据集测试脚本
验证数据集是否可以正确加载和使用
"""

import os
import sys

def test_imports():
    """测试导入"""
    print("1. 测试模块导入...")
    try:
        from datasets.plt_net_mini import PLTNetMini, get_plt_net_mini_classes, select_random_forget_classes
        from forget_cls import all_ds, forget_classes_all
        print("✅ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_dataset_config():
    """测试数据集配置"""
    print("\n2. 测试数据集配置...")
    try:
        from forget_cls import all_ds, forget_classes_all
        
        # 检查PLTNetMini是否在数据集列表中
        if 'PLTNetMini' not in all_ds:
            print("❌ PLTNetMini 不在数据集列表中")
            return False
        
        # 检查遗忘类别是否配置
        if 'PLTNetMini' not in forget_classes_all:
            print("❌ PLTNetMini 遗忘类别未配置")
            return False
        
        forget_classes = forget_classes_all['PLTNetMini']
        print(f"✅ 配置检查通过，遗忘类别: {forget_classes}")
        return True
    except Exception as e:
        print(f"❌ 配置检查失败: {e}")
        return False

def test_class_functions():
    """测试类别函数"""
    print("\n3. 测试类别函数...")
    try:
        from datasets.plt_net_mini import get_plt_net_mini_classes, select_random_forget_classes
        
        all_classes = get_plt_net_mini_classes()
        print(f"✅ 总类别数: {len(all_classes)}")
        
        random_classes = select_random_forget_classes(3)
        print(f"✅ 随机选择3个类别: {random_classes}")
        return True
    except Exception as e:
        print(f"❌ 类别函数测试失败: {e}")
        return False

def test_config_file():
    """测试配置文件"""
    print("\n4. 测试配置文件...")
    config_path = "configs/datasets/plt_net_mini.yaml"
    if os.path.exists(config_path):
        print(f"✅ 配置文件存在: {config_path}")
        return True
    else:
        print(f"❌ 配置文件不存在: {config_path}")
        return False

def main():
    """主测试函数"""
    print("=== PLT_NET_Mini 数据集集成测试 ===\n")
    
    tests = [
        test_imports,
        test_dataset_config, 
        test_class_functions,
        test_config_file
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== 测试结果: {passed}/{total} 通过 ===")
    
    if passed == total:
        print("🎉 所有测试通过！PLT_NET_Mini 数据集已成功集成")
        print("\n下一步可以运行:")
        print("  python main.py --run_ds PLTNetMini --backbone_arch RN50")
        print("  或者直接运行: run_plt_net_mini.bat")
    else:
        print("❌ 部分测试失败，请检查配置")
        sys.exit(1)

if __name__ == "__main__":
    main()