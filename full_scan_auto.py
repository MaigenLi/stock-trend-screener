#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全市场股票扫描 - 自动版本（无需交互）
"""

import os
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试多种导入方式
try:
    # 方法1: 从core目录导入
    from core.trend_volume_screener import TrendVolumeScreener
    print("✅ 从core目录导入成功")
except ImportError:
    try:
        # 方法2: 直接导入
        from trend_volume_screener import TrendVolumeScreener
        print("✅ 直接导入成功")
    except ImportError:
        # 方法3: 添加路径后导入
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
        from trend_volume_screener import TrendVolumeScreener
        print("✅ 添加路径后导入成功")

def main():
    print("=" * 80)
    print("📈 全市场股票扫描 - 自动版本")
    print("=" * 80)
    
    # 使用中等参数（推荐）
    params = {
        'price_above_ma': 'ma5',
        'ma_trend': True,
        'min_trend_days': 5,
        'min_three_day_change': 3.0,
        'max_three_day_change': 30.0,
        'min_up_days': 2,
        'min_volume_ratio': 1.0,
        'consecutive_volume': False,
        'max_ten_day_change': 40.0,
        'min_price': 3.0,
        'max_price': 200.0,
        'min_score': 70.0,
    }
    param_name = "中等参数"
    
    print(f"📋 使用参数: {param_name}")
    print()
    
    # 创建筛选器
    screener = TrendVolumeScreener(params)
    
    # 加载股票代码（全市场扫描）
    print("📋 加载股票代码...")
    codes = screener.load_stock_codes(sample_size=0, all_stocks=True)
    print(f"扫描股票数量: {len(codes)}只")
    print()
    
    # 运行筛选
    print("🔍 开始筛选...")
    print("⏳ 请耐心等待...")
    print()
    
    start_time = time.time()
    selected_stocks = screener.run_screening(codes, max_workers=15)
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ 筛选完成！")
    print(f"   处理股票: {len(codes)}只")
    print(f"   筛选出: {len(selected_stocks)}只")
    print(f"   耗时: {elapsed_time:.1f}秒")
    print()
    
    # 显示结果
    screener.print_results(selected_stocks, len(codes))
    
    # 保存结果
    if selected_stocks:
        screener.save_results(selected_stocks, len(codes))
    
    print(f"\n⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()