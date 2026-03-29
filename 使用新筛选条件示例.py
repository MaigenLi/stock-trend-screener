#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用新筛选条件的示例
新的筛选条件包括：
1. 排除ST股票
2. 60日均线上涨
3. 短期均线多头排列 (MA5 > MA10 > MA20)
4. 所有均线都在60日均线之上 (MA5 > MA10 > MA20 > MA60)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.trend_volume_screener import TrendVolumeScreener

def main():
    print("🎯 股票趋势筛选器 - 增强版")
    print("=" * 60)
    print("新增筛选条件：")
    print("1. ✅ 排除ST股票")
    print("2. ✅ 60日均线上涨")
    print("3. ✅ 短期均线多头排列 (MA5 > MA10 > MA20)")
    print("4. ✅ 所有均线都在60日均线之上 (MA5 > MA10 > MA20 > MA60)")
    print("=" * 60)
    
    # 创建筛选器
    screener = TrendVolumeScreener()
    
    # 可以调整参数
    params = {
        # 趋势参数
        'min_trend_days': 5,
        'price_above_ma': 'ma5',
        'ma_trend': True,
        
        # 三天表现参数
        'min_three_day_change': 8.0,
        'max_three_day_change': 30.0,
        'min_up_days': 2,
        
        # 量能参数
        'min_volume_ratio': 1.0,
        'consecutive_volume': False,
        
        # 风险参数
        'max_ten_day_change': 40.0,
        
        # 其他参数
        'min_price': 3.0,
        'max_price': 200.0,
        'min_score': 70.0,
    }
    
    screener.params.update(params)
    
    print("\n📋 当前筛选参数：")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print("\n💡 使用说明：")
    print("1. 运行全市场扫描: python full_scan_auto.py")
    print("2. 运行快速测试: python quick_test.py")
    print("3. 使用主筛选器: python screener.py")
    print("\n新的筛选条件已自动集成到核心筛选逻辑中。")

if __name__ == "__main__":
    main()