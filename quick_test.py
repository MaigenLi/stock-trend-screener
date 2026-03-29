#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试 - 验证修复后的筛选器
"""

import os
import sys
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入筛选器
try:
    from core.trend_volume_screener import TrendVolumeScreener
    print("✅ 成功导入 TrendVolumeScreener")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_stock_name():
    """测试股票名称获取"""
    print("\n" + "=" * 60)
    print("📊 测试股票名称获取")
    print("=" * 60)
    
    screener = TrendVolumeScreener()
    
    test_codes = [
        'sh600519',  # 贵州茅台
        'sz000001',  # 平安银行
        'sz002460',  # 赣锋锂业
        'sh600036',  # 招商银行
        'sz000858',  # 五粮液
    ]
    
    for code in test_codes:
        name = screener.get_stock_name(code)
        print(f"  {code}: {name}")
    
    return True

def test_sector_info():
    """测试板块信息获取"""
    print("\n" + "=" * 60)
    print("📊 测试板块信息获取")
    print("=" * 60)
    
    # 导入板块信息模块
    try:
        from core.stock_sector import get_sector_info
        sector_info = get_sector_info()
        print("✅ 成功导入板块信息模块")
    except ImportError as e:
        print(f"❌ 导入板块信息模块失败: {e}")
        return False
    
    test_codes = [
        'sh600519',  # 贵州茅台
        'sz002460',  # 赣锋锂业
    ]
    
    for code in test_codes:
        info = sector_info.get_stock_sector_info(code)
        print(f"\n  {code}:")
        print(f"    主要板块: {info.get('main_sector', '未知')}")
        print(f"    热度: {info.get('sector_hotness', 40)}")
        print(f"    人气: {info.get('sector_popularity', 30)}")
        print(f"    来源: {info.get('source', 'unknown')}")
    
    return True

def test_small_screening():
    """测试小规模筛选"""
    print("\n" + "=" * 60)
    print("📊 测试小规模筛选")
    print("=" * 60)
    
    # 使用宽松参数
    params = {
        'price_above_ma': 'ma5',
        'ma_trend': False,  # 放宽均线排列要求
        'min_trend_days': 3,
        'min_three_day_change': 1.0,  # 降低涨幅要求
        'max_three_day_change': 50.0,
        'min_up_days': 1,
        'min_volume_ratio': 0.8,  # 降低量比要求
        'consecutive_volume': False,
        'max_ten_day_change': 50.0,
        'min_price': 1.0,
        'max_price': 500.0,
        'min_score': 50.0,  # 降低评分要求
    }
    
    screener = TrendVolumeScreener(params)
    
    # 只测试少量股票
    test_codes = [
        'sh600519', 'sz000001', 'sz002460', 'sh600036', 'sz000858',
        'sh600030', 'sz300750', 'sh600276', 'sh600000', 'sh600016',
    ]
    
    print(f"测试 {len(test_codes)} 只股票...")
    print()
    
    selected_stocks = []
    
    for code in test_codes:
        try:
            result = screener.evaluate_stock(code)
            if result:
                selected_stocks.append(result)
                print(f"  ✅ {code} {result['name']}: 评分 {result['score']:.1f}")
            else:
                print(f"  ❌ {code}: 不符合条件")
        except Exception as e:
            print(f"  ⚠️  {code}: 评估失败 - {e}")
    
    print(f"\n筛选结果: {len(selected_stocks)}/{len(test_codes)} 只股票符合条件")
    
    if selected_stocks:
        print("\n符合条件股票详情:")
        for stock in selected_stocks:
            print(f"  {stock['code']} {stock['name']} (评分:{stock['score']:.1f})")
            print(f"    价格: {stock['latest_price']:.2f} 三天涨幅: {stock['three_day_change']:+.2f}%")
            print(f"    量比: {stock['avg_volume_ratio']:.2f} 趋势强度: {stock['trend_strength']:.1%}")
            print(f"    板块: {stock.get('main_sector', '未知')}")
            print()
    
    return len(selected_stocks) > 0

def main():
    print("=" * 80)
    print("🔧 快速测试 - 验证修复后的筛选器")
    print("=" * 80)
    
    start_time = time.time()
    
    # 运行测试
    tests = [
        ("股票名称获取", test_stock_name),
        ("板块信息获取", test_sector_info),
        ("小规模筛选", test_small_screening),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n▶️  运行测试: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ 通过" if success else "❌ 失败"
            print(f"  结果: {status}")
        except Exception as e:
            print(f"  ⚠️  异常: {e}")
            results.append((test_name, False))
    
    # 显示测试总结
    print("\n" + "=" * 80)
    print("📋 测试总结")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"测试总数: {total}")
    print(f"通过测试: {passed}")
    print(f"通过率: {passed/total*100:.1f}%")
    
    print("\n详细结果:")
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    elapsed = time.time() - start_time
    print(f"\n⏰ 总耗时: {elapsed:.1f}秒")
    print("=" * 80)
    
    # 返回总体结果
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)