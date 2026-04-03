#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试 - 验证趋势放量筛选器终极版
"""

import os
import sys
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入终极版筛选器
try:
    from core.trend_volume_screener_ultimate import UltimateTrendVolumeScreener
    print("✅ 成功导入 UltimateTrendVolumeScreener")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def test_small_screening():
    """测试小规模筛选"""
    print("\n" + "=" * 60)
    print("📊 测试小规模筛选（UltimateTrendVolumeScreener）")
    print("=" * 60)

    screener = UltimateTrendVolumeScreener()

    test_codes = [
        'sh600519', 'sz000001', 'sz002460', 'sh600036', 'sz000858',
        'sh600030', 'sz300750', 'sh600276', 'sh600000', 'sh600016',
    ]

    print(f"测试 {len(test_codes)} 只股票...")
    print()

    results = []
    for code in test_codes:
        try:
            result = screener.analyze_stock_comprehensive(code)
            if result and result.get('score', 0) > 0:
                print(f"  ✅ {code}: 评分 {result['score']:.1f}")
                results.append(result)
            else:
                print(f"  ❌ {code}: 不符合条件")
        except Exception as e:
            print(f"  ⚠️  {code}: 分析失败 - {e}")

    print(f"\n筛选结果: {len(results)}/{len(test_codes)} 只股票")
    return len(results) > 0


def test_comprehensive_screening():
    """测试完整筛选流程"""
    print("\n" + "=" * 60)
    print("📊 测试完整筛选流程")
    print("=" * 60)

    screener = UltimateTrendVolumeScreener()

    test_codes = [
        'sh600519', 'sh600036', 'sz000858',
        'sh600030', 'sz300750',
    ]

    try:
        stocks = screener.run_comprehensive_screening(test_codes)
        print(f"\n  符合条件: {len(stocks)} 只")
        for s in stocks[:5]:
            print(f"  {s['code']} {s.get('name', '未知')}: 评分 {s['score']:.1f}")
        return len(stocks) > 0
    except Exception as e:
        print(f"  ⚠️  筛选异常: {e}")
        return False


def main():
    print("=" * 80)
    print("🔧 快速测试 - 趋势放量筛选器终极版")
    print("=" * 80)

    start_time = time.time()

    tests = [
        ("小规模筛选", test_small_screening),
        ("完整筛选流程", test_comprehensive_screening),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n▶️  运行测试: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"  结果: {'✅ 通过' if success else '❌ 失败'}")
        except Exception as e:
            print(f"  ⚠️  异常: {e}")
            results.append((test_name, False))

    passed = sum(1 for _, s in results if s)
    print(f"\n{'='*80}")
    print(f"📋 测试总结: {passed}/{len(results)} 通过")
    elapsed = time.time() - start_time
    print(f"⏰ 总耗时: {elapsed:.1f}s")
    print("=" * 80)

    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
