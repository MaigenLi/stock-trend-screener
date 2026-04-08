#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终全市场扫描 - 使用优化参数
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trend_volume_screener_ultimate import UltimateTrendVolumeScreener as TrendVolumeScreener

def main():
    print("=" * 80)
    print("📈 最终全市场股票扫描")
    print("=" * 80)
    
    # 参数选择菜单
    print("请选择参数组合:")
    print("  1. 严格参数 (筛选比例约1-3%)")
    print("  2. 中等参数 (筛选比例约5-10%) - 推荐")
    print("  3. 宽松参数 (筛选比例约10-20%)")
    print("  4. 自定义参数")
    
    choice = input("\n请输入选择 (1-4, 默认2): ").strip()
    
    if choice == "1":
        # 严格参数
        params = {
            'price_above_ma': 'ma10',
            'ma_trend': True,
            'min_trend_days': 5,
            'min_three_day_change': 5.0,
            'max_three_day_change': 25.0,
            'min_up_days': 2,
            'min_volume_ratio': 1.2,
            'consecutive_volume': False,
            'max_ten_day_change': 30.0,
            'min_price': 5.0,
            'max_price': 150.0,
            'min_score': 80.0,
        }
        param_name = "严格参数"
        
    elif choice == "3":
        # 宽松参数
        params = {
            'price_above_ma': 'ma5',
            'ma_trend': True,
            'min_trend_days': 5,
            'min_three_day_change': 3.0,
            'max_three_day_change': 30.0,
            'min_up_days': 2,
            'min_volume_ratio': 0.8,
            'consecutive_volume': False,
            'max_ten_day_change': 50.0,
            'min_price': 3.0,
            'max_price': 200.0,
            'min_score': 60.0,
        }
        param_name = "宽松参数"
        
    elif choice == "4":
        # 自定义参数
        print("\n📝 自定义参数设置:")
        params = {}
        params['price_above_ma'] = input("价格在哪个均线之上? (ma5/ma10/ma20, 默认ma5): ").strip() or 'ma5'
        params['ma_trend'] = input("要求均线多头排列? (y/n, 默认y): ").strip().lower() != 'n'
        params['min_three_day_change'] = float(input("最小三天涨幅(%)? (默认3.0): ").strip() or "3.0")
        params['max_three_day_change'] = float(input("最大三天涨幅(%)? (默认30.0): ").strip() or "30.0")
        params['min_volume_ratio'] = float(input("最小平均量比? (默认1.0): ").strip() or "1.0")
        params['min_score'] = float(input("最小综合评分? (默认70.0): ").strip() or "70.0")
        params['min_trend_days'] = 5
        params['min_up_days'] = 2
        params['consecutive_volume'] = False
        params['max_ten_day_change'] = 40.0
        params['min_price'] = 3.0
        params['max_price'] = 200.0
        param_name = "自定义参数"
        
    else:
        # 中等参数 (默认)
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
    
    print(f"\n✅ 选择参数组合: {param_name}")
    print("参数设置:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()
    
    # 创建筛选器
    screener = TrendVolumeScreener(params)
    
    # 加载全部股票
    print("📋 加载全部股票代码...")
    stock_codes_file = str(Path.home() / "stock_code" / "results" / "stock_codes.txt")
    all_codes = []
    
    try:
        with open(stock_codes_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if line.startswith('sh') or line.startswith('sz'):
                        all_codes.append(line)
    except Exception as e:
        print(f"❌ 加载股票代码失败: {e}")
        return
    
    total_stocks = len(all_codes)
    print(f"全市场股票数量: {total_stocks}只")
    print()
    
    # 确认开始扫描
    confirm = input(f"⚠️  即将扫描全部 {total_stocks} 只股票，这可能需要一些时间。\n确认开始扫描? (y/n, 默认y): ").strip().lower()
    
    if confirm == 'n':
        print("❌ 用户取消扫描")
        return
    
    # 运行参数
    max_workers = 20
    batch_size = 100
    
    print(f"\n🚀 开始全市场扫描...")
    print(f"并行线程: {max_workers}")
    print(f"预计时间: 约{total_stocks/max_workers/10:.0f}-{total_stocks/max_workers/5:.0f}分钟")
    print()
    
    start_time = time.time()
    selected_stocks = []
    processed_count = 0
    
    # 分批处理，避免内存问题
    for batch_start in range(0, total_stocks, batch_size):
        batch_end = min(batch_start + batch_size, total_stocks)
        batch_codes = all_codes[batch_start:batch_end]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(screener.evaluate_stock, code): code for code in batch_codes}
            
            for future in as_completed(futures):
                processed_count += 1
                code = futures[future]
                
                try:
                    result = future.result(timeout=10)
                    if result:
                        selected_stocks.append(result)
                except Exception:
                    pass
                
                # 显示进度
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    progress = processed_count / total_stocks * 100
                    estimated_total = elapsed / (processed_count / total_stocks) if processed_count > 0 else 0
                    remaining = max(0, estimated_total - elapsed)
                    
                    print(f"  进度: {processed_count}/{total_stocks} ({progress:.1f}%) | "
                          f"筛选出: {len(selected_stocks)}只 | "
                          f"用时: {elapsed/60:.1f}分钟 | "
                          f"剩余: {remaining/60:.1f}分钟")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ 全市场扫描完成！")
    print(f"   总用时: {elapsed_time/60:.1f}分钟")
    print(f"   扫描股票: {total_stocks}只")
    print(f"   筛选出: {len(selected_stocks)}只")
    print(f"   筛选比例: {len(selected_stocks)/total_stocks*100:.2f}%")
    print()
    
    if selected_stocks:
        # 按评分排序
        selected_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        print("=" * 80)
        print("🎯 全市场筛选结果")
        print("=" * 80)
        
        # 显示前20只
        display_count = min(20, len(selected_stocks))
        print(f"📈 找到 {len(selected_stocks)} 只符合条件的股票，显示前 {display_count} 只:")
        print()
        
        for i, stock in enumerate(selected_stocks[:display_count], 1):
            # 趋势描述
            trend_desc = []
            if stock['price_above_ma5']:
                trend_desc.append("MA5↑")
            if stock['price_above_ma10']:
                trend_desc.append("MA10↑")
            if stock['ma5_above_ma10']:
                trend_desc.append("MA5>MA10")
            
            # 量能描述
            volume_desc = ""
            if stock['consecutive_volume']:
                volume_desc = "连续放量"
            elif stock['avg_volume_ratio'] >= 1.2:
                volume_desc = "明显放量"
            elif stock['avg_volume_ratio'] >= 1.0:
                volume_desc = "放量"
            
            # 板块热度和人气描述
            hotness = stock.get('sector_hotness', 40)
            popularity = stock.get('sector_popularity', 30)
            
            # 热度图标
            if hotness >= 80:
                hotness_emoji = "🔥🔥"
                hotness_desc = "高热"
            elif hotness >= 60:
                hotness_emoji = "🔥"
                hotness_desc = "中热"
            elif hotness >= 40:
                hotness_emoji = "♨️"
                hotness_desc = "温热"
            else:
                hotness_emoji = "⚪"
                hotness_desc = "一般"
            
            # 人气图标
            if popularity >= 80:
                popularity_emoji = "👥👥"
                popularity_desc = "高人氣"
            elif popularity >= 60:
                popularity_emoji = "👥"
                popularity_desc = "中人氣"
            elif popularity >= 40:
                popularity_emoji = "👤"
                popularity_desc = "溫人氣"
            else:
                popularity_emoji = "👤"
                popularity_desc = "一般人氣"
            
            main_sector = stock.get('main_sector', '未知')
            sector_category = stock.get('sector_category', '其他')
            
            print(f"{i:3d}. {stock['code']} {stock['name']} (评分:{stock['score']:.1f})")
            print(f"     价格: {stock['latest_price']:7.2f} ({stock['latest_change']:+.2f}%)")
            print(f"     三天: {stock['three_day_change']:+.2f}% ({stock['up_days']}/3上涨)")
            print(f"     量比: {stock['avg_volume_ratio']:.2f} {volume_desc}")
            print(f"     趋势: {stock['trend_strength']:.0%} ({' '.join(trend_desc)})")
            print(f"     板块: {main_sector} ({sector_category})")
            print(f"     热度: {hotness_emoji}{hotness_desc}({hotness}) 人气: {popularity_emoji}{popularity_desc}({popularity})")
            print()
        
        # 统计信息
        import numpy as np
        scores = [s['score'] for s in selected_stocks]
        three_day_changes = [s['three_day_change'] for s in selected_stocks]
        volume_ratios = [s['avg_volume_ratio'] for s in selected_stocks]
        ten_day_changes = [s['ten_day_change'] for s in selected_stocks]
        trend_strengths = [s['trend_strength'] for s in selected_stocks]
        
        print(f"\n📊 全市场特征统计:")
        print(f"   平均评分: {np.mean(scores):.1f}")
        print(f"   平均三天涨幅: {np.mean(three_day_changes):.2f}%")
        print(f"   平均量比: {np.mean(volume_ratios):.2f}")
        print(f"   平均十日涨幅: {np.mean(ten_day_changes):.2f}%")
        print(f"   平均趋势强度: {np.mean(trend_strengths):.0%}")
        
        # 优质特征统计
        high_score = sum(1 for s in selected_stocks if s['score'] >= 80)
        consecutive_up = sum(1 for s in selected_stocks if s['consecutive_up'])
        consecutive_volume = sum(1 for s in selected_stocks if s['consecutive_volume'])
        strong_trend = sum(1 for s in selected_stocks if s['trend_strength'] >= 0.7)
        perfect_trend = sum(1 for s in selected_stocks if s['price_above_ma5'] and s['price_above_ma10'] and s['ma5_above_ma10'])
        
        print(f"\n⭐ 优质特征分布:")
        print(f"   高分股票(≥80分): {high_score}只 ({high_score/len(selected_stocks)*100:.1f}%)")
        print(f"   完美趋势: {perfect_trend}只 ({perfect_trend/len(selected_stocks)*100:.1f}%)")
        print(f"   强势趋势(≥70%): {strong_trend}只 ({strong_trend/len(selected_stocks)*100:.1f}%)")
        print(f"   连续三天上涨: {consecutive_up}只 ({consecutive_up/len(selected_stocks)*100:.1f}%)")
        print(f"   连续三天放量: {consecutive_volume}只 ({consecutive_volume/len(selected_stocks)*100:.1f}%)")
        
        # TOP 10推荐
        if len(selected_stocks) >= 10:
            print(f"\n🏆 TOP 10 推荐:")
            for i, stock in enumerate(selected_stocks[:10], 1):
                print(f"   {i:2d}. {stock['code']} {stock['name']} (评分:{stock['score']:.1f})")
                print(f"       三天: {stock['three_day_change']:+.2f}% | 量比: {stock['avg_volume_ratio']:.2f}")
                print(f"       趋势: {stock['trend_strength']:.0%} | 十日涨幅: {stock['ten_day_change']:+.2f}%")
        
        # 保存结果 - 使用screener的save_results方法
        # 这会自动生成3个文件：详细结果、股票代码列表、按板块分类的代码列表
        screener.save_results(selected_stocks, total_stocks)
        
        # 同时保存一个全市场专用的文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = "./results/current"
        os.makedirs(results_dir, exist_ok=True)
        
        full_market_file = os.path.join(results_dir, f"full_market_{param_name}_{timestamp}.txt")
        
        with open(full_market_file, "w", encoding='utf-8') as f:
            f.write(f"# 全市场扫描结果 - {param_name}\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 扫描股票: {total_stocks}只\n")
            f.write(f"# 筛选出: {len(selected_stocks)}只\n")
            f.write(f"# 筛选比例: {len(selected_stocks)/total_stocks*100:.2f}%\n")
            f.write(f"# 参数: {params}\n")
            f.write(f"# 用时: {elapsed_time/60:.1f}分钟\n")
            f.write("\n")
            
            # 板块统计
            sector_stats = {}
            for stock in selected_stocks:
                sector = stock.get('main_sector', '未知')
                if sector not in sector_stats:
                    sector_stats[sector] = 0
                sector_stats[sector] += 1
            
            f.write(f"# 板块分布:\n")
            for sector, count in sorted(sector_stats.items(), key=lambda x: x[1], reverse=True):
                f.write(f"#   {sector}: {count}只 ({count/len(selected_stocks)*100:.1f}%)\n")
            f.write("\n")
            
            for stock in selected_stocks:
                # 板块热度和人气描述
                hotness = stock.get('sector_hotness', 40)
                popularity = stock.get('sector_popularity', 30)
                
                # 热度描述
                if hotness >= 80:
                    hotness_desc = "🔥🔥高热"
                elif hotness >= 60:
                    hotness_desc = "🔥中热"
                elif hotness >= 40:
                    hotness_desc = "♨️温热"
                else:
                    hotness_desc = "⚪一般"
                
                # 人气描述
                if popularity >= 80:
                    popularity_desc = "👥👥高人氣"
                elif popularity >= 60:
                    popularity_desc = "👥中人氣"
                elif popularity >= 40:
                    popularity_desc = "👤溫人氣"
                else:
                    popularity_desc = "👤一般人氣"
                
                main_sector = stock.get('main_sector', '未知')
                sector_category = stock.get('sector_category', '其他')
                
                f.write(f"{stock['code']} {stock['name']} (评分:{stock['score']:.1f})\n")
                f.write(f"  价格: {stock['latest_price']:.2f} 涨跌: {stock['latest_change']:+.2f}%\n")
                f.write(f"  三天涨幅: {stock['three_day_change']:+.2f}% 上涨天数: {stock['up_days']}/3\n")
                f.write(f"  量比: {stock['avg_volume_ratio']:.2f} {'连续放量' if stock['consecutive_volume'] else ''}\n")
                f.write(f"  十日涨幅: {stock['ten_day_change']:+.2f}% 趋势强度: {stock['trend_strength']:.0%}\n")
                f.write(f"  板块: {main_sector} ({sector_category})\n")
                f.write(f"  热度: {hotness_desc}({hotness}) 人气: {popularity_desc}({popularity})\n")
                f.write(f"  MA5: {stock['ma5']:.2f} MA10: {stock['ma10']:.2f} MA20: {stock['ma20']:.2f}\n")
                f.write("\n")
        
        print(f"\n💾 全市场专用结果已保存到: {full_market_file}")
        
        print(f"\n💡 投资建议:")
        print(f"   1. 优先选择评分≥80分的股票")
        print(f"   2. 关注趋势完美且连续上涨的标的")
        print(f"   3. 量价配合良好的股票更具潜力")
        print(f"   4. 建议分散投资5-10只，控制风险")
        
    else:
        print("❌ 未找到符合条件的股票")
        print("\n📉 市场分析:")
        print("  当前市场环境下，符合趋势条件的股票较少")
        print("  可能原因:")
        print("  1. 市场整体处于调整期")
        print("  2. 缺乏明显的趋势性机会")
        print("  3. 参数设置可能过严")
        print("\n💡 建议:")
        print("  1. 使用更宽松的参数组合重新扫描")
        print("  2. 关注超跌反弹机会")
        print("  3. 等待市场趋势明确后再操作")
    
    print(f"\n⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
