#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hot_sector.py — 当日市场资金热点板块 & 热点个股实时监控
============================================================
用法：
  python hot_sector.py                      # 实时热点（一次性）
  python hot_sector.py --mode monitor       # 持续监控（每30秒刷新）
  python hot_sector.py --mode monitor -i 60 # 每60秒刷新
  python hot_sector.py --mode monitor -n 8  # 每次显示每个板块TOP8只
  python hot_sector.py --cache              # 使用今日缓存数据
"""

import sys, time, argparse, os, signal
from datetime import date, datetime
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw/workspace"
sys.path.insert(0, str(WORKSPACE / "stock_trend"))

import numpy as np
import pandas as pd
import akshare as ak

CACHE_DIR = WORKSPACE / ".cache" / "hot_sector"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── 网络请求 ─────────────────────────────────────
def fetch(fn, *args, retries=2, wait=3, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise

# ── 数据抓取 ─────────────────────────────────────

def get_industry_fund_flow():
    """返回行业板块资金流（stock_fund_flow_industry），按净额降序"""
    df = fetch(ak.stock_fund_flow_industry, retries=2, wait=4)
    for col in ['流入资金', '流出资金', '净额', '行业-涨跌幅']:
        if col in df.columns and df[col].dtype == object:
            df[col] = (
                df[col].astype(str)
                .str.replace('亿', '', regex=False)
                .str.replace('万', '', regex=False)
                .str.strip()
                .replace(['None', ''], np.nan)
                .apply(pd.to_numeric, errors='coerce')
            )
    if '净额' in df.columns:
        df = df.sort_values('净额', ascending=False).reset_index(drop=True)
    return df

def get_concept_fund_flow():
    """返回概念板块资金流（stock_fund_flow_concept），按净额降序"""
    df = fetch(ak.stock_fund_flow_concept, retries=2, wait=4)
    for col in ['流入资金', '流出资金', '净额', '行业-涨跌幅']:
        if col in df.columns and df[col].dtype == object:
            df[col] = (
                df[col].astype(str)
                .str.replace('亿', '', regex=False)
                .str.replace('万', '', regex=False)
                .str.strip()
                .replace(['None', ''], np.nan)
                .apply(pd.to_numeric, errors='coerce')
            )
    if '净额' in df.columns:
        df = df.sort_values('净额', ascending=False).reset_index(drop=True)
    return df

def get_spot_label_map():
    """stock_sector_spot 的 label->板块名 映射"""
    spot = fetch(ak.stock_sector_spot, retries=1, wait=2)
    return dict(zip(spot['label'], spot['板块']))

# ── 细分行业 → stock_sector_spot label 映射 ────────
# key: stock_fund_flow_industry 行业名
# value: stock_sector_spot 宽基板块名（用于查 label）
INDUSTRY_TO_BOARD = {
    '半导体':        '电子器件',
    '电子化学品':    '电子器件',
    '电池':          '电子器件',
    '光伏设备':       '电子器件',
    '电子元件':      '电子器件',
    '光学光电子':    '电子器件',
    '消费电子':      '电子器件',
    '汽车电子':      '汽车制造',
    '通信设备':       '电子器件',
    '输配电气':       '发电设备',
    '电网设备':       '发电设备',
    '风电设备':       '发电设备',
    '电源设备':       '发电设备',
    '军工':          '飞机制造',
    '航天航空':       '飞机制造',
    '光刻胶':        '电子器件',
    '芯片概念':       '电子器件',
    '存储芯片':       '电子器件',
    '先进封装':       '电子器件',
    '第三代半导体':   '电子器件',
    'MCU芯片':        '电子器件',
    '汽车芯片':      '汽车制造',
    '国家大基金持股': '电子器件',
    '中芯国际概念':   '电子器件',
    'OLED':          '电子器件',
    'F5G概念':        '电子器件',
    '固态电池':       '电子器件',
    '有色金属':       '有色金属',
    '钢铁行业':       '钢铁行业',
    '煤炭行业':       '煤炭行业',
    '石油行业':       '石油行业',
    '油气开采及服务':  '石油行业',
    '水泥行业':       '建筑建材',
    '建筑建材':       '建筑建材',
    '装修建材':       '建筑建材',
    '非金属材料':     '建筑建材',
    '玻璃行业':       '玻璃行业',
    '电力行业':       '电力行业',
    '公用事业':       '供水供气',
    '燃气':           '供水供气',
    '水务':           '供水供气',
    '风电':           '发电设备',
    '核电':           '电力行业',
    '能源金属':       '有色金属',
    '化工行业':       '化工行业',
    '化学制品':       '化工行业',
    '化学原料':       '化工行业',
    '化纤行业':       '化纤行业',
    '氟化工概念':     '化工行业',
    '农药化肥':       '农药化肥',
    '塑料制品':       '塑料制品',
    '橡胶制品':       '化学行业',
    '陶瓷行业':       '陶瓷行业',
    '酿酒行业':       '酿酒行业',
    '饮料制造':       '酿酒行业',
    '白酒':           '酿酒行业',
    '食品行业':       '食品行业',
    '家电行业':       '家电行业',
    '纺织行业':       '纺织行业',
    '纺织机械':       '纺织机械',
    '服装鞋类':       '服装鞋类',
    '美容护理':       '化学行业',
    '造纸行业':       '造纸行业',
    '包装印刷':       '印刷包装',
    '印刷包装':       '印刷包装',
    '商贸百货':       '商业百货',
    '商业百货':       '商业百货',
    '酒店旅游':       '酒店旅游',
    '旅游酒店':       '酒店旅游',
    '文化传媒':       '传媒娱乐',
    '传媒娱乐':       '传媒娱乐',
    '教培行业':       '传媒娱乐',
    '房地产':         '房地产',
    '银行':           '金融行业',
    '证券':           '金融行业',
    '保险':           '金融行业',
    '多元金融':       '金融行业',
    '医药':           '生物制药',
    '医药行业':       '生物制药',
    '生物制药':       '生物制药',
    '医疗器械':       '医疗器械',
    '中药':           '生物制药',
    '农林牧渔':       '农林牧渔',
    '农业':           '农林牧渔',
    '饲料':           '农林牧渔',
    '畜禽养殖':       '农林牧渔',
    '渔业':           '农林牧渔',
    '林业':           '农林牧渔',
    '汽车制造':       '汽车制造',
    '汽车零部件':     '汽车制造',
    '摩托车':         '摩托车',
    '机械行业':       '机械行业',
    '通用设备':       '机械行业',
    '专用设备':       '机械行业',
    '仪器仪表':       '仪器仪表',
    '电子信息':       '电子信息',
    '软件开发':       '电子信息',
    '互联网服务':     '电子信息',
    '计算机设备':     '电子信息',
    '通信服务':       '电子信息',
    '交通运输':       '交通运输',
    '港口航运':       '交通运输',
    '上海自贸区':     '交通运输',
    '公路桥梁':       '公路桥梁',
    '铁路公路':       '交通运输',
    '航空机场':       '交通运输',
    '航运港口':       '交通运输',
    '物流行业':       '交通运输',
    '交运物流':       '交通运输',
    '仓储物流':       '交通运输',
    '环保行业':       '环保行业',
    '环保工程':       '环保行业',
    '综合行业':       '其它行业',
    '其它行业':       '其它行业',
    '次新股':         '次新股',
    '白酒概念':       '酿酒行业',
    '燃气':           '供水供气',
    '物资外贸':       '物资外贸',
    '开发区':         '开发区',
}

def resolve_spot_label(sector_name, label_map):
    """将细分行业名解析为 stock_sector_spot 的 label，不存在返回None"""
    board_name = INDUSTRY_TO_BOARD.get(sector_name, sector_name)
    # 精确匹配
    for lbl, name in label_map.items():
        if name == board_name:
            return lbl
    # 模糊
    for lbl, name in label_map.items():
        if board_name in name or name in board_name:
            return lbl
    return None

def get_top_stocks_in_sectors(sector_names, label_map, top_n=8):
    """
    批量获取多个板块的成分股，按涨跌幅排序取TOP。
    sector_names: list of 行业/板块名称
    label_map: {label: 板块名} from stock_sector_spot
    返回: dict[sector_name] -> DataFrame(top_n rows)
    """
    result = {}
    for sname in sector_names:
        lbl = resolve_spot_label(sname, label_map)
        if not lbl:
            result[sname] = pd.DataFrame()
            continue
        try:
            sub = fetch(ak.stock_sector_detail, sector=lbl, retries=1, wait=2)
            if sub is None or sub.empty:
                result[sname] = pd.DataFrame()
                continue
            if 'changepercent' in sub.columns:
                sub = sub.sort_values('changepercent', ascending=False)
            elif '涨跌幅' in sub.columns:
                sub = sub.sort_values('涨跌幅', ascending=False)
            result[sname] = sub.head(top_n).copy()
        except Exception:
            result[sname] = pd.DataFrame()
    return result

# ── 格式化 ───────────────────────────────────────
def fmt_chg(v):
    if pd.isna(v): return '—'
    return f'{v:+.2f}%'

def fmt_amt(v):
    if pd.isna(v): return '—'
    v = float(v)
    if abs(v) >= 1e4:
        return f'{v/1e4:.2f}万'
    elif abs(v) >= 1e2:
        return f'{v:.2f}亿'
    return f'{v:.2f}亿'

def fmt_price(v):
    if pd.isna(v): return '—'
    return f'{v:.2f}'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ── 报告打印函数 ─────────────────────────────────

def print_market_overview():
    """主要指数涨跌"""
    try:
        idx = ak.index_spot_em()
        if idx is None or idx.empty:
            return
        targets = {'上证指数': '000001', '深证成指': '399001',
                   '创业板指': '399006', '科创50': '000688'}
        print("📊 主要指数")
        print(f"{'─'*55}")
        print(f"{'指数':<12} {'最新价':>10} {'涨跌幅':>10}")
        print(f"{'─'*55}")
        for name, code in targets.items():
            row = idx[idx.apply(lambda r: str(r.get('代码','')).endswith(code), axis=1)]
            if not row.empty:
                price = row.iloc[0].get('最新价', '—')
                chg    = row.iloc[0].get('涨跌幅', '—')
                price_s = f"{price:.2f}" if isinstance(price, (int,float)) else str(price)
                chg_s   = f"{chg:+.2f}%"   if isinstance(chg,   (int,float)) else str(chg)
                print(f"{name:<12} {price_s:>10} {chg_s:>10}")
        print()
    except Exception:
        pass

def print_hot_summary(ind_df, con_df, top_n=10):
    """资金热点综合排名"""
    ind_top = ind_df.head(top_n)[['行业','净额','行业-涨跌幅']].rename(columns={'行业':'板块'})
    ind_top['类型'] = '行业'
    con_top = con_df.head(top_n)[['行业','净额','行业-涨跌幅']].rename(columns={'行业':'板块'})
    con_top['类型'] = '概念'
    combined = (pd.concat([ind_top, con_top])
                .sort_values('净额', ascending=False)
                .head(top_n)
                .reset_index(drop=True))

    print(f"🔥 资金热点综合排名 TOP{top_n}（行业+概念，按净额）")
    print(f"{'─'*60}")
    print(f"{'排名':>4} {'板块':<14} {'类型':>6} {'净额(亿)':>10} {'涨跌幅':>8}")
    print(f"{'─'*60}")
    for i, row in combined.iterrows():
        rank = i + 1
        chg_s = fmt_chg(row.get('行业-涨跌幅', np.nan))
        net_s = fmt_amt(row.get('净额', np.nan))
        print(f"{rank:>4} {row['板块']:<14} {row['类型']:>6} {net_s:>10} {chg_s:>8}")
    print()
    return list(combined['板块'])

def print_industry_sectors(ind_df):
    print(f"🏭 行业板块 TOP15（按净额）")
    print(f"{'─'*70}")
    if ind_df.empty:
        print("  （暂无数据）")
        return
    print(f"{'板块':<12} {'涨跌幅':>8} {'流入(亿)':>10} {'流出(亿)':>10} {'净额(亿)':>10} {'家数':>6}")
    print(f"{'─'*70}")
    for _, r in ind_df.head(15).iterrows():
        chg_s = fmt_chg(r.get('行业-涨跌幅', np.nan))
        print(f"{r['行业']:<12} {chg_s:>8} "
              f"{fmt_amt(r.get('流入资金',np.nan)):>10} "
              f"{fmt_amt(r.get('流出资金',np.nan)):>10} "
              f"{fmt_amt(r.get('净额',np.nan)):>10} "
              f"{r.get('公司家数','—'):>6}")
    print()

def print_concept_sectors(con_df):
    print(f"🔮 概念板块 TOP15（按净额）")
    print(f"{'─'*70}")
    if con_df.empty:
        print("  （暂无数据）")
        return
    print(f"{'板块':<14} {'涨跌幅':>8} {'流入(亿)':>10} {'流出(亿)':>10} {'净额(亿)':>10} {'家数':>6}")
    print(f"{'─'*70}")
    for _, r in con_df.head(15).iterrows():
        chg_s = fmt_chg(r.get('行业-涨跌幅', np.nan))
        print(f"{r['行业']:<14} {chg_s:>8} "
              f"{fmt_amt(r.get('流入资金',np.nan)):>10} "
              f"{fmt_amt(r.get('流出资金',np.nan)):>10} "
              f"{fmt_amt(r.get('净额',np.nan)):>10} "
              f"{r.get('公司家数','—'):>6}")
    print()

def print_sector_stocks(sector_stocks, top_sector_n=6, top_stock_n=8):
    """
    打印热点板块的成分股。
    sector_stocks: dict[sector_name] -> DataFrame
    """
    if not sector_stocks:
        return 0
    print(f"📈 热点板块成分股（每个板块 TOP{top_stock_n}）")
    print(f"{'─'*72}")
    shown = 0
    for sname, sub in sector_stocks.items():
        if sub is None or sub.empty:
            continue
        shown += 1
        tag = '🏭' if '板块' in str(type(sub)) and len(sub) > 0 else '🔮'
        print(f"\n  {tag} {sname}")
        print(f"{'─'*72}")
        print(f"{'代码':<10} {'名称':<8} {'现价':>7} {'涨跌幅':>8} {'成交额(亿)':>12} {'换手率%':>8}")
        print(f"{'─'*72}")
        for _, row in sub.head(top_stock_n).iterrows():
            code  = str(row.get('code', ''))
            name  = str(row.get('name', ''))[:8]
            price = fmt_price(row.get('trade', row.get('price', np.nan)))
            chg   = fmt_chg(row.get('changepercent', row.get('涨跌幅', np.nan)))
            amt_raw = row.get('amount', np.nan)
            amt_s = f"{amt_raw/1e8:.2f}" if not pd.isna(amt_raw) else '—'
            turn_raw = row.get('turnoverratio', row.get('turnover', np.nan))
            turn_s = f"{turn_raw:.2f}%" if not pd.isna(turn_raw) else '—'
            print(f"{code:<10} {name:<8} {price:>7} {chg:>8} {amt_s:>12} {turn_s:>8}")
    print()
    return shown

# ── 监控主循环 ───────────────────────────────────
def monitor_loop(interval=30, top_sector_n=6, top_stock_n=8):
    round_num = 0
    start_time = time.time()
    last_ind = pd.DataFrame()
    last_con = pd.DataFrame()
    stop_requested = False

    def handle_signal(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while not stop_requested:
        round_num += 1
        t0 = time.time()
        failed = []

        try:
            ind_df = get_industry_fund_flow()
        except Exception as e:
            ind_df = pd.DataFrame()
            failed.append('行业')

        try:
            con_df = get_concept_fund_flow()
        except Exception as e:
            con_df = pd.DataFrame()
            failed.append('概念')

        # 获取 label_map（每次重取保证新鲜）
        label_map = {}
        try:
            label_map = get_spot_label_map()
        except Exception:
            pass

        # 合并行业+概念热点，取前 top_sector_n 个板块名
        top_sector_names = []
        if not ind_df.empty:
            for sname in ind_df['行业'].head(top_sector_n):
                if sname not in top_sector_names:
                    top_sector_names.append(sname)
        if not con_df.empty:
            for sname in con_df['行业'].head(top_sector_n):
                if sname not in top_sector_names:
                    top_sector_names.append(sname)
                    if len(top_sector_names) >= top_sector_n:
                        break

        # 批量取成分股
        sector_stocks = {}
        if top_sector_names and label_map:
            try:
                sector_stocks = get_top_stocks_in_sectors(top_sector_names, label_map, top_n=top_stock_n)
            except Exception as e:
                print(f"   ⚠️ 成分股获取失败: {e}")

        elapsed = time.time() - start_time
        clear_screen()

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'═'*72}")
        print(f"  📊 热点板块实时监控  |  {now_str}  |  第{round_num}轮  |  已运行 {elapsed/60:.1f}分钟")
        print(f"{'═'*72}")
        if failed:
            print(f"  ⚠️ 失败: {', '.join(failed)}  |  刷新 {time.time()-t0:.1f}s")
        else:
            print(f"  ⏱ 刷新耗时 {time.time()-t0:.1f}s")

        print_market_overview()
        hot_secs = print_hot_summary(ind_df, con_df, top_n=10)
        shown = print_sector_stocks(sector_stocks, top_sector_n=top_sector_n, top_stock_n=top_stock_n)
        print(f"\n  💡 下次刷新: {interval}秒后  |  Ctrl+C 停止")

        # 检测热点变化
        changes = []
        if not ind_df.empty and not last_ind.empty:
            old = set(last_ind['行业'].head(3))
            new = set(ind_df['行业'].head(3))
            for s in new - old:
                changes.append(f"📍 新晋热门行业: {s}")
        if not con_df.empty and not last_con.empty:
            old = set(last_con['行业'].head(3))
            new = set(con_df['行业'].head(3))
            for s in new - old:
                changes.append(f"📍 新晋热门概念: {s}")
        if changes:
            print(f"\n  {'  |  '.join(changes)}")

        last_ind = ind_df.copy()
        last_con = con_df.copy()

        for sec in range(interval, 0, -1):
            time.sleep(1)
            if stop_requested:
                break

    print(f"\n✅ 监控已停止（共{round_num}轮，{(time.time()-start_time)/60:.1f}分钟）")

# ── 一次性报告 ───────────────────────────────────
def report_once():
    t0 = time.time()
    failed = []

    print("📡 正在抓取行业板块资金流...")
    try:
        ind_df = get_industry_fund_flow()
        print(f"   ✅ {len(ind_df)} 个行业板块")
    except Exception as e:
        ind_df = pd.DataFrame()
        failed.append('行业')

    print("📡 正在抓取概念板块资金流...")
    try:
        con_df = get_concept_fund_flow()
        print(f"   ✅ {len(con_df)} 个概念板块")
    except Exception as e:
        con_df = pd.DataFrame()
        failed.append('概念')

    label_map = {}
    try:
        label_map = get_spot_label_map()
    except Exception:
        pass

    print(f"\n总耗时: {time.time()-t0:.1f}s")

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"\n{'='*72}")
    print(f"  📊 当日市场资金热点报告  ({now})")
    print(f"{'='*72}\n")

    print_market_overview()
    hot_secs = print_hot_summary(ind_df, con_df, top_n=10)
    print_industry_sectors(ind_df)
    print_concept_sectors(con_df)

    # 取热点板块成分股
    top_sector_names = []
    for sname in ind_df['行业'].head(6):
        if sname not in top_sector_names:
            top_sector_names.append(sname)
    for sname in con_df['行业'].head(6):
        if sname not in top_sector_names:
            top_sector_names.append(sname)

    if top_sector_names and label_map:
        print(f"{'─'*72}")
        sector_stocks = get_top_stocks_in_sectors(top_sector_names, label_map, top_n=8)
        print_sector_stocks(sector_stocks, top_sector_n=6, top_stock_n=8)

    if failed:
        print(f"⚠️ 失败模块: {', '.join(failed)}")

    # 保存缓存
    try:
        import json
        cache_file = CACHE_DIR / f"hot_sector_{date.today()}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                'industry': ind_df.to_dict(orient='records') if not ind_df.empty else [],
                'concept': con_df.to_dict(orient='records') if not con_df.empty else [],
            }, f, ensure_ascii=False)
        print(f"\n💾 缓存已保存: {cache_file}")
    except Exception as e:
        print(f"⚠️ 缓存保存失败: {e}")

# ── 主程序 ───────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='当日市场资金热点板块 & 个股实时监控',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python hot_sector.py                              # 一次性热点报告
  python hot_sector.py --mode monitor               # 持续监控（每30秒）
  python hot_sector.py --mode monitor -i 60        # 每60秒刷新
  python hot_sector.py --mode monitor -n 8         # 每板块显示8只
  python hot_sector.py --cache                      # 使用今日缓存
        """
    )
    parser.add_argument('--mode',    choices=['once','monitor'], default='once',
                        help='once=一次性报告, monitor=持续监控')
    parser.add_argument('-i', '--interval', type=int, default=30,
                        help='监控刷新间隔秒数（默认30）')
    parser.add_argument('-n', '--top-stock', type=int, default=8,
                        help='每个板块显示的股票数量（默认8）')
    parser.add_argument('--cache',  action='store_true', help='使用今日缓存数据')
    args = parser.parse_args()

    if args.cache:
        cache_file = CACHE_DIR / f"hot_sector_{date.today()}.json"
        if not cache_file.exists():
            print("无今日缓存，请去掉 --cache 参数运行")
            return
        import json
        with open(cache_file) as f:
            data = json.load(f)
        print(f"[使用缓存: {cache_file.name}]\n")
        ind_df = pd.DataFrame(data.get('industry', []))
        con_df = pd.DataFrame(data.get('concept', []))
        if ind_df.empty and con_df.empty:
            print("缓存数据为空，请去掉 --cache 重试")
            return
        print_market_overview()
        print_hot_summary(ind_df, con_df)
        print_industry_sectors(ind_df)
        print_concept_sectors(con_df)
        label_map = {}
        try:
            label_map = get_spot_label_map()
        except Exception:
            pass
        if label_map:
            top_sector_names = []
            for sname in ind_df['行业'].head(6):
                if sname not in top_sector_names:
                    top_sector_names.append(sname)
            for sname in con_df['行业'].head(6):
                if sname not in top_sector_names:
                    top_sector_names.append(sname)
            sector_stocks = get_top_stocks_in_sectors(top_sector_names, label_map, top_n=args.top_stock)
            print_sector_stocks(sector_stocks, top_sector_n=6, top_stock_n=args.top_stock)
        return

    if args.mode == 'monitor':
        print(f"\n🚀 启动持续监控（每 {args.interval} 秒，Ctrl+C 停止）\n")
        monitor_loop(interval=args.interval, top_sector_n=6, top_stock_n=args.top_stock)
    else:
        report_once()

if __name__ == '__main__':
    main()