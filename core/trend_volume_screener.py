#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势放量筛选器 - 可调节参数
专注于真正的上升趋势和放量上涨
"""

import os
import struct
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# 导入板块信息模块
try:
    # 尝试多种导入方式
    import sys
    import os
    
    # 方法1: 相对导入（当作为模块导入时）
    try:
        from .stock_sector import get_sector_info
        HAS_SECTOR_INFO = True
        print("✅ 板块信息模块已加载 (相对导入)")
    except ImportError:
        # 方法2: 绝对导入（当直接运行时）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from stock_sector import get_sector_info
        HAS_SECTOR_INFO = True
        print("✅ 板块信息模块已加载 (绝对导入)")
except ImportError as e:
    # 如果无法导入，创建虚拟函数
    HAS_SECTOR_INFO = False
    print(f"⚠️ 无法加载板块信息模块: {e}")
    class DummySectorInfo:
        def get_stock_sector_info(self, code, name=""):
            return {
                'main_sector': '未知',
                'sector_hotness': 30,
                'sector_category': '其他',
                'sector_popularity': 30,
                'source': 'dummy',
            }
        def format_sector_info(self, sector_info):
            return "板块: 未知 (模块未加载)"
    
    def get_sector_info():
        return DummySectorInfo()

# 数据目录
TDX_DIR = "/mnt/d/new_tdx/vipdoc/"
WORK_DIR = "./"
RESULTS_DIR = os.path.join(WORK_DIR, "results/current")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 股票代码文件
STOCK_CODES_FILE = "/home/hfie/stock_code/results/stock_codes.txt"

class TrendVolumeScreener:
    def __init__(self, params: Dict = None):
        # 默认参数
        self.params = {
            # 趋势参数
            'min_trend_days': 5,           # 最小上升趋势天数
            'price_above_ma': 'ma5',       # 价格在哪个均线之上: ma5, ma10, ma20
            'ma_trend': True,              # 是否要求均线多头排列
            
            # 三天表现参数
            'min_three_day_change': 8.0,   # 最小三天涨幅(%)
            'max_three_day_change': 30.0,  # 最大三天涨幅(%)
            'min_up_days': 2,              # 最小上涨天数(3天内)
            
            # 量能参数
            'min_volume_ratio': 1.0,       # 最小平均量比
            'consecutive_volume': False,   # 是否要求连续三天放量
            
            # 风险参数
            'max_ten_day_change': 40.0,    # 最大十日涨幅(%)
            
            # 其他参数
            'min_price': 3.0,              # 最小价格(元)
            'max_price': 200.0,            # 最大价格(元)
            'min_score': 70.0,             # 最小综合评分
        }
        
        # 更新用户参数
        if params:
            self.params.update(params)
    
    def read_tdx_day(self, code: str) -> Optional[pd.DataFrame]:
        """读取通达信日线数据"""
        if code.startswith('sh'):
            market = 'sh'
            code_num = code[2:]
        elif code.startswith('sz'):
            market = 'sz'
            code_num = code[2:]
        else:
            if code.startswith('6'):
                market = 'sh'
                code_num = code
            else:
                market = 'sz'
                code_num = code
        
        path = os.path.join(TDX_DIR, market, "lday", f"{market}{code_num}.day")
        
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, 'rb') as f:
                data = f.read()
            
            rows = []
            for i in range(0, len(data), 32):
                d = data[i:i+32]
                if len(d) < 32:
                    continue

                date_int = struct.unpack('I', d[0:4])[0]
                open_price = struct.unpack('I', d[4:8])[0] / 100.0
                high = struct.unpack('I', d[8:12])[0] / 100.0
                low = struct.unpack('I', d[12:16])[0] / 100.0
                close = struct.unpack('I', d[16:20])[0] / 100.0
                volume = struct.unpack('I', d[24:28])[0]
                rows.append([date_int, open_price, high, low, close, volume])
            
            if not rows:
                return None
                
            df = pd.DataFrame(rows, columns=['date_int','open','high','low','close','volume'])
            df['date'] = pd.to_datetime(df['date_int'].astype(str))
            df = df.sort_values("date").reset_index(drop=True)
            
            df = df[(df['close'] > 0) & (df['volume'] > 0)]
            return df
            
        except Exception:
            return None
    
    def get_stock_name(self, code: str) -> str:
        """获取股票名称（优化网络请求版）"""
        # 首先尝试从本地数据库获取
        local_name = self._get_stock_name_from_local(code)
        if local_name != "未知":
            return local_name
        
        # 如果本地数据库没有，尝试优化的网络请求
        try:
            # 转换股票代码格式
            if code.startswith('sh'):
                api_code = f'sh{code[2:]}'
            elif code.startswith('sz'):
                api_code = f'sz{code[2:]}'
            else:
                if code.startswith('6'):
                    api_code = f'sh{code}'
                else:
                    api_code = f'sz{code}'
            
            # 使用新浪财经接口
            url = f'http://hq.sinajs.cn/list={api_code}'
            
            # 设置请求头，避免被屏蔽
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'http://finance.sina.com.cn/',
                'Accept': '*/*',
                'Accept-Language': 'zh-CN,zh;q=0.9',
            }
            
            # 使用会话保持连接
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=3, verify=False)
            
            if response.status_code == 200 and response.text:
                # 解析返回数据
                match = re.search(r'="([^"]+)"', response.text)
                if match:
                    parts = match.group(1).split(',')
                    if len(parts) > 0 and parts[0]:
                        name = parts[0].strip()
                        if name and name not in ['', 'null', 'NULL', 'None']:
                            # 缓存获取到的名称
                            self._cache_stock_name(code, name)
                            return name
                
                # 如果新浪接口失败，尝试东方财富接口
                return self._get_stock_name_from_eastmoney(code)
                
        except requests.exceptions.Timeout:
            print(f"⏰ {code}: 股票名称请求超时")
        except requests.exceptions.ConnectionError:
            print(f"🔌 {code}: 股票名称连接错误")
        except Exception as e:
            print(f"⚠️ {code}: 股票名称获取异常 - {e}")
        
        return "未知"
    
    def _get_stock_name_from_eastmoney(self, code: str) -> str:
        """从东方财富获取股票名称"""
        try:
            # 转换股票代码格式
            if code.startswith('sh'):
                stock_code = code[2:]
                market = 'sh'
            elif code.startswith('sz'):
                stock_code = code[2:]
                market = 'sz'
            else:
                return "未知"
            
            # 东方财富接口
            url = f'http://quote.eastmoney.com/{market}{stock_code}.html'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'http://quote.eastmoney.com/',
            }
            
            response = requests.get(url, headers=headers, timeout=3, verify=False)
            if response.status_code == 200:
                html = response.text
                
                # 从JavaScript变量中提取股票名称
                pattern = r'var quotedata = ({[^}]+})'
                match = re.search(pattern, html)
                if match:
                    import json
                    try:
                        quotedata = json.loads(match.group(1))
                        name = quotedata.get('name')
                        if name and name not in ['', 'null', 'NULL', 'None']:
                            # 缓存获取到的名称
                            self._cache_stock_name(code, name)
                            return name
                    except:
                        pass
                
                # 从HTML标题中提取
                title_pattern = r'<title>([^<]+)</title>'
                title_match = re.search(title_pattern, html)
                if title_match:
                    title = title_match.group(1)
                    # 提取股票名称（通常格式为"股票名称(股票代码) - 东方财富网"）
                    name_match = re.search(r'^([^(]+)', title)
                    if name_match:
                        name = name_match.group(1).strip()
                        if name and len(name) < 20:  # 股票名称通常不会太长
                            self._cache_stock_name(code, name)
                            return name
        except:
            pass
        
        return "未知"
    
    def _is_st_stock(self, stock_name: str) -> bool:
        """判断是否为ST股票"""
        if not stock_name:
            return False
        
        # 检查是否包含ST标识
        st_keywords = ['ST', '*ST', 'st', '*st', '退市']
        for keyword in st_keywords:
            if keyword in stock_name:
                return True
        
        return False
    
    def _is_ma60_rising(self, df: pd.DataFrame, lookback_days: int = 5) -> bool:
        """判断60日均线是否上涨
        参数:
            df: 股票数据DataFrame
            lookback_days: 查看最近多少天的MA60趋势，默认5天
        """
        if len(df) < 60 + lookback_days:
            return False
        
        # 计算MA60
        df['ma60'] = df['close'].rolling(60).mean()
        
        # 获取最近lookback_days天的MA60值
        ma60_values = df['ma60'].iloc[-lookback_days:].values
        
        # 检查是否有NaN值
        if any(pd.isna(v) for v in ma60_values):
            return False
        
        # 方法1: 简单检查 - 当前MA60是否大于前一天的MA60
        simple_rising = ma60_values[-1] > ma60_values[-2]
        
        # 方法2: 趋势检查 - 最近lookback_days天的MA60是否整体呈上升趋势
        # 计算线性回归斜率
        x = np.arange(len(ma60_values))
        y = ma60_values
        slope = np.polyfit(x, y, 1)[0]  # 线性回归的斜率
        
        # 如果斜率为正，说明MA60整体呈上升趋势
        trend_rising = slope > 0
        
        # 两种条件都满足才认为是真正的上涨
        return simple_rising and trend_rising
    
    def _cache_stock_name(self, code: str, name: str):
        """缓存股票名称到本地文件"""
        try:
            import json
            import os
            
            cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_name_cache.json")
            
            # 读取现有缓存
            cache = {}
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            
            # 更新缓存
            cache[code] = name
            
            # 保存缓存
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
                
        except:
            pass  # 缓存失败不影响主要功能
    
    def _get_stock_name_from_local(self, code: str) -> str:
        """从本地数据库获取股票名称（简化版）"""
        # 首先尝试从缓存文件读取
        try:
            import json
            import os
            
            cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_name_cache.json")
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                    if code in cache:
                        return cache[code]
        except:
            pass
        
        # 如果缓存中没有，使用最小化的内置数据
        common_stocks = {
            # 最常见的几只股票
            'sh600519': '贵州茅台', 'sz000001': '平安银行', 'sz002460': '赣锋锂业',
            'sh600036': '招商银行', 'sz000858': '五粮液', 'sh600030': '中信证券',
            'sz300750': '宁德时代', 'sh600276': '恒瑞医药', 'sh600000': '浦发银行',
            'sh600016': '民生银行', 'sh600028': '中国石化', 'sh600031': '三一重工',
            'sh600048': '保利发展', 'sh600050': '中国联通', 'sh600104': '上汽集团',
            'sh600309': '万华化学', 'sh600547': '山东黄金', 'sh600585': '海螺水泥',
            'sh600690': '海尔智家', 'sh600837': '海通证券', 'sh600887': '伊利股份',
            'sh600900': '长江电力', 'sh601318': '中国平安', 'sh601328': '交通银行',
            'sh601398': '工商银行', 'sh601668': '中国建筑', 'sh601857': '中国石油',
            'sh601888': '中国国旅', 'sh601919': '中远海控', 'sh601988': '中国银行',
            'sh601998': '中信银行', 'sz000002': '万科A', 'sz000063': '中兴通讯',
            'sz000066': '中国长城', 'sz000100': 'TCL科技', 'sz000157': '中联重科',
            'sz000333': '美的集团', 'sz000338': '潍柴动力', 'sz000425': '徐工机械',
            'sz000538': '云南白药', 'sz000568': '泸州老窖', 'sz000625': '长安汽车',
            'sz000651': '格力电器', 'sz000725': '京东方A', 'sz000876': '新希望',
            'sz000895': '双汇发展', 'sz000938': '紫光股份', 'sz000963': '华东医药',
            'sz000977': '浪潮信息', 'sz002024': '苏宁易购', 'sz002142': '宁波银行',
            'sz002230': '科大讯飞', 'sz002241': '歌尔股份', 'sz002304': '洋河股份',
            'sz002415': '海康威视', 'sz002475': '立讯精密', 'sz002594': '比亚迪',
            'sz002714': '牧原股份', 'sz002736': '国信证券', 'sz300059': '东方财富',
            'sz300122': '智飞生物', 'sz300124': '汇川技术', 'sz300142': '沃森生物',
            'sz300347': '泰格医药', 'sz300498': '温氏股份', 'sz300760': '迈瑞医疗',
        }
        
        return common_stocks.get(code, "未知")
    
    def analyze_trend(self, df: pd.DataFrame, lookback: int = 60) -> Dict:
        """分析趋势（需要至少60天数据来计算MA60）"""
        if len(df) < lookback:
            return {}
        
        recent = df.iloc[-lookback:].copy()
        recent = recent.reset_index(drop=True)
        
        # 计算均线
        recent['ma5'] = recent['close'].rolling(5).mean()
        recent['ma10'] = recent['close'].rolling(10).mean()
        recent['ma20'] = recent['close'].rolling(20).mean()
        recent['ma60'] = recent['close'].rolling(60).mean()
        
        # 计算量能指标
        recent['vol_ma5'] = recent['volume'].rolling(5).mean()
        recent['volume_ratio'] = recent['volume'] / recent['vol_ma5']
        
        # 计算涨跌幅
        recent['change_pct'] = recent['close'].pct_change() * 100
        
        latest = recent.iloc[-1]
        
        # 1. 价格位置分析
        price_above_ma5 = latest['close'] > latest['ma5']
        price_above_ma10 = latest['close'] > latest['ma10']
        price_above_ma20 = latest['close'] > latest['ma20']
        price_above_ma60 = latest['close'] > latest['ma60']
        
        # 2. 均线排列分析
        ma5_above_ma10 = latest['ma5'] > latest['ma10']
        ma10_above_ma20 = latest['ma10'] > latest['ma20']
        ma20_above_ma60 = latest['ma20'] > latest['ma60']
        
        # 3. 趋势强度分析
        # 计算最近N日的上涨天数
        trend_days = self.params.get('min_trend_days', 5)
        if len(recent) >= trend_days:
            last_n = recent.iloc[-trend_days:]
            up_days_in_trend = sum(last_n['change_pct'] > 0)
            trend_strength = up_days_in_trend / trend_days
        else:
            trend_strength = 0
        
        # 4. 量能趋势分析
        if len(recent) >= 5:
            last_5 = recent.iloc[-5:]
            volume_trend = last_5['volume_ratio'].mean()
            volume_increasing = last_5['volume_ratio'].iloc[-1] > last_5['volume_ratio'].iloc[0]
        else:
            volume_trend = 0
            volume_increasing = False
        
        return {
            'price_above_ma5': price_above_ma5,
            'price_above_ma10': price_above_ma10,
            'price_above_ma20': price_above_ma20,
            'price_above_ma60': price_above_ma60,
            'ma5_above_ma10': ma5_above_ma10,
            'ma10_above_ma20': ma10_above_ma20,
            'ma20_above_ma60': ma20_above_ma60,
            'trend_strength': trend_strength,
            'volume_trend': volume_trend,
            'volume_increasing': volume_increasing,
            'latest_price': latest['close'],
            'latest_change': latest['change_pct'],
            'ma5': latest['ma5'],
            'ma10': latest['ma10'],
            'ma20': latest['ma20'],
            'ma60': latest['ma60'],
        }
    
    def analyze_three_day_performance(self, df: pd.DataFrame) -> Dict:
        """分析最近三天表现
        说明：
        - 三天涨幅 = 今天收盘价 相对 3 个交易日前收盘价 的涨跌幅
        - 十日涨幅 = 今天收盘价 相对 10 个交易日前收盘价 的涨跌幅
        这样统计的才是完整区间涨幅，而不是少算一天。
        """
        if len(df) < 4:
            return {}
        
        # 至少保留11天，便于正确计算10日涨幅（需要取到10个交易日前的收盘价）
        recent = df.iloc[-11:].copy()
        recent = recent.reset_index(drop=True)
        
        # 计算量能指标
        recent['vol_ma5'] = recent['volume'].rolling(5).mean()
        recent['volume_ratio'] = recent['volume'] / recent['vol_ma5']
        recent['change_pct'] = recent['close'].pct_change() * 100
        
        # 分析最近三天（这3天分别对应 3 个单日涨跌幅）
        last_3 = recent.iloc[-3:]
        
        # 三天累计涨幅：从3个交易日前收盘价算到今天收盘价
        start_price = recent.iloc[-4]['close']
        end_price = recent.iloc[-1]['close']
        three_day_change = (end_price - start_price) / start_price * 100
        
        # 上涨天数：最近3个交易日中上涨的天数
        up_days = int((last_3['change_pct'] > 0).sum())
        
        # 是否连续上涨
        consecutive_up = bool((last_3['change_pct'] > 0).all())
        
        # 量能分析
        avg_volume_ratio = last_3['volume_ratio'].mean()
        consecutive_volume = bool((last_3['volume_ratio'] > 1.0).all())
        volume_increasing = last_3['volume_ratio'].iloc[-1] > last_3['volume_ratio'].iloc[0]
        
        # 十日涨幅（风险控制）：从10个交易日前收盘价算到今天收盘价
        if len(recent) >= 11:
            start_price_10 = recent.iloc[-11]['close']
            end_price_10 = recent.iloc[-1]['close']
            ten_day_change = (end_price_10 - start_price_10) / start_price_10 * 100
        else:
            ten_day_change = 0
        
        return {
            'three_day_change': three_day_change,
            'up_days': up_days,
            'consecutive_up': consecutive_up,
            'avg_volume_ratio': avg_volume_ratio,
            'consecutive_volume': consecutive_volume,
            'volume_increasing': volume_increasing,
            'ten_day_change': ten_day_change,
        }
    
    def evaluate_stock(self, code: str) -> Optional[Dict]:
        """评估单只股票"""
        df = self.read_tdx_day(code)
        if df is None or len(df) < 61:  # 需要至少61天数据来计算MA60
            return None
        
        # 获取股票名称
        stock_name = self.get_stock_name(code)
        
        # 1. 排除ST股票
        if self._is_st_stock(stock_name):
            return None
        
        # 分析趋势
        trend_analysis = self.analyze_trend(df)
        if not trend_analysis:
            return None
        
        # 分析三天表现
        three_day_analysis = self.analyze_three_day_performance(df)
        if not three_day_analysis:
            return None
        
        # 获取最新价格
        latest_price = trend_analysis['latest_price']
        
        # 2. 价格过滤
        if latest_price < self.params['min_price'] or latest_price > self.params['max_price']:
            return None
        
        # 3. 检查60日均线上涨
        if not self._is_ma60_rising(df):
            return None
        
        # 4. 检查短期均线多头排列 (MA5 > MA10 > MA20)
        if not (trend_analysis['ma5_above_ma10'] and trend_analysis['ma10_above_ma20']):
            return None
        
        # 5. 检查所有均线都在60日均线之上 (MA5 > MA10 > MA20 > MA60)
        if not (trend_analysis['ma5'] > trend_analysis['ma10'] > 
                trend_analysis['ma20'] > trend_analysis['ma60']):
            return None
        
        # 6. 趋势条件检查（原有逻辑）
        trend_ok = False
        ma_type = self.params['price_above_ma']
        
        if ma_type == 'ma5':
            trend_ok = trend_analysis['price_above_ma5']
        elif ma_type == 'ma10':
            trend_ok = trend_analysis['price_above_ma10']
        elif ma_type == 'ma20':
            trend_ok = trend_analysis['price_above_ma20']
        
        # 均线多头排列条件
        if self.params['ma_trend']:
            if ma_type == 'ma5':
                trend_ok = trend_ok and trend_analysis['ma5_above_ma10']
            elif ma_type == 'ma10':
                trend_ok = trend_ok and trend_analysis['ma10_above_ma20']
        
        # 趋势强度条件
        trend_ok = trend_ok and (trend_analysis['trend_strength'] >= 0.6)
        
        if not trend_ok:
            return None
        
        # 3. 三天表现条件检查
        three_day_change = three_day_analysis['three_day_change']
        up_days = three_day_analysis['up_days']
        avg_volume_ratio = three_day_analysis['avg_volume_ratio']
        
        three_day_ok = (
            self.params['min_three_day_change'] <= three_day_change <= self.params['max_three_day_change'] and
            up_days >= self.params['min_up_days'] and
            avg_volume_ratio >= self.params['min_volume_ratio']
        )
        
        # 连续放量条件
        if self.params['consecutive_volume']:
            three_day_ok = three_day_ok and three_day_analysis['consecutive_volume']
        
        if not three_day_ok:
            return None
        
        # 4. 风险条件检查
        ten_day_change = three_day_analysis['ten_day_change']
        risk_ok = ten_day_change <= self.params['max_ten_day_change']
        
        if not risk_ok:
            return None
        
        # 计算综合评分
        score = self.calculate_score(trend_analysis, three_day_analysis)
        
        if score < self.params['min_score']:
            return None
        
        # 获取板块信息（包含热度和人气）
        sector_info = get_sector_info().get_stock_sector_info(code, stock_name)
        
        # 返回股票信息
        return {
            'code': code,
            'name': stock_name,
            'score': score,
            'latest_price': latest_price,
            'latest_change': trend_analysis['latest_change'],
            'three_day_change': three_day_change,
            'ten_day_change': ten_day_change,
            'up_days': up_days,
            'consecutive_up': three_day_analysis['consecutive_up'],
            'avg_volume_ratio': avg_volume_ratio,
            'consecutive_volume': three_day_analysis['consecutive_volume'],
            'volume_increasing': three_day_analysis['volume_increasing'],
            'trend_strength': trend_analysis['trend_strength'],
            'price_above_ma5': trend_analysis['price_above_ma5'],
            'price_above_ma10': trend_analysis['price_above_ma10'],
            'price_above_ma20': trend_analysis['price_above_ma20'],
            'ma5_above_ma10': trend_analysis['ma5_above_ma10'],
            # 板块信息（包含热度和人气）
            'main_sector': sector_info.get('main_sector', '未知'),
            'sector_hotness': sector_info.get('sector_hotness', 40),
            'sector_popularity': sector_info.get('sector_popularity', 30),
            'sector_category': sector_info.get('sector_category', '其他'),
            'sector_source': sector_info.get('source', 'unknown'),
            'ma10_above_ma20': trend_analysis['ma10_above_ma20'],
            'ma5': trend_analysis['ma5'],
            'ma10': trend_analysis['ma10'],
            'ma20': trend_analysis['ma20'],
        }
    
    def calculate_score(self, trend_analysis: Dict, three_day_analysis: Dict) -> float:
        """计算综合评分"""
        score = 0
        max_score = 0
        
        # 趋势分 (40%)
        max_score += 40
        
        # 价格位置
        if trend_analysis['price_above_ma5']:
            score += 5
        if trend_analysis['price_above_ma10']:
            score += 10
        if trend_analysis['price_above_ma20']:
            score += 15
        if trend_analysis['price_above_ma60']:
            score += 10
        
        # 均线排列
        if trend_analysis['ma5_above_ma10']:
            score += 5
        if trend_analysis['ma10_above_ma20']:
            score += 5
        if trend_analysis['ma20_above_ma60']:
            score += 5
        
        # 趋势强度
        trend_strength = trend_analysis['trend_strength']
        score += trend_strength * 15  # 最高15分
        
        # 量能趋势
        if trend_analysis['volume_increasing']:
            score += 5
        
        # 三天表现分 (40%)
        max_score += 40
        
        three_day_change = three_day_analysis['three_day_change']
        up_days = three_day_analysis['up_days']
        avg_volume_ratio = three_day_analysis['avg_volume_ratio']
        
        # 涨幅分
        if three_day_change >= 5.0:
            score += 15
        elif three_day_change >= 3.0:
            score += 10
        elif three_day_change > 0:
            score += 5
        
        # 上涨天数分
        if up_days == 3:
            score += 10
        elif up_days == 2:
            score += 7
        elif up_days == 1:
            score += 3
        
        # 量能分
        if avg_volume_ratio >= 1.5:
            score += 15
        elif avg_volume_ratio >= 1.2:
            score += 12
        elif avg_volume_ratio >= 1.0:
            score += 8
        elif avg_volume_ratio >= 0.8:
            score += 5
        
        # 连续上涨/放量加分
        if three_day_analysis['consecutive_up']:
            score += 5
        if three_day_analysis['consecutive_volume']:
            score += 5
        
        # 风险控制分 (20%)
        max_score += 20
        
        ten_day_change = three_day_analysis['ten_day_change']
        
        # 十日涨幅控制
        if ten_day_change <= 20.0:
            score += 15
        elif ten_day_change <= 30.0:
            score += 10
        elif ten_day_change <= 40.0:
            score += 5
        
        # 价格稳定性（避免暴涨暴跌）
        if three_day_change <= 25.0:
            score += 5
        
        # 计算总分
        total_score = (score / max_score) * 100 if max_score > 0 else 0
        
        return total_score
    
    def load_stock_codes(self, sample_size: int = 800, all_stocks: bool = False) -> List[str]:
        """加载股票代码样本"""
        codes = []
        try:
            with open(STOCK_CODES_FILE, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            
            if all_stocks or sample_size == 0:
                # 扫描全部股票
                for line in all_lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if line.startswith('sh') or line.startswith('sz'):
                            codes.append(line)
            else:
                # 均匀采样
                total_lines = len(all_lines)
                step = max(1, total_lines // sample_size)
                
                for i in range(0, min(total_lines, sample_size * step), step):
                    line = all_lines[i].strip()
                    if line and not line.startswith('#'):
                        if line.startswith('sh') or line.startswith('sz'):
                            codes.append(line)
            
            return codes
        except Exception:
            # 返回测试代码
            return [
                'sh600098', 'sh600107', 'sh600167', 'sh600236', 'sh600250',
                'sh600698', 'sh600719', 'sh600780', 'sh600792', 'sh600844',
                'sh600869', 'sh601869', 'sh601908', 'sh600136', 'sh600163',
                'sh600241', 'sh600310', 'sh600316', 'sh600207', 'sh600032',
                'sh600023', 'sh600017', 'sh600125', 'sh600257', 'sh600323',
                'sh600358', 'sh600428', 'sh600433', 'sh600468', 'sh600477',
            ]
    
    def run_screening(self, codes: List[str], max_workers: int = 15) -> List[Dict]:
        """运行筛选"""
        selected_stocks = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.evaluate_stock, code): code for code in codes}
            
            for i, future in enumerate(as_completed(futures), 1):
                code = futures[future]
                
                try:
                    result = future.result(timeout=10)
                    if result:
                        selected_stocks.append(result)
                    
                    # 显示进度
                    if i % 50 == 0:
                        print(f"  已处理 {i}/{len(codes)} 只，筛选出 {len(selected_stocks)} 只")
                        
                except Exception:
                    pass
        
        return selected_stocks
    
    def print_results(self, stocks: List[Dict], total_codes: int):
        """打印结果"""
        if not stocks:
            print("❌ 未找到符合条件的股票")
            return
        
        # 按评分排序
        stocks.sort(key=lambda x: x['score'], reverse=True)
        
        print("=" * 80)
        print("🎯 筛选结果")
        print("=" * 80)
        
        # 显示前30只
        display_count = min(30, len(stocks))
        
        print(f"📈 找到 {len(stocks)} 只符合条件的股票，显示前 {display_count} 只:")
        print()
        
        for i, stock in enumerate(stocks[:display_count], 1):
            # 趋势描述
            trend_desc = []
            if stock['price_above_ma5']:
                trend_desc.append("MA5↑")
            if stock['price_above_ma10']:
                trend_desc.append("MA10↑")
            if stock['price_above_ma20']:
                trend_desc.append("MA20↑")
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
            print(f"     趋势: {stock['trend_strength']:.1%} {' '.join(trend_desc)}")
            print(f"     板块: {main_sector} ({sector_category})")
            print(f"     热度: {hotness_emoji}{hotness_desc}({hotness}) 人气: {popularity_emoji}{popularity_desc}({popularity})")
            print()
        
        # 统计信息
        print(f"\n📊 统计信息:")
        print(f"   扫描股票: {total_codes}只")
        print(f"   筛选出: {len(stocks)}只")
        print(f"   筛选比例: {len(stocks)/total_codes*100:.2f}%")
        
        if stocks:
            scores = [s['score'] for s in stocks]
            three_day_changes = [s['three_day_change'] for s in stocks]
            volume_ratios = [s['avg_volume_ratio'] for s in stocks]
            ten_day_changes = [s['ten_day_change'] for s in stocks]
            trend_strengths = [s['trend_strength'] for s in stocks]
            
            print(f"\n🔍 特征统计:")
            print(f"   平均评分: {np.mean(scores):.1f}")
            print(f"   平均三天涨幅: {np.mean(three_day_changes):.2f}%")
            print(f"   平均量比: {np.mean(volume_ratios):.2f}")
            print(f"   平均十日涨幅: {np.mean(ten_day_changes):.2f}%")
            print(f"   平均趋势强度: {np.mean(trend_strengths):.1%}")
            
            # 优质特征
            high_score = sum(1 for s in stocks if s['score'] >= 80)
            consecutive_up = sum(1 for s in stocks if s['consecutive_up'])
            consecutive_volume = sum(1 for s in stocks if s['consecutive_volume'])
            strong_trend = sum(1 for s in stocks if s['trend_strength'] >= 0.8)
            
            print(f"\n⭐ 优质特征:")
            print(f"   高分股票(≥80分): {high_score}只 ({high_score/len(stocks)*100:.1f}%)")
            print(f"   强势趋势(≥80%): {strong_trend}只 ({strong_trend/len(stocks)*100:.1f}%)")
            print(f"   连续三天上涨: {consecutive_up}只 ({consecutive_up/len(stocks)*100:.1f}%)")
            print(f"   连续三天放量: {consecutive_volume}只 ({consecutive_volume/len(stocks)*100:.1f}%)")
        
        # TOP 5推荐
        if len(stocks) >= 5:
            print(f"\n🏆 TOP 5 推荐:")
            for i, stock in enumerate(stocks[:5], 1):
                print(f"   {i}. {stock['code']} {stock['name']} (评分:{stock['score']:.1f})")
                print(f"      三天: {stock['three_day_change']:+.2f}% | 量比: {stock['avg_volume_ratio']:.2f}")
                print(f"      趋势强度: {stock['trend_strength']:.1%}")
    
    def save_results(self, stocks: List[Dict], total_codes: int):
        """保存结果"""
        if not stocks:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"trend_volume_stocks_{timestamp}"
        
        # 1. 保存详细结果文件
        detailed_file = os.path.join(RESULTS_DIR, f"{base_name}.txt")
        
        with open(detailed_file, "w", encoding='utf-8') as f:
            f.write(f"# 趋势放量筛选结果\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 筛选参数:\n")
            for key, value in self.params.items():
                f.write(f"#   {key}: {value}\n")
            f.write(f"# 扫描股票: {total_codes}只\n")
            f.write(f"# 筛选出: {len(stocks)}只\n")
            f.write(f"\n")
            
            for stock in stocks:
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
                f.write(f"  十日涨幅: {stock['ten_day_change']:+.2f}% 趋势强度: {stock['trend_strength']:.1%}\n")
                f.write(f"  板块: {main_sector} ({sector_category})\n")
                f.write(f"  热度: {hotness_desc}({hotness}) 人气: {popularity_desc}({popularity})\n")
                f.write(f"  MA5: {stock['ma5']:.2f} MA10: {stock['ma10']:.2f} MA20: {stock['ma20']:.2f}\n")
                f.write(f"\n")
        
        # 2. 保存股票代码列表文件（只有代码）
        codes_file = os.path.join(RESULTS_DIR, f"{base_name}_codes.txt")
        
        with open(codes_file, "w", encoding='utf-8') as f:
            f.write(f"# 股票代码列表\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 筛选参数: 同{base_name}.txt\n")
            f.write(f"# 股票数量: {len(stocks)}只\n")
            f.write(f"\n")
            
            for stock in stocks:
                f.write(f"{stock['code']}\n")
        
        # 3. 保存按板块分类的代码列表
        sector_codes_file = os.path.join(RESULTS_DIR, f"{base_name}_sector_codes.txt")
        
        # 按板块分类
        sector_dict = {}
        for stock in stocks:
            sector = stock.get('main_sector', '未知')
            if sector not in sector_dict:
                sector_dict[sector] = []
            sector_dict[sector].append(stock['code'])
        
        with open(sector_codes_file, "w", encoding='utf-8') as f:
            f.write(f"# 按板块分类的股票代码\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n")
            
            # 按板块股票数量排序
            for sector, codes in sorted(sector_dict.items(), key=lambda x: len(x[1]), reverse=True):
                f.write(f"## {sector} ({len(codes)}只)\n")
                for code in codes:
                    f.write(f"{code}\n")
                f.write(f"\n")
        
        print(f"\n💾 结果文件已保存:")
        print(f"   详细结果: {detailed_file}")
        print(f"   股票代码: {codes_file}")
        print(f"   板块分类: {sector_codes_file}")

def main():
    parser = argparse.ArgumentParser(description="趋势放量股票筛选器")
    
    # 趋势参数
    parser.add_argument('--price-above', choices=['ma5', 'ma10', 'ma20'], default='ma5',
                       help='价格在哪个均线之上 (默认: ma5)')
    parser.add_argument('--ma-trend', action='store_true', default=True,
                       help='要求均线多头排列 (默认: True)')
    parser.add_argument('--min-trend-days', type=int, default=5,
                       help='最小上升趋势天数 (默认: 5)')
    
    # 三天表现参数
    parser.add_argument('--min-three-day', type=float, default=8.0,
                       help='最小三天涨幅(%) (默认: 8.0)')
    parser.add_argument('--max-three-day', type=float, default=30.0,
                       help='最大三天涨幅(%) (默认: 30.0)')
    parser.add_argument('--min-up-days', type=int, default=2,
                       help='最小上涨天数(3天内) (默认: 2)')
    
    # 量能参数
    parser.add_argument('--min-volume-ratio', type=float, default=1.0,
                       help='最小平均量比 (默认: 1.0)')
    parser.add_argument('--consecutive-volume', action='store_true', default=False,
                       help='要求连续三天放量 (默认: False)')
    
    # 风险参数
    parser.add_argument('--max-ten-day', type=float, default=40.0,
                       help='最大十日涨幅(%) (默认: 40.0)')
    
    # 其他参数
    parser.add_argument('--min-price', type=float, default=3.0,
                       help='最小价格(元) (默认: 3.0)')
    parser.add_argument('--max-price', type=float, default=200.0,
                       help='最大价格(元) (默认: 200.0)')
    parser.add_argument('--min-score', type=float, default=70.0,
                       help='最小综合评分 (默认: 70.0)')
    
    # 运行参数
    parser.add_argument('--sample-size', type=int, default=800,
                       help='采样股票数量 (默认: 800，设为0扫描全部)')
    parser.add_argument('--workers', type=int, default=15,
                       help='并行工作线程数 (默认: 15)')
    parser.add_argument('--all', action='store_true', default=False,
                       help='扫描全部股票 (覆盖sample-size参数)')
    
    args = parser.parse_args()
    
    # 检查是否没有提供任何参数（除了默认值）
    # 如果用户只运行了 `python trend_volume_screener.py`，则默认进行全市场扫描
    default_args = {
        'price_above': 'ma5',
        'ma_trend': True,
        'min_trend_days': 5,
        'min_three_day': 8.0,
        'max_three_day': 30.0,
        'min_up_days': 2,
        'min_volume_ratio': 1.0,
        'consecutive_volume': False,
        'max_ten_day': 40.0,
        'min_price': 3.0,
        'max_price': 200.0,
        'min_score': 70.0,
        'sample_size': 800,
        'workers': 15,
        'all': False,
    }
    
    # 检查用户是否提供了任何非默认参数
    user_provided_args = False
    for arg_name, default_value in default_args.items():
        arg_value = getattr(args, arg_name.replace('-', '_'))
        if arg_value != default_value:
            user_provided_args = True
            break
    
    # 智能判断是否进行全市场扫描
    if not user_provided_args:
        # 情况1：完全没有提供任何参数
        print("=" * 80)
        print("🎯 检测到无参数运行，默认进行全市场扫描")
        print("=" * 80)
        args.all = True
    elif user_provided_args and not args.all and args.sample_size == 800:
        # 情况2：用户提供了筛选参数，但没有指定扫描范围
        # 这种情况下也进行全市场扫描，因为用户可能想用自定义参数扫描全市场
        print("=" * 80)
        print("🎯 检测到自定义筛选参数，自动进行全市场扫描")
        print("💡 提示：如需限制扫描数量，请使用 --sample-size 参数")
        print("=" * 80)
        args.all = True
    
    # 创建参数字典
    params = {
        'price_above_ma': args.price_above,
        'ma_trend': args.ma_trend,
        'min_trend_days': args.min_trend_days,
        'min_three_day_change': args.min_three_day,
        'max_three_day_change': args.max_three_day,
        'min_up_days': args.min_up_days,
        'min_volume_ratio': args.min_volume_ratio,
        'consecutive_volume': args.consecutive_volume,
        'max_ten_day_change': args.max_ten_day,
        'min_price': args.min_price,
        'max_price': args.max_price,
        'min_score': args.min_score,
    }
    
    # 创建筛选器
    screener = TrendVolumeScreener(params)
    
    print("=" * 80)
    print("📈 趋势放量股票筛选器 - 可调节参数版")
    print("=" * 80)
    
    # 显示当前参数
    print("📋 当前筛选参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()
    
    # 加载股票代码
    print("📋 加载股票代码...")
    codes = screener.load_stock_codes(sample_size=args.sample_size, all_stocks=args.all)
    print(f"扫描股票数量: {len(codes)}只")
    print()
    
    # 运行筛选
    print("🔍 开始筛选...")
    print("⏳ 请耐心等待...")
    print()
    
    selected_stocks = screener.run_screening(codes, max_workers=args.workers)
    
    print(f"\n✅ 筛选完成！")
    print(f"   处理股票: {len(codes)}只")
    print(f"   筛选出: {len(selected_stocks)}只")
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
