#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票板块信息获取模块 - 修复版
从多个数据源获取股票所属板块、人气和热度信息
"""

import requests
import re
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import os
import random
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings

# 禁用SSL警告
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

class StockSectorInfo:
    """股票板块信息获取器 - 优化网络请求版"""
    
    def __init__(self, cache_dir: str = None, use_proxy: bool = False, proxy_url: str = None):
        # 缓存目录
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sector_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 用户代理列表（更多选择）
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
        
        # 代理设置
        self.use_proxy = use_proxy
        self.proxy_url = proxy_url
        self.proxies = None
        if use_proxy and proxy_url:
            self.proxies = {
                'http': proxy_url,
                'https': proxy_url,
            }
        
        # 初始化请求会话
        self.session = self._create_session()
        
        # 初始化板块热度数据
        self.init_sector_data()
    
    def _create_session(self):
        """创建优化的HTTP会话"""
        session = requests.Session()
        
        # 设置重试策略
        retry_strategy = Retry(
            total=3,  # 总重试次数
            backoff_factor=1,  # 重试等待时间因子
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码
            allowed_methods=["GET", "POST"]  # 允许重试的方法
        )
        
        # 创建适配器
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=100)
        
        # 挂载适配器
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 设置默认请求头
        session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        return session
    
    def get_random_user_agent(self):
        """获取随机User-Agent"""
        return random.choice(self.user_agents)
    
    def init_sector_data(self):
        """初始化板块数据"""
        # 板块热度映射（基于市场关注度）
        self.sector_hotness = {
            # 高热度板块 (80-100)
            "人工智能": 95, "AI": 95, "芯片": 90, "半导体": 90, "算力": 88,
            "新能源": 85, "光伏": 85, "储能": 83, "锂电池": 82, "新能源汽车": 88,
            "创新药": 82, "医疗器械": 80, "生物医药": 78, "医药": 75,
            "云计算": 84, "大数据": 82, "数字经济": 80, "信创": 78,
            "机器人": 83, "工业母机": 80, "智能制造": 78,
            
            # 中热度板块 (60-79)
            "白酒": 70, "食品饮料": 65, "消费": 68, "家电": 62,
            "证券": 72, "金融科技": 68, "保险": 60, "银行": 58,
            "有色金属": 68, "稀土": 70, "黄金": 65, "煤炭": 62,
            "电力": 64, "电网": 66, "特高压": 68,
            "军工": 72, "航天航空": 70, "船舶": 65,
            "房地产": 55, "基建": 62, "建筑": 58,
            
            # 低热度板块 (40-59)
            "化工": 58, "化肥": 56, "化纤": 52,
            "钢铁": 50, "水泥": 48, "建材": 52,
            "汽车": 56, "零部件": 54, "整车": 52,
            "零售": 48, "商贸": 46, "物流": 50,
            "传媒": 54, "游戏": 56, "影视": 50,
            "农业": 52, "种植": 48, "养殖": 50,
            
            # 冷门板块 (0-39)
            "纺织服装": 38, "轻工制造": 36, "造纸": 34,
            "港口": 32, "航运": 34, "机场": 30,
            "旅游": 42, "酒店": 38, "餐饮": 36,
            "教育": 28, "环保": 40, "公用事业": 35,
        }
        
        # 板块分类
        self.sector_categories = {
            "科技": ["人工智能", "AI", "芯片", "半导体", "算力", "云计算", "大数据", "数字经济", "信创", "5G", "物联网", "区块链", "消费电子"],
            "新能源": ["新能源", "光伏", "储能", "锂电池", "新能源汽车", "氢能源", "风电", "核电", "绿色电力"],
            "医药": ["创新药", "医疗器械", "生物医药", "医药", "中药", "医疗服务", "CRO", "疫苗"],
            "高端制造": ["机器人", "工业母机", "智能制造", "高端装备", "数控机床", "自动化"],
            "消费": ["白酒", "食品饮料", "消费", "家电", "零售", "商贸", "服装", "化妆品"],
            "金融": ["证券", "金融科技", "保险", "银行", "互联网金融"],
            "周期": ["有色金属", "稀土", "黄金", "煤炭", "化工", "化肥", "化纤", "钢铁", "水泥", "建材"],
            "军工": ["军工", "航天航空", "船舶", "国防", "卫星导航"],
            "基建": ["房地产", "基建", "建筑", "工程机械", "轨道交通"],
            "汽车": ["汽车", "零部件", "整车", "新能源汽车", "智能驾驶"],
            "其他": ["农业", "传媒", "游戏", "影视", "旅游", "酒店", "餐饮", "教育", "环保", "公用事业", "物流", "港口", "航运", "机场"],
        }
        
        # 板块人气指标（基于搜索量、讨论热度等）
        self.sector_popularity = {
            # 高人气板块
            "人工智能": 95, "AI": 95, "芯片": 92, "新能源": 90, "白酒": 88,
            "医药": 85, "证券": 82, "光伏": 80, "云计算": 78,
            
            # 中人气板块
            "锂电池": 75, "储能": 72, "机器人": 70, "军工": 68,
            "消费电子": 65, "食品饮料": 62, "有色金属": 60,
            
            # 低人气板块
            "银行": 45, "保险": 42, "房地产": 40, "煤炭": 38,
            "钢铁": 35, "化工": 40, "建筑": 38,
        }
    
    def get_random_user_agent(self):
        """获取随机User-Agent"""
        return random.choice(self.user_agents)
    
    def get_cache_key(self, code: str) -> str:
        """生成缓存键"""
        return hashlib.md5(code.encode()).hexdigest()[:8]
    
    def get_cache_file(self, code: str) -> str:
        """获取缓存文件路径"""
        cache_key = self.get_cache_key(code)
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def load_from_cache(self, code: str) -> Optional[Dict]:
        """从缓存加载数据"""
        cache_file = self.get_cache_file(code)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 检查缓存是否过期（7天）
                    cache_time = datetime.fromisoformat(data.get('cache_time', '2000-01-01'))
                    if (datetime.now() - cache_time).days < 7:
                        return data
            except Exception:
                pass
        return None
    
    def save_to_cache(self, code: str, data: Dict):
        """保存数据到缓存"""
        cache_file = self.get_cache_file(code)
        data['cache_time'] = datetime.now().isoformat()
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def get_stock_sector_from_10jqka(self, code: str) -> Optional[Dict]:
        """从同花顺获取股票板块信息（优化版）"""
        try:
            # 转换股票代码格式
            if code.startswith('sh'):
                stock_code = code[2:]
                market = 'sh'
            elif code.startswith('sz'):
                stock_code = code[2:]
                market = 'sz'
            else:
                return None
            
            # 同花顺接口
            url = f'http://quote.eastmoney.com/{market}{stock_code}.html'
            
            # 设置请求头
            headers = {
                'User-Agent': self.get_random_user_agent(),
                'Referer': 'http://quote.eastmoney.com/',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
            }
            
            # 分级超时设置
            timeout_settings = (5, 10)  # (连接超时, 读取超时)
            
            # 发送请求
            response = self.session.get(
                url, 
                headers=headers, 
                timeout=timeout_settings,
                proxies=self.proxies,
                verify=False  # 跳过SSL验证（谨慎使用）
            )
            
            if response.status_code == 200:
                html = response.text
                
                # 提取板块信息（使用更健壮的正则表达式）
                sectors = []
                
                # 方法1: 从JavaScript变量中提取
                js_pattern = r'var quotedata = ({[^}]+})'
                js_match = re.search(js_pattern, html)
                if js_match:
                    try:
                        quotedata = json.loads(js_match.group(1))
                        bk_name = quotedata.get('bk_name')
                        if bk_name and bk_name not in ['--', '暂无数据', '未知']:
                            sectors.append(bk_name)
                    except:
                        pass
                
                # 方法2: 查找breadcrumb中的板块信息
                breadcrumb_pattern = r'<a[^>]*>([^<]+)</a></span><span class="breadcrumb_item">([^<]+)</span>'
                breadcrumb_matches = re.findall(breadcrumb_pattern, html)
                for match in breadcrumb_matches:
                    if '板块' in match[0] or 'board' in match[0].lower():
                        sector = match[1]
                        if sector and sector not in ['--', '暂无数据', '未知']:
                            sectors.append(sector)
                
                # 方法3: 查找行业板块 - 多种模式匹配
                if not sectors:
                    industry_patterns = [
                        r'所属行业[：:]\s*<a[^>]*>([^<]+)</a>',
                        r'行业分类[：:]\s*<a[^>]*>([^<]+)</a>',
                        r'industry:"([^"]+)"',
                        r'所属行业[：:]\s*([^<]+)',
                    ]
                    
                    for pattern in industry_patterns:
                        industry_match = re.search(pattern, html)
                        if industry_match:
                            industry = industry_match.group(1).strip()
                            if industry and industry not in ['--', '暂无数据', '未知']:
                                sectors.append(industry)
                                break
                
                # 方法4: 查找概念板块 - 多种模式匹配
                concept_patterns = [
                    r'概念板块[：:]\s*([^<]+)',
                    r'概念题材[：:]\s*([^<]+)',
                    r'所属概念[：:]\s*([^<]+)',
                    r'concept:"([^"]+)"',
                ]
                
                for pattern in concept_patterns:
                    concept_match = re.search(pattern, html)
                    if concept_match:
                        concepts = concept_match.group(1).strip()
                        if concepts and concepts not in ['--', '暂无数据', '未知']:
                            # 分割多个概念（支持多种分隔符）
                            concept_list = []
                            for sep in [' ', ',', '、', '，', ';', '；']:
                                if sep in concepts:
                                    concept_list = [c.strip() for c in concepts.split(sep) if c.strip()]
                                    break
                            if not concept_list:
                                concept_list = [concepts]
                            sectors.extend(concept_list)
                            break
                
                # 方法5: 查找地区板块
                area_patterns = [
                    r'地区板块[：:]\s*<a[^>]*>([^<]+)</a>',
                    r'所属地区[：:]\s*<a[^>]*>([^<]+)</a>',
                    r'地区[：:]\s*([^<]+)',
                ]
                
                for pattern in area_patterns:
                    area_match = re.search(pattern, html)
                    if area_match:
                        area = area_match.group(1).strip()
                        if area and area not in ['--', '暂无数据', '未知']:
                            sectors.append(area)
                            break
                
                # 去重并过滤无效值
                sectors = list(set([s for s in sectors if s and len(s) > 1]))
                
                if sectors:
                    return {
                        'sectors': sectors,
                        'source': '10jqka',
                        'response_time': response.elapsed.total_seconds(),
                    }
                else:
                    # 如果没有找到板块信息，记录日志
                    print(f"⚠️  {code}: 从同花顺获取板块信息，但未找到有效数据")
                    
        except requests.exceptions.Timeout:
            print(f"⏰  {code}: 同花顺请求超时")
        except requests.exceptions.ConnectionError:
            print(f"🔌  {code}: 同花顺连接错误")
        except requests.exceptions.TooManyRedirects:
            print(f"🔄  {code}: 同花顺重定向过多")
        except requests.exceptions.RequestException as e:
            print(f"❌  {code}: 同花顺请求异常 - {e}")
        except Exception as e:
            print(f"⚠️  {code}: 同花顺处理异常 - {e}")
        
        return None
    
    def get_stock_sector_from_xueqiu(self, code: str) -> Optional[Dict]:
        """从雪球获取股票板块信息（优化版）"""
        try:
            # 转换股票代码格式
            if code.startswith('sh'):
                stock_code = f'SH{code[2:]}'
            elif code.startswith('sz'):
                stock_code = f'SZ{code[2:]}'
            else:
                return None
            
            # 雪球接口
            url = f'https://stock.xueqiu.com/v5/stock/quote.json'
            params = {
                'symbol': stock_code,
                'extend': 'detail',
                'timestamp': int(time.time() * 1000),  # 添加时间戳避免缓存
            }
            
            # 雪球需要一些基本的cookie
            cookies = {
                'device_id': 'test_device_' + str(random.randint(10000, 99999)),
                'xq_a_token': 'test_token_' + str(random.randint(10000, 99999)),
            }
            
            headers = {
                'User-Agent': self.get_random_user_agent(),
                'Referer': 'https://xueqiu.com/',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Origin': 'https://xueqiu.com',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-site',
            }
            
            # 发送请求
            response = self.session.get(
                url, 
                params=params, 
                headers=headers,
                cookies=cookies,
                timeout=(3, 5),
                proxies=self.proxies,
                verify=False
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    stock_data = data['data'].get('quote', {})
                    
                    sectors = []
                    
                    # 行业
                    industry = stock_data.get('industry')
                    if industry and industry not in ['--', '暂无数据', '未知']:
                        sectors.append(industry)
                    
                    # 概念板块（雪球可能在其他字段中）
                    concept = stock_data.get('concept')
                    if concept and concept not in ['--', '暂无数据', '未知']:
                        sectors.append(concept)
                    
                    # 主营业务
                    main_business = stock_data.get('main_business')
                    if main_business and main_business not in ['--', '暂无数据', '未知']:
                        # 从主营业务中提取关键词
                        business_keywords = self._extract_keywords_from_business(main_business)
                        sectors.extend(business_keywords)
                    
                    # 去重
                    sectors = list(set([s for s in sectors if s and len(s) > 1]))
                    
                    if sectors:
                        return {
                            'sectors': sectors,
                            'source': 'xueqiu',
                            'response_time': response.elapsed.total_seconds(),
                        }
                    
        except requests.exceptions.Timeout:
            print(f"⏰  {code}: 雪球请求超时")
        except requests.exceptions.ConnectionError:
            print(f"🔌  {code}: 雪球连接错误")
        except requests.exceptions.JSONDecodeError:
            print(f"📄  {code}: 雪球返回非JSON数据")
        except Exception as e:
            # 雪球接口可能限制访问，失败是正常的
            pass
        
        return None
    
    def _extract_keywords_from_business(self, business: str) -> List[str]:
        """从主营业务描述中提取关键词"""
        if not business:
            return []
        
        keywords = []
        business_lower = business.lower()
        
        # 常见业务关键词映射
        business_keywords_map = {
            "新能源": ["新能源", "光伏", "风电", "储能", "电池", "锂电", "太阳能", "清洁能源"],
            "医药": ["医药", "制药", "生物", "医疗", "健康", "药业", "医院", "器械", "疫苗", "中药"],
            "科技": ["科技", "技术", "软件", "信息", "数据", "网络", "电子", "通信", "智能", "AI", "人工智能"],
            "制造": ["制造", "生产", "加工", "装配", "工厂", "工业", "设备"],
            "服务": ["服务", "咨询", "管理", "运营", "维护", "支持"],
            "贸易": ["贸易", "销售", "经销", "代理", "批发", "零售"],
            "金融": ["金融", "投资", "证券", "银行", "保险", "信托", "基金"],
            "房地产": ["房地产", "地产", "房产", "物业", "开发", "建设"],
            "运输": ["运输", "物流", "快递", "航运", "航空", "港口"],
            "农业": ["农业", "农", "牧", "渔", "种植", "养殖"],
        }
        
        for sector, keyword_list in business_keywords_map.items():
            for keyword in keyword_list:
                if keyword in business_lower:
                    keywords.append(sector)
                    break  # 每个板块只添加一次
        
        return list(set(keywords))
    
    def get_stock_sector_from_local_db(self, code: str, name: str) -> Dict:
        """从本地数据库/预设数据获取板块信息"""
        # 这里可以扩展为从本地数据库读取
        # 暂时使用名称推断
        return self.infer_sector_from_name(code, name)
    
    def get_sector_hotness(self, sector_name: str) -> int:
        """获取板块热度（0-100）"""
        # 首先检查精确匹配
        if sector_name in self.sector_hotness:
            return self.sector_hotness[sector_name]
        
        # 然后检查包含关系
        for sector, hotness in self.sector_hotness.items():
            if sector in sector_name or sector_name in sector:
                return hotness
        
        # 默认热度
        return 40
    
    def get_sector_popularity(self, sector_name: str) -> int:
        """获取板块人气（0-100）"""
        # 首先检查精确匹配
        if sector_name in self.sector_popularity:
            return self.sector_popularity[sector_name]
        
        # 然后检查包含关系
        for sector, popularity in self.sector_popularity.items():
            if sector in sector_name or sector_name in sector:
                return popularity
        
        # 默认人气（基于热度推算）
        hotness = self.get_sector_hotness(sector_name)
        return max(30, min(80, hotness * 0.8))
    
    def analyze_sectors(self, sectors: List[str]) -> Dict:
        """分析板块信息"""
        if not sectors:
            return {
                'main_sector': '未知',
                'hotness': 40,
                'popularity': 30,
                'category': '其他',
                'all_sectors': [],
            }
        
        # 计算平均热度和人气
        hotness_sum = 0
        popularity_sum = 0
        valid_sectors = []
        
        for sector in sectors:
            hotness = self.get_sector_hotness(sector)
            popularity = self.get_sector_popularity(sector)
            hotness_sum += hotness
            popularity_sum += popularity
            valid_sectors.append({
                'name': sector,
                'hotness': hotness,
                'popularity': popularity,
            })
        
        # 按热度排序
        valid_sectors.sort(key=lambda x: x['hotness'], reverse=True)
        
        # 主要板块（热度最高的）
        main_sector = valid_sectors[0]['name'] if valid_sectors else '未知'
        avg_hotness = hotness_sum // len(sectors) if sectors else 40
        avg_popularity = popularity_sum // len(sectors) if sectors else 30
        
        # 确定分类
        category = '其他'
        for cat, cat_sectors in self.sector_categories.items():
            for cat_sector in cat_sectors:
                if cat_sector in main_sector or main_sector in cat_sector:
                    category = cat
                    break
            if category != '其他':
                break
        
        return {
            'main_sector': main_sector,
            'hotness': avg_hotness,
            'popularity': avg_popularity,
            'category': category,
            'all_sectors': valid_sectors,
        }
    
    def get_stock_sector_info(self, code: str, name: str = "", force_online: bool = False) -> Dict:
        """获取股票板块信息（简化版）"""
        # 先检查缓存（除非强制在线获取）
        if not force_online:
            cached = self.load_from_cache(code)
            if cached:
                # 检查缓存是否新鲜（1天内）
                cache_time = datetime.fromisoformat(cached.get('cache_time', '2000-01-01'))
                if (datetime.now() - cache_time).days < 1:
                    return cached
        
        # 简化：直接尝试从同花顺获取，失败则使用本地推断
        sector_info = None
        source_used = 'inferred'
        network_used = False
        
        try:
            # 尝试从同花顺获取（带超时）
            print(f"🌐  {code}: 尝试从同花顺获取板块信息...")
            sector_info = self.get_stock_sector_from_10jqka(code)
            if sector_info and sector_info.get('sectors'):
                source_used = '10jqka'
                network_used = True
                print(f"✅  {code}: 从同花顺获取成功")
            else:
                print(f"⚠️  {code}: 同花顺获取失败，使用本地推断")
                sector_info = self.get_stock_sector_from_local_db(code, name)
        except Exception as e:
            print(f"❌  {code}: 获取异常 - {e}")
            sector_info = self.get_stock_sector_from_local_db(code, name)
        
        # 分析板块信息
        if sector_info:
            sectors = sector_info.get('sectors', [])
            analysis = self.analyze_sectors(sectors)
            
            result = {
                'code': code,
                'name': name,
                'sectors': sectors,
                'main_sector': analysis['main_sector'],
                'sector_hotness': analysis['hotness'],
                'sector_popularity': analysis['popularity'],
                'sector_category': analysis['category'],
                'all_sectors_info': analysis['all_sectors'],
                'source': source_used,
                'network_used': network_used,
                'cache_time': datetime.now().isoformat(),
            }
            
            # 如果是从网络获取的，保存到缓存
            if network_used:
                self.save_to_cache(code, result)
        else:
            # 默认信息
            result = {
                'code': code,
                'name': name,
                'sectors': [],
                'main_sector': '未知',
                'sector_hotness': 40,
                'sector_popularity': 30,
                'sector_category': '其他',
                'all_sectors_info': [],
                'source': 'default',
                'network_used': False,
                'cache_time': datetime.now().isoformat(),
            }
        
        return result
    
    def _check_network_connection(self, test_url: str = "http://www.baidu.com", timeout: int = 3) -> bool:
        """检测网络连接是否可用"""
        try:
            # 尝试连接测试URL
            print(f"🔍 网络检测: 尝试连接 {test_url}...")
            response = self.session.get(test_url, timeout=timeout, verify=False)
            print(f"🔍 网络检测: 状态码 {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"🔍 网络检测: 失败 - {e}")
            return False
    
    def infer_sector_from_name(self, code: str, name: str) -> Dict:
        """从股票名称推断板块"""
        if not name or name == "未知":
            # 从代码推断
            if code.startswith('sh'):
                if code[2:].startswith('60'):
                    return {'sectors': ['上证主板'], 'source': 'inferred'}
                elif code[2:].startswith('68'):
                    return {'sectors': ['科创板'], 'source': 'inferred'}
            elif code.startswith('sz'):
                if code[2:].startswith('00'):
                    return {'sectors': ['深证主板'], 'source': 'inferred'}
                elif code[2:].startswith('30'):
                    return {'sectors': ['创业板'], 'source': 'inferred'}
            return {'sectors': [], 'source': 'inferred'}
        
        name_lower = name.lower()
        sectors = []
        
        # 根据名称关键词推断
        sector_keywords = {
            "银行": ["银行", "农商行", "商业银行", "商行"],
            "证券": ["证券", "券商", "投行", "中信", "华泰", "海通"],
            "保险": ["保险", "人寿", "财险", "太保", "平安"],
            "医药": ["医药", "制药", "生物", "医疗", "健康", "药业", "医院", "器械", "药", "生物技术"],
            "科技": ["科技", "技术", "软件", "信息", "数据", "网络", "电子", "通信", "数码", "智能", "AI", "人工智能"],
            "新能源": ["新能源", "能源", "光伏", "风电", "储能", "电池", "锂电", "锂业", "太阳能", "清洁能源", "赣锋", "天齐"],
            "汽车": ["汽车", "车", "汽配", "零部件", "整车", "乘用车", "商用车"],
            "消费": ["消费", "食品", "饮料", "酒", "零售", "商超", "百货", "超市", "购物", "餐饮"],
            "房地产": ["地产", "房产", "置业", "物业", "万科", "保利", "招商蛇口"],
            "化工": ["化工", "化学", "材料", "石化", "石油", "化纤"],
            "机械": ["机械", "装备", "设备", "制造", "工程", "重工"],
            "电力": ["电力", "电网", "电气", "能源", "发电", "供电"],
            "建筑": ["建筑", "建设", "工程", "路桥", "中铁", "中建", "交建"],
            "有色": ["有色", "金属", "矿业", "矿", "铝", "铜", "黄金", "稀土"],
            "煤炭": ["煤炭", "煤业", "煤矿", "神华"],
            "钢铁": ["钢铁", "钢", "宝钢", "鞍钢"],
            "运输": ["运输", "物流", "快递", "航运", "航空", "港口", "机场", "海运", "空运"],
            "农业": ["农业", "农", "牧", "渔", "种子", "化肥", "农药", "养殖"],
            "传媒": ["传媒", "文化", "影视", "娱乐", "游戏", "出版", "广告"],
            "旅游": ["旅游", "旅行", "景区", "酒店", "旅行社"],
        }
        
        for sector, keywords in sector_keywords.items():
            for keyword in keywords:
                if keyword in name_lower:
                    sectors.append(sector)
                    break
        
        # 去重
        sectors = list(set(sectors))
        
        # 如果没有推断出板块，根据代码前缀推断
        if not sectors:
            if code.startswith('sh'):
                if code[2:].startswith('60'):
                    sectors = ['上证主板']
                elif code[2:].startswith('68'):
                    sectors = ['科创板']
            elif code.startswith('sz'):
                if code[2:].startswith('00'):
                    sectors = ['深证主板']
                elif code[2:].startswith('30'):
                    sectors = ['创业板']
        
        return {
            'sectors': sectors,
            'source': 'inferred',
        }
    
    def format_sector_info(self, sector_info: Dict) -> str:
        """格式化板块信息用于显示"""
        if not sector_info or sector_info.get('main_sector') == '未知':
            return "板块: 未知"
        
        main_sector = sector_info.get('main_sector', '未知')
        hotness = sector_info.get('sector_hotness', 40)
        popularity = sector_info.get('sector_popularity', 30)
        category = sector_info.get('sector_category', '其他')
        
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
            popularity_desc = "👥高人氣"
        elif popularity >= 60:
            popularity_desc = "👥中人氣"
        elif popularity >= 40:
            popularity_desc = "👤溫人氣"
        else:
            popularity_desc = "👤一般人氣"
        
        # 其他板块（最多显示2个）
        other_sectors = []
        all_sectors_info = sector_info.get('all_sectors_info', [])
        for i, sector_info_item in enumerate(all_sectors_info):
            if i == 0:
                continue  # 跳过主要板块
            if i >= 3:  # 最多显示2个其他板块
                break
            other_sectors.append(sector_info_item['name'])
        
        result = f"板块: {main_sector} ({category}) | 热度: {hotness_desc}({hotness}) | 人气: {popularity_desc}({popularity})"
        if other_sectors:
            result += f" | 相关: {', '.join(other_sectors)}"
        
        return result
    
    def get_hot_sectors(self, limit: int = 10) -> List[Dict]:
        """获取当前热门板块"""
        hot_sectors = []
        for sector, hotness in sorted(self.sector_hotness.items(), key=lambda x: x[1], reverse=True):
            if hotness >= 70:  # 高热度板块
                popularity = self.get_sector_popularity(sector)
                hot_sectors.append({
                    'name': sector,
                    'hotness': hotness,
                    'popularity': popularity,
                    'category': next((cat for cat, sectors in self.sector_categories.items() if sector in sectors), '其他'),
                })
                if len(hot_sectors) >= limit:
                    break
        
        return hot_sectors

# 全局实例
_sector_info_instance = None

def get_sector_info() -> StockSectorInfo:
    """获取板块信息实例（单例）"""
    global _sector_info_instance
    if _sector_info_instance is None:
        _sector_info_instance = StockSectorInfo()
    return _sector_info_instance

if __name__ == "__main__":
    # 测试代码
    sector_info = get_sector_info()
    
    # 测试获取板块信息
    test_cases = [
        ('sh600000', '浦发银行'),
        ('sz000001', '平安银行'),
        ('sh600519', '贵州茅台'),
        ('sz002415', '海康威视'),
        ('sh600036', '招商银行'),
        ('sh600030', '中信证券'),
        ('sh600276', '恒瑞医药'),
        ('sz300750', '宁德时代'),
    ]
    
    print("测试股票板块信息获取:")
    print("=" * 80)
    
    for code, name in test_cases:
        info = sector_info.get_stock_sector_info(code, name)
        formatted = sector_info.format_sector_info(info)
        print(f"{code} {name}:")
        print(f"  {formatted}")
        print(f"  来源: {info.get('source', 'unknown')}")
        print()
    
    print("\n当前热门板块:")
    print("=" * 80)
    hot_sectors = sector_info.get_hot_sectors(5)
    for sector in hot_sectors:
        print(f"{sector['name']}: 热度{sector['hotness']}分, 人气{sector['popularity']}分 ({sector['category']})")