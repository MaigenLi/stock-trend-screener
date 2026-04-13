#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地股票名称获取模块
使用本地JSON文件获取股票代码和名称对应关系，避免网络请求
"""

import json
import os
from typing import Dict, Optional

# 本地股票名称文件路径
LOCAL_STOCK_NAMES_FILE = "~/stock_code/results/all_stock_names_final.json"

# 缓存加载的股票名称数据
_stock_names_cache = None

def load_stock_names() -> Dict:
    """加载本地股票名称数据"""
    global _stock_names_cache
    
    if _stock_names_cache is not None:
        return _stock_names_cache
    
    try:
        if not os.path.exists(LOCAL_STOCK_NAMES_FILE):
            print(f"⚠️ 本地股票名称文件不存在: {LOCAL_STOCK_NAMES_FILE}")
            _stock_names_cache = {}
            return _stock_names_cache
        
        with open(LOCAL_STOCK_NAMES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取股票信息
        stocks_data = data.get('stocks', {})
        _stock_names_cache = {}
        
        for stock_key, stock_info in stocks_data.items():
            code = stock_info.get('code', '')
            name = stock_info.get('name', '')
            if code and name:
                # 存储两种格式：带市场前缀和不带市场前缀
                _stock_names_cache[stock_key] = name  # 例如: sh600000 -> 浦发银行
                _stock_names_cache[code] = name       # 例如: 600000 -> 浦发银行
        
        print(f"✅ 已加载本地股票名称数据，共 {len(_stock_names_cache)//2} 只股票")
        return _stock_names_cache
        
    except Exception as e:
        print(f"❌ 加载本地股票名称文件失败: {e}")
        _stock_names_cache = {}
        return _stock_names_cache

def get_stock_name(code: str) -> str:
    """获取股票名称
    
    参数:
        code: 股票代码，可以是带市场前缀的格式（如'sh600000'）或不带前缀的格式（如'600000'）
    
    返回:
        股票名称，如果找不到则返回"未知"
    """
    stock_names = load_stock_names()
    
    # 尝试直接查找
    if code in stock_names:
        return stock_names[code]
    
    # 如果代码不带市场前缀，尝试添加前缀查找
    if not code.startswith('sh') and not code.startswith('sz'):
        # 尝试上海市场
        sh_code = f"sh{code}"
        if sh_code in stock_names:
            return stock_names[sh_code]
        
        # 尝试深圳市场
        sz_code = f"sz{code}"
        if sz_code in stock_names:
            return stock_names[sz_code]
    
    # 如果带市场前缀但找不到，尝试去掉前缀查找
    elif code.startswith('sh') or code.startswith('sz'):
        pure_code = code[2:]
        if pure_code in stock_names:
            return stock_names[pure_code]
    
    return "未知"

def get_stock_info(code: str) -> Optional[Dict]:
    """获取完整的股票信息
    
    参数:
        code: 股票代码
    
    返回:
        包含code和name的字典，如果找不到则返回None
    """
    name = get_stock_name(code)
    if name == "未知":
        return None
    
    return {
        'code': code,
        'name': name,
        'full_code': code if code.startswith(('sh', 'sz')) else f"sh{code}" if code.startswith('6') else f"sz{code}"
    }

def get_all_stock_codes() -> list:
    """获取所有股票代码列表（带市场前缀）"""
    stock_names = load_stock_names()
    # 只返回带市场前缀的代码
    return [code for code in stock_names.keys() if code.startswith(('sh', 'sz'))]

def get_stock_count() -> int:
    """获取股票总数"""
    stock_names = load_stock_names()
    # 计算带市场前缀的股票数量
    return len([code for code in stock_names.keys() if code.startswith(('sh', 'sz'))])

def test_local_stock_names():
    """测试本地股票名称获取功能"""
    test_cases = [
        ('sh600000', '浦发银行'),
        ('600000', '浦发银行'),
        ('sz000001', '平安银行'),
        ('000001', '平安银行'),
        ('sh600519', '贵州茅台'),
        ('600519', '贵州茅台'),
        ('sz300750', '宁德时代'),
        ('300750', '宁德时代'),
        ('sh000001', '上证指数'),  # 指数
        ('sz399001', '深证成指'),  # 指数
    ]
    
    print("测试本地股票名称获取:")
    print("=" * 60)
    
    for code, expected_name in test_cases:
        name = get_stock_name(code)
        status = "✅" if name == expected_name else "❌"
        print(f"{status} {code:15} -> {name:10} (期望: {expected_name})")
    
    print(f"\n股票总数: {get_stock_count()} 只")
    
    # 显示前10只股票
    print("\n前10只股票:")
    all_codes = get_all_stock_codes()[:10]
    for code in all_codes:
        print(f"  {code}: {get_stock_name(code)}")

if __name__ == "__main__":
    test_local_stock_names()
