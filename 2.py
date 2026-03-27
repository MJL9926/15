"""
ALL-IN GUI (Python 3.11) - 超增强版完整代码 - 修复版 + 自动下单功能
- 20+机器学习模型
- 8个技术指标模型
- 完整的Tkinter GUI
- 智能集成投票
- 实时预测系统
- 模拟点击自动下单功能
"""

import threading
import time
import math
import os
import sys
import pickle
import traceback
from datetime import datetime, timedelta
import queue
import warnings
warnings.filterwarnings('ignore')

import requests
import pandas as pd
import numpy as np

# ML 核心库
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, \
    HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern

# Optional libs (try/except)
try:
    from lightgbm import LGBMClassifier
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    # XGBoost 1.5+ 版本使用以下方式
    XGBClassifier = xgb.XGBClassifier
except Exception:
    XGB_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
    # CatBoost 需要特殊处理以避免兼容性问题
except Exception:
    CATBOOST_AVAILABLE = False

# Tkinter GUI
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

# 自动下单相关库
try:
    import pyautogui
    import pygetwindow as gw
    AUTO_CLICK_AVAILABLE = True
except Exception:
    AUTO_CLICK_AVAILABLE = False
    pyautogui = None
    gw = None

# Colors in console (optional)
try:
    import colorama
    colorama.init()
    RED_C = colorama.Fore.RED
    GREEN_C = colorama.Fore.GREEN
    YELLOW_C = colorama.Fore.YELLOW
    RESET_C = colorama.Style.RESET_ALL
except Exception:
    RED_C = GREEN_C = YELLOW_C = RESET_C = ""

# ---------------- CONFIG ----------------
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
DATA_LIMIT = 2500

LOG_CSV = "btc_prediction_log.csv"
LOG_XLSX = "btc_prediction_log.xlsx"
MODEL_DIR = "models"
COORD_CONFIG = "click_coordinates.pkl"
os.makedirs(MODEL_DIR, exist_ok=True)

# 模型文件路径
MODEL_FILES = {
    'rf1': os.path.join(MODEL_DIR, "btc_model_rf1.pkl"),
    'rf2': os.path.join(MODEL_DIR, "btc_model_rf2.pkl"),
    'rf3': os.path.join(MODEL_DIR, "btc_model_rf3.pkl"),
    'lgb1': os.path.join(MODEL_DIR, "btc_model_lgb1.pkl"),
    'lgb2': os.path.join(MODEL_DIR, "btc_model_lgb2.pkl"),
    'xgb1': os.path.join(MODEL_DIR, "btc_model_xgb1.pkl"),
    'xgb2': os.path.join(MODEL_DIR, "btc_model_xgb2.pkl"),
    'cat1': os.path.join(MODEL_DIR, "btc_model_cat1.pkl"),
    'cat2': os.path.join(MODEL_DIR, "btc_model_cat2.pkl"),
    'ada': os.path.join(MODEL_DIR, "btc_model_ada.pkl"),
    'gb': os.path.join(MODEL_DIR, "btc_model_gb.pkl"),
    'et': os.path.join(MODEL_DIR, "btc_model_et.pkl"),
    'hgb': os.path.join(MODEL_DIR, "btc_model_hgb.pkl"),
    'svm1': os.path.join(MODEL_DIR, "btc_model_svm1.pkl"),
    'svm2': os.path.join(MODEL_DIR, "btc_model_svm2.pkl"),
    'knn': os.path.join(MODEL_DIR, "btc_model_knn.pkl"),
    'mlp': os.path.join(MODEL_DIR, "btc_model_mlp.pkl"),
    'lda': os.path.join(MODEL_DIR, "btc_model_lda.pkl"),
    'qda': os.path.join(MODEL_DIR, "btc_model_qda.pkl"),
    'nb': os.path.join(MODEL_DIR, "btc_model_nb.pkl"),
    'ensemble': os.path.join(MODEL_DIR, "btc_model_ensemble.pkl")
}

TRAIN_WINDOW = 1000
RETRAIN_EVERY = 3
MIN_TRAIN_SAMPLES = 400
ONLINE_LEARN = True

CONF_THRESHOLD = 0.65
PAUSE_IF_RECENT_WINRATE_LT = 0.45
PAUSE_RECENT_WINDOW = 25
PAUSE_COOLDOWN_MIN = 45
CONSECUTIVE_FAIL_SWITCH = 5

COUNTDOWN_SECONDS = 10 * 60
REFRESH_INTERVAL = 30

SESSION = requests.Session()
SESSION.headers.update({"Connection": "keep-alive", "User-Agent": "Mozilla/5.0"})
MAX_RETRIES = 3

# 特征列
FEATURE_COLS_BASE = [
    'ret1', 'ret3', 'ret5', 'ret10', 'ret15', 'ret20',
    'ma3_diff', 'ma5_diff', 'ma8_diff', 'ma13_diff', 'ma21_diff', 'ma34_diff', 
    'ma55_diff', 'ma89_diff', 'ma144_diff',
    'ema8_diff', 'ema13_diff', 'ema21_diff', 'ema34_diff',
    'macd', 'macd_sig', 'macd_hist', 'macd_signal', 'macd_hist_norm',
    'rsi', 'rsi_ma', 'rsi_slope',
    'stoch_k', 'stoch_d', 'stoch_diff',
    'bbw', 'bb_position', 'bb_squeeze',
    'atr14', 'atr_percent', 'atr_ratio',
    'vol_chg', 'vol_over_ma', 'vol_ratio',
    'price_accel', 'price_velocity',
    'obv', 'obv_ma_diff',
    'vwap_diff', 'price_vwap_ratio',
    'williams_r', 'cci', 'adx', 'adx_trend',
    'price_vol_corr', 'price_vol_corr_ma',
    'hurts', 'fractal_dim',
    'market_regime', 'trend_strength', 'volatility_regime', 'volume_regime'
]

# ---------------- UTIL: requests ----------------
def safe_get(url, params=None, timeout=8):
    for attempt in range(MAX_RETRIES):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5 + attempt*0.5)
    raise Exception("HTTP 多次请求失败")

def get_realtime_price_multi(symbol):
    sources = [
        ("binance", f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"),
        ("okx", f"https://www.okx.com/api/v5/market/ticker?instId={symbol.replace('USDT', '-USDT')}"),
        ("huobi", f"https://api.huobi.pro/market/trade?symbol={symbol.lower()}"),
        ("bybit", f"https://api.bybit.com/v2/public/tickers?symbol={symbol}"),
        ("kucoin", f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={symbol}")
    ]
    
    for source_name, url in sources:
        try:
            if source_name == "binance":
                r = safe_get("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol}, timeout=5)
                return float(r.json().get("price"))
            elif source_name == "okx":
                r = safe_get("https://www.okx.com/api/v5/market/ticker", params={"instId": symbol.replace('USDT', '-USDT')}, timeout=5)
                j = r.json()
                if 'data' in j and len(j['data'])>0:
                    return float(j['data'][0]['last'])
            elif source_name == "huobi":
                r = safe_get("https://api.huobi.pro/market/trade", params={"symbol": symbol.lower()}, timeout=5)
                j = r.json()
                if 'tick' in j and 'data' in j['tick'] and len(j['tick']['data'])>0:
                    return float(j['tick']['data'][0]['price'])
            elif source_name == "bybit":
                r = safe_get("https://api.bybit.com/v2/public/tickers", params={"symbol": symbol}, timeout=5)
                j = r.json()
                if 'result' in j and j['result']:
                    return float(j['result'][0]['last_price'])
            elif source_name == "kucoin":
                r = safe_get("https://api.kucoin.com/api/v1/market/orderbook/level1", params={"symbol": symbol}, timeout=5)
                j = r.json()
                if 'data' in j:
                    return float(j['data']['price'])
        except Exception:
            continue
    
    return float('nan')

def get_klines_multi(symbol, interval, limit=500):
    for source in ["binance", "okx", "bybit"]:
        try:
            if source == "binance":
                r = safe_get("https://api.binance.com/api/v3/klines", 
                           params={"symbol": symbol, "interval": interval, "limit": limit}, 
                           timeout=10)
                data = r.json()
                df = pd.DataFrame(data, columns=[
                    'open_time','open','high','low','close','volume','close_time',
                    'quote_asset_volume','number_of_trades','taker_buy_base','taker_buy_quote','ignore'])
                df = df[['open_time','open','high','low','close','volume']].astype(float)
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                return df
            elif source == "okx":
                r = safe_get("https://www.okx.com/api/v5/market/history-candles", 
                           params={"instId": symbol.replace('USDT', '-USDT'), "bar": interval, "limit": limit}, 
                           timeout=10)
                j = r.json()
                if 'data' in j:
                    arr = j['data']
                    df = pd.DataFrame(arr, columns=['open_time','open','high','low','close','volume'])
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df = df[['open_time','open','high','low','close','volume']].astype(float)
                    return df
            elif source == "bybit":
                r = safe_get("https://api.bybit.com/v5/market/kline", 
                           params={"category": "spot", "symbol": symbol, "interval": interval, "limit": limit}, 
                           timeout=10)
                j = r.json()
                if 'result' in j and 'list' in j['result']:
                    arr = j['result']['list']
                    df = pd.DataFrame(arr, columns=['open_time','open','high','low','close','volume'])
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df = df[['open_time','open','high','low','close','volume']].astype(float)
                    return df
        except Exception:
            continue
    
    raise Exception("无法从任何数据源获取 K 线")

# ---------------- 特征工程 ----------------
def calculate_advanced_indicators(df):
    """计算高级技术指标"""
    df = df.copy()
    
    # 收益率
    for period in [1, 3, 5, 10, 15, 20]:
        df[f'ret{period}'] = df['close'].pct_change(period)
    
    # 移动平均
    for w in [3, 5, 8, 13, 21, 34, 55, 89, 144]:
        df[f'ma{w}'] = df['close'].rolling(w).mean()
        df[f'ma{w}_diff'] = (df['close'] - df[f'ma{w}']) / (df[f'ma{w}'] + 1e-9)
    
    # 指数移动平均
    for span in [8, 13, 21, 34]:
        df[f'ema{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        df[f'ema{span}_diff'] = (df['close'] - df[f'ema{span}']) / (df[f'ema{span}'] + 1e-9)
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']
    df['macd_signal'] = np.where(df['macd'] > df['macd_sig'], 1, -1)
    df['macd_hist_norm'] = df['macd_hist'] / (df['close'] + 1e-9)
    
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / (roll_down + 1e-9)
    df['rsi'] = 100 - (100 / (1 + RS))
    df['rsi_ma'] = df['rsi'].rolling(5).mean()
    df['rsi_slope'] = df['rsi'].diff(3)
    
    # 随机指标
    low_min = df['low'].rolling(14).min()
    high_max = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-9))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['stoch_diff'] = df['stoch_k'] - df['stoch_d']
    
    # 布林带
    for period in [20, 50]:
        ma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        
        df[f'bbw_{period}'] = (upper - lower) / (ma + 1e-9)
        df[f'bb_position_{period}'] = (df['close'] - lower) / (upper - lower + 1e-9)
        df[f'bb_squeeze_{period}'] = (std / ma).rolling(5).mean()
    
    df['bbw'] = df['bbw_20'] if 'bbw_20' in df.columns else df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    df['bb_position'] = df['bb_position_20'] if 'bb_position_20' in df.columns else (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min() + 1e-9)
    df['bb_squeeze'] = df['bb_squeeze_20'] if 'bb_squeeze_20' in df.columns else (df['close'].rolling(20).std() / df['close'].rolling(20).mean()).rolling(5).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()
    df['atr_percent'] = df['atr14'] / df['close']
    df['atr_ratio'] = df['atr14'] / df['atr14'].rolling(20).mean()
    
    # 成交量
    df['vol_chg'] = df['volume'].pct_change()
    df['vol_ma21'] = df['volume'].rolling(21).mean()
    df['vol_over_ma'] = df['volume'] / (df['vol_ma21'] + 1e-9)
    df['vol_ratio'] = df['volume'].rolling(10).mean() / df['volume'].rolling(50).mean()
    
    # 价格加速度
    df['price_accel'] = df['close'].diff().diff()
    df['price_velocity'] = df['close'].diff()
    
    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma_diff'] = df['obv'] - df['obv'].rolling(20).mean()
    
    # VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_diff'] = df['close'] - df['vwap']
    df['price_vwap_ratio'] = df['close'] / (df['vwap'] + 1e-9)
    
    # 威廉指标
    df['williams_r'] = 100 * ((df['high'].rolling(14).max() - df['close']) / 
                              (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-9))
    
    # CCI
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad_tp = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad_tp + 1e-9)
    
    # ADX
    high_diff = df['high'].diff()
    low_diff = df['low'].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = df['high'] - df['low']
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14).mean() / (tr + 1e-9)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14).mean() / (tr + 1e-9)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['adx'] = pd.Series(dx).ewm(alpha=1/14).mean()
    df['adx_trend'] = np.where(df['adx'] > 25, 1, np.where(df['adx'] < 20, -1, 0))
    
    # 相关性
    df['price_vol_corr'] = df['close'].rolling(20).corr(df['volume'])
    df['price_vol_corr_ma'] = df['price_vol_corr'].rolling(5).mean()
    
    # Hurst指数
    lags = range(2, 100)
    tau = []
    for lag in lags:
        differences = df['close'].diff(lag).dropna()
        if len(differences) > 10:
            tau.append(np.std(differences))
        else:
            tau.append(np.nan)
    
    if len([t for t in tau if not np.isnan(t)]) > 10:
        try:
            poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
            df['hurts'] = poly[0]
        except:
            df['hurts'] = 0.5
    else:
        df['hurts'] = 0.5
    
    # 分形维度
    window = 50
    fd_values = []
    for i in range(len(df) - window):
        window_data = df['close'].iloc[i:i+window].values
        L = np.sum(np.abs(np.diff(window_data)))
        d = np.log(window) / (np.log(window) + np.log(L/(window_data[-1]-window_data[0] + 1e-9)))
        fd_values.append(d)
    
    fd_series = pd.Series(fd_values, index=df.index[window:])
    df['fractal_dim'] = fd_series.reindex(df.index).fillna(method='ffill')
    
    # 市场状态
    regime = pd.Series(0, index=df.index)
    trend = np.where(df['close'] > df['close'].rolling(50).mean(), 1, -1)
    volatility = df['atr_percent'].rolling(20).mean()
    high_vol = volatility > volatility.quantile(0.7)
    low_vol = volatility < volatility.quantile(0.3)
    volume_status = np.where(df['vol_over_ma'] > 1.5, 1, np.where(df['vol_over_ma'] < 0.7, -1, 0))
    regime = trend + np.where(high_vol, -0.5, np.where(low_vol, 0.5, 0)) + volume_status * 0.5
    df['market_regime'] = np.where(regime > 1, 2, np.where(regime < -1, -2, np.where(regime > 0, 1, -1)))
    
    # 趋势强度
    df['trend_strength'] = df['adx'] / 100
    
    # 波动率状态
    atr_percent = df['atr_percent']
    df['volatility_regime'] = np.where(atr_percent > atr_percent.rolling(50).mean() * 1.5, 2,
                                     np.where(atr_percent < atr_percent.rolling(50).mean() * 0.7, -2,
                                             np.where(atr_percent > atr_percent.rolling(50).mean(), 1, -1)))
    
    # 成交量状态
    vol_ratio = df['vol_over_ma']
    df['volume_regime'] = np.where(vol_ratio > 1.5, 2,
                                 np.where(vol_ratio < 0.7, -2,
                                         np.where(vol_ratio > 1.0, 1, -1)))
    
    return df

def add_features(df, symbol=None, include_orderbook=False, include_oi=False):
    """添加特征"""
    df = calculate_advanced_indicators(df)
    
    # 订单簿特征
    if include_orderbook:
        try:
            r = requests.get(f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=50", timeout=5)
            if r.status_code == 200:
                data = r.json()
                bids = np.array([[float(b[0]), float(b[1])] for b in data['bids']])
                asks = np.array([[float(a[0]), float(a[1])] for a in data['asks']])
                bid_volume = bids[:, 1].sum()
                ask_volume = asks[:, 1].sum()
                df['orderbook_imbalance'] = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-9)
                bid_pressure = (bids[:10, 0] * bids[:10, 1]).sum()
                ask_pressure = (asks[:10, 0] * asks[:10, 1]).sum()
                df['orderbook_pressure'] = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure + 1e-9)
        except Exception:
            df['orderbook_imbalance'] = 0
            df['orderbook_pressure'] = 0
    
    # OI和资金费率
    if include_oi:
        try:
            r = requests.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}", timeout=5)
            if r.status_code == 200:
                data = r.json()
                df['open_interest'] = float(data['openInterest'])
                df['open_interest_change'] = df['open_interest'].pct_change()
            
            r = requests.get(f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}", timeout=5)
            if r.status_code == 200:
                data = r.json()
                df['funding_rate'] = float(data.get('lastFundingRate', 0))
        except Exception:
            df['open_interest'] = 0
            df['funding_rate'] = 0
    
    df = df.dropna()
    return df

def create_label(df, future_periods=10):
    """创建标签"""
    for period in [5, 10, 15, 20]:
        df[f'future_ret{period}'] = df['close'].shift(-period) / df['close'] - 1
    
    future_rets = df[[f'future_ret{period}' for period in [5, 10, 15]]].mean(axis=1)
    df['label'] = (future_rets > future_rets.rolling(50).mean()).astype(int)
    
    return df.dropna()

# ---------------- 超级模型集合 ----------------
class SuperModelCollection:
    """超级模型集合"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        
    def create_all_models(self):
        """创建所有模型"""
        models = {}
        
        # RandomForest 变体
        models['rf1'] = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        models['rf2'] = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=43, n_jobs=-1)
        models['rf3'] = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=44, n_jobs=-1)
        
        # LightGBM
        if LGB_AVAILABLE:
            models['lgb1'] = LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)
            models['lgb2'] = LGBMClassifier(n_estimators=250, random_state=43, n_jobs=-1, verbose=-1)
        
        # XGBoost - 移除过时的参数
        if XGB_AVAILABLE:
            try:
                models['xgb1'] = XGBClassifier(n_estimators=200, random_state=42)
                models['xgb2'] = XGBClassifier(n_estimators=250, random_state=43)
            except Exception as e:
                print(f"XGBoost 初始化失败: {e}")
        
        # CatBoost - 特殊处理
        if CATBOOST_AVAILABLE:
            models['cat1'] = CatBoostClassifier(iterations=200, random_seed=42, verbose=False, allow_writing_files=False)
            models['cat2'] = CatBoostClassifier(iterations=250, random_seed=43, verbose=False, allow_writing_files=False)
        
        # 其他模型
        models['ada'] = AdaBoostClassifier(n_estimators=100, random_state=42)
        models['gb'] = GradientBoostingClassifier(n_estimators=150, random_state=42)
        models['et'] = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        models['hgb'] = HistGradientBoostingClassifier(max_iter=150, random_state=42)
        models['svm1'] = SVC(C=1.0, probability=True, random_state=42)
        models['svm2'] = LinearSVC(C=1.0, random_state=42)
        models['knn'] = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
        models['mlp'] = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        models['lda'] = LinearDiscriminantAnalysis()
        models['qda'] = QuadraticDiscriminantAnalysis()
        models['nb'] = GaussianNB()
        
        self.models = models
        return models
    
    def train_models(self, X, y, features, test_size=0.2):
        """训练所有模型"""
        trained_models = {}
        performances = {}
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        scaler = RobustScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        self.scalers['all'] = scaler
        
        for name, model in self.models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                accuracy = accuracy_score(y_val, y_pred)
                
                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(X_val_scaled)[:, 1]
                        auc = roc_auc_score(y_val, y_proba)
                    except:
                        auc = 0.5
                    f1 = f1_score(y_val, y_pred, zero_division=0)
                else:
                    auc = 0.5
                    f1 = accuracy
                
                score = accuracy * 0.4 + auc * 0.4 + f1 * 0.2
                
                trained_models[name] = model
                performances[name] = {
                    'accuracy': accuracy,
                    'auc': auc,
                    'f1': f1,
                    'score': score
                }
                
            except Exception as e:
                print(f"训练模型 {name} 失败: {e}")
                continue
        
        # 训练集成模型（排除CatBoost以避免兼容性问题）
        ensemble_model = self.train_ensemble_safe(X_train_scaled, y_train, performances)
        if ensemble_model:
            y_pred = ensemble_model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            
            if hasattr(ensemble_model, "predict_proba"):
                try:
                    y_proba = ensemble_model.predict_proba(X_val_scaled)[:, 1]
                    auc = roc_auc_score(y_val, y_proba)
                except:
                    auc = 0.5
                f1 = f1_score(y_val, y_pred, zero_division=0)
            else:
                auc = 0.5
                f1 = accuracy
            
            score = accuracy * 0.4 + auc * 0.4 + f1 * 0.2
            
            trained_models['ensemble'] = ensemble_model
            performances['ensemble'] = {
                'accuracy': accuracy,
                'auc': auc,
                'f1': f1,
                'score': score
            }
        
        self.models = trained_models
        self.model_performance = performances
        
        return trained_models, performances, scaler, features
    
    def train_ensemble_safe(self, X, y, performances):
        """安全的集成模型训练 - 排除与sklearn不兼容的模型"""
        sorted_models = sorted(performances.items(), key=lambda x: x[1]['score'], reverse=True)
        top_models = sorted_models[:min(10, len(sorted_models))]
        
        if not top_models:
            return None
        
        estimators = []
        weights = []
        
        for name, perf in top_models:
            if name in self.models:
                model = self.models[name]
                # 排除CatBoost模型（与sklearn不兼容）
                if 'cat' in name and CATBOOST_AVAILABLE:
                    continue
                weight = perf['score']
                estimators.append((name, model))
                weights.append(weight)
        
        if len(estimators) < 2:
            return None
        
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        
        voting_clf.fit(X, y)
        return voting_clf

# ---------------- 技术指标模型 ----------------
class SuperTechnicalModels:
    """技术指标模型集合"""
    
    @staticmethod
    def trend_following_model(df):
        try:
            trend_score = 0
            for period in [5, 20, 50]:
                ma_key = f'ma{period}'
                if ma_key in df.columns:
                    ma = df[ma_key].iloc[-1]
                    ma_prev = df[ma_key].iloc[-2] if len(df) > 1 else ma
                    trend_score += 1 if ma > ma_prev else -1
            
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            trend_score += 2 if momentum > 0.01 else -2 if momentum < -0.01 else 0
            
            if 'adx_trend' in df.columns:
                trend_score += df['adx_trend'].iloc[-1] * 1.5
            
            signal = 1 if trend_score > 0 else 0
            confidence = min(0.9, abs(trend_score) * 0.2)
            
            return signal, confidence, "TREND_FOLLOW"
        except:
            return 0, 0.5, "TREND_FOLLOW"
    
    @staticmethod
    def mean_reversion_model(df):
        try:
            mean_reversion_score = 0
            
            if 'bb_position' in df.columns:
                bb_pos = df['bb_position'].iloc[-1]
                if bb_pos > 0.8:
                    mean_reversion_score -= 2
                elif bb_pos < 0.2:
                    mean_reversion_score += 2
            
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if rsi > 75:
                    mean_reversion_score -= 1.5
                elif rsi < 25:
                    mean_reversion_score += 1.5
            
            signal = 1 if mean_reversion_score > 0 else 0
            confidence = min(0.85, abs(mean_reversion_score) * 0.25)
            
            return signal, confidence, "MEAN_REVERSION"
        except:
            return 0, 0.5, "MEAN_REVERSION"
    
    @staticmethod
    def momentum_model(df):
        try:
            momentum_score = 0
            
            ret5 = df['ret5'].iloc[-1] if 'ret5' in df.columns else 0
            ret10 = df['ret10'].iloc[-1] if 'ret10' in df.columns else 0
            
            momentum_score += ret5 * 100
            momentum_score += ret10 * 50
            
            if len(df) > 10 and 'ret5' in df.columns:
                mom_accel = ret5 - df['ret5'].iloc[-5]
                momentum_score += mom_accel * 200
            
            if 'macd_hist' in df.columns:
                macd_hist = df['macd_hist'].iloc[-1]
                momentum_score += macd_hist * 10
            
            signal = 1 if momentum_score > 0 else 0
            confidence = min(0.8, abs(momentum_score) * 0.3)
            
            return signal, confidence, "MOMENTUM"
        except:
            return 0, 0.5, "MOMENTUM"
    
    @staticmethod
    def volume_model(df):
        try:
            volume_score = 0
            
            price_change = df['ret1'].iloc[-1] if 'ret1' in df.columns else 0
            volume_change = df['vol_chg'].iloc[-1] if 'vol_chg' in df.columns else 0
            
            if price_change > 0 and volume_change > 0.5:
                volume_score += 2
            elif price_change < 0 and volume_change > 0.5:
                volume_score -= 2
            
            if 'obv_ma_diff' in df.columns:
                obv_trend = df['obv_ma_diff'].iloc[-1]
                volume_score += obv_trend * 0.1
            
            if 'vol_over_ma' in df.columns:
                vol_ratio = df['vol_over_ma'].iloc[-1]
                if vol_ratio > 2.0:
                    volume_score += 1 if price_change > 0 else -1
            
            signal = 1 if volume_score > 0 else 0
            confidence = min(0.75, abs(volume_score) * 0.4)
            
            return signal, confidence, "VOLUME"
        except:
            return 0, 0.5, "VOLUME"
    
    @staticmethod
    def volatility_model(df):
        try:
            volatility_score = 0
            
            if 'atr_percent' in df.columns:
                atr_pct = df['atr_percent'].iloc[-1]
                atr_ma = df['atr_percent'].rolling(20).mean().iloc[-1]
                
                if atr_pct > atr_ma * 1.5:
                    volatility_score -= 1.5
                elif atr_pct < atr_ma * 0.7:
                    volatility_score += 1
            
            price_std = df['close'].rolling(20).std().iloc[-1] / df['close'].rolling(20).mean().iloc[-1]
            volatility_score -= price_std * 10
            
            signal = 1 if volatility_score > 0 else 0
            confidence = min(0.7, abs(volatility_score) * 0.5)
            
            return signal, confidence, "VOLATILITY"
        except:
            return 0, 0.5, "VOLATILITY"
    
    @staticmethod
    def breakout_model(df):
        try:
            breakout_score = 0
            
            resistance = df['high'].rolling(20).max().iloc[-1]
            support = df['low'].rolling(20).min().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price > resistance * 1.002:
                breakout_score += 2
            elif current_price < support * 0.998:
                breakout_score -= 2
            
            if 'vol_over_ma' in df.columns:
                vol_ratio = df['vol_over_ma'].iloc[-1]
                if vol_ratio > 1.5:
                    breakout_score *= 1.5
            
            signal = 1 if breakout_score > 0 else 0
            confidence = min(0.85, abs(breakout_score) * 0.4)
            
            return signal, confidence, "BREAKOUT"
        except:
            return 0, 0.5, "BREAKOUT"
    
    @staticmethod
    def pattern_model(df):
        try:
            pattern_score = 0
            
            if 'body_ratio' in df.columns:
                body_ratio = df['body_ratio'].iloc[-1]
                if body_ratio < 0.1:
                    pattern_score -= 0.5
            
            if len(df) > 10:
                up_count = sum(1 for i in range(1, 6) if df['close'].iloc[-i] > df['close'].iloc[-i-1])
                if up_count >= 4:
                    pattern_score -= 0.5
                elif up_count <= 1:
                    pattern_score += 0.5
            
            signal = 1 if pattern_score > 0 else 0
            confidence = min(0.7, abs(pattern_score) * 0.6)
            
            return signal, confidence, "PATTERN"
        except:
            return 0, 0.5, "PATTERN"
    
    @staticmethod
    def market_sentiment_model(df):
        try:
            sentiment_score = 0
            
            if 'market_regime' in df.columns:
                regime = df['market_regime'].iloc[-1]
                sentiment_score += regime * 0.5
            
            if 'trend_strength' in df.columns:
                trend_strength = df['trend_strength'].iloc[-1]
                sentiment_score += trend_strength
            
            signal = 1 if sentiment_score > 0 else 0
            confidence = min(0.65, abs(sentiment_score) * 0.8)
            
            return signal, confidence, "SENTIMENT"
        except:
            return 0, 0.5, "SENTIMENT"
    
    @staticmethod
    def get_all_predictions(df):
        models = [
            SuperTechnicalModels.trend_following_model,
            SuperTechnicalModels.mean_reversion_model,
            SuperTechnicalModels.momentum_model,
            SuperTechnicalModels.volume_model,
            SuperTechnicalModels.volatility_model,
            SuperTechnicalModels.breakout_model,
            SuperTechnicalModels.pattern_model,
            SuperTechnicalModels.market_sentiment_model
        ]
        
        predictions = []
        for model_func in models:
            try:
                signal, confidence, name = model_func(df)
                predictions.append((name, signal, confidence))
            except Exception:
                continue
        
        return predictions

# ---------------- 智能集成投票器 ----------------
class SmartEnsembleVoter:
    """智能集成投票器"""
    
    def __init__(self):
        self.model_weights = {}
        self.model_history = {}
        self.adaptive_weights = True
        
    def update_model_performance(self, model_name, correct):
        if model_name not in self.model_history:
            self.model_history[model_name] = []
        
        self.model_history[model_name].append(correct)
        
        if len(self.model_history[model_name]) > 100:
            self.model_history[model_name] = self.model_history[model_name][-100:]
    
    def get_model_performance(self, model_name):
        if model_name not in self.model_history or not self.model_history[model_name]:
            return 0.5
        
        recent = self.model_history[model_name][-20:] if len(self.model_history[model_name]) >= 20 else self.model_history[model_name]
        if not recent:
            return 0.5
        
        return sum(recent) / len(recent)
    
    def calculate_adaptive_weights(self, predictions, market_conditions):
        weights = {}
        
        for name, signal, conf in predictions:
            base_weight = conf
            
            if 'TREND' in name and market_conditions.get('trend_strength', 0) > 0.6:
                base_weight *= 1.4
            
            if 'MEAN_REVERSION' in name and market_conditions.get('volatility', 0) > 0.7:
                base_weight *= 1.3
            
            if 'MOMENTUM' in name and market_conditions.get('momentum', 0) > 0.5:
                base_weight *= 1.2
            
            historical_perf = self.get_model_performance(name)
            base_weight *= (historical_perf * 0.5 + 0.5)
            
            weights[name] = min(base_weight, 1.0)
        
        return weights
    
    def analyze_market_conditions(self, df):
        conditions = {
            'trend_strength': abs(df['adx'].iloc[-1] / 100) if 'adx' in df.columns else 0.5,
            'volatility': df['atr_percent'].iloc[-1] if 'atr_percent' in df.columns else 0.5,
            'momentum': df['ret5'].iloc[-1] * 10 if 'ret5' in df.columns else 0,
            'market_regime': df['market_regime'].iloc[-1] if 'market_regime' in df.columns else 0
        }
        return conditions
    
    def weighted_voting(self, ml_predictions, tech_predictions, market_conditions):
        all_predictions = ml_predictions + tech_predictions
        
        if not all_predictions:
            return None, 0.0, "无预测", {}, {}
        
        weights = self.calculate_adaptive_weights(all_predictions, market_conditions)
        
        up_weight = 0.0
        down_weight = 0.0
        total_weight = 0.0
        
        ml_up = 0.0
        ml_down = 0.0
        ml_total = 0.0
        
        tech_up = 0.0
        tech_down = 0.0
        tech_total = 0.0
        
        vote_details = []
        for name, pred, conf in all_predictions:
            weight = weights.get(name, conf)
            total_weight += weight
            
            if 'ML_' in name or name in ['RF1', 'RF2', 'RF3', 'LGB1', 'LGB2', 'XGB1', 'XGB2', 'CAT1', 'CAT2', 
                                        'ADA', 'GB', 'ET', 'HGB', 'SVM1', 'SVM2', 'KNN', 'MLP', 'LDA', 'QDA', 
                                        'NB', 'ENSEMBLE']:
                ml_total += weight
                if pred == 1:
                    ml_up += weight
                    up_weight += weight
                else:
                    ml_down += weight
                    down_weight += weight
            else:
                tech_total += weight
                if pred == 1:
                    tech_up += weight
                    up_weight += weight
                else:
                    tech_down += weight
                    down_weight += weight
            
            vote_details.append(f"{name}:{'↑' if pred==1 else '↓'}({conf:.2f})")
        
        ml_ratio = ml_up / ml_total if ml_total > 0 else 0
        tech_ratio = tech_up / tech_total if tech_total > 0 else 0
        
        if total_weight == 0:
            return None, 0.0, "权重为零", {'ml': 0.5, 'tech': 0.5}, {'ml': 0, 'tech': 0}
        
        vote_ratio = up_weight / total_weight
        
        if vote_ratio > 0.5:
            final_pred = 1
            final_conf = 2 * (vote_ratio - 0.5)
        else:
            final_pred = 0
            final_conf = 2 * (0.5 - vote_ratio)
        
        volatility_factor = 1 - market_conditions['volatility']
        final_conf *= volatility_factor
        
        vote_summary = f"{len(all_predictions)}模型: 涨{up_weight:.2f}/跌{down_weight:.2f} | ML:{ml_ratio:.2f} | Tech:{tech_ratio:.2f}"
        
        model_ratios = {'ml': ml_ratio, 'tech': tech_ratio}
        model_weights = {'ml': ml_total, 'tech': tech_total}
        
        return final_pred, min(max(final_conf, 0.0), 1.0), vote_summary, model_ratios, model_weights
    
    def calculate_model_consistency(self, predictions):
        if not predictions:
            return 0.0
        
        signals = [pred for _, pred, _ in predictions]
        if len(signals) == 0:
            return 0.0
        
        up_count = sum(1 for s in signals if s == 1)
        down_count = sum(1 for s in signals if s == 0)
        
        consistency = abs(up_count - down_count) / len(signals)
        return consistency

# ---------------- 自动下单管理器 ----------------
class AutoClickManager:
    """自动下单管理器 - 处理模拟点击功能"""
    
    def __init__(self):
        self.coordinates = {
            'amount_input': None,    # 金额输入框坐标
            'buy_up': None,          # 买涨按钮坐标
            'buy_down': None,        # 买跌按钮坐标
            'confirm': None          # 确认按钮坐标
        }
        self.auto_trade_enabled = False
        self.trade_amount = "10"  # 默认交易金额
        
        # 加载保存的坐标
        self.load_coordinates()
        
    def load_coordinates(self):
        """从文件加载保存的坐标"""
        try:
            if os.path.exists(COORD_CONFIG):
                with open(COORD_CONFIG, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.coordinates.update(saved_data.get('coordinates', {}))
                    self.trade_amount = saved_data.get('trade_amount', "10")
                return True
            return False
        except Exception as e:
            print(f"加载坐标配置失败: {e}")
            return False
    
    def save_coordinates(self):
        """保存坐标到文件"""
        try:
            data = {
                'coordinates': self.coordinates,
                'trade_amount': self.trade_amount
            }
            with open(COORD_CONFIG, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"保存坐标配置失败: {e}")
            return False
    
    def set_coordinate(self, key, x, y):
        """设置坐标点"""
        if key in self.coordinates:
            self.coordinates[key] = (x, y)
            return True
        return False
    
    def get_coordinate(self, key):
        """获取坐标点"""
        return self.coordinates.get(key)
    
    def capture_coordinate(self, key, delay=3):
        """捕获坐标（需要在指定时间内将鼠标移动到目标位置）"""
        if not AUTO_CLICK_AVAILABLE:
            return None
        
        print(f"请在 {delay} 秒内将鼠标移动到目标位置...")
        time.sleep(delay)
        x, y = pyautogui.position()
        self.set_coordinate(key, x, y)
        print(f"已捕获 {key} 坐标: ({x}, {y})")
        return (x, y)
    
    def execute_auto_trade(self, direction, confidence):
        """执行自动交易点击"""
        if not AUTO_CLICK_AVAILABLE:
            return False, "自动点击库未安装"
        
        if not self.auto_trade_enabled:
            return False, "自动交易未启用"
        
        # 检查所有必要的坐标是否已设置
        required_keys = ['amount_input', 'buy_up', 'buy_down', 'confirm']
        missing_keys = [key for key in required_keys if self.coordinates[key] is None]
        
        if missing_keys:
            return False, f"缺失坐标: {', '.join(missing_keys)}"
        
        try:
            # 激活交易窗口（可选）
            self.bring_trade_window_to_front()
            
            # 执行点击序列
            self._click_sequence(direction)
            
            # 记录交易
            trade_log = {
                'time': datetime.now(),
                'direction': '涨' if direction == 1 else '跌',
                'confidence': confidence,
                'amount': self.trade_amount,
                'status': '成功'
            }
            
            return True, f"自动交易执行成功: 方向={'涨' if direction==1 else '跌'}, 置信度={confidence:.2%}, 金额={self.trade_amount}"
            
        except Exception as e:
            return False, f"自动交易执行失败: {str(e)}"
    
    def _click_sequence(self, direction):
        """执行点击序列"""
        # 确保pyautogui可用
        if not AUTO_CLICK_AVAILABLE:
            return
        
        # 添加小延迟确保窗口激活
        time.sleep(0.5)
        
        # 1. 点击金额输入框并输入金额
        self._safe_click(self.coordinates['amount_input'])
        time.sleep(0.2)
        pyautogui.hotkey('ctrl', 'a')  # 全选
        time.sleep(0.1)
        pyautogui.press('delete')      # 删除原有内容
        time.sleep(0.1)
        pyautogui.write(self.trade_amount)  # 输入金额
        time.sleep(0.2)
        
        # 2. 根据方向点击买涨或买跌按钮
        if direction == 1:  # 买涨
            self._safe_click(self.coordinates['buy_up'])
        else:  # 买跌
            self._safe_click(self.coordinates['buy_down'])
        time.sleep(0.3)
        
        # 3. 点击确认按钮
        self._safe_click(self.coordinates['confirm'])
        time.sleep(0.2)
    
    def _safe_click(self, coords):
        """安全点击，确保坐标有效"""
        if coords and len(coords) == 2:
            x, y = coords
            pyautogui.click(x, y)
            return True
        return False
    
    def bring_trade_window_to_front(self):
        """将交易窗口置顶（可选功能）"""
        try:
            # 尝试找到包含"币安"、"Binance"、"交易"等关键词的窗口
            windows = gw.getAllTitles()
            for window_title in windows:
                if any(keyword in window_title for keyword in ['币安', 'Binance', '交易', 'Trade']):
                    win = gw.getWindowsWithTitle(window_title)[0]
                    win.activate()
                    time.sleep(0.5)
                    return True
        except:
            pass  # 如果无法找到窗口，继续执行
        return False

# ---------------- 完整的GUI应用程序 ----------------
class EnhancedBTCApp:
    def __init__(self, root):
        self.root = root
        root.title("BTC 10分钟预测 — 超增强版 (20+模型) - 修复版 + 自动下单")
        root.geometry("1300x900")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 字体
        self.default_font = ("微软雅黑", 10)
        self.title_font = ("微软雅黑", 12, "bold")
        self.big_font = ("微软雅黑", 14, "bold")
        self.huge_font = ("微软雅黑", 24, "bold")
        
        # State
        self.running = False
        self.queue = queue.Queue()
        self.thread = None
        
        # 模型管理器
        self.super_models = SuperModelCollection()
        self.super_tech_models = SuperTechnicalModels()
        self.smart_voter = SmartEnsembleVoter()
        
        # 自动下单管理器
        self.auto_click_manager = AutoClickManager()
        
        # 模型开关变量
        self.model_vars = {}
        self.create_model_vars()
        
        # 初始化GUI
        self._init_gui()
        
        # 加载模型和配置
        self.load_all_models()
        self.load_config()
        
        # GUI循环
        self.root.after(500, self.process_queue)
    
    def create_model_vars(self):
        """创建模型开关变量"""
        ml_models = ['rf1', 'rf2', 'rf3', 'lgb1', 'lgb2', 'xgb1', 'xgb2', 
                    'cat1', 'cat2', 'ada', 'gb', 'et', 'hgb', 'svm1', 'svm2',
                    'knn', 'mlp', 'lda', 'qda', 'nb', 'ensemble']
        
        for model in ml_models:
            self.model_vars[model] = tk.BooleanVar(value=model in ['rf1', 'ensemble'])
        
        tech_models = ['TREND_FOLLOW', 'MEAN_REVERSION', 'MOMENTUM', 'VOLUME',
                      'VOLATILITY', 'BREAKOUT', 'PATTERN', 'SENTIMENT']
        
        for model in tech_models:
            self.model_vars[model] = tk.BooleanVar(value=True)
        
        self.include_orderbook = tk.BooleanVar(value=False)
        self.include_oi = tk.BooleanVar(value=False)
        self.adaptive_voting = tk.BooleanVar(value=True)
        self.online_learning = tk.BooleanVar(value=ONLINE_LEARN)
        self.auto_trade_enabled = tk.BooleanVar(value=False)
        
        # 交易金额变量
        self.trade_amount_var = tk.StringVar(value=self.auto_click_manager.trade_amount)
    
    def _init_gui(self):
        """初始化GUI界面"""
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 顶部面板
        top_frame = tk.LabelFrame(main_frame, text="实时预测", font=self.title_font)
        top_frame.pack(fill='x', pady=(0, 10))
        
        # 价格显示
        price_frame = tk.Frame(top_frame)
        price_frame.pack(side='left', fill='both', expand=True, padx=20, pady=10)
        tk.Label(price_frame, text="当前价格", font=self.default_font).pack()
        self.price_label = tk.Label(price_frame, text="-- USD", font=("Arial", 32, "bold"))
        self.price_label.pack(pady=(5, 0))
        
        # 预测显示
        pred_frame = tk.Frame(top_frame)
        pred_frame.pack(side='left', fill='both', expand=True, padx=20, pady=10)
        tk.Label(pred_frame, text="10分钟预测", font=self.default_font).pack()
        self.direction_label = tk.Label(pred_frame, text="等待预测...", font=("Arial", 28, "bold"), fg="gray")
        self.direction_label.pack(pady=(5, 0))
        
        # 置信度显示
        conf_frame = tk.Frame(top_frame)
        conf_frame.pack(side='left', fill='both', expand=True, padx=20, pady=10)
        tk.Label(conf_frame, text="置信度", font=self.default_font).pack()
        self.confidence_label = tk.Label(conf_frame, text="--", font=("Arial", 24), fg="blue")
        self.confidence_label.pack(pady=(5, 0))
        
        # 自动交易状态
        auto_frame = tk.Frame(top_frame)
        auto_frame.pack(side='left', fill='both', expand=True, padx=20, pady=10)
        tk.Label(auto_frame, text="自动交易", font=self.default_font).pack()
        self.auto_status_label = tk.Label(auto_frame, text="关闭", font=("Arial", 18), fg="red")
        self.auto_status_label.pack(pady=(5, 0))
        
        # 中间左侧 - 统计和状态
        mid_left = tk.LabelFrame(main_frame, text="统计信息", font=self.title_font)
        mid_left.pack(side='left', fill='both', expand=True, padx=(0, 5), pady=10)
        
        stats_grid = tk.Frame(mid_left)
        stats_grid.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 统计行
        tk.Label(stats_grid, text="累计胜率", font=self.default_font).grid(row=0, column=0, sticky='w', pady=5)
        self.cum_var = tk.StringVar(value="—")
        tk.Label(stats_grid, textvariable=self.cum_var, font=("Arial", 16)).grid(row=0, column=1, sticky='e', pady=5)
        
        tk.Label(stats_grid, text="最近50次", font=self.default_font).grid(row=1, column=0, sticky='w', pady=5)
        self.recent_var = tk.StringVar(value="—")
        tk.Label(stats_grid, textvariable=self.recent_var, font=("Arial", 16)).grid(row=1, column=1, sticky='e', pady=5)
        
        tk.Label(stats_grid, text="连胜/连败", font=self.default_font).grid(row=2, column=0, sticky='w', pady=5)
        self.streak_var = tk.StringVar(value="—")
        tk.Label(stats_grid, textvariable=self.streak_var, font=("Arial", 16)).grid(row=2, column=1, sticky='e', pady=5)
        
        tk.Label(stats_grid, text="总交易次数", font=self.default_font).grid(row=3, column=0, sticky='w', pady=5)
        self.total_var = tk.StringVar(value="—")
        tk.Label(stats_grid, textvariable=self.total_var, font=("Arial", 16)).grid(row=3, column=1, sticky='e', pady=5)
        
        tk.Label(stats_grid, text="ML模型比例", font=self.default_font).grid(row=4, column=0, sticky='w', pady=5)
        self.ml_ratio_var = tk.StringVar(value="—")
        tk.Label(stats_grid, textvariable=self.ml_ratio_var, font=("Arial", 14)).grid(row=4, column=1, sticky='e', pady=5)
        
        tk.Label(stats_grid, text="技术模型比例", font=self.default_font).grid(row=5, column=0, sticky='w', pady=5)
        self.tech_ratio_var = tk.StringVar(value="—")
        tk.Label(stats_grid, textvariable=self.tech_ratio_var, font=("Arial", 14)).grid(row=5, column=1, sticky='e', pady=5)
        
        # 市场状态
        status_frame = tk.Frame(mid_left)
        status_frame.pack(fill='x', padx=10, pady=10)
        tk.Label(status_frame, text="市场状态", font=self.title_font).pack(anchor='w')
        self.market_status_var = tk.StringVar(value="等待分析...")
        tk.Label(status_frame, textvariable=self.market_status_var, font=self.default_font).pack(anchor='w', pady=5)
        
        # 中间右侧 - 控制面板
        mid_right = tk.LabelFrame(main_frame, text="控制面板", font=self.title_font)
        mid_right.pack(side='right', fill='both', expand=True, padx=(5, 0), pady=10)
        
        # 控制按钮
        ctrl_frame = tk.Frame(mid_right)
        ctrl_frame.pack(fill='x', padx=10, pady=10)
        self.start_btn = tk.Button(ctrl_frame, text="▶ 开始预测", bg="#4CAF50", fg="white", 
                                  font=self.big_font, command=self.start, width=15)
        self.start_btn.pack(side='left', padx=5)
        self.stop_btn = tk.Button(ctrl_frame, text="■ 停止", bg="#F44336", fg="white",
                                 font=self.big_font, command=self.stop, state='disabled', width=15)
        self.stop_btn.pack(side='left', padx=5)
        
        # 训练按钮
        train_frame = tk.Frame(mid_right)
        train_frame.pack(fill='x', padx=10, pady=5)
        tk.Button(train_frame, text="🚀 高级训练", command=self.advanced_train, 
                 font=self.default_font, width=20).pack(side='left', padx=2)
        tk.Button(train_frame, text="⚡ 快速训练", command=self.quick_train,
                 font=self.default_font, width=20).pack(side='left', padx=2)
        
        # 自动交易开关
        auto_trade_frame = tk.Frame(mid_right)
        auto_trade_frame.pack(fill='x', padx=10, pady=5)
        tk.Checkbutton(auto_trade_frame, text="启用自动交易", 
                      variable=self.auto_trade_enabled, font=self.big_font,
                      command=self.toggle_auto_trade).pack()
        
        # 数据源选项
        data_frame = tk.LabelFrame(mid_right, text="数据源", font=self.default_font)
        data_frame.pack(fill='x', padx=10, pady=10)
        tk.Checkbutton(data_frame, text="盘口数据 (Orderbook)", 
                      variable=self.include_orderbook, font=self.default_font).pack(anchor='w', pady=2)
        tk.Checkbutton(data_frame, text="持仓量/资金费率 (OI/Funding)", 
                      variable=self.include_oi, font=self.default_font).pack(anchor='w', pady=2)
        
        # 投票设置
        vote_frame = tk.LabelFrame(mid_right, text="投票设置", font=self.default_font)
        vote_frame.pack(fill='x', padx=10, pady=10)
        tk.Checkbutton(vote_frame, text="自适应权重投票", 
                      variable=self.adaptive_voting, font=self.default_font).pack(anchor='w', pady=2)
        tk.Checkbutton(vote_frame, text="在线学习", 
                      variable=self.online_learning, font=self.default_font).pack(anchor='w', pady=2)
        
        # 倒计时显示
        countdown_frame = tk.Frame(mid_right)
        countdown_frame.pack(fill='x', padx=10, pady=10)
        tk.Label(countdown_frame, text="倒计时", font=self.title_font).pack()
        self.count_var = tk.StringVar(value="10:00")
        tk.Label(countdown_frame, textvariable=self.count_var, 
                font=("Arial", 36, "bold"), fg="blue").pack()
        
        # 价格显示
        price_info_frame = tk.Frame(mid_right)
        price_info_frame.pack(fill='x', padx=10, pady=10)
        tk.Label(price_info_frame, text="开始价", font=self.default_font).grid(row=0, column=0, sticky='w')
        self.start_price_var = tk.StringVar(value="—")
        tk.Label(price_info_frame, textvariable=self.start_price_var, 
                font=("Arial", 14)).grid(row=0, column=1, sticky='e')
        tk.Label(price_info_frame, text="结束价", font=self.default_font).grid(row=1, column=0, sticky='w', pady=5)
        self.end_price_var = tk.StringVar(value="—")
        tk.Label(price_info_frame, textvariable=self.end_price_var, 
                font=("Arial", 14)).grid(row=1, column=1, sticky='e', pady=5)
        
        # 底部 - 模型选择和日志
        bottom_frame = tk.Frame(main_frame)
        bottom_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        # 左侧 - 模型选择和自动交易设置
        left_bottom_frame = tk.Frame(bottom_frame)
        left_bottom_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # 模型选择
        model_frame = tk.LabelFrame(left_bottom_frame, text="模型选择 (20+模型)", font=self.title_font)
        model_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # 创建可滚动区域
        model_canvas = tk.Canvas(model_frame)
        scrollbar = tk.Scrollbar(model_frame, orient="vertical", command=model_canvas.yview)
        scrollable_frame = tk.Frame(model_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: model_canvas.configure(scrollregion=model_canvas.bbox("all"))
        )
        
        model_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        model_canvas.configure(yscrollcommand=scrollbar.set)
        
        # ML模型组
        ml_frame = tk.LabelFrame(scrollable_frame, text="机器学习模型", font=self.default_font)
        ml_frame.pack(fill='x', padx=10, pady=5)
        
        ml_models_groups = [
            ['rf1', 'rf2', 'rf3'],
            ['lgb1', 'lgb2'],
            ['xgb1', 'xgb2'],
            ['cat1', 'cat2'],
            ['ada', 'gb', 'et', 'hgb'],
            ['svm1', 'svm2', 'knn', 'mlp'],
            ['lda', 'qda', 'nb', 'ensemble']
        ]
        
        row_idx = 0
        for group in ml_models_groups:
            frame = tk.Frame(ml_frame)
            frame.grid(row=row_idx, column=0, columnspan=2, sticky='w', pady=2)
            for model in group:
                if model in self.model_vars:
                    cb = tk.Checkbutton(frame, text=model.upper(), 
                                       variable=self.model_vars[model],
                                       font=self.default_font)
                    cb.pack(side='left', padx=5)
            row_idx += 1
        
        # 技术指标模型组
        tech_frame = tk.LabelFrame(scrollable_frame, text="技术指标模型", font=self.default_font)
        tech_frame.pack(fill='x', padx=10, pady=5)
        
        tech_col1 = ['TREND_FOLLOW', 'MEAN_REVERSION', 'MOMENTUM', 'VOLUME']
        tech_col2 = ['VOLATILITY', 'BREAKOUT', 'PATTERN', 'SENTIMENT']
        
        for col, models in enumerate([tech_col1, tech_col2]):
            frame = tk.Frame(tech_frame)
            frame.grid(row=0, column=col, padx=10, pady=5)
            for model in models:
                if model in self.model_vars:
                    cb = tk.Checkbutton(frame, text=model.replace('_', ' '), 
                                       variable=self.model_vars[model],
                                       font=self.default_font)
                    cb.pack(anchor='w', pady=2)
        
        # 模型控制按钮
        model_btn_frame = tk.Frame(scrollable_frame)
        model_btn_frame.pack(fill='x', padx=10, pady=10)
        tk.Button(model_btn_frame, text="全选", command=self.select_all_models,
                 font=self.default_font).pack(side='left', padx=5)
        tk.Button(model_btn_frame, text="全不选", command=self.deselect_all_models,
                 font=self.default_font).pack(side='left', padx=5)
        tk.Button(model_btn_frame, text="仅ML模型", command=self.select_ml_only,
                 font=self.default_font).pack(side='left', padx=5)
        tk.Button(model_btn_frame, text="仅技术模型", command=self.select_tech_only,
                 font=self.default_font).pack(side='left', padx=5)
        
        model_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 自动交易设置
        auto_setup_frame = tk.LabelFrame(left_bottom_frame, text="自动交易设置", font=self.title_font)
        auto_setup_frame.pack(fill='x', pady=(0, 10))
        
        # 交易金额设置
        amount_frame = tk.Frame(auto_setup_frame)
        amount_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(amount_frame, text="交易金额:", font=self.default_font).pack(side='left')
        amount_entry = tk.Entry(amount_frame, textvariable=self.trade_amount_var, width=15, font=self.default_font)
        amount_entry.pack(side='left', padx=5)
        tk.Button(amount_frame, text="设置金额", command=self.set_trade_amount,
                 font=self.default_font).pack(side='left', padx=5)
        
        # 坐标设置按钮
        coord_frame = tk.Frame(auto_setup_frame)
        coord_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(coord_frame, text="设置金额坐标", command=lambda: self.capture_coordinate('amount_input'),
                 font=self.default_font).pack(side='left', padx=2, pady=2)
        tk.Button(coord_frame, text="设置买涨坐标", command=lambda: self.capture_coordinate('buy_up'),
                 font=self.default_font).pack(side='left', padx=2, pady=2)
        tk.Button(coord_frame, text="设置买跌坐标", command=lambda: self.capture_coordinate('buy_down'),
                 font=self.default_font).pack(side='left', padx=2, pady=2)
        tk.Button(coord_frame, text="设置确认坐标", command=lambda: self.capture_coordinate('confirm'),
                 font=self.default_font).pack(side='left', padx=2, pady=2)
        
        # 保存/测试按钮
        save_test_frame = tk.Frame(auto_setup_frame)
        save_test_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(save_test_frame, text="保存坐标", command=self.save_coordinates,
                 font=self.default_font).pack(side='left', padx=2)
        tk.Button(save_test_frame, text="测试买涨", command=lambda: self.test_auto_trade(1),
                 font=self.default_font).pack(side='left', padx=2)
        tk.Button(save_test_frame, text="测试买跌", command=lambda: self.test_auto_trade(0),
                 font=self.default_font).pack(side='left', padx=2)
        
        # 坐标显示
        coord_display_frame = tk.Frame(auto_setup_frame)
        coord_display_frame.pack(fill='x', padx=10, pady=5)
        
        self.coord_display_text = tk.Text(coord_display_frame, height=4, width=50, font=self.default_font)
        self.coord_display_text.pack(fill='x')
        self.update_coord_display()
        
        # 右侧 - 日志显示
        log_frame = tk.LabelFrame(bottom_frame, text="系统日志", font=self.title_font)
        log_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', 
                                                 height=15, font=("Courier New", 9))
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 日志控制
        log_ctrl_frame = tk.Frame(log_frame)
        log_ctrl_frame.pack(fill='x', padx=10, pady=(0, 10))
        tk.Button(log_ctrl_frame, text="清空日志", command=self.clear_log,
                 font=self.default_font).pack(side='left', padx=5)
        tk.Button(log_ctrl_frame, text="导出日志", command=self.export_log,
                 font=self.default_font).pack(side='left', padx=5)
        tk.Button(log_ctrl_frame, text="保存配置", command=self.save_config,
                 font=self.default_font).pack(side='left', padx=5)
    
    def toggle_auto_trade(self):
        """切换自动交易状态"""
        if self.auto_trade_enabled.get():
            if not AUTO_CLICK_AVAILABLE:
                messagebox.showwarning("警告", "自动点击库未安装，请安装pyautogui和pygetwindow")
                self.auto_trade_enabled.set(False)
                return
            
            # 检查坐标是否已设置
            required_keys = ['amount_input', 'buy_up', 'buy_down', 'confirm']
            missing_keys = [key for key in required_keys if self.auto_click_manager.coordinates[key] is None]
            
            if missing_keys:
                response = messagebox.askyesno("确认", f"以下坐标未设置: {', '.join(missing_keys)}\n是否继续启用自动交易？")
                if not response:
                    self.auto_trade_enabled.set(False)
                    return
            
            self.auto_click_manager.auto_trade_enabled = True
            self.auto_status_label.config(text="开启", fg="green")
            self.log("自动交易已启用")
        else:
            self.auto_click_manager.auto_trade_enabled = False
            self.auto_status_label.config(text="关闭", fg="red")
            self.log("自动交易已禁用")
    
    def set_trade_amount(self):
        """设置交易金额"""
        amount = self.trade_amount_var.get()
        try:
            # 验证金额格式
            float(amount)
            self.auto_click_manager.trade_amount = amount
            self.log(f"交易金额已设置为: {amount}")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的金额数字")
    
    def capture_coordinate(self, key_name):
        """捕获坐标"""
        if not AUTO_CLICK_AVAILABLE:
            messagebox.showerror("错误", "请先安装pyautogui和pygetwindow库")
            return
        
        # 在新线程中捕获坐标，避免阻塞GUI
        def capture_thread():
            self.log(f"开始捕获 {key_name} 坐标，请在3秒内将鼠标移动到目标位置...")
            coord = self.auto_click_manager.capture_coordinate(key_name, delay=3)
            if coord:
                self.log(f"{key_name} 坐标已设置: {coord}")
                self.update_coord_display()
        
        threading.Thread(target=capture_thread, daemon=True).start()
    
    def update_coord_display(self):
        """更新坐标显示"""
        text = ""
        for key, coord in self.auto_click_manager.coordinates.items():
            if coord:
                text += f"{key}: ({coord[0]}, {coord[1]})\n"
            else:
                text += f"{key}: 未设置\n"
        
        self.coord_display_text.delete(1.0, tk.END)
        self.coord_display_text.insert(1.0, text)
    
    def save_coordinates(self):
        """保存坐标配置"""
        if self.auto_click_manager.save_coordinates():
            self.log("坐标配置已保存")
        else:
            self.log("坐标配置保存失败")
    
    def test_auto_trade(self, direction):
        """测试自动交易"""
        if not AUTO_CLICK_AVAILABLE:
            messagebox.showerror("错误", "请先安装pyautogui和pygetwindow库")
            return
        
        # 在新线程中测试
        def test_thread():
            success, msg = self.auto_click_manager.execute_auto_trade(direction, 0.75)
            if success:
                self.log(f"测试成功: {msg}")
                messagebox.showinfo("测试成功", f"自动交易测试成功: {msg}")
            else:
                self.log(f"测试失败: {msg}")
                messagebox.showerror("测试失败", f"自动交易测试失败: {msg}")
        
        threading.Thread(target=test_thread, daemon=True).start()
    
    def select_all_models(self):
        for var in self.model_vars.values():
            var.set(True)
    
    def deselect_all_models(self):
        for var in self.model_vars.values():
            var.set(False)
    
    def select_ml_only(self):
        self.deselect_all_models()
        ml_keys = ['rf1', 'rf2', 'rf3', 'lgb1', 'lgb2', 'xgb1', 'xgb2', 
                  'cat1', 'cat2', 'ada', 'gb', 'et', 'hgb', 'svm1', 'svm2',
                  'knn', 'mlp', 'lda', 'qda', 'nb', 'ensemble']
        for key in ml_keys:
            if key in self.model_vars:
                self.model_vars[key].set(True)
    
    def select_tech_only(self):
        self.deselect_all_models()
        tech_keys = ['TREND_FOLLOW', 'MEAN_REVERSION', 'MOMENTUM', 'VOLUME',
                    'VOLATILITY', 'BREAKOUT', 'PATTERN', 'SENTIMENT']
        for key in tech_keys:
            if key in self.model_vars:
                self.model_vars[key].set(True)
    
    def save_config(self):
        config = {
            'models': {name: var.get() for name, var in self.model_vars.items()},
            'include_orderbook': self.include_orderbook.get(),
            'include_oi': self.include_oi.get(),
            'adaptive_voting': self.adaptive_voting.get(),
            'online_learning': self.online_learning.get(),
            'auto_trade_enabled': self.auto_trade_enabled.get(),
            'trade_amount': self.trade_amount_var.get()
        }
        try:
            with open('btc_predictor_config.pkl', 'wb') as f:
                pickle.dump(config, f)
            self.log("配置已保存")
        except Exception as e:
            self.log(f"保存配置失败: {str(e)}")
    
    def load_config(self):
        try:
            if os.path.exists('btc_predictor_config.pkl'):
                with open('btc_predictor_config.pkl', 'rb') as f:
                    config = pickle.load(f)
                for name, value in config.get('models', {}).items():
                    if name in self.model_vars:
                        self.model_vars[name].set(value)
                self.include_orderbook.set(config.get('include_orderbook', False))
                self.include_oi.set(config.get('include_oi', False))
                self.adaptive_voting.set(config.get('adaptive_voting', True))
                self.online_learning.set(config.get('online_learning', ONLINE_LEARN))
                self.auto_trade_enabled.set(config.get('auto_trade_enabled', False))
                self.trade_amount_var.set(config.get('trade_amount', "10"))
                
                # 更新自动交易管理器状态
                self.auto_click_manager.trade_amount = config.get('trade_amount', "10")
                self.toggle_auto_trade()
                
                self.log("配置已加载")
        except Exception as e:
            self.log(f"加载配置失败: {str(e)}")
    
    def clear_log(self):
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
    
    def log(self, msg, newline=True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{timestamp}] {msg}"
        print(text)
        self.queue.put(('log', text))
    
    def update_price_display(self, price, pred, conf):
        self.queue.put(('price', (price, pred, conf)))
    
    def process_queue(self):
        try:
            while not self.queue.empty():
                typ, payload = self.queue.get_nowait()
                if typ == 'log':
                    self.log_text.configure(state='normal')
                    self.log_text.insert('end', payload + "\n")
                    self.log_text.yview('end')
                    self.log_text.configure(state='disabled')
                elif typ == 'price':
                    price, pred, conf = payload
                    self.price_label.config(text=f"{price:,.2f} USD")
                    if pred is None:
                        self.direction_label.config(text="等待预测...", fg="gray")
                        self.confidence_label.config(text="--", fg="gray")
                    else:
                        if pred == 1:
                            self.direction_label.config(text="预测：上涨 📈", fg="green")
                        else:
                            self.direction_label.config(text="预测：下跌 📉", fg="red")
                        self.confidence_label.config(text=f"{conf:.1%}")
                        if conf > 0.75:
                            self.confidence_label.config(fg="dark green")
                        elif conf > 0.65:
                            self.confidence_label.config(fg="blue")
                        elif conf > 0.55:
                            self.confidence_label.config(fg="orange")
                        else:
                            self.confidence_label.config(fg="red")
                elif typ == 'stats':
                    cum, recent50, streak, total = payload
                    self.cum_var.set(f"{cum:.1f}%" if not math.isnan(cum) else "—")
                    self.recent_var.set(f"{recent50:.1f}%" if not math.isnan(recent50) else "—")
                    self.streak_var.set(streak)
                    self.total_var.set(str(total))
                elif typ == 'model_ratios':
                    ml_ratio, tech_ratio = payload
                    self.ml_ratio_var.set(f"{ml_ratio:.1%}" if ml_ratio is not None else "—")
                    self.tech_ratio_var.set(f"{tech_ratio:.1%}" if tech_ratio is not None else "—")
                elif typ == 'market_status':
                    status, volatility = payload
                    self.market_status_var.set(status)
                elif typ == 'count':
                    self.count_var.set(payload)
                elif typ == 'start_end':
                    s, e = payload
                    self.start_price_var.set(f"{s:,.2f}" if not math.isnan(s) else "—")
                    self.end_price_var.set(f"{e:,.2f}" if not math.isnan(e) else "—")
                elif typ == 'auto_status':
                    status, color = payload
                    self.auto_status_label.config(text=status, fg=color)
        except Exception as e:
            print("process_queue error:", e)
        finally:
            self.root.after(500, self.process_queue)
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.thread = threading.Thread(target=self.run_loop, daemon=True)
        self.thread.start()
        self.log("开始预测循环 - 超增强版 (20+模型) + 自动下单")
    
    def stop(self):
        if not self.running:
            return
        self.running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.log("停止中...")
    
    def on_close(self):
        if messagebox.askokcancel("退出", "确定退出程序？"):
            self.running = False
            self.save_config()
            self.auto_click_manager.save_coordinates()
            self.root.destroy()
    
    def export_log(self):
        if os.path.exists(LOG_CSV):
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if path:
                try:
                    import shutil
                    if path.endswith('.csv'):
                        shutil.copy(LOG_CSV, path)
                    elif path.endswith('.xlsx'):
                        df = pd.read_csv(LOG_CSV)
                        df.to_excel(path, index=False)
                    messagebox.showinfo("导出", f"已导出到 {path}")
                except Exception as e:
                    messagebox.showerror("导出失败", str(e))
        else:
            messagebox.showinfo("导出", "无日志文件")
    
    def advanced_train(self):
        if messagebox.askyesno("确认", "高级训练将训练20+个模型，可能需要较长时间。继续吗？"):
            thr = threading.Thread(target=self._advanced_train, daemon=True)
            thr.start()
    
    def quick_train(self):
        thr = threading.Thread(target=self._quick_train, daemon=True)
        thr.start()
    
    def load_all_models(self):
        self.log("加载模型中...")
        loaded_count = 0
        for name, path in MODEL_FILES.items():
            try:
                if os.path.exists(path):
                    obj = load_model(path)
                    if obj:
                        self.super_models.models[name] = obj['model']
                        if 'scaler' in obj:
                            self.super_models.scalers[name] = obj['scaler']
                        loaded_count += 1
            except Exception as e:
                self.log(f"加载模型 {name} 失败: {str(e)[:50]}")
        self.log(f"已加载 {loaded_count} 个模型")
    
    def save_model(self, path, model, features, scaler):
        obj = {'model': model, 'features': features, 'scaler': scaler}
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    
    def _advanced_train(self):
        self.log("🚀 开始高级训练 - 训练20+模型...")
        try:
            df_raw = get_klines_multi(SYMBOL, INTERVAL, DATA_LIMIT * 2)
            df = add_features(df_raw, symbol=SYMBOL, 
                            include_orderbook=self.include_orderbook.get(), 
                            include_oi=self.include_oi.get())
            df = create_label(df)
            
            if len(df) < MIN_TRAIN_SAMPLES * 2:
                self.log("样本不足，训练取消")
                return
            
            feature_cols = []
            for col in FEATURE_COLS_BASE:
                if col in df.columns:
                    feature_cols.append(col)
            
            if self.include_orderbook.get():
                for col in ['orderbook_imbalance', 'orderbook_pressure']:
                    if col in df.columns:
                        feature_cols.append(col)
            
            if self.include_oi.get():
                for col in ['open_interest', 'funding_rate']:
                    if col in df.columns:
                        feature_cols.append(col)
            
            self.log(f"使用 {len(feature_cols)} 个特征进行训练")
            
            X = df[feature_cols].astype(float).values
            y = df['label'].astype(int).values
            
            self.super_models.create_all_models()
            trained_models, performances, scaler, features = self.super_models.train_models(X, y, feature_cols)
            
            for name, model in trained_models.items():
                if name in MODEL_FILES:
                    try:
                        self.save_model(MODEL_FILES[name], model, features, scaler)
                    except Exception as e:
                        self.log(f"保存模型 {name} 失败: {str(e)[:50]}")
            
            self.log("=" * 60)
            self.log("训练完成总结:")
            self.log("=" * 60)
            
            for name, perf in sorted(performances.items(), key=lambda x: x[1]['score'], reverse=True):
                self.log(f"{name:10s}: 准确率={perf['accuracy']:.4f}, AUC={perf['auc']:.4f}, "
                        f"F1={perf['f1']:.4f}, 综合={perf['score']:.4f}")
            
            self.super_models.models = trained_models
            self.super_models.model_performance = performances
            
            best_model = max(performances.items(), key=lambda x: x[1]['score'])[0]
            self.log("=" * 60)
            self.log(f"共训练 {len(trained_models)} 个模型")
            self.log(f"最佳模型: {best_model} (综合评分: {performances[best_model]['score']:.4f})")
            self.log("=" * 60)
            
        except Exception as e:
            self.log(f"高级训练异常: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _quick_train(self):
        self.log("⚡ 开始快速训练...")
        try:
            df_raw = get_klines_multi(SYMBOL, INTERVAL, DATA_LIMIT)
            df = add_features(df_raw, symbol=SYMBOL, include_orderbook=False, include_oi=False)
            df = create_label(df)
            
            if len(df) < MIN_TRAIN_SAMPLES:
                self.log("样本不足，训练取消")
                return
            
            feature_cols = []
            basic_features = FEATURE_COLS_BASE[:30]
            for col in basic_features:
                if col in df.columns:
                    feature_cols.append(col)
            
            X = df[feature_cols].astype(float).values
            y = df['label'].astype(int).values
            
            # 只训练核心模型
            key_models = {}
            key_models['rf1'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
            if LGB_AVAILABLE:
                key_models['lgb1'] = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
            
            if XGB_AVAILABLE:
                try:
                    key_models['xgb1'] = XGBClassifier(n_estimators=100, random_state=42)
                except:
                    pass
            
            key_models = {k: v for k, v in key_models.items() if v is not None}
            
            scaler = RobustScaler().fit(X)
            X_scaled = scaler.transform(X)
            
            performances = {}
            trained_models = {}
            
            for name, model in key_models.items():
                try:
                    tscv = TimeSeriesSplit(n_splits=3)
                    scores = []
                    for train_idx, val_idx in tscv.split(X_scaled):
                        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        scores.append(accuracy_score(y_val, y_pred))
                    avg_score = np.mean(scores)
                    model.fit(X_scaled, y)
                    trained_models[name] = model
                    performances[name] = avg_score
                    self.log(f"{name}: 准确率={avg_score:.4f}")
                except Exception as e:
                    self.log(f"训练模型 {name} 失败: {str(e)[:50]}")
            
            if len(trained_models) >= 2:
                estimators = [(name, model) for name, model in trained_models.items()]
                ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
                ensemble.fit(X_scaled, y)
                trained_models['ensemble'] = ensemble
                y_pred = ensemble.predict(X_scaled[-100:])
                ensemble_score = accuracy_score(y[-100:], y_pred)
                performances['ensemble'] = ensemble_score
                self.log(f"集成模型: 准确率={ensemble_score:.4f}")
            
            for name, model in trained_models.items():
                if name in MODEL_FILES:
                    try:
                        self.save_model(MODEL_FILES[name], model, feature_cols, scaler)
                    except Exception as e:
                        self.log(f"保存模型 {name} 失败: {str(e)[:50]}")
            
            self.log(f"快速训练完成，共训练 {len(trained_models)} 个模型")
            
        except Exception as e:
            self.log(f"快速训练异常: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def run_loop(self):
        round_idx = 0
        while self.running:
            try:
                start_time = time.time()
                
                try:
                    df_raw = get_klines_multi(SYMBOL, INTERVAL, DATA_LIMIT)
                except Exception as e:
                    self.log(f"获取K线失败: {str(e)}")
                    time.sleep(5)
                    continue

                df = add_features(df_raw, symbol=SYMBOL, 
                                include_orderbook=self.include_orderbook.get(), 
                                include_oi=self.include_oi.get())
                df = create_label(df)
                
                if len(df) < MIN_TRAIN_SAMPLES:
                    self.log("样本太少，等待补齐...")
                    time.sleep(5)
                    continue

                market_analysis = self.smart_voter.analyze_market_conditions(df)
                regime = market_analysis.get('market_regime', 0)
                if regime > 0:
                    status = f"上涨趋势 ({abs(regime)}级)"
                elif regime < 0:
                    status = f"下跌趋势 ({abs(regime)}级)"
                else:
                    status = "震荡整理"
                
                volatility = market_analysis.get('volatility', 0)
                vol_status = f"{volatility:.2%}"
                self.queue.put(('market_status', (status, vol_status)))

                feat_cols = []
                for col in FEATURE_COLS_BASE:
                    if col in df.columns:
                        feat_cols.append(col)
                
                if self.include_orderbook.get():
                    for col in ['orderbook_imbalance', 'orderbook_pressure']:
                        if col in df.columns:
                            feat_cols.append(col)
                
                if self.include_oi.get():
                    for col in ['open_interest', 'funding_rate']:
                        if col in df.columns:
                            feat_cols.append(col)
                
                all_ml_predictions = []
                all_tech_predictions = []
                
                for name, model in self.super_models.models.items():
                    if name not in self.model_vars or not self.model_vars[name].get():
                        continue
                    try:
                        if name in self.super_models.scalers:
                            scaler = self.super_models.scalers[name]
                        else:
                            scaler = RobustScaler()
                            available_features = [f for f in feat_cols if f in df.columns]
                            if len(available_features) > 10:
                                X_sample = df[available_features].values[-100:]
                                scaler.fit(X_sample)
                                self.super_models.scalers[name] = scaler
                        
                        available_features = [f for f in feat_cols if f in df.columns]
                        if len(available_features) < len(feat_cols) * 0.5:
                            continue
                        X_current = df[available_features].iloc[-1:].values
                        X_scaled = scaler.transform(X_current)
                        
                        # CatBoost需要特殊处理
                        if 'cat' in name and CATBOOST_AVAILABLE:
                            try:
                                pred = model.predict(X_scaled)[0]
                                if hasattr(model, "predict_proba"):
                                    proba = model.predict_proba(X_scaled)[0]
                                    conf = float(max(proba))
                                else:
                                    conf = 0.7
                                all_ml_predictions.append((name.upper(), int(pred), conf))
                            except:
                                continue
                        elif hasattr(model, "predict_proba"):
                            try:
                                proba = model.predict_proba(X_scaled)[0]
                                pred = int(np.argmax(proba))
                                conf = float(max(proba))
                                all_ml_predictions.append((name.upper(), pred, conf))
                            except:
                                pred = model.predict(X_scaled)[0]
                                all_ml_predictions.append((name.upper(), int(pred), 0.6))
                        else:
                            pred = model.predict(X_scaled)[0]
                            all_ml_predictions.append((name.upper(), int(pred), 0.6))
                    except Exception as e:
                        self.log(f"模型 {name} 预测失败: {str(e)[:50]}")
                        continue
                
                if any(self.model_vars[m].get() for m in ['TREND_FOLLOW', 'MEAN_REVERSION', 'MOMENTUM', 
                                                         'VOLUME', 'VOLATILITY', 'BREAKOUT', 'PATTERN', 'SENTIMENT']):
                    tech_predictions = self.super_tech_models.get_all_predictions(df)
                    for name, signal, conf in tech_predictions:
                        if name in self.model_vars and self.model_vars[name].get():
                            all_tech_predictions.append((name, signal, conf))
                
                if all_ml_predictions or all_tech_predictions:
                    final_pred, final_conf, vote_summary, model_ratios, model_weights = self.smart_voter.weighted_voting(
                        all_ml_predictions, all_tech_predictions, market_analysis
                    )
                    
                    consistency = self.smart_voter.calculate_model_consistency(all_ml_predictions + all_tech_predictions)
                    self.queue.put(('model_ratios', (model_ratios['ml'], model_ratios['tech'])))
                    self.log(f"第{round_idx+1}轮 | 一致性: {consistency:.3f} | {vote_summary}")
                    
                    if consistency > 0.7:
                        final_conf = min(final_conf * 1.2, 1.0)
                else:
                    final_pred = None
                    final_conf = 0.0
                    self.log(f"第{round_idx+1}轮 | 无模型可用，跳过")

                start_price = get_realtime_price_multi(SYMBOL)
                if math.isnan(start_price):
                    self.log("无法获取开始实时价格，跳过本轮")
                    time.sleep(3)
                    continue

                self.update_price_display(start_price, final_pred, final_conf)
                
                # 执行自动交易（如果有预测且置信度足够）
                if (final_pred is not None and final_conf >= CONF_THRESHOLD and 
                    self.auto_trade_enabled.get() and self.auto_click_manager.auto_trade_enabled):
                    try:
                        # 在新线程中执行自动交易，避免阻塞预测循环
                        def execute_auto_trade_thread():
                            success, msg = self.auto_click_manager.execute_auto_trade(final_pred, final_conf)
                            if success:
                                self.log(f"自动交易执行成功: {msg}")
                            else:
                                self.log(f"自动交易执行失败: {msg}")
                        
                        threading.Thread(target=execute_auto_trade_thread, daemon=True).start()
                    except Exception as e:
                        self.log(f"自动交易执行异常: {str(e)}")

                df_log_existing = pd.read_csv(LOG_CSV) if os.path.exists(LOG_CSV) else pd.DataFrame()
                cum, recent50, wins, total = compute_winrates(df_log_existing)
                streak_text = self._calc_streak_text(df_log_existing)
                self.queue.put(('stats', (cum, recent50, streak_text, total)))

                if final_pred is None or final_conf < CONF_THRESHOLD:
                    self.log(f"置信度不足（{final_conf:.3f}），本轮跳过记录")
                    log_prediction(start_price, -1, -1, 'low_conf', final_conf)
                    time.sleep(REFRESH_INTERVAL)
                    round_idx += 1
                    continue

                end_ts = time.time() + COUNTDOWN_SECONDS
                cur_price = start_price
                while time.time() < end_ts and self.running:
                    now = time.time()
                    rem = int(end_ts - now)
                    m = rem // 60; s = rem % 60
                    self.queue.put(('count', f"{m:02d}:{s:02d}"))
                    if rem % REFRESH_INTERVAL == 0:
                        p = get_realtime_price_multi(SYMBOL)
                        if not math.isnan(p):
                            cur_price = p
                    time.sleep(1)

                final_price = get_realtime_price_multi(SYMBOL)
                if math.isnan(final_price):
                    self.log("无法获取结束价格，记录为跳过")
                    log_prediction(start_price, -1, -1, 'no_final_price', final_conf)
                    round_idx += 1
                    time.sleep(REFRESH_INTERVAL)
                    continue

                actual = int(final_price > start_price)
                correct = (final_pred == actual)
                
                self.log(f"开始价 {start_price:.2f} | 结束价 {final_price:.2f} | 实际 {'涨' if actual==1 else '跌'} | 预测 {'正确' if correct else '错误'} | conf={final_conf:.3f}")
                
                df_log = log_prediction(start_price, final_pred, actual, 'auto', final_conf)
                
                for name, _, _ in all_ml_predictions + all_tech_predictions:
                    self.smart_voter.update_model_performance(name, correct)
                
                cum, recent50, wins, total = compute_winrates(df_log)
                streak_text = self._calc_streak_text(df_log)
                self.queue.put(('stats', (cum, recent50, streak_text, total)))
                self.queue.put(('start_end', (start_price, final_price)))

                consec_fail = self._consecutive_fail_count(df_log)
                if consec_fail >= CONSECUTIVE_FAIL_SWITCH:
                    self.log(f"检测到连续失败 {consec_fail} 次，短暂暂停在线学习")
                    time.sleep(60)
                    self.log("恢复运行")

                round_idx += 1
                
                if round_idx % RETRAIN_EVERY == 0 and self.online_learning.get():
                    self.log(f"第{round_idx}轮，触发在线学习")
                    self._online_learn(df)
                
                elapsed = time.time() - start_time
                to_wait = max(REFRESH_INTERVAL - elapsed, 1)
                time.sleep(to_wait)
                
            except Exception as e:
                self.log(f"主循环异常: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(3)
    
    def _online_learn(self, df):
        try:
            if len(df) < MIN_TRAIN_SAMPLES * 1.5:
                return
            self.log("开始在线学习...")
            recent_df = df.tail(MIN_TRAIN_SAMPLES * 2)
            feature_cols = []
            for col in FEATURE_COLS_BASE:
                if col in recent_df.columns:
                    feature_cols.append(col)
            X = recent_df[feature_cols].values
            y = recent_df['label'].values
            scaler = RobustScaler().fit(X)
            Xs = scaler.transform(X)
            enabled_models = []
            for name, model in self.super_models.models.items():
                if name in self.model_vars and self.model_vars[name].get() and name != 'ensemble':
                    # 排除CatBoost模型（与sklearn VotingClassifier不兼容）
                    if 'cat' in name:
                        continue
                    enabled_models.append((name, model))
            if len(enabled_models) >= 2:
                ensemble_model = VotingClassifier(estimators=enabled_models, voting='soft', n_jobs=-1)
                ensemble_model.fit(Xs, y)
                if 'ensemble' in MODEL_FILES:
                    self.save_model(MODEL_FILES['ensemble'], ensemble_model, feature_cols, scaler)
                    self.super_models.models['ensemble'] = ensemble_model
                    self.super_models.scalers['ensemble'] = scaler
                    self.log("在线学习完成")
        except Exception as e:
            self.log(f"在线学习失败: {str(e)}")
    
    def _calc_streak_text(self, df_log):
        if df_log is None or df_log.empty:
            return "—"
        valid_data = df_log[(df_log['actual'] != -1) & (df_log['correct'].notna())]
        if valid_data.empty:
            return "—"
        arr = valid_data['correct'].astype(int).values
        if len(arr) == 0:
            return "—"
        current_streak = 0
        latest_result = arr[-1]
        for i in range(len(arr)-1, -1, -1):
            if arr[i] == latest_result:
                current_streak += 1
            else:
                break
        if latest_result == 1:
            return f"{current_streak} 连胜"
        else:
            return f"{current_streak} 连败"
    
    def _consecutive_fail_count(self, df_log):
        if df_log is None or df_log.empty:
            return 0
        valid_data = df_log[(df_log['actual'] != -1) & (df_log['correct'].notna())]
        if valid_data.empty:
            return 0
        arr = valid_data['correct'].astype(int).values
        cnt = 0
        for i in range(len(arr)-1, -1, -1):
            if arr[i] == 0:
                cnt += 1
            else:
                break
        return cnt

# 辅助函数
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path,'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

def log_prediction(start_price, predicted, actual, regime, conf):
    new_row = {
        "time": datetime.now(),
        "start_price": start_price,
        "predicted": int(predicted) if predicted is not None else -1,
        "actual": int(actual) if actual is not None else -1,
        "correct": int(predicted==actual) if predicted is not None and actual is not None else np.nan,
        "regime": regime,
        "conf": float(conf) if conf is not None else np.nan
    }
    os.makedirs(os.path.dirname(LOG_CSV) if os.path.dirname(LOG_CSV) else '.', exist_ok=True)
    if os.path.exists(LOG_CSV):
        df = pd.read_csv(LOG_CSV)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(LOG_CSV, index=False)
    try:
        df.to_excel(LOG_XLSX, index=False)
    except Exception:
        pass
    return df

def compute_winrates(df):
    if df is None or df.empty:
        return float('nan'), float('nan'), 0, 0
    valid_data = df[(df['actual'] != -1) & (df['correct'].notna())]
    if valid_data.empty:
        return float('nan'), float('nan'), 0, 0
    cum = valid_data['correct'].mean() * 100
    recent50 = valid_data['correct'].tail(50).mean() * 100
    wins = int(valid_data['correct'].sum())
    total = int(valid_data['correct'].shape[0])
    return cum, recent50, wins, total

def main():
    root = tk.Tk()
    app = EnhancedBTCApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()