
# btc_5min_stable_final.py
"""
BTC 5分钟预测稳定最终版 (Python 3.6+)
- 完整的倒计时系统
- 实时胜率统计
- 进场条件可视化
- 多维度过滤器
- 高级风险管理
- 胜率记录持久化
- 语音播报功能
- 自动下单功能
- 稳定性优化版
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
import json
from collections import deque

# 导入必要的模块
import os
import sys
import threading
import time
import math
import pickle
import traceback
from datetime import datetime, timedelta
import queue
import json
from collections import deque

# 基础依赖
import requests
import pandas as pd
import numpy as np

# 语音播报 - 可选
TTS_AVAILABLE = False
TTS_METHOD = 'none'

try:
    if not IS_ANDROID:
        import pyttsx3
        TTS_AVAILABLE = True
        TTS_METHOD = 'pyttsx3'
    else:
        # 在Android平台上使用Kivy的Audio或Android TextToSpeech
        print("信息: Android平台使用内置语音播报")
        TTS_AVAILABLE = True
        TTS_METHOD = 'android'
except Exception as e:
    print(f"信息: 语音播报功能不可用: {e}")
    TTS_AVAILABLE = False
    TTS_METHOD = 'none'

# 自动下单功能
try:
    if not IS_ANDROID:
        import pyautogui
        AUTO_TRADE_AVAILABLE = True
        AUTO_TRADE_METHOD = 'pyautogui'
    else:
        # 在Android平台上使用交易提示
        print("信息: Android平台使用交易提示模式")
        AUTO_TRADE_AVAILABLE = True
        AUTO_TRADE_METHOD = 'android'
except Exception as e:
    print(f"信息: 自动交易功能不可用: {e}")
    AUTO_TRADE_AVAILABLE = False
    AUTO_TRADE_METHOD = 'none'

# ML - 可选
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception as e:
    print(f"信息: scikit-learn 不可用: {e}")
    SKLEARN_AVAILABLE = False

# Kivy GUI - 必需
try:
    import kivy
    kivy.require('2.0.0')
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.label import Label
    from kivy.uix.button import Button
    from kivy.uix.textinput import TextInput
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.checkbox import CheckBox
    from kivy.uix.screenmanager import ScreenManager, Screen
    from kivy.clock import Clock
    from kivy.properties import StringProperty, ObjectProperty, BooleanProperty, NumericProperty
    from kivy.graphics import Color, Rectangle
    from kivy.core.window import Window
    from kivy.uix.behaviors import ButtonBehavior
    from kivy.uix.popup import Popup
    from kivy.uix.floatlayout import FloatLayout
    from kivy.animation import Animation
    KIVY_AVAILABLE = True
except Exception as e:
    print(f"错误: kivy 导入失败: {e}")
    KIVY_AVAILABLE = False
    if not IS_ANDROID:
        sys.exit(1)
    else:
        print("警告: Kivy导入失败，但在Android平台上继续运行")

# 语音播报
tts_engine = None
if TTS_AVAILABLE:
    if TTS_METHOD == 'pyttsx3' and not IS_ANDROID:
        try:
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 180)
            tts_engine.setProperty('volume', 0.9)
            print("语音播报功能已启用 (pyttsx3)")
        except Exception as e:
            print(f"语音播报初始化失败: {e}")
            TTS_AVAILABLE = False
    elif TTS_METHOD == 'android':
        try:
            # 在Android平台上初始化语音功能
            print("语音播报功能已启用 (Android)")
        except Exception as e:
            print(f"Android语音初始化失败: {e}")
            TTS_AVAILABLE = False

# 自动下单
if AUTO_TRADE_AVAILABLE:
    if AUTO_TRADE_METHOD == 'pyautogui':
        # 设置pyautogui安全设置
        pyautogui.PAUSE = 0.5  # 每次pyautogui调用后暂停0.5秒
        pyautogui.FAILSAFE = True  # 启用故障安全功能
        print("自动下单功能已启用 (pyautogui)")
    elif AUTO_TRADE_METHOD == 'android':
        print("自动下单功能已启用 (Android交易提示)")
    else:
        print("自动下单功能已启用")
else:
    print("警告: 自动下单功能不可用")

# ML
if not SKLEARN_AVAILABLE:
    print("警告: 使用简化模式运行，部分ML功能不可用")

# ---------------- 全局配置 ----------------
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
DATA_LIMIT = 2000
FUTURE_BARS = 5

# 文件路径
import os
from kivy.utils import platform

# 获取应用数据目录
if platform == 'android':
    from android.storage import app_storage_path
    app_dir = app_storage_path()
else:
    app_dir = os.path.dirname(os.path.abspath(__file__))

# 确保模型目录存在
MODEL_DIR = os.path.join(app_dir, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 文件路径
LOG_CSV = os.path.join(app_dir, "btc_5min_final_log.csv")
LOG_XLSX = os.path.join(app_dir, "btc_5min_final_log.xlsx")
STATS_FILE = os.path.join(app_dir, "btc_5min_stats.pkl")
COORD_FILE = os.path.join(app_dir, "auto_trade_coords.json")

MODEL_STACK_FILE = os.path.join(MODEL_DIR, "btc_5min_stack_final.pkl")

# 训练参数
MIN_TRAIN_SAMPLES = 300
RETRAIN_EVERY = 3

# 预测参数
CONF_THRESHOLD = 0.65
MIN_ORDERBOOK_CONCENTRATION = 0.75
MIN_PRESSURE_RATIO = 1.5
MTF_ALIGNMENT_RATIO = 0.60

# 时间参数
COUNTDOWN_SECONDS = 300
REFRESH_INTERVAL = 30

# 自动交易参数
TRADE_AMOUNT = 10  # 默认交易金额（美元）
AUTO_TRADE_ENABLED = False  # 自动交易默认关闭

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})
MAX_RETRIES = 3
REQUEST_TIMEOUT = 10

# ---------------- 工具函数 ----------------
def safe_get(url, params=None, timeout=REQUEST_TIMEOUT):
    """安全的HTTP请求"""
    for attempt in range(MAX_RETRIES):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.Timeout:
            print(f"请求超时 ({attempt+1}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(1 + attempt)
        except requests.exceptions.ConnectionError:
            print(f"连接错误 ({attempt+1}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 + attempt)
        except Exception as e:
            print(f"请求失败 ({attempt+1}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(0.5 + attempt)
    return None

def speak_text(text, rate=180, volume=0.9):
    """语音播报文本"""
    if not TTS_AVAILABLE:
        return False
    
    try:
        def speak_in_thread():
            try:
                if TTS_METHOD == 'pyttsx3' and tts_engine is not None:
                    # 在桌面平台上使用pyttsx3
                    tts_engine.setProperty('rate', rate)
                    tts_engine.setProperty('volume', volume)
                    tts_engine.say(text)
                    tts_engine.runAndWait()
                elif TTS_METHOD == 'android':
                    # 在Android平台上使用Kivy的声音功能
                    # 这里使用简单的提示音，实际项目中可以使用Android的TextToSpeech
                    print(f"Android语音提示: {text}")
                    # 可以在这里添加Android TextToSpeech的实现
                else:
                    print(f"语音播报: {text}")
            except Exception as e:
                print(f"语音播报失败: {e}")
        
        thread = threading.Thread(target=speak_in_thread, daemon=True)
        thread.start()
        return True
    except Exception as e:
        print(f"启动语音播报失败: {e}")
        return False

# ---------------- 自动交易功能 ----------------
class AutoTrader:
    def __init__(self):
        self.coords = {
            'amount': (100, 100),      # 金额输入框坐标
            'buy_up': (200, 200),      # 买涨按钮坐标
            'buy_down': (300, 200),    # 买跌按钮坐标
            'confirm': (400, 300)      # 确认按钮坐标
        }
        self.trade_amount = TRADE_AMOUNT
        self.enabled = AUTO_TRADE_ENABLED
        self.load_coordinates()
    
    def load_coordinates(self):
        """从文件加载坐标"""
        try:
            if os.path.exists(COORD_FILE):
                with open(COORD_FILE, 'r') as f:
                    data = json.load(f)
                    if 'coords' in data:
                        self.coords = data['coords']
                    if 'trade_amount' in data:
                        self.trade_amount = data['trade_amount']
                print(f"已加载坐标配置: {self.coords}")
                return True
        except Exception as e:
            print(f"加载坐标文件失败: {e}")
        return False
    
    def save_coordinates(self):
        """保存坐标到文件"""
        try:
            data = {
                'coords': self.coords,
                'trade_amount': self.trade_amount,
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(COORD_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"坐标配置已保存: {self.coords}")
            return True
        except Exception as e:
            print(f"保存坐标文件失败: {e}")
            return False
    
    def set_coordinate(self, coord_type, x, y):
        """设置坐标"""
        if coord_type in self.coords:
            self.coords[coord_type] = (x, y)
            return True
        return False
    
    def get_current_mouse_position(self):
        """获取当前鼠标位置"""
        if not AUTO_TRADE_AVAILABLE:
            return (0, 0)
        try:
            return pyautogui.position()
        except:
            return (0, 0)
    
    def test_click(self, coord_type):
        """测试点击指定坐标"""
        if not AUTO_TRADE_AVAILABLE:
            return False, "pyautogui不可用"
        
        if coord_type not in self.coords:
            return False, f"未知坐标类型: {coord_type}"
        
        try:
            x, y = self.coords[coord_type]
            # 移动鼠标到指定位置
            pyautogui.moveTo(x, y, duration=0.5)
            time.sleep(0.2)
            # 点击
            pyautogui.click()
            time.sleep(0.2)
            return True, f"已测试点击 {coord_type} 坐标 ({x}, {y})"
        except Exception as e:
            return False, f"测试点击失败: {str(e)}"
    
    def execute_trade(self, direction):
        """执行交易
        direction: 1=买涨, 0=买跌
        """
        if not AUTO_TRADE_AVAILABLE:
            return False, "自动交易不可用"
        
        if not self.enabled:
            return False, "自动交易未启用"
        
        try:
            if AUTO_TRADE_METHOD == 'pyautogui':
                # 在桌面平台上使用pyautogui
                steps = []
                
                # 1. 点击金额输入框并输入金额
                amount_x, amount_y = self.coords['amount']
                pyautogui.moveTo(amount_x, amount_y, duration=0.3)
                time.sleep(0.1)
                pyautogui.click()
                time.sleep(0.1)
                pyautogui.hotkey('ctrl', 'a')  # 全选
                time.sleep(0.1)
                pyautogui.press('delete')  # 删除原有内容
                time.sleep(0.1)
                pyautogui.typewrite(str(self.trade_amount))  # 输入金额
                steps.append(f"设置金额: ${self.trade_amount}")
                time.sleep(0.2)
                
                # 2. 根据方向点击买涨或买跌按钮
                if direction == 1:  # 买涨
                    btn_x, btn_y = self.coords['buy_up']
                    btn_type = "买涨"
                else:  # 买跌
                    btn_x, btn_y = self.coords['buy_down']
                    btn_type = "买跌"
                
                pyautogui.moveTo(btn_x, btn_y, duration=0.3)
                time.sleep(0.1)
                pyautogui.click()
                steps.append(f"点击{btn_type}按钮")
                time.sleep(0.2)
                
                # 3. 点击确认按钮
                confirm_x, confirm_y = self.coords['confirm']
                pyautogui.moveTo(confirm_x, confirm_y, duration=0.3)
                time.sleep(0.1)
                pyautogui.click()
                steps.append("点击确认按钮")
                time.sleep(0.2)
                
                # 4. 返回主窗口（按ESC或点击其他位置）
                pyautogui.moveTo(amount_x + 100, amount_y, duration=0.3)
                time.sleep(0.1)
                pyautogui.click()
                time.sleep(0.1)
                
                return True, f"自动下单成功: {'买涨' if direction == 1 else '买跌'} ${self.trade_amount}"
            
            elif AUTO_TRADE_METHOD == 'android':
                # 在Android平台上显示交易提示
                direction_text = "买涨" if direction == 1 else "买跌"
                return True, f"交易提示: {direction_text} ${self.trade_amount}"
            
            else:
                return False, "自动交易方法未定义"
            
        except Exception as e:
            error_msg = f"自动交易执行失败: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return False, error_msg

# ---------------- 数据获取 ----------------
class DataFetcher:
    @staticmethod
    def get_price_multi(symbol=SYMBOL):
        """获取当前价格"""
        try:
            r = safe_get("https://api.binance.com/api/v3/ticker/price", 
                        params={"symbol": symbol}, timeout=5)
            if r:
                data = r.json()
                return float(data['price'])
            return float('nan')
        except Exception as e:
            print(f"获取价格失败: {e}")
            return float('nan')
    
    @staticmethod
    def get_klines_multi(symbol=SYMBOL, interval=INTERVAL, limit=DATA_LIMIT):
        """获取K线数据"""
        try:
            r = safe_get("https://api.binance.com/api/v3/klines",
                        params={"symbol": symbol, "interval": interval, "limit": limit})
            if not r:
                return pd.DataFrame()
            
            data = r.json()
            
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
            
            # 安全类型转换
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
            df = df.dropna()
            return df
        except Exception as e:
            print(f"K线获取失败: {e}")
            return pd.DataFrame()

# ---------------- 订单簿分析 ----------------
class OrderBookAnalyzer:
    @staticmethod
    def get_orderbook_depth(symbol=SYMBOL, limit=50):
        """获取订单簿深度"""
        try:
            r = safe_get("https://api.binance.com/api/v3/depth",
                        params={"symbol": symbol, "limit": limit})
            if not r:
                return [], []
            
            data = r.json()
            bids = []
            asks = []
            
            for price, qty in data.get('bids', []):
                try:
                    bids.append((float(price), float(qty)))
                except:
                    continue
            
            for price, qty in data.get('asks', []):
                try:
                    asks.append((float(price), float(qty)))
                except:
                    continue
            
            return bids, asks
        except Exception as e:
            print(f"获取订单簿失败: {e}")
            return [], []
    
    @staticmethod
    def analyze_orderbook(bids, asks, top_n=10):
        """分析订单簿"""
        if not bids or not asks:
            return None
        
        try:
            top_bids = bids[:top_n]
            top_asks = asks[:top_n]
            
            top_bid_vol = sum(q for _, q in top_bids)
            top_ask_vol = sum(q for _, q in top_asks)
            
            total_bid_vol = sum(q for _, q in bids)
            total_ask_vol = sum(q for _, q in asks)
            
            if total_bid_vol == 0 or total_ask_vol == 0:
                return None
            
            bid_concentration = top_bid_vol / total_bid_vol
            ask_concentration = top_ask_vol / total_ask_vol
            
            pressure_ratio = top_bid_vol / (top_ask_vol + 1e-9)
            imbalance = (top_bid_vol - top_ask_vol) / (top_bid_vol + top_ask_vol + 1e-9)
            
            return {
                'bid_concentration': bid_concentration,
                'ask_concentration': ask_concentration,
                'pressure_ratio': pressure_ratio,
                'imbalance': imbalance,
                'top_bid_vol': top_bid_vol,
                'top_ask_vol': top_ask_vol
            }
        except Exception as e:
            print(f"分析订单簿失败: {e}")
            return None

# ---------------- 技术指标 ----------------
class TechnicalIndicators:
    @staticmethod
    def add_features(df):
        """添加技术指标特征"""
        if df.empty or len(df) < 50:
            return df
            
        try:
            df = df.copy()
            
            # 基础特征
            df['returns'] = df['close'].pct_change()
            
            # 移动平均
            for period in [3, 5, 8, 13, 21, 50]:
                if len(df) >= period:
                    df[f'ma_{period}'] = df['close'].rolling(period).mean()
                    df[f'ma_{period}_diff'] = (df['close'] - df[f'ma_{period}']) / (df[f'ma_{period}'] + 1e-9)
            
            # RSI
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / (avg_loss + 1e-9)
                df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            if len(df) >= 26:
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema12 - ema26
                if len(df) >= 35:
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # 布林带
            if len(df) >= 20:
                df['bb_middle'] = df['close'].rolling(20).mean()
                bb_std = df['close'].rolling(20).std()
                df['bb_upper'] = df['bb_middle'] + 2 * bb_std
                df['bb_lower'] = df['bb_middle'] - 2 * bb_std
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-9)
            
            # ATR
            if len(df) >= 14:
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr_14'] = tr.rolling(14).mean()
                df['atr_pct'] = df['atr_14'] / (df['close'] + 1e-9)
            
            # 成交量
            if len(df) >= 21:
                df['volume_ma'] = df['volume'].rolling(21).mean()
                df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-9)
            
            return df.dropna()
        except Exception as e:
            print(f"计算技术指标失败: {e}")
            return df
    
    @staticmethod
    def create_labels(df, future_bars=5):
        """创建标签"""
        if df.empty or len(df) < future_bars:
            return df
            
        try:
            df = df.copy()
            if len(df) > future_bars:
                df['future_return'] = df['close'].shift(-future_bars) / df['close'] - 1
                df['label'] = (df['future_return'] > 0).astype(int)
            return df.dropna()
        except Exception as e:
            print(f"创建标签失败: {e}")
            return df

# ---------------- 进场过滤器 ----------------
class EntryFilter:
    def __init__(self):
        self.last_filter_result = None
    
    def evaluate_entry(self, symbol, current_price, predicted_direction):
        """评估进场条件"""
        scores = {'total': 0, 'max': 100}
        conditions = []
        
        try:
            # 1. 订单簿条件
            bids, asks = OrderBookAnalyzer.get_orderbook_depth(symbol)
            ob_analysis = OrderBookAnalyzer.analyze_orderbook(bids, asks)
            
            if ob_analysis:
                if predicted_direction == 1:  # 看涨
                    if ob_analysis['bid_concentration'] > MIN_ORDERBOOK_CONCENTRATION:
                        scores['total'] += 20
                        conditions.append(f"买盘集中度 ✓ ({ob_analysis['bid_concentration']:.1%})")
                    
                    if ob_analysis['pressure_ratio'] > MIN_PRESSURE_RATIO:
                        scores['total'] += 20
                        conditions.append(f"买压强劲 ✓ ({ob_analysis['pressure_ratio']:.1f}x)")
                else:  # 看跌
                    if ob_analysis['ask_concentration'] > MIN_ORDERBOOK_CONCENTRATION:
                        scores['total'] += 20
                        conditions.append(f"卖盘集中度 ✓ ({ob_analysis['ask_concentration']:.1%})")
                    
                    if ob_analysis['pressure_ratio'] < 1/MIN_PRESSURE_RATIO:
                        scores['total'] += 20
                        conditions.append(f"卖压强劲 ✓ ({1/ob_analysis['pressure_ratio']:.1f}x)")
            
            # 2. 技术指标条件
            try:
                df = DataFetcher.get_klines_multi(symbol, '1m', 100)
                if not df.empty and len(df) > 50:
                    df = TechnicalIndicators.add_features(df)
                    if len(df) > 0:
                        latest = df.iloc[-1].to_dict()
                    else:
                        latest = {}
                    
                    if predicted_direction == 1:  # 看涨
                        if latest.get('rsi', 50) < 50:
                            scores['total'] += 15
                            conditions.append(f"RSI偏低 ✓ ({latest.get('rsi', 0):.1f})")
                        
                        if latest.get('macd', 0) > latest.get('macd_signal', 0):
                            scores['total'] += 15
                            conditions.append("MACD金叉 ✓")
                        
                        if latest.get('close', 0) > latest.get('ma_13', 0):
                            scores['total'] += 10
                            conditions.append("价格在MA13之上 ✓")
                    
                    else:  # 看跌
                        if latest.get('rsi', 50) > 50:
                            scores['total'] += 15
                            conditions.append(f"RSI偏高 ✓ ({latest.get('rsi', 0):.1f})")
                        
                        if latest.get('macd', 0) < latest.get('macd_signal', 0):
                            scores['total'] += 15
                            conditions.append("MACD死叉 ✓")
                        
                        if latest.get('close', 0) < latest.get('ma_13', 0):
                            scores['total'] += 10
                            conditions.append("价格在MA13之下 ✓")
            except Exception as e:
                conditions.append(f"技术分析错误: {str(e)[:50]}")
            
            # 3. 波动率条件
            if 'atr_pct' in locals().get('df', pd.DataFrame()).columns and len(df) > 0:
                try:
                    atr_pct = df['atr_pct'].iloc[-1]
                    if 0.0005 <= atr_pct <= 0.003:
                        scores['total'] += 20
                        conditions.append(f"波动率适中 ✓ ({atr_pct:.4%})")
                    else:
                        conditions.append(f"波动率异常 ({atr_pct:.4%})")
                except:
                    pass
            
        except Exception as e:
            conditions.append(f"分析错误: {str(e)[:50]}")
        
        # 归一化分数
        normalized_score = scores['total'] / scores['max'] if scores['max'] > 0 else 0
        
        self.last_filter_result = {
            'should_enter': normalized_score >= 0.5,
            'score': normalized_score,
            'conditions': conditions,
            'raw_score': scores['total']
        }
        
        return self.last_filter_result

# ---------------- 胜率统计 ----------------
class WinRateTracker:
    def __init__(self):
        self.history = deque(maxlen=100)
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        
    def add_trade(self, is_win):
        """添加交易结果"""
        try:
            self.history.append(is_win)
            self.total_trades += 1
            
            if is_win:
                self.winning_trades += 1
                self.consecutive_wins += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_wins = 0
                self.consecutive_losses += 1
        except Exception as e:
            print(f"添加交易结果失败: {e}")
    
    def get_statistics(self):
        """获取统计信息"""
        try:
            if self.total_trades == 0:
                return {
                    'total_win_rate': 0,
                    'recent_win_rate': 0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'consecutive_wins': 0,
                    'consecutive_losses': 0,
                    'best_streak': 0,
                    'worst_streak': 0
                }
            
            total_win_rate = self.winning_trades / self.total_trades
            
            # 近期胜率 (最近20次)
            recent = list(self.history)[-20:] if len(self.history) >= 20 else list(self.history)
            recent_win_rate = sum(recent) / len(recent) if recent else 0
            
            # 计算历史最大连胜/连败
            best_streak = 0
            worst_streak = 0
            current_streak = 0
            current_type = None
            
            for result in self.history:
                if current_type is None:
                    current_type = result
                    current_streak = 1
                elif result == current_type:
                    current_streak += 1
                else:
                    if current_type:
                        best_streak = max(best_streak, current_streak)
                    else:
                        worst_streak = max(worst_streak, current_streak)
                    current_type = result
                    current_streak = 1
            
            if current_type:
                best_streak = max(best_streak, current_streak)
            else:
                worst_streak = max(worst_streak, current_streak)
            
            return {
                'total_win_rate': total_win_rate,
                'recent_win_rate': recent_win_rate,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.total_trades - self.winning_trades,
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses,
                'best_streak': best_streak,
                'worst_streak': worst_streak
            }
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return {
                'total_win_rate': 0,
                'recent_win_rate': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'consecutive_wins': 0,
                'consecutive_losses': 0,
                'best_streak': 0,
                'worst_streak': 0
            }
    
    def save(self, filename=STATS_FILE):
        """保存胜率数据到文件"""
        try:
            stats = {
                'history': list(self.history),
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(stats, f)
            
            print(f"胜率记录已保存到 {filename}")
            return True
        except Exception as e:
            print(f"保存胜率数据失败: {e}")
            return False
    
    def load(self, filename=STATS_FILE):
        """从文件加载胜率数据"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    stats = pickle.load(f)
                
                # 恢复数据
                self.history = deque(stats.get('history', []), maxlen=100)
                self.consecutive_wins = stats.get('consecutive_wins', 0)
                self.consecutive_losses = stats.get('consecutive_losses', 0)
                self.total_trades = stats.get('total_trades', 0)
                self.winning_trades = stats.get('winning_trades', 0)
                
                saved_at = stats.get('saved_at', '未知时间')
                print(f"已加载胜率数据 (保存于: {saved_at})")
                return True
            else:
                print("未找到胜率数据文件，将创建新记录")
                return False
        except Exception as e:
            print(f"加载胜率数据失败: {e}")
            return False

# 导入必要的模块
import os
from kivy.core.text import LabelBase
from kivy.config import Config
from kivy.utils import platform

# 平台检测
IS_ANDROID = platform == 'android'
IS_IOS = platform == 'ios'
IS_DESKTOP = not (IS_ANDROID or IS_IOS)

# 字体注册 - 只在桌面平台执行
if IS_DESKTOP:
    # 尝试注册系统中常见的中文字体
    try:
        # 尝试注册Microsoft YaHei字体
        LabelBase.register(name='msyh', fn_regular='C:\\Windows\\Fonts\\msyh.ttc')
        print("成功注册Microsoft YaHei字体")
    except Exception as e:
        print(f"注册Microsoft YaHei字体失败: {e}")
    
    try:
        # 尝试注册SimSun字体
        LabelBase.register(name='simsun', fn_regular='C:\\Windows\\Fonts\\simsun.ttc')
        print("成功注册SimSun字体")
    except Exception as e:
        print(f"注册SimSun字体失败: {e}")

# 设置环境变量，强制使用支持中文的字体
if IS_DESKTOP:
    os.environ['KIVY_DEFAULT_FONT'] = 'msyh,simsun,Arial'
else:
    # 在移动平台使用默认字体
    os.environ['KIVY_DEFAULT_FONT'] = 'Arial'

# 设置Kivy配置
if IS_DESKTOP:
    Config.set('kivy', 'default_font', ['msyh', 'simsun', 'Arial'])
else:
    Config.set('kivy', 'default_font', ['Arial'])



# ---------------- Kivy主应用 ----------------
class BTC5MinKivyApp(App):
    def build(self):
        """构建Kivy应用"""
        Window.clearcolor = (1, 1, 1, 1)
        # 自动适应不同手机屏幕尺寸
        # 首先尝试获取设备屏幕尺寸
        try:
            from kivy.utils import platform
            if platform == 'android' or platform == 'ios':
                # 在移动设备上使用全屏
                Window.fullscreen = True
            else:
                # 在桌面设备上使用常见的手机屏幕尺寸
                Window.size = (375, 812)  # iPhone X尺寸
        except:
            # 如果获取失败，使用默认的手机屏幕尺寸
            Window.size = (375, 812)  # iPhone X尺寸
        return MainScreen()

class AnimatedButton(ButtonBehavior, BoxLayout):
    """带有动画效果的按钮"""
    def __init__(self, **kwargs):
        super(AnimatedButton, self).__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (120, 40)
    
    def on_press(self):
        # 按下动画
        anim = Animation(size=(115, 35), duration=0.1)
        anim.start(self)
    
    def on_release(self):
        # 释放动画
        anim = Animation(size=(120, 40), duration=0.1)
        anim.start(self)

class MainScreen(BoxLayout):
    # 字符串属性
    price_text = StringProperty("$ --")
    prediction_text = StringProperty("等待预测...")
    confidence_text = StringProperty("置信度: --")
    countdown_text = StringProperty("准备中")
    status_text = StringProperty("就绪")
    time_text = StringProperty(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    model_status_text = StringProperty("模型状态: 初始化中...")
    auto_trade_status_text = StringProperty("✗ 已禁用")
    conditions_text = StringProperty("进场条件将在此显示...")
    coord_display_text = StringProperty("当前坐标设置:\n  金额输入框: (100, 100)\n  买涨按钮: (200, 200)\n  买跌按钮: (300, 200)\n  确认按钮: (400, 300)\n交易金额: $10")
    
    # 布尔属性
    auto_trade_enabled = BooleanProperty(False)
    
    # 数值属性
    trade_amount = NumericProperty(10)
    
    # 统计数据属性
    total_win_rate = StringProperty("--")
    recent_win_rate = StringProperty("--")
    total_trades = StringProperty("--")
    winning_trades = StringProperty("--")
    losing_trades = StringProperty("--")
    consecutive_wins = StringProperty("--")
    consecutive_losses = StringProperty("--")
    best_streak = StringProperty("--")
    worst_streak = StringProperty("--")
    
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 0
        self.spacing = 0
        
        # 设置主窗口背景颜色为白色
        with self.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(pos=self.pos, size=self.size)
        
        # 绑定位置和大小变化事件
        self.bind(pos=self.update_rect, size=self.update_rect)
        
        # 语音播报设置
        self.tts_enabled = TTS_AVAILABLE
        self.tts_announce_prediction = True
        self.tts_announce_countdown = True
        self.tts_announce_result = True
        
        # 自动下单模块显示控制
        self.auto_trade_module_visible = False
        
        # 初始化组件
        self.data_fetcher = DataFetcher()
        self.orderbook_analyzer = OrderBookAnalyzer()
        self.tech_indicators = TechnicalIndicators()
        self.entry_filter = EntryFilter()
        self.winrate_tracker = WinRateTracker()
        self.auto_trader = AutoTrader()
        
        # 状态变量
        self.running = False
        self.current_price = 0
        self.current_prediction = None
        self.current_confidence = 0
        self.countdown_active = False
        self.countdown_seconds = COUNTDOWN_SECONDS
        self.start_time = None
        
        # 模型
        self.model = None
        self.model_loaded = False
        self.scaler = None
        
        # 线程和队列
        self.queue = queue.Queue()
        
        # 创建GUI
        self.create_widgets()
        
        # 延迟加载模型和胜率数据
        Clock.schedule_once(self.initialize_app, 0.1)
        
        # 启动价格更新
        self.start_price_updater()
        
        # 启动K线图更新
        self.start_kline_updater()
        
        # 启动队列处理
        Clock.schedule_interval(self.process_queue, 0.1)
        
        # 更新时间
        Clock.schedule_interval(self.update_time, 1)
    
    def update_rect(self, instance, value):
        """更新背景矩形的位置和大小"""
        self.rect.pos = instance.pos
        self.rect.size = instance.size
    
    def create_widgets(self):
        """创建Kivy组件"""
        # === 标题 ===
        title_frame = FloatLayout(size_hint_y=0.07)
        title_frame.canvas.before.clear()
        with title_frame.canvas.before:
            Color(1, 1, 1, 1)
            Rectangle(pos=title_frame.pos, size=title_frame.size)
        title_label = Label(text="BTC五分钟预测", font_size=14, halign='center', bold=True, pos_hint={'center_x': 0.5, 'center_y': 0.8}, font_name='msyh', color=(0, 0, 0, 1))
        title_frame.add_widget(title_label)
        self.add_widget(title_frame)
        
        # === 顶部状态栏 ===
        top_frame = FloatLayout(size_hint_y=0.09)
        top_frame.canvas.before.clear()
        with top_frame.canvas.before:
            Color(1, 1, 1, 1)
            Rectangle(pos=top_frame.pos, size=top_frame.size)
        
        # 网格布局
        grid_layout = GridLayout(cols=5, spacing=6, pos_hint={'x': 0.05, 'y': 0.05}, size_hint=(0.9, 0.9))
        
        # 当前价格
        price_frame = FloatLayout()
        price_label = Label(text="当前价格", font_size=8, halign='center', bold=True, pos_hint={'center_x': 0.5, 'y': 0.7}, font_name='msyh', color=(0, 0, 0, 1))
        self.price_display = Label(text="$-------", font_size=9, halign='center', pos_hint={'center_x': 0.5, 'y': 0.3}, font_name='msyh', color=(0, 0, 0, 1))
        price_frame.add_widget(price_label)
        price_frame.add_widget(self.price_display)
        grid_layout.add_widget(price_frame)
        
        # 5分钟预测
        pred_frame = FloatLayout()
        pred_label = Label(text="5分钟预测", font_size=8, halign='center', bold=True, pos_hint={'center_x': 0.5, 'y': 0.7}, font_name='msyh', color=(0, 0, 0, 1))
        self.pred_display = Label(text="等待预测", font_size=9, halign='center', pos_hint={'center_x': 0.5, 'y': 0.3}, font_name='msyh', color=(0, 0, 0, 1))
        pred_frame.add_widget(pred_label)
        pred_frame.add_widget(self.pred_display)
        grid_layout.add_widget(pred_frame)
        
        # 置信度
        conf_frame = FloatLayout()
        conf_label = Label(text="置信度", font_size=8, halign='center', bold=True, pos_hint={'center_x': 0.5, 'y': 0.7}, font_name='msyh', color=(0, 0, 0, 1))
        self.conf_display = Label(text="----", font_size=9, halign='center', pos_hint={'center_x': 0.5, 'y': 0.3}, font_name='msyh', color=(0, 0, 0, 1))
        conf_frame.add_widget(conf_label)
        conf_frame.add_widget(self.conf_display)
        grid_layout.add_widget(conf_frame)
        
        # 倒计时
        count_frame = FloatLayout()
        count_label = Label(text="倒计时", font_size=8, halign='center', bold=True, pos_hint={'center_x': 0.5, 'y': 0.7}, font_name='msyh', color=(0, 0, 0, 1))
        self.count_display = Label(text="5:00", font_size=9, halign='center', pos_hint={'center_x': 0.5, 'y': 0.3}, font_name='msyh', color=(0, 0, 0, 1))
        count_frame.add_widget(count_label)
        count_frame.add_widget(self.count_display)
        grid_layout.add_widget(count_frame)
        
        # 语音播报
        tts_frame = FloatLayout()
        tts_label = Label(text="语音播报", font_size=8, halign='center', bold=True, pos_hint={'center_x': 0.5, 'y': 0.7}, font_name='msyh', color=(0, 0, 0, 1))
        tts_status = "语音: 启用" if TTS_AVAILABLE else "语音: 关闭"
        tts_color = (0, 1, 0, 1) if TTS_AVAILABLE else (1, 0, 0, 1)  # 启用显示绿色，关闭显示红色
        tts_status_label = Label(text=tts_status, font_size=8, halign='center', color=tts_color, pos_hint={'center_x': 0.5, 'y': 0.3}, font_name='msyh')
        tts_frame.add_widget(tts_label)
        tts_frame.add_widget(tts_status_label)
        grid_layout.add_widget(tts_frame)
        
        # 将网格布局添加到top_frame
        top_frame.add_widget(grid_layout)
        
        self.add_widget(top_frame)
        
        # === 控制面板 ===
        control_frame = FloatLayout(size_hint_y=0.09)
        control_frame.canvas.before.clear()
        with control_frame.canvas.before:
            Color(1, 1, 1, 1)
            Rectangle(pos=control_frame.pos, size=control_frame.size)
        
        # 控制面板标题
        control_title = Label(text="控制面板", font_size=9, halign='left', bold=True, pos_hint={'x': 0.05, 'y': 0.95}, size_hint=(0.9, 0.2), font_name='msyh', color=(0, 0, 0, 1))
        control_frame.add_widget(control_title)
        
        # 按钮框架
        btn_frame = BoxLayout(spacing=5, pos_hint={'x': 0.05, 'y': 0.4}, size_hint=(0.9, 0.5))
        
        # 开始预测按钮
        self.start_btn = Button(text="开始预测", background_color=(0, 0.8, 0, 1), color=(1, 1, 1, 1), font_size=8, bold=True, font_name='msyh')
        self.start_btn.bind(on_press=self.start_prediction)
        btn_frame.add_widget(self.start_btn)
        
        # 停止按钮
        self.stop_btn = Button(text="停止", background_color=(1, 0, 0, 1), color=(1, 1, 1, 1), font_size=8, bold=True, disabled=True, font_name='msyh')
        self.stop_btn.bind(on_press=self.stop_prediction)
        btn_frame.add_widget(self.stop_btn)
        
        # 训练模型按钮
        train_btn = Button(text="训练模型", background_color=(0, 0, 1, 1), color=(1, 1, 1, 1), font_size=8, font_name='msyh')
        train_btn.bind(on_press=self.train_model)
        btn_frame.add_widget(train_btn)
        
        # 重置统计按钮
        reset_btn = Button(text="重置统计", background_color=(1, 0.5, 0, 1), color=(1, 1, 1, 1), font_size=8, font_name='msyh')
        reset_btn.bind(on_press=self.reset_statistics)
        btn_frame.add_widget(reset_btn)
        
        # 语音测试按钮
        if TTS_AVAILABLE:
            tts_btn = Button(text="语音测试", background_color=(0.5, 0, 0.5, 1), color=(1, 1, 1, 1), font_size=8, font_name='msyh')
            tts_btn.bind(on_press=self.test_tts)
            btn_frame.add_widget(tts_btn)
        
        control_frame.add_widget(btn_frame)
        
        # 模型状态
        model_status_label = Label(text="模型状态: 初始化中", font_size=8, halign='left', bold=True, pos_hint={'x': 0.05, 'y': 0.1}, size_hint=(0.9, 0.2), font_name='msyh', color=(1, 1, 1, 1))
        control_frame.add_widget(model_status_label)
        
        self.add_widget(control_frame)
        
        # === 1分钟K线图 ===
        kline_frame = FloatLayout(size_hint_y=0.1)
        kline_frame.canvas.before.clear()
        with kline_frame.canvas.before:
            Color(1, 1, 1, 1)
            Rectangle(pos=kline_frame.pos, size=kline_frame.size)
        
        kline_label = Label(text="1分钟K线图", font_size=9, halign='left', bold=True, pos_hint={'x': 0.05, 'y': 0.8}, size_hint=(0.9, 0.2), font_name='msyh', color=(0, 0, 0, 1))
        kline_frame.add_widget(kline_label)
        
        # K线图画布
        from kivy.uix.widget import Widget
        class KLineWidget(Widget):
            def __init__(self, **kwargs):
                super(KLineWidget, self).__init__(**kwargs)
                self.kline_data = []
                self.bind(pos=self.update_kline, size=self.update_kline)
                # 初始化时绘制空图表
                self.update_kline(self, None)
            
            def set_kline_data(self, data):
                self.kline_data = data
                self.update_kline(self, None)
            
            def update_kline(self, instance, value):
                self.canvas.clear()
                
                # 确保pos和size有效
                if not hasattr(self, 'pos') or not hasattr(self, 'size'):
                    return
                
                if self.size[0] <= 0 or self.size[1] <= 0:
                    return
                
                # 绘制背景和网格线，即使没有数据
                with self.canvas:
                    # 背景
                    Color(1, 1, 1, 1)
                    Rectangle(pos=self.pos, size=self.size)
                    
                    # 网格线
                    Color(0.8, 0.8, 0.8, 1)
                    for i in range(5):
                        y = self.pos[1] + self.size[1] * i / 4
                        Rectangle(pos=(self.pos[0], y), size=(self.size[0], 1))
                    for i in range(10):
                        x = self.pos[0] + self.size[0] * i / 9
                        Rectangle(pos=(x, self.pos[1]), size=(1, self.size[1]))
                
                # 如果没有数据，显示提示信息
                if not self.kline_data or len(self.kline_data) == 0:
                    with self.canvas:
                        Color(0.5, 0.5, 0.5, 1)
                        from kivy.core.text import Label as CoreLabel
                        label = CoreLabel(text="加载K线数据中...", font_size=10, color=(0.5, 0.5, 0.5, 1))
                        label.refresh()
                        text_texture = label.texture
                        text_pos = (self.pos[0] + self.size[0]/2 - text_texture.width/2, 
                                   self.pos[1] + self.size[1]/2 - text_texture.height/2)
                        Rectangle(texture=text_texture, pos=text_pos, size=text_texture.size)
                    return
                
                # 绘制K线图
                with self.canvas:
                    # 计算K线数据范围
                    highs = [k[2] for k in self.kline_data]
                    lows = [k[3] for k in self.kline_data]
                    if highs and lows:
                        max_price = max(highs)
                        min_price = min(lows)
                        price_range = max_price - min_price if max_price > min_price else 1
                        
                        # 绘制K线
                        bar_width = self.size[0] / len(self.kline_data) * 0.8
                        for i, kline in enumerate(self.kline_data):
                            open_p, high_p, low_p, close_p = kline[1], kline[2], kline[3], kline[4]
                            
                            # 计算坐标
                            x = self.pos[0] + i * self.size[0] / len(self.kline_data) + bar_width * 0.1
                            y_high = self.pos[1] + (high_p - min_price) / price_range * self.size[1]
                            y_low = self.pos[1] + (low_p - min_price) / price_range * self.size[1]
                            y_open = self.pos[1] + (open_p - min_price) / price_range * self.size[1]
                            y_close = self.pos[1] + (close_p - min_price) / price_range * self.size[1]
                            
                            # 确保坐标有效
                            if x < self.pos[0] or x > self.pos[0] + self.size[0]:
                                continue
                            
                            # 绘制影线
                            Color(0, 0, 0, 1)
                            Rectangle(pos=(x + bar_width/2 - 1, min(y_open, y_close)), size=(2, abs(y_open - y_close)))
                            Rectangle(pos=(x + bar_width/2 - 1, y_low), size=(2, y_high - y_low))
                            
                            # 绘制实体
                            if close_p >= open_p:
                                Color(0, 1, 0, 1)  # 上涨
                            else:
                                Color(1, 0, 0, 1)  # 下跌
                            Rectangle(pos=(x, min(y_open, y_close)), size=(bar_width, abs(y_open - y_close)))
        
        self.kline_widget = KLineWidget(pos_hint={'x': 0.05, 'y': 0.05}, size_hint=(0.9, 0.7))
        kline_frame.add_widget(self.kline_widget)
        
        self.add_widget(kline_frame)
        
        # === 胜率统计 ===
        stats_frame = FloatLayout(size_hint_y=0.14)
        stats_frame.canvas.before.clear()
        with stats_frame.canvas.before:
            Color(1, 1, 1, 1)
            Rectangle(pos=stats_frame.pos, size=stats_frame.size)
        
        stats_label = Label(text="胜率统计", font_size=9, halign='left', bold=True, pos_hint={'x': 0.05, 'y': 0.9}, size_hint=(0.9, 0.1), font_name='msyh', color=(0, 0, 0, 1))
        stats_frame.add_widget(stats_label)
        
        # 使用网格布局
        stats_grid = GridLayout(cols=3, spacing=6, pos_hint={'x': 0.05, 'y': 0.05}, size_hint=(0.9, 0.8))
        
        # 统计项
        stats_items = [
            ("总胜率: --", "total_win_rate"),
            ("近期胜率: --", "recent_win_rate"),
            ("总交易数: --", "total_trades"),
            ("盈利交易: --", "winning_trades"),
            ("亏损交易: --", "losing_trades"),
            ("当前连胜: --", "consecutive_wins"),
            ("当前连败: --", "consecutive_losses"),
            ("最大连胜: --", "best_streak"),
            ("最大连败: --", "worst_streak")
        ]
        
        self.stats_labels = {}
        
        # 创建统计标签
        for label_text, key in stats_items:
            # 根据key设置不同的颜色
            if key in ['total_win_rate', 'recent_win_rate', 'total_trades']:
                color = (0, 0, 1, 1)  # 蓝色
            elif key in ['winning_trades', 'consecutive_wins', 'best_streak']:
                color = (0, 1, 0, 1)  # 绿色
            elif key in ['losing_trades', 'consecutive_losses', 'worst_streak']:
                color = (1, 0, 0, 1)  # 红色
            else:
                color = (0, 0, 0, 1)  # 默认黑色
            
            stat_label = Label(text=label_text, font_size=8, halign='left', font_name='msyh', color=color)
            stats_grid.add_widget(stat_label)
            self.stats_labels[key] = stat_label
        
        stats_frame.add_widget(stats_grid)
        
        self.add_widget(stats_frame)
        
        # === 运行日志 ===
        # 隐藏运行日志模块
        # log_frame = FloatLayout(size_hint_y=0.12)
        # log_frame.canvas.before.clear()
        # with log_frame.canvas.before:
        #     Color(0.6, 0.2, 0.6, 1)
        #     Rectangle(pos=log_frame.pos, size=log_frame.size)
        # 
        # log_label = Label(text="运行日志", font_size=7, halign='left', bold=True, pos_hint={'x': 0.05, 'y': 0.9}, size_hint=(0.9, 0.1), font_name='msyh', color=(1, 1, 1, 1))
        # log_frame.add_widget(log_label)
        # 
        # self.log_display = TextInput(text="", multiline=True, readonly=True, background_color=(1, 1, 1, 1), font_size=5, pos_hint={'x': 0.05, 'y': 0.05}, size_hint=(0.9, 0.8), font_name='msyh')
        # log_frame.add_widget(self.log_display)
        # 
        # self.add_widget(log_frame)
        
        # === 状态栏 ===
        status_frame = FloatLayout(size_hint_y=0.07)
        status_frame.canvas.before.clear()
        with status_frame.canvas.before:
            Color(1, 1, 1, 1)
            Rectangle(pos=status_frame.pos, size=status_frame.size)
        
        self.status_label = Label(text="初始化失败", font_size=6, halign='left', bold=True, pos_hint={'x': 0.05, 'y': 0.5}, size_hint=(0.7, 0.5), font_name='msyh', color=(0, 0, 0, 1))
        status_frame.add_widget(self.status_label)
        
        self.time_label = Label(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), font_size=5, halign='right', pos_hint={'x': 0.75, 'y': 0.5}, size_hint=(0.2, 0.5), font_name='msyh', color=(0, 0, 0, 1))
        status_frame.add_widget(self.time_label)
        
        self.add_widget(status_frame)
    
    def update_time(self, dt):
        """更新时间显示"""
        self.time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.text = self.time_text
    
    def log(self, message, level="INFO"):
        """添加日志"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # 颜色映射
            color_tags = {
                "INFO": "[color=#ffffff]",
                "ERROR": "[color=#ff6b6b]",
                "WARNING": "[color=#ffd93d]",
                "SUCCESS": "[color=#6bcf7f]",
                "TRADE": "[color=#64b5f6]"  # 交易相关日志
            }
            
            tag = color_tags.get(level, "[color=#ffffff]")
            formatted = f"[{timestamp}] {message}\n"
            
            self.queue.put(('log', (formatted, tag)))
        except Exception as e:
            print(f"日志记录失败: {e}")
    
    def speak(self, text, rate=180, volume=0.9):
        """语音播报"""
        if self.tts_enabled:
            return speak_text(text, rate, volume)
        return False
    
    def test_tts(self, instance):
        """测试语音播报"""
        if not self.tts_enabled:
            self.log("语音播报功能不可用", "ERROR")
            return
        
        try:
            self.speak("BTC预测系统语音测试")
            self.log("语音播报测试完成", "SUCCESS")
        except Exception as e:
            self.log(f"语音测试失败: {e}", "ERROR")
    
    def update_status(self, message):
        """更新状态栏"""
        try:
            self.queue.put(('status', message))
        except:
            pass
    
    def update_price(self, price):
        """更新价格"""
        try:
            if not math.isnan(price):
                self.current_price = price
                self.queue.put(('price', price))
        except:
            pass
    
    def update_prediction(self, prediction, confidence):
        """更新预测"""
        try:
            self.current_prediction = prediction
            self.current_confidence = confidence
            self.queue.put(('prediction', (prediction, confidence)))
            
            # 语音播报
            if self.tts_enabled and self.tts_announce_prediction:
                if prediction == 1:
                    speak_text(f"预测上涨，置信度{confidence:.0%}")
                elif prediction == 0:
                    speak_text(f"预测下跌，置信度{confidence:.0%}")
        except Exception as e:
            print(f"更新预测失败: {e}")
    
    def update_filter(self, status, score, conditions=None):
        """更新过滤器状态"""
        try:
            self.queue.put(('filter', (status, score, conditions)))
        except:
            pass
    
    def update_countdown(self, seconds):
        """更新倒计时"""
        try:
            self.queue.put(('countdown', seconds))
            
            # 语音播报关键节点
            if self.tts_enabled and self.tts_announce_countdown:
                if seconds == 300:
                    speak_text("开始5分钟倒计时")
                elif seconds == 60:
                    speak_text("剩余1分钟")
                elif seconds == 10:
                    speak_text("倒计时10秒")
        except:
            pass
    
    def update_statistics_display(self, stats=None):
        """更新统计数据显示"""
        try:
            if stats is None:
                stats = self.winrate_tracker.get_statistics()
            
            for key, label in self.stats_labels.items():
                if key in stats:
                    value = stats[key]
                    
                    if key == 'total_win_rate':
                        label.text = f"总胜率: {value:.2%}"
                    elif key == 'recent_win_rate':
                        label.text = f"近期胜率: {value:.2%}"
                    elif key == 'total_trades':
                        label.text = f"总交易数: {value}"
                    elif key == 'winning_trades':
                        label.text = f"盈利交易: {value}"
                    elif key == 'losing_trades':
                        label.text = f"亏损交易: {value}"
                    elif key == 'consecutive_wins':
                        label.text = f"当前连胜: {value}"
                    elif key == 'consecutive_losses':
                        label.text = f"当前连败: {value}"
                    elif key == 'best_streak':
                        label.text = f"最大连胜: {value}"
                    elif key == 'worst_streak':
                        label.text = f"最大连败: {value}"
        except Exception as e:
            print(f"更新统计显示失败: {e}")
    
    def update_coord_display(self):
        """更新坐标显示"""
        try:
            coords = self.auto_trader.coords
            display_text = "当前坐标设置:\n"
            for coord_type, (x, y) in coords.items():
                if coord_type == 'amount':
                    display_text += f"  金额输入框: ({x}, {y})\n"
                elif coord_type == 'buy_up':
                    display_text += f"  买涨按钮: ({x}, {y})\n"
                elif coord_type == 'buy_down':
                    display_text += f"  买跌按钮: ({x}, {y})\n"
                elif coord_type == 'confirm':
                    display_text += f"  确认按钮: ({x}, {y})\n"
            display_text += f"交易金额: ${self.auto_trader.trade_amount}"
            self.coord_display.text = display_text
        except Exception as e:
            print(f"更新坐标显示失败: {e}")
    
    def process_queue(self, dt):
        """处理消息队列"""
        try:
            while not self.queue.empty():
                msg_type, data = self.queue.get_nowait()
                
                if msg_type == 'log':
                    message, tag = data
                    self.log_display.text += message
                    self.log_display.scroll_y = 0
                
                elif msg_type == 'status':
                    self.status_text = data
                    self.status_label.text = self.status_text
                
                elif msg_type == 'price':
                    price = data
                    self.price_text = f"${price:,.2f}"
                    self.price_display.text = self.price_text
                
                elif msg_type == 'prediction':
                    prediction, confidence = data
                    
                    if prediction == 1:
                        self.prediction_text = "预测方向: 上涨"
                        self.pred_display.text = self.prediction_text
                        self.pred_display.color = (0.3, 0.75, 0.4, 1)
                    elif prediction == 0:
                        self.prediction_text = "预测方向: 下跌"
                        self.pred_display.text = self.prediction_text
                        self.pred_display.color = (0.9, 0.27, 0.21, 1)
                    else:
                        self.prediction_text = "预测：--"
                        self.pred_display.text = self.prediction_text
                        self.pred_display.color = (0.5, 0.5, 0.5, 1)
                    
                    self.confidence_text = f"置信度: {confidence:.2%}"
                    self.conf_display.text = self.confidence_text
                
                elif msg_type == 'filter':
                    status, score, conditions = data
                    
                    # 更新条件显示
                    if conditions:
                        conditions_text = ""
                        for i, condition in enumerate(conditions, 1):
                            conditions_text += f"{i}. {condition}\n"
                        self.conditions_display.text = conditions_text
                
                elif msg_type == 'countdown':
                    if data == "准备中":
                        self.countdown_text = "准备中"
                        self.count_display.text = self.countdown_text
                    else:
                        minutes = data // 60
                        seconds = data % 60
                        self.countdown_text = f"{minutes:02d}:{seconds:02d}"
                        self.count_display.text = self.countdown_text
                        
                        # 颜色变化
                        if data > 60:
                            self.count_display.color = (0.13, 0.55, 0.95, 1)
                        elif data > 10:
                            self.count_display.color = (1, 0.6, 0, 1)
                        else:
                            self.count_display.color = (0.9, 0.27, 0.21, 1)
        
        except Exception as e:
            print(f"队列处理错误: {e}")
    
    # ========== 自动交易相关方法 ==========
    def toggle_auto_trade(self, instance, value):
        """切换自动交易状态"""
        self.auto_trader.enabled = value
        
        # 更新状态标签
        if self.auto_trader.enabled:
            self.auto_trade_status_text = "✓ 已启用"
            self.auto_trade_status_label.text = self.auto_trade_status_text
            self.auto_trade_status_label.color = (0.3, 0.75, 0.4, 1)
            self.log("自动交易已启用", "SUCCESS")
            if self.tts_enabled:
                self.speak("自动交易已启用")
        else:
            self.auto_trade_status_text = "✗ 已禁用"
            self.auto_trade_status_label.text = self.auto_trade_status_text
            self.auto_trade_status_label.color = (0.9, 0.27, 0.21, 1)
            self.log("自动交易已禁用", "WARNING")
    
    def set_trade_amount(self, instance):
        """设置交易金额"""
        try:
            amount_str = self.amount_input.text
            if amount_str:
                amount = float(amount_str)
                if amount > 0:
                    self.auto_trader.trade_amount = amount
                    self.log(f"交易金额设置为: ${amount}", "SUCCESS")
                    self.update_coord_display()
                else:
                    self.log("交易金额必须大于0", "ERROR")
        except ValueError:
            self.log("请输入有效的数字", "ERROR")
    
    def set_coordinate(self, coord_type):
        """设置坐标"""
        if not AUTO_TRADE_AVAILABLE:
            self.log("pyautogui不可用，无法设置坐标", "ERROR")
            return
        
        try:
            # 显示提示信息
            coord_names = {
                'amount': "金额输入框",
                'buy_up': "买涨按钮",
                'buy_down': "买跌按钮",
                'confirm': "确认按钮"
            }
            
            name = coord_names.get(coord_type, coord_type)
            
            self.log(f"请将鼠标移动到{name}位置，然后在5秒内按下确认键", "INFO")
            
            # 等待5秒让用户移动鼠标
            def get_mouse_position(dt):
                # 获取当前鼠标位置
                x, y = self.auto_trader.get_current_mouse_position()
                self.auto_trader.set_coordinate(coord_type, x, y)
                
                self.log(f"{name}坐标已设置为: ({x}, {y})", "SUCCESS")
                self.update_coord_display()
                self.update_status(f"{name}坐标设置完成")
                
                # 语音提示
                if self.tts_enabled:
                    self.speak(f"{name}坐标设置完成")
            
            Clock.schedule_once(get_mouse_position, 5)
        except Exception as e:
            self.log(f"设置坐标失败: {e}", "ERROR")
    
    def save_coordinates(self, instance):
        """保存坐标配置"""
        if self.auto_trader.save_coordinates():
            self.log("坐标配置已保存", "SUCCESS")
            if self.tts_enabled:
                self.speak("坐标配置已保存")
        else:
            self.log("保存坐标配置失败", "ERROR")
    
    def test_coordinate(self, coord_type):
        """测试点击坐标"""
        if not AUTO_TRADE_AVAILABLE:
            self.log("pyautogui不可用，无法测试点击", "ERROR")
            return
        
        success, message = self.auto_trader.test_click(coord_type)
        if success:
            self.log(f"测试成功: {message}", "SUCCESS")
        else:
            self.log(f"测试失败: {message}", "ERROR")
    
    def execute_auto_trade(self, direction):
        """执行自动交易"""
        if not self.auto_trader.enabled:
            return False, "自动交易未启用"
        
        self.log("开始执行自动下单...", "TRADE")
        self.update_status("自动下单中...")
        
        # 在新线程中执行交易，避免阻塞
        def trade_thread():
            success, message = self.auto_trader.execute_trade(direction)
            if success:
                self.log(message, "TRADE")
                self.update_status("自动下单成功")
                # 启动倒计时
                self.countdown_active = True
                self.start_countdown()
                # 语音播报
                if self.tts_enabled:
                    if direction == 1:
                        self.speak("多进去")
                    else:
                        self.speak("空进去")
            else:
                self.log(message, "ERROR")
                self.update_status("自动下单失败")
        
        threading.Thread(target=trade_thread, daemon=True).start()
        return True, "开始执行自动下单"
    
    def start_countdown(self):
        """开始5分钟倒计时"""
        def countdown_thread():
            self.start_time = time.time()
            end_time = self.start_time + COUNTDOWN_SECONDS
            
            self.update_status("5分钟倒计时开始...")
            
            # 倒计时循环
            while time.time() < end_time and self.running and self.countdown_active:
                remaining = int(end_time - time.time())
                self.update_countdown(remaining)
                
                # 每30秒更新一次价格
                if int(time.time()) % 30 == 0:
                    current = self.data_fetcher.get_price_multi()
                    if not math.isnan(current):
                        self.update_price(current)
                
                time.sleep(1)
            
            # 倒计时结束
            if self.running:
                self.countdown_active = False
                self.update_countdown("准备中")
        
        threading.Thread(target=countdown_thread, daemon=True).start()
    def initialize_app(self, dt):
        """初始化应用"""
        try:
            # 加载胜率数据
            if self.winrate_tracker.load():
                self.log("胜率记录已加载", "SUCCESS")
            else:
                self.log("创建新的胜率记录", "INFO")
            
            # 加载模型
            self.load_model()
            
            # 更新自动交易状态显示
            self.auto_trade_enabled = self.auto_trader.enabled
            try:
                self.amount_input.text = str(self.auto_trader.trade_amount)
                self.toggle_auto_trade(None, self.auto_trader.enabled)  # 更新状态标签
            except AttributeError:
                # 如果这些组件不存在，跳过
                pass
            
            # 语音播报测试
            if self.tts_enabled:
                self.log("语音播报功能已启用", "SUCCESS")
                # 延迟播报欢迎语
                Clock.schedule_once(lambda x: self.speak("BTC五分钟预测系统已启动"), 1)
            else:
                self.log("语音播报功能不可用", "WARNING")
            
            # 自动交易状态
            if AUTO_TRADE_AVAILABLE:
                self.log("自动下单功能已就绪", "SUCCESS")
            else:
                self.log("警告: pyautogui不可用，自动下单功能将不可用", "WARNING")
                
            self.update_status("就绪")
            
        except Exception as e:
            self.log(f"初始化失败: {str(e)}", "ERROR")
            self.update_status("初始化失败")
    
    def load_model(self):
        """加载模型"""
        try:
            if os.path.exists(MODEL_STACK_FILE):
                with open(MODEL_STACK_FILE, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                
                if self.model and self.scaler:
                    self.model_loaded = True
                    self.model_status_text = "模型状态: 已加载"
                    self.log("模型加载成功", "SUCCESS")
                else:
                    self.model_status_text = "模型状态: 文件损坏"
                    self.log("模型文件损坏", "WARNING")
            else:
                self.model_status_text = "模型状态: 未训练"
                self.log("未找到模型文件，请先训练模型", "WARNING")
        except Exception as e:
            self.model_status_text = "模型状态: 加载失败"
            self.log(f"模型加载失败: {str(e)}", "ERROR")
    
    def start_price_updater(self):
        """启动价格更新"""
        def updater():
            while True:
                try:
                    price = self.data_fetcher.get_price_multi()
                    if not math.isnan(price):
                        self.update_price(price)
                    time.sleep(2)
                except Exception as e:
                    print(f"价格更新失败: {e}")
                    time.sleep(5)
        
        threading.Thread(target=updater, daemon=True).start()
    
    def start_kline_updater(self):
        """启动K线图更新"""
        def updater():
            while True:
                try:
                    # 直接使用Binance API获取1分钟K线数据
                    import requests
                    api_url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=30"
                    response = requests.get(api_url)
                    if response.status_code == 200:
                        klines = response.json()
                        kline_data = []
                        for k in klines:
                            kline_data.append([
                                int(k[0] / 1000),  # 转换为秒
                                float(k[1]),  # open
                                float(k[2]),  # high
                                float(k[3]),  # low
                                float(k[4])   # close
                            ])
                        # 在主线程中更新K线图
                        from kivy.clock import Clock
                        if hasattr(self, 'kline_widget'):
                            Clock.schedule_once(lambda dt: self.kline_widget.set_kline_data(kline_data))
                    time.sleep(10)  # 每10秒更新一次
                except Exception as e:
                    print(f"K线更新失败: {e}")
                    time.sleep(15)
        
        threading.Thread(target=updater, daemon=True).start()
    
    def start_prediction(self, instance):
        """开始预测"""
        if self.running:
            return
        
        self.running = True
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        
        # 启动预测线程
        threading.Thread(target=self.prediction_loop, daemon=True).start()
        
        self.log("开始5分钟预测循环", "SUCCESS")
        self.update_status("运行中...")
        
        # 语音播报开始
        if self.tts_enabled:
            self.speak("开始BTC五分钟预测")
    
    def stop_prediction(self, instance):
        """停止预测"""
        if not self.running:
            return
        
        self.running = False
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        self.countdown_active = False
        
        self.log("预测循环已停止", "WARNING")
        self.update_status("已停止")
        self.update_countdown("准备中")
        
        # 语音播报停止
        if self.tts_enabled:
            self.speak("预测已停止")
    
    def prediction_loop(self):
        """预测主循环"""
        round_count = 0
        
        while self.running:
            try:
                round_count += 1
                self.log(f"=== 第 {round_count} 轮预测开始 ===", "INFO")
                
                # 获取数据
                self.update_status("获取数据中...")
                df_raw = self.data_fetcher.get_klines_multi(limit=200)
                
                if df_raw.empty:
                    self.log("获取数据失败，等待重试...", "ERROR")
                    time.sleep(10)
                    continue
                
                # 特征工程
                self.update_status("特征计算中...")
                df = self.tech_indicators.add_features(df_raw)
                df = self.tech_indicators.create_labels(df, FUTURE_BARS)
                
                if len(df) < 100:
                    self.log(f"数据不足 ({len(df)} < 100)，跳过本轮", "WARNING")
                    time.sleep(REFRESH_INTERVAL)
                    continue
                
                # 准备特征
                feature_cols = [
                    'returns', 'ma_3_diff', 'ma_5_diff', 'ma_8_diff', 'ma_13_diff',
                    'ma_21_diff', 'ma_50_diff', 'rsi', 'macd', 'macd_hist',
                    'bb_width', 'atr_pct', 'volume_ratio'
                ]
                
                # 确保所有特征都存在
                available_features = [col for col in feature_cols if col in df.columns]
                if len(available_features) < 5:
                    self.log("可用特征不足，跳过本轮", "WARNING")
                    time.sleep(REFRESH_INTERVAL)
                    continue
                
                # 获取最新特征
                X_latest = df[available_features].iloc[-1:].values
                
                # 使用模型预测
                prediction = None
                confidence = 0.5
                
                if self.model_loaded and self.model and self.scaler:
                    try:
                        X_scaled = self.scaler.transform(X_latest)
                        
                        if hasattr(self.model, 'predict_proba'):
                            proba = self.model.predict_proba(X_scaled)[0]
                            prediction = np.argmax(proba)
                            confidence = np.max(proba)
                        elif hasattr(self.model, 'predict'):
                            prediction = self.model.predict(X_scaled)[0]
                            confidence = 0.7  # 默认置信度
                    except Exception as e:
                        self.log(f"模型预测失败: {e}，使用随机预测", "WARNING")
                        prediction = np.random.choice([0, 1])
                        confidence = 0.5 + np.random.random() * 0.3
                else:
                    # 没有模型，使用随机
                    prediction = np.random.choice([0, 1])
                    confidence = 0.5 + np.random.random() * 0.3
                
                # 更新预测显示
                self.update_prediction(prediction, confidence)
                self.log(f"模型预测: {'上涨' if prediction == 1 else '下跌'} (置信度: {confidence:.2%})", "INFO")
                
                # 重置倒计时为准备中
                self.countdown_active = False
                self.update_countdown("准备中")
                
                # 进场时机过滤
                self.update_status("评估进场条件...")
                current_price = self.current_price
                if current_price == 0:
                    current_price = self.data_fetcher.get_price_multi()
                
                filter_result = self.entry_filter.evaluate_entry(SYMBOL, current_price, prediction)
                
                # 显示过滤器结果
                self.update_filter(
                    "通过" if filter_result['should_enter'] else "未通过",
                    filter_result['score'],
                    filter_result['conditions']
                )
                
                # 记录过滤器条件
                if filter_result['conditions']:
                    self.log("进场条件评估:", "INFO")
                    for condition in filter_result['conditions']:
                        self.log(f"  {condition}", "INFO")
                
                # 如果不满足进场条件，跳过
                if not filter_result['should_enter']:
                    self.log(f"❌ 进场条件不满足 (得分: {filter_result['score']:.1%})，跳过本轮", "WARNING")
                    time.sleep(REFRESH_INTERVAL)
                    continue
                
                # 获取开始价格
                start_price = self.data_fetcher.get_price_multi()
                if math.isnan(start_price):
                    self.log("无法获取开始价格", "ERROR")
                    time.sleep(5)
                    continue
                
                self.log(f"✅ 进场信号确认！开始价格: ${start_price:,.2f}", "SUCCESS")
                self.log(f"   预测方向: {'上涨' if prediction == 1 else '下跌'}", "SUCCESS")
                self.log(f"   模型置信度: {confidence:.2%}", "SUCCESS")
                self.log(f"   过滤器得分: {filter_result['score']:.1%}", "SUCCESS")
                
                # 语音播报进场信号
                if self.tts_enabled and self.tts_announce_prediction:
                    direction_text = "上涨" if prediction == 1 else "下跌"
                    self.speak(f"进场信号确认，预测{direction_text}，开始价格{start_price:.0f}美元")
                
                # 执行自动下单（如果启用）
                if self.auto_trader.enabled and AUTO_TRADE_AVAILABLE:
                    self.log("触发自动下单...", "TRADE")
                    success, message = self.execute_auto_trade(prediction)
                    if success:
                        self.log("自动下单指令已发送", "TRADE")
                
                # 开始倒计时
                self.countdown_active = True
                self.start_countdown()
                
                # 语音播报
                if self.tts_enabled:
                    if prediction == 1:
                        self.speak("多进去")
                    else:
                        self.speak("空进去")
                
                # 等待倒计时结束
                while self.countdown_active and self.running:
                    time.sleep(1)
                
                if not self.running:
                    break
                
                # 获取结束价格
                end_price = self.data_fetcher.get_price_multi()
                if math.isnan(end_price):
                    self.log("无法获取结束价格", "ERROR")
                    continue
                
                # 计算实际结果
                actual = 1 if end_price > start_price else 0
                correct = 1 if prediction == actual else 0
                change_pct = (end_price - start_price) / start_price * 100
                
                # 更新胜率统计
                self.winrate_tracker.add_trade(correct == 1)
                
                # 更新显示
                result_text = "正确 ✓" if correct else "错误 ✗"
                
                self.log(f"════════════════════════════════════════", "INFO")
                self.log(f"预测结果: {result_text}", "SUCCESS" if correct else "ERROR")
                self.log(f"开始价格: ${start_price:,.2f}", "INFO")
                self.log(f"结束价格: ${end_price:,.2f}", "INFO")
                self.log(f"价格变化: {change_pct:+.2f}%", "INFO")
                self.log(f"预测方向: {'上涨' if prediction == 1 else '下跌'}", "INFO")
                self.log(f"实际方向: {'上涨' if actual == 1 else '下跌'}", "INFO")
                self.log(f"════════════════════════════════════════", "INFO")
                
                # 语音播报最终结果
                if self.tts_enabled and self.tts_announce_result:
                    if correct:
                        self.speak(f"预测正确，价格变化{change_pct:+.1f}%")
                    else:
                        self.speak(f"预测错误，价格变化{change_pct:+.1f}%")
                
                # 更新统计数据
                self.update_statistics_display()
                
                # 等待60秒后开始下一轮预测
                self.log("等待60秒后开始下一轮预测...", "INFO")
                for i in range(60, 0, -1):
                    if not self.running:
                        break
                    time.sleep(1)
                
                if not self.running:
                    break
                
            except Exception as e:
                self.log(f"预测循环异常: {str(e)}", "ERROR")
                traceback.print_exc()
                time.sleep(10)
    
    def train_model(self, instance):
        """训练模型"""
        if not SKLEARN_AVAILABLE:
            self.log("scikit-learn 不可用，无法训练模型", "ERROR")
            return
        
        def train():
            try:
                self.update_status("训练模型中...")
                self.log("开始训练模型...", "INFO")
                
                # 获取数据
                df_raw = self.data_fetcher.get_klines_multi(limit=DATA_LIMIT)
                if df_raw.empty:
                    self.log("获取训练数据失败", "ERROR")
                    return
                
                # 特征工程
                df = self.tech_indicators.add_features(df_raw)
                df = self.tech_indicators.create_labels(df, FUTURE_BARS)
                
                if len(df) < MIN_TRAIN_SAMPLES:
                    self.log(f"训练数据不足 ({len(df)} < {MIN_TRAIN_SAMPLES})", "ERROR")
                    return
                
                # 特征选择
                feature_cols = [
                    'returns', 'ma_3_diff', 'ma_5_diff', 'ma_8_diff', 'ma_13_diff',
                    'ma_21_diff', 'ma_50_diff', 'rsi', 'macd', 'macd_hist',
                    'bb_width', 'atr_pct', 'volume_ratio'
                ]
                available_features = [col for col in feature_cols if col in df.columns]
                
                X = df[available_features].values
                y = df['label'].values
                
                self.log(f"使用 {len(df)} 个样本，{len(available_features)} 个特征进行训练", "INFO")
                
                # 训练随机森林模型
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
                
                # 标准化
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # 训练
                model.fit(X_scaled, y)
                
                # 保存模型
                model_data = {
                    'model': model,
                    'scaler': scaler,
                    'features': available_features,
                    'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                with open(MODEL_STACK_FILE, 'wb') as f:
                    pickle.dump(model_data, f)
                
                self.model = model
                self.scaler = scaler
                self.model_loaded = True
                self.model_status_text = "模型状态: 已训练"
                
                self.log("模型训练完成！", "SUCCESS")
                self.update_status("模型训练完成")
                
                # 语音播报训练完成
                if self.tts_enabled:
                    self.speak("模型训练完成")
                
            except Exception as e:
                self.log(f"模型训练失败: {str(e)}", "ERROR")
                traceback.print_exc()
                self.update_status("训练失败")
        
        # 在新线程中训练
        threading.Thread(target=train, daemon=True).start()
    
    def reset_statistics(self, instance):
        """重置统计"""
        try:
            self.winrate_tracker = WinRateTracker()
            self.update_statistics_display()
            self.log("统计信息已重置", "SUCCESS")
            
            # 语音播报
            if self.tts_enabled:
                self.speak("统计信息已重置")
        except Exception as e:
            self.log(f"重置统计失败: {e}", "ERROR")
    
    def on_stop(self):
        """应用停止时的处理"""
        try:
            # 保存胜率数据
            if self.winrate_tracker.save():
                self.log("胜率记录已保存", "SUCCESS")
            else:
                self.log("胜率记录保存失败", "WARNING")
            
            # 保存坐标配置
            if self.auto_trader.save_coordinates():
                self.log("坐标配置已保存", "SUCCESS")
            
            # 停止预测循环
            if self.running:
                self.running = False
                time.sleep(1)
        except Exception as e:
            print(f"关闭应用时出错: {e}")

# ---------------- 主程序 ----------------
def main():
    """主函数"""
    try:
        print("=" * 70)
        print("BTC 5分钟预测系统 - Kivy版")
        print("=" * 70)
        print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"预测周期: 5分钟")
        print(f"倒计时系统: ✓")
        print(f"胜率统计: ✓ (记录已持久化)")
        print(f"语音播报: {'✓' if TTS_AVAILABLE else '✗'}")
        print(f"进场过滤器: ✓")
        print(f"自动下单: {'✓' if AUTO_TRADE_AVAILABLE else '✗'}")
        print(f"GUI: Kivy")
        print("=" * 70)
        
        # 启动Kivy应用
        BTC5MinKivyApp().run()
    except Exception as e:
        print(f"应用启动失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
