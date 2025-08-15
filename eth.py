# eth_bot_super_analyst.py - Bot ETH với 10 tính năng phân tích nâng cao

import requests
import time
import schedule
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import logging
import os
import sys
from collections import deque
import ta  # Thư viện phân tích kỹ thuật nâng cao

# Fix Unicode encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# ================== 🔐 CẤU HÌNH ==================
BOT_TOKEN = os.getenv("BOT_TOKEN", "7621331832:AAEAdFhGCHvqggE8ZwgpxoSPZZ729MDV-UA")
CHAT_ID = os.getenv("CHAT_ID", "5752214928")

# 🌐 URLs
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
TELEGRAM_PHOTO = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_ORDERBOOK = "https://fapi.binance.com/fapi/v1/depth"
BINANCE_WALLET = "https://api.binance.com/sapi/v1/accountSnapshot" # Ví dụ, cần API key

# 📊 Lưu trữ tín hiệu để backtest & thống kê
signal_history = deque(maxlen=100)
performance_stats = {
    'total_signals': 0,
    'win_signals': 0,
    'total_pnl': 0.0
}

# ==================== 🔧 CẤU HÌNH LOGGING ====================
def setup_logging():
    """Setup logging với UTF-8 encoding"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    file_handler = logging.FileHandler("ethbot_super.log", encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

def safe_log(message, level="info"):
    """Hàm log an toàn với Unicode"""
    try:
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
    except UnicodeEncodeError:
        clean_message = message.encode('ascii', 'ignore').decode('ascii')
        if level == "info":
            logger.info(f"[CLEANED] {clean_message}")
        elif level == "error":
            logger.error(f"[CLEANED] {clean_message}")
        elif level == "warning":
            logger.warning(f"[CLEANED] {clean_message}")
            # ==================== 🔧 HÀM HỖ TRỢ CƠ BẢN ====================

def send_telegram_message(text: str):
    """Gửi tin nhắn qua Telegram với retry"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(TELEGRAM_API, data={
                "chat_id": CHAT_ID,
                "text": text,
                "parse_mode": "Markdown"
            }, timeout=15)
            
            if response.status_code == 200:
                safe_log("✅ Tin nhắn đã gửi!")
                return True
            else:
                safe_log(f"❌ Gửi tin nhắn thất bại: {response.status_code} - {response.text}", "error")
                
        except Exception as e:
            safe_log(f"❌ Lỗi gửi tin nhắn (lần {attempt + 1}): {e}", "error")
            
        time.sleep(2)
    
    return False

def send_telegram_photo(photo_buffer: BytesIO, caption: str = ""):
    """Gửi ảnh qua Telegram"""
    try:
        response = requests.post(TELEGRAM_PHOTO, data={
            "chat_id": CHAT_ID,
            "caption": caption
        }, files={"photo": photo_buffer}, timeout=20)
        if response.status_code == 200:
            safe_log("✅ Ảnh đã gửi!")
            return True
        else:
            safe_log(f"❌ Gửi ảnh thất bại: {response.status_code} - {response.text}", "error")
            return False
    except Exception as e:
        safe_log(f"❌ Lỗi gửi ảnh: {e}", "error")
        return False

def test_telegram_connection():
    """Test kết nối Telegram"""
    safe_log("🔍 Đang test kết nối Telegram...")
    try:
        response = requests.post(TELEGRAM_API, data={
            "chat_id": CHAT_ID,
            "text": "🤖 Bot ETH Test - Kết nối thành công!"
        }, timeout=10)
        if response.status_code == 200:
            safe_log("✅ Kết nối Telegram OK!")
            return True
        else:
            safe_log(f"❌ Lỗi Telegram: {response.status_code} - {response.text}", "error")
            return False
    except Exception as e:
        safe_log(f"❌ Lỗi kết nối: {e}", "error")
        return False

def get_klines(symbol="ETHUSDT", interval="15m", limit=300):
    """Lấy dữ liệu nến từ Binance"""
    try:
        response = requests.get(BINANCE_KLINES, params={
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }, timeout=10)
        if response.status_code != 200:
            safe_log(f"❌ Lỗi Binance: {response.status_code}")
            return None
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["close"] = pd.to_numeric(df["close"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["open"] = pd.to_numeric(df["open"])
        df["volume"] = pd.to_numeric(df["volume"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
        return df
    except Exception as e:
        safe_log(f"❌ Lỗi kết nối Binance: {e}")
        return None
    # ==================== 🔧 PHÂN TÍCH KỸ THUẬT NÂNG CAO ====================

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal

def compute_bollinger_bands(series, window=20, num_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return ma, upper, lower

def compute_atr(high, low, close, window=14):
    """Tính Average True Range"""
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift())
    tr2 = abs(low - close.shift())
    tr = pd.DataFrame({'tr0': tr0, 'tr1': tr1, 'tr2': tr2}).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def detect_candlestick_patterns(df):
    """Phát hiện mô hình nến Nhật (Hammer, Engulfing, Doji, v.v.)"""
    signals = []
    
    # Hammer & Inverted Hammer
    body = abs(df['close'] - df['open'])
    lower_wick = np.where(df['close'] > df['open'], 
                         df['open'] - df['low'], 
                         df['close'] - df['low'])
    upper_wick = np.where(df['close'] > df['open'], 
                         df['high'] - df['close'], 
                         df['high'] - df['open'])
    
    # Hammer: body nhỏ, chân dưới dài
    hammer = (body < (df['high'] - df['low']) * 0.3) & (lower_wick > body * 2)
    if hammer.iloc[-1]:
        signals.append("🔨 Mô hình Hammer")
    
    # Shooting Star: body nhỏ, chân trên dài
    shooting_star = (body < (df['high'] - df['low']) * 0.3) & (upper_wick > body * 2)
    if shooting_star.iloc[-1]:
        signals.append("⭐ Mô hình Shooting Star")
    
    # Engulfing Bullish
    prev_body = df['close'].shift() - df['open'].shift()
    curr_body = df['close'] - df['open']
    engulfing_bull = (prev_body < 0) & (curr_body > 0) & (df['close'] > df['open'].shift()) & (df['open'] < df['close'].shift())
    if engulfing_bull.iloc[-1]:
        signals.append("🟢 Mô hình Bullish Engulfing")
    
    # Engulfing Bearish
    engulfing_bear = (prev_body > 0) & (curr_body < 0) & (df['close'] < df['open'].shift()) & (df['open'] > df['close'].shift())
    if engulfing_bear.iloc[-1]:
        signals.append("🔴 Mô hình Bearish Engulfing")
    
    # Doji
    doji = abs(df['close'] - df['open']) < (df['high'] - df['low']) * 0.1
    if doji.iloc[-1]:
        signals.append("⚪ Mô hình Doji")
    
    return signals

def detect_supply_demand_zones(df, lookback=50):
    """Phát hiện vùng Supply/Demand Zone"""
    zones = []
    df_recent = df.tail(lookback)
    
    # Demand Zone: Giá giảm mạnh rồi bật lên
    for i in range(2, len(df_recent)-2):
        if (df_recent.iloc[i-1]['close'] > df_recent.iloc[i]['close'] and 
            df_recent.iloc[i]['close'] < df_recent.iloc[i+1]['close'] and
            df_recent.iloc[i+1]['close'] < df_recent.iloc[i+2]['close']):
            zone_price = df_recent.iloc[i]['low']
            zones.append(f"🟢 Demand Zone tại ${zone_price:.2f}")
    
    # Supply Zone: Giá tăng mạnh rồi giảm xuống
    for i in range(2, len(df_recent)-2):
        if (df_recent.iloc[i-1]['close'] < df_recent.iloc[i]['close'] and 
            df_recent.iloc[i]['close'] > df_recent.iloc[i+1]['close'] and
            df_recent.iloc[i+1]['close'] > df_recent.iloc[i+2]['close']):
            zone_price = df_recent.iloc[i]['high']
            zones.append(f"🔴 Supply Zone tại ${zone_price:.2f}")
    
    return zones[:3]  # Trả về 3 vùng gần nhất

def market_phase_analysis(df):
    """Phân tích giai đoạn thị trường (Accumulation, Markup, Distribution, Markdown)"""
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Đơn giản hóa: dựa trên xu hướng SMA và RSI
    if current['sma20'] > current['sma50'] and current['rsi'] < 70:
        return "📈 Giai đoạn MARKUP (Tăng giá)"
    elif current['sma20'] < current['sma50'] and current['rsi'] > 30:
        return "📉 Giai đoạn MARKDOWN (Giảm giá)"
    elif current['sma20'] > current['sma50'] and current['rsi'] > 70:
        return "📊 Giai đoạn DISTRIBUTION (Phân phối)"
    elif current['sma20'] < current['sma50'] and current['rsi'] < 30:
        return "🏦 Giai đoạn ACCUMULATION (Tích lũy)"
    else:
        return "🔄 Giai đoạn SIDEWAYS (Đi ngang)"

def detect_fake_breakout(df):
    """Phát hiện tín hiệu giả (Fake Breakout)"""
    if len(df) < 3:
        return False
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Nếu volume tăng mạnh nhưng giá không đi tiếp
    volume_spike = current['volume'] > df['volume'].tail(10).mean() * 2
    price_weak = abs((current['close'] - prev['close']) / prev['close']) < 0.005  # < 0.5%
    
    if volume_spike and price_weak:
        return True
    return False
# ==================== 🔧 PHÂN TÍCH NÂNG CAO (BINANCE FUTURES) ====================

def get_funding_rate(symbol="ETHUSDT"):
    """Lấy Funding Rate từ Binance Futures"""
    try:
        response = requests.get(BINANCE_FUNDING, params={
            "symbol": symbol,
            "limit": 1
        }, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                rate = float(data[0]['fundingRate']) * 100
                return rate
        return 0
    except Exception as e:
        safe_log(f"❌ Lỗi lấy Funding Rate: {e}")
        return 0

def get_orderbook(symbol="ETHUSDT", limit=50):
    """Lấy Order Book từ Binance Futures"""
    try:
        response = requests.get(BINANCE_ORDERBOOK, params={
            "symbol": symbol,
            "limit": limit
        }, timeout=10)
        if response.status_code == 200:
            data = response.json()
            bids = pd.DataFrame(data['bids'], columns=['price', 'qty'])
            asks = pd.DataFrame(data['asks'], columns=['price', 'qty'])
            bids['price'] = pd.to_numeric(bids['price'])
            bids['qty'] = pd.to_numeric(bids['qty'])
            asks['price'] = pd.to_numeric(asks['price'])
            asks['qty'] = pd.to_numeric(asks['qty'])
            return bids, asks
        return None, None
    except Exception as e:
        safe_log(f"❌ Lỗi lấy Order Book: {e}")
        return None, None

def analyze_orderbook(bids, asks, current_price):
    """Phân tích Order Book để tìm bức tường bid/ask"""
    signals = []
    if bids is None or asks is None:
        return signals
    
    # Tìm bức tường bid (mua lớn)
    bid_wall = bids[bids['qty'] > bids['qty'].quantile(0.95)]
    if not bid_wall.empty:
        wall_price = bid_wall.iloc[0]['price']
        if abs(wall_price - current_price) / current_price < 0.01:  # Trong 1%
            signals.append(f"🟢 Bức tường BID tại ${wall_price:.2f}")
    
    # Tìm bức tường ask (bán lớn)
    ask_wall = asks[asks['qty'] > asks['qty'].quantile(0.95)]
    if not ask_wall.empty:
        wall_price = ask_wall.iloc[0]['price']
        if abs(wall_price - current_price) / current_price < 0.01:  # Trong 1%
            signals.append(f"🔴 Bức tường ASK tại ${wall_price:.2f}")
    
    return signals
# ==================== 🔧 SO SÁNH VÀ BÁO CÁO ====================

def compare_with_altcoins():
    """So sánh ETH với các altcoin khác"""
    altcoins = ["BTCUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT"]
    eth_change = 0
    avg_alt_change = 0
    
    try:
        df_eth = get_klines("ETHUSDT", "1h", 25)
        if df_eth is not None:
            eth_change = ((df_eth.iloc[-1]['close'] - df_eth.iloc[0]['close']) / df_eth.iloc[0]['close']) * 100
        
        changes = []
        for alt in altcoins:
            df_alt = get_klines(alt, "1h", 25)
            if df_alt is not None:
                change = ((df_alt.iloc[-1]['close'] - df_alt.iloc[0]['close']) / df_alt.iloc[0]['close']) * 100
                changes.append(change)
        
        if changes:
            avg_alt_change = np.mean(changes)
        
        diff = eth_change - avg_alt_change
        if diff > 1:
            return f"💎 ETH MẠNH hơn thị trường (+{diff:.2f}%)"
        elif diff < -1:
            return f"🔻 ETH YẾU hơn thị trường ({diff:.2f}%)"
        else:
            return f"📊 ETH/TRUNG TÍNH ({diff:.2f}%)"
    except Exception as e:
        safe_log(f"❌ Lỗi so sánh altcoin: {e}")
        return "📊 ETH/Không xác định"

def daily_performance_report():
    """Gửi báo cáo hiệu suất cuối ngày"""
    if len(signal_history) == 0:
        return
    
    total = len(signal_history)
    wins = sum(1 for s in signal_history if s.get('profit', False))
    winrate = (wins / total) * 100 if total > 0 else 0
    avg_pnl = np.mean([s.get('pnl', 0) for s in signal_history]) if signal_history else 0
    
    report = f"""📊 *BÁO CÁO HIỆU SUẤT NGÀY*
━━━━━━━━━━━━━━━━━━━━━━━━
📈 *Tổng tín hiệu:* {total}
✅ *Tín hiệu thắng:* {wins}
📊 *Tỷ lệ thắng:* {winrate:.1f}%
💰 *Lợi nhuận TB:* {avg_pnl:.2f}%

⏰ *Thời gian:* {datetime.now().strftime("%d/%m/%Y")}
"""
    
    send_telegram_message(report)
    # ==================== 🔧 HÀM PHÂN TÍCH CHÍNH ====================

def analyze_signals_advanced(df_15m):
    """Phân tích tín hiệu nâng cao với tất cả các tính năng"""
    if df_15m is None or len(df_15m) < 50:
        return 0, 0, []
    
    # Tính các chỉ báo
    df_15m['rsi'] = compute_rsi(df_15m['close'])
    macd_line, signal_line = compute_macd(df_15m['close'])
    df_15m['macd'] = macd_line
    df_15m['macd_signal'] = signal_line
    df_15m['ema12'] = df_15m['close'].ewm(span=12).mean()
    df_15m['ema26'] = df_15m['close'].ewm(span=26).mean()
    df_15m['sma50'] = df_15m['close'].rolling(50).mean()
    df_15m['bb_mid'], df_15m['bb_upper'], df_15m['bb_lower'] = compute_bollinger_bands(df_15m['close'])
    df_15m['atr'] = compute_atr(df_15m['high'], df_15m['low'], df_15m['close'])

    current = df_15m.iloc[-1]
    prev = df_15m.iloc[-2]

    buy_score = 0
    sell_score = 0
    signals = []

    # ============ PHÂN TÍCH RSI ============
    if current['rsi'] < 25:
        buy_score += 4
        signals.append("🔥 RSI cực Oversold (<25)")
    elif current['rsi'] < 35:
        buy_score += 3
        signals.append("🟢 RSI thấp (<35)")
    elif current['rsi'] > 75:
        sell_score += 4
        signals.append("🔥 RSI cực Overbought (>75)")
    elif current['rsi'] > 65:
        sell_score += 3
        signals.append("🔴 RSI cao (>65)")

    # ============ PHÂN TÍCH EMA CROSS ============
    ema_cross_up = (prev['ema12'] <= prev['ema26']) and (current['ema12'] > current['ema26'])
    ema_cross_down = (prev['ema12'] >= prev['ema26']) and (current['ema12'] < current['ema26'])

    if ema_cross_up:
        buy_score += 4
        signals.append("🚀 EMA12 vượt EMA26")
    elif ema_cross_down:
        sell_score += 4
        signals.append("💥 EMA12 xuống dưới EMA26")

    # ============ PHÂN TÍCH MACD ============
    macd_cross_up = (prev['macd'] <= prev['macd_signal']) and (current['macd'] > current['macd_signal'])
    macd_cross_down = (prev['macd'] >= prev['macd_signal']) and (current['macd'] < current['macd_signal'])

    if macd_cross_up:
        buy_score += 3
        signals.append("⬆️ MACD tăng")
    elif macd_cross_down:
        sell_score += 3
        signals.append("⬇️ MACD giảm")

    # ============ PHÂN TÍCH GIÁ TREND ============
    price_change = ((current['close'] - prev['close']) / prev['close']) * 100
    if abs(price_change) > 1.5:
        if price_change > 0:
            buy_score += 2
            signals.append(f"📈 Giá tăng nhanh (+{price_change:.1f}%)")
        else:
            sell_score += 2
            signals.append(f"📉 Giá giảm nhanh ({price_change:.1f}%)")

    # ============ PHÂN TÍCH BOLLINGER BANDS ============
    if current['close'] < current['bb_lower']:
        buy_score += 3
        signals.append("🔵 Giá chạm dải dưới Bollinger")
    elif current['close'] > current['bb_upper']:
        sell_score += 3
        signals.append("🔵 Giá chạm dải trên Bollinger")

    # ============ PHÂN TÍCH VOLUME ============
    avg_volume = df_15m['volume'].tail(15).mean()
    current_volume = current['volume']
    if current_volume > avg_volume * 2:
        if buy_score > sell_score:
            buy_score += 2
            signals.append("💪 Volume bùng nổ")
        else:
            sell_score += 2
            signals.append("💪 Volume bùng nổ")

    # ============ PHÂN TÍCH MÔ HÌNH NẾN NHẬT ============
    candle_signals = detect_candlestick_patterns(df_15m)
    signals.extend(candle_signals)
    buy_score += len([s for s in candle_signals if "🟢" in s or "🔨" in s]) * 2
    sell_score += len([s for s in candle_signals if "🔴" in s or "⭐" in s]) * 2

    # ============ PHÂN TÍCH VÙNG SUPPLY/DEMAND ============
    zones = detect_supply_demand_zones(df_15m)
    signals.extend(zones)
    
    # ============ PHÂN TÍCH GIAI ĐOẠN THỊ TRƯỜNG ============
    market_phase = market_phase_analysis(df_15m)
    signals.append(market_phase)
    
    # ============ PHÁT HIỆN TÍN HIỆU GIẢ ============
    if detect_fake_breakout(df_15m):
        signals.append("⚠️ Cảnh báo: Có thể là tín hiệu giả")
        # Giảm điểm để thận trọng
        buy_score = max(0, buy_score - 2)
        sell_score = max(0, sell_score - 2)

    return buy_score, sell_score, signals

def send_analysis_alert():
    """Hàm chính để phân tích và gửi tín hiệu nâng cao"""
    try:
        safe_log("🔄 Đang phân tích tín hiệu nâng cao...")
        
        # 1. Lấy dữ liệu 15m
        df_15m = get_klines("ETHUSDT", "15m", 200)
        if df_15m is None or len(df_15m) < 100:
            safe_log("❌ Không lấy được dữ liệu hợp lệ")
            return
        
        current_price = df_15m.iloc[-1]['close']
        
        # 2. Phân tích tín hiệu nâng cao
        buy_score, sell_score, all_signals = analyze_signals_advanced(df_15m)
        
        # 3. Phân tích Funding Rate
        funding_rate = get_funding_rate("ETHUSDT")
        if funding_rate > 0.1:
            all_signals.append(f"📈 Funding Rate cao ({funding_rate:.3f}%) - Cẩn thận đảo chiều")
            sell_score += 1
        elif funding_rate < -0.1:
            all_signals.append(f"📉 Funding Rate thấp ({funding_rate:.3f}%) - Cơ hội mua")
            buy_score += 1
        
        # 4. Phân tích Order Book
        bids, asks = get_orderbook("ETHUSDT", 50)
        ob_signals = analyze_orderbook(bids, asks, current_price)
        all_signals.extend(ob_signals)
        
        # 5. So sánh với altcoin
        altcoin_status = compare_with_altcoins()
        all_signals.append(altcoin_status)
        if "MẠNH" in altcoin_status:
            buy_score += 1
        elif "YẾU" in altcoin_status:
            sell_score += 1
        
        # 6. Xác định hành động
        if buy_score >= 8:
            action, action_emoji, strength = "MUA MẠNH", "🟢🔥", "RẤT MẠNH"
        elif buy_score >= 5:
            action, action_emoji, strength = "MUA", "🟢", "MẠNH"
        elif sell_score >= 8:
            action, action_emoji, strength = "BÁN MẠNH", "🔴🔥", "RẤT MẠNH"
        elif sell_score >= 5:
            action, action_emoji, strength = "BÁN", "🔴", "MẠNH"
        elif buy_score > sell_score:
            action, action_emoji, strength = "NGHIÊNG MUA", "🟡", "TRUNG BÌNH"
        elif sell_score > buy_score:
            action, action_emoji, strength = "NGHIÊNG BÁN", "🟡", "TRUNG BÌNH"
        else:
            action, action_emoji, strength = "HOLD", "⚪", "YẾU"
        
        # 7. Tạo danh sách tín hiệu (giới hạn 12)
        signal_text = "\n".join([f"• {s}" for s in all_signals[:12]])
        if len(all_signals) > 12:
            signal_text += f"\n• ... và {len(all_signals) - 12} tín hiệu khác"
        if not signal_text:
            signal_text = "• Không có tín hiệu đặc biệt"
        
        # 8. Tạo tin nhắn
        message = f"""{action_emoji} *TÍN HIỆU {action}* {action_emoji}
━━━━━━━━━━━━━━━━━━━━━━━━
🪙 *Tài sản:* ETH/USDT
💰 *Giá hiện tại:* ${current_price:.2f}
📊 *RSI:* {df_15m.iloc[-1]['rsi']:.1f}
📈 *Độ mạnh:* {strength}
⚖️ *Điểm MUA:* {buy_score} | *Điểm BÁN:* {sell_score}
💱 *Funding Rate:* {funding_rate:.3f}%

🎯 *CÁC TÍN HIỆU:*
{signal_text}

⏰ *Thời gian:* {datetime.now().strftime("%d/%m/%Y %H:%M")}

⚠️ *Lưu ý:* Đây chỉ là phân tích kỹ thuật, không phải lời khuyên đầu tư."""

        # 9. Gửi tin nhắn
        send_telegram_message(message)
        
        # 10. Lưu vào lịch sử để thống kê
        signal_record = {
            'time': datetime.now(),
            'action': action,
            'price': current_price,
            'buy_score': buy_score,
            'sell_score': sell_score
        }
        signal_history.append(signal_record)
        performance_stats['total_signals'] += 1
        
    except Exception as e:
        safe_log(f"❌ Lỗi tổng quát trong send_analysis_alert: {e}")

# ==================== 🔧 CHẠY BOT ====================

def main():
    print("🚀 Bot ETH Trading Super Analyst - 10 Tính năng nâng cao")
    print("📋 Chat ID:", CHAT_ID)
    print("=" * 60)

    # Test kết nối trước
    if not test_telegram_connection():
        print("❌ DỪNG: Không kết nối được Telegram!")
        print("\n🔧 Hãy kiểm tra:")
        print("1. Mở Telegram và tìm bot của bạn")
        print("2. Gửi /start cho bot")
        print("3. Kiểm tra Chat ID có đúng không")
        return

    # Chạy test ngay lập tức
    print("\n🧪 Chạy phân tích đầu tiên...")
    send_analysis_alert()

    # Lên lịch chạy mỗi 5 phút
    schedule.every(5).minutes.do(send_analysis_alert)
    
    # Gửi báo cáo cuối ngày
    schedule.every().day.at("23:55").do(daily_performance_report)

    print("\n⏰ Bot đã sẵn sàng! Sẽ phân tích mỗi 5 phút...")
    print("📊 Báo cáo hiệu suất sẽ được gửi lúc 23:55")
    print("🛑 Nhấn Ctrl+C để dừng bot")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n🛑 Bot đã dừng!")
        safe_log("Bot stopped by user")

if __name__ == "__main__":
    main()