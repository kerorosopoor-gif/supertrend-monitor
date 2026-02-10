import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ccxt.async_support as ccxt_async
import time
import asyncio
from telegram import Bot
import json
import os
from datetime import datetime
import logging  # 新增：用於除錯系統

# ==================== 除錯系統設定 ====================
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_log.txt"),  # 記錄到檔案
        logging.StreamHandler()  # 也輸出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 輔助函式：記錄錯誤並在 Streamlit 中顯示
def log_and_show_error(exception, context=""):
    error_msg = f"錯誤發生在 {context}: {str(exception)}"
    logger.error(error_msg)
    st.error(error_msg)  # 在 Streamlit 介面顯示錯誤

# ==================== Telegram 資訊 ====================
BOT_TOKEN = '8422928305:AAFZd3Ogcmw5jj4K3ib4suNF0ey_uGlY_c4'
CHAT_ID = '502442494'
bot = Bot(token=BOT_TOKEN)

# ==================== 參數設定 ====================
ATR_PERIOD = 12
MULTIPLIER = 3.0
SOURCE = 'hl2'
SMA_LEN1 = 60
SMA_LEN2 = 100
SMA_LEN3 = 200
TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT',
    'SOL/USDT', 'TRX/USDT', 'DOGE/USDT', 'ADA/USDT',
    'BCH/USDT', 'LINK/USDT', 'ZEC/USDT', 'XLM/USDT',
    'XMR/USDT', 'LTC/USDT', 'HBAR/USDT', 'AVAX/USDT',
    'USDC/USDT'
]

# ==================== 狀態管理 (Session State) ====================
def get_empty_signal_structure():
    return {'buy_0': None, 'buy_1': None, 'buy_2': None, 'buy_3': None,
            'sell_0': None, 'sell_1': None, 'sell_2': None, 'sell_3': None,
            'pending_buy': None, 'pending_sell': None}

if 'last_signals' not in st.session_state:
    st.session_state.last_signals = {symbol: {tf: get_empty_signal_structure() for tf in TIMEFRAMES} for symbol in SYMBOLS}
else:
    for symbol in SYMBOLS:
        if symbol not in st.session_state.last_signals:
            st.session_state.last_signals[symbol] = {}
        for tf in TIMEFRAMES:
            if tf not in st.session_state.last_signals[symbol]:
                st.session_state.last_signals[symbol][tf] = get_empty_signal_structure()

last_signals = st.session_state.last_signals

# 初始化通知紀錄 (Notification Log)
if 'notification_log' not in st.session_state:
    st.session_state.notification_log = []

if 'last_signal_emoji' not in st.session_state:
    json_file = 'last_signal_emoji.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            st.session_state.last_signal_emoji = json.load(f)
    else:
        st.session_state.last_signal_emoji = {symbol: {tf: None for tf in TIMEFRAMES} for symbol in SYMBOLS}

for symbol in SYMBOLS:
    if symbol not in st.session_state.last_signal_emoji:
        st.session_state.last_signal_emoji[symbol] = {}
    for tf in TIMEFRAMES:
        if tf not in st.session_state.last_signal_emoji[symbol]:
            st.session_state.last_signal_emoji[symbol][tf] = None

last_signal_emoji = st.session_state.last_signal_emoji

if 'new_signal_detected' not in st.session_state:
    st.session_state.new_signal_detected = False

# ==================== 核心函式 (Async 修復版) ====================

sem = asyncio.Semaphore(5)

async def fetch_ohlcv_async(exchange, symbol, timeframe):
    async with sem:
        retries = 2
        for i in range(retries):
            try:
                bars = await exchange.fetch_ohlcv(symbol, timeframe, limit=300)
                df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Hong_Kong')
                return df
            except Exception as e:
                if i < retries - 1:
                    print(f"⚠️ {symbol} {timeframe} 抓取失敗，1秒後重試... ({e})")
                    await asyncio.sleep(1)
                else:
                    log_and_show_error(e, f"fetch_ohlcv_async for {symbol} {timeframe}")
                    return None

async def fetch_all_data(symbols, timeframes):
    exchange = ccxt_async.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    tasks = []
    keys = []
    
    for symbol in symbols:
        for tf in timeframes:
            tasks.append(fetch_ohlcv_async(exchange, symbol, tf))
            keys.append((symbol, tf))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    await exchange.close()
    
    data_map = {}
    for (symbol, tf), result in zip(keys, results):
        if isinstance(result, pd.DataFrame):
            data_map[(symbol, tf)] = result
        else:
            data_map[(symbol, tf)] = None
    return data_map

def calculate_supertrend(df):
    try:
        if SOURCE == 'hl2':
            df['src'] = (df['high'] + df['low']) / 2
        df['tr'] = np.maximum.reduce([df['high'] - df['low'], np.abs(df['high'] - df['close'].shift()), np.abs(df['low'] - df['close'].shift())])
        df['atr'] = df['tr'].ewm(alpha=1/ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
        df['up'] = df['src'] - (MULTIPLIER * df['atr'])
        df['dn'] = df['src'] + (MULTIPLIER * df['atr'])
        df['up1'] = np.nan
        df['dn1'] = np.nan
        df['trend'] = 1

        for i in range(1, len(df)):
            up = df['up'].iloc[i]
            up1_prev = df['up1'].iloc[i-1] if not np.isnan(df['up1'].iloc[i-1]) else up
            df.loc[i, 'up1'] = max(up, up1_prev) if df['close'].iloc[i-1] > up1_prev else up

            dn = df['dn'].iloc[i]
            dn1_prev = df['dn1'].iloc[i-1] if not np.isnan(df['dn1'].iloc[i-1]) else dn
            df.loc[i, 'dn1'] = min(dn, dn1_prev) if df['close'].iloc[i-1] < dn1_prev else dn

            trend_prev = df['trend'].iloc[i-1]
            close_i = df['close'].iloc[i]
            if trend_prev == -1 and close_i > df['dn1'].iloc[i-1]:
                df.loc[i, 'trend'] = 1
            elif trend_prev == 1 and close_i < df['up1'].iloc[i-1]:
                df.loc[i, 'trend'] = -1
            else:
                df.loc[i, 'trend'] = trend_prev

        df['buy_signal'] = (df['trend'] == 1) & (df['trend'].shift() == -1)
        df['sell_signal'] = (df['trend'] == -1) & (df['trend'].shift() == 1)
        df.fillna(0, inplace=True)
        return df
    except Exception as e:
        log_and_show_error(e, "calculate_supertrend")
        return df  # 返回原始 df 以避免程式崩潰

def calculate_smmas(df):
    try:
        df['smma60'] = df['close'].ewm(alpha=1/SMA_LEN1, adjust=False).mean()
        df['smma100'] = df['close'].ewm(alpha=1/SMA_LEN2, adjust=False).mean()
        df['smma200'] = df['close'].ewm(alpha=1/SMA_LEN3, adjust=False).mean()
        return df
    except Exception as e:
        log_and_show_error(e, "calculate_smmas")
        return df

async def send_notification(message):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=message)
        print(f"[{time.strftime('%H:%M:%S')}] ✅ Telegram 已發送")
    except Exception as e:
        log_and_show_error(e, "send_notification")

# 輔助函式：新增紀錄到 Session Log
def add_to_log(symbol, timeframe, action, level, price, time_obj, emoji):
    try:
        log_entry = {
            'time': time_obj.strftime('%m/%d %H:%M'),
            'symbol': symbol,
            'timeframe': timeframe,
            'action': action,
            'level': level,
            'price': f"{price:.4f}",
            'emoji': emoji
        }
        st.session_state.notification_log.insert(0, log_entry)
        st.session_state.notification_log = st.session_state.notification_log[:50]
    except Exception as e:
        log_and_show_error(e, "add_to_log")

# ==================== 主分析流程 (Async) ====================
async def run_analysis_async():
    try:
        start_time = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 開始新一輪檢查 (Async 安全模式)...")
        
        data_map = await fetch_all_data(SYMBOLS, TIMEFRAMES)
        
        summary = []
        dfs = {symbol: {} for symbol in SYMBOLS}
        st.session_state.new_signal_detected = False

        for symbol in SYMBOLS:
            symbol_summary = {'幣種': symbol}
            latest_close_disp = None
            latest_time_disp = None
            
            for timeframe in TIMEFRAMES:
                df = data_map.get((symbol, timeframe))
                
                if df is None or len(df) < 200:
                    symbol_summary[f'{timeframe} 信號'] = 'N/A'
                    continue
                
                df = calculate_supertrend(df)
                df = calculate_smmas(df)
                
                current_candle = df.iloc[-1]
                closed_candle = df.iloc[-2]

                if timeframe == '5m':
                    latest_close_disp = current_candle['close']
                    latest_time_disp = current_candle['timestamp']
                elif latest_close_disp is None and timeframe == '15m':
                    latest_close_disp = current_candle['close']
                    latest_time_disp = current_candle['timestamp']

                smma_list = [closed_candle['smma60'], closed_candle['smma100'], closed_candle['smma200']]
                buy_smma_count = sum(closed_candle['close'] > sma for sma in smma_list)
                sell_smma_count = sum(closed_candle['close'] < sma for sma in smma_list)
                
                buy_signal = closed_candle['buy_signal']
                sell_signal = closed_candle['sell_signal']
                
                emoji = ''
                signal_str = '無'
                
                # --- 買入邏輯 ---
                if buy_signal:
                    last_signals[symbol][timeframe]['pending_sell'] = None
                    count = buy_smma_count
                    if count == 3:
                        key, level, emoji = 'buy_3', '極強買入', '🟢🟢🟢'
                    elif count == 2:
                        key, level, emoji = 'buy_2', '很強買入', '🟢🟢'
                    elif count == 1:
                        key, level, emoji = 'buy_1', '強買入', '🟢'
                    else:
                        key, level, emoji = 'buy_0', '留意買入', '🟡'
                    
                    signal_str = emoji
                    last_signal_emoji[symbol][timeframe] = emoji
                    
                    if last_signals[symbol][timeframe][key] is None or closed_candle['timestamp'] > last_signals[symbol][timeframe][key]:
                        msg = f"{emoji} {symbol} {timeframe} SuperTrend {level}\n價格：{closed_candle['close']:.4f}"
                        await send_notification(msg)
                        last_signals[symbol][timeframe][key] = closed_candle['timestamp']
                        
                        st.session_state.new_signal_detected = True
                        st.toast(f"{emoji} {symbol} {timeframe} 買入信號!", icon="🟢")
                        add_to_log(symbol, timeframe, 'buy', level, closed_candle['close'], closed_candle['timestamp'], emoji)
                        
                        if count < 3:
                            last_signals[symbol][timeframe]['pending_buy'] = closed_candle['timestamp']
                
                # --- 賣出邏輯 ---
                elif sell_signal:
                    last_signals[symbol][timeframe]['pending_buy'] = None
                    count = sell_smma_count
                    if count == 3:
                        key, level, emoji = 'sell_3', '極強賣出', '🔴🔴🔴'
                    elif count == 2:
                        key, level, emoji = 'sell_2', '很強賣出', '🔴🔴'
                    elif count == 1:
                        key, level, emoji = 'sell_1', '強賣出', '🔴'
                    else:
                        key, level, emoji = 'sell_0', '留意賣出', '🟡'
                    
                    signal_str = emoji
                    last_signal_emoji[symbol][timeframe] = emoji
                    
                    if last_signals[symbol][timeframe][key] is None or closed_candle['timestamp'] > last_signals[symbol][timeframe][key]:
                        msg = f"{emoji} {symbol} {timeframe} SuperTrend {level}\n價格：{closed_candle['close']:.4f}"
                        await send_notification(msg)
                        last_signals[symbol][timeframe][key] = closed_candle['timestamp']
                        
                        st.session_state.new_signal_detected = True
                        st.toast(f"{emoji} {symbol} {timeframe} 賣出信號!", icon="🔴")
                        add_to_log(symbol, timeframe, 'sell', level, closed_candle['close'], closed_candle['timestamp'], emoji)

                        if count < 3:
                            last_signals[symbol][timeframe]['pending_sell'] = closed_candle['timestamp']
                
                # --- 無新信號 (回溯或舊紀錄) ---
                else:
                    if last_signal_emoji[symbol][timeframe] is not None:
                        # 如果有上次 emoji，但沒有時間，我們需要回溯找時間（但為了簡單，假設 json 已存，但這裡我們不存時間，所以仍需回溯）
                        # 為了效率，我們在這裡也回溯找時間
                        last_sig_time = None
                        search_range = len(df)
                        for i in range(len(df) - 3, len(df) - search_range - 1, -1):
                            row = df.iloc[i]
                            if row['buy_signal'] or row['sell_signal']:
                                last_sig_time = row['timestamp'].strftime('%m/%d %H:%M')
                                break
                        if last_sig_time:
                            signal_str = last_signal_emoji[symbol][timeframe] + f' ({last_sig_time})'
                        else:
                            signal_str = last_signal_emoji[symbol][timeframe] + ' (無時間)'
                    else:
                        found_history = False
                        search_range = len(df)  # 修改：搜尋整個 df，而不是限 50 根
                        last_sig_time = None
                        for i in range(len(df) - 3, len(df) - search_range - 1, -1):  # 修改：擴大到整個範圍
                            try:
                                row = df.iloc[i]
                                if row['buy_signal']:
                                    past_smma_list = [row['smma60'], row['smma100'], row['smma200']]
                                    count = sum(row['close'] > sma for sma in past_smma_list)
                                    if count == 3: emoji = '🟢🟢🟢'
                                    elif count == 2: emoji = '🟢🟢'
                                    elif count == 1: emoji = '🟢'
                                    else: emoji = '🟡'
                                    last_signal_emoji[symbol][timeframe] = emoji
                                    last_sig_time = row['timestamp'].strftime('%m/%d %H:%M')
                                    signal_str = emoji + f' ({last_sig_time})'
                                    found_history = True
                                    break
                                elif row['sell_signal']:
                                    past_smma_list = [row['smma60'], row['smma100'], row['smma200']]
                                    count = sum(row['close'] < sma for sma in past_smma_list)
                                    if count == 3: emoji = '🔴🔴🔴'
                                    elif count == 2: emoji = '🔴🔴'
                                    elif count == 1: emoji = '🔴'
                                    else: emoji = '🟡'
                                    last_signal_emoji[symbol][timeframe] = emoji
                                    last_sig_time = row['timestamp'].strftime('%m/%d %H:%M')
                                    signal_str = emoji + f' ({last_sig_time})'
                                    found_history = True
                                    break
                            except Exception as e:
                                log_and_show_error(e, f"回溯信號時 for {symbol} {timeframe} at index {i}")
                        if not found_history:
                            signal_str = "無歷史信號"  # 修改：更明確的訊息

                # 計算上次時間 (現在為每個 timeframe 計算，但不加到 summary，只用在 signal_str)
                if timeframe == '5m':
                    last_sig_idx = -1
                    for i in range(len(df) - 2, -1, -1):
                        if df['buy_signal'].iloc[i] or df['sell_signal'].iloc[i]:
                            last_sig_idx = i
                            break
                    if last_sig_idx != -1:
                        last_ts = df['timestamp'].iloc[last_sig_idx]
                        delta = current_candle['timestamp'] - last_ts
                        mins = int(delta.total_seconds() / 60)
                        hrs = mins // 60
                        mins_rem = mins % 60
                        dur_str = f"{hrs}h {mins_rem}m" if hrs > 0 else f"{mins_rem}m"
                        time_str = last_ts.strftime('%m/%d %H:%M')
                    else:
                        time_str, dur_str = "無", "N/A"
                    symbol_summary['上次信號'] = time_str
                    symbol_summary['持續'] = dur_str

                symbol_summary[f'{timeframe} 信號'] = signal_str
                dfs[symbol][timeframe] = df
            
            if latest_close_disp is not None:
                symbol_summary['價格'] = f"{latest_close_disp:.4f}"
                symbol_summary['時間'] = latest_time_disp.strftime('%H:%M')
            summary.append(symbol_summary)

        with open('last_signal_emoji.json', 'w') as f:
            json.dump(last_signal_emoji, f)
            
        print(f"[{time.strftime('%H:%M:%S')}] ✅ 檢查完成 (耗時 {time.time()-start_time:.2f}秒)")
        return pd.DataFrame(summary), dfs
    except Exception as e:
        log_and_show_error(e, "run_analysis_async 主流程")
        return pd.DataFrame(), {}

# ==================== Streamlit 頁面 ====================
st.set_page_config(page_title="SuperTrend Monitor", layout="wide")
st_autorefresh(interval=300000, key="data_refresh")

st.title('🚀 SuperTrend Pro 監控')

# 執行分析
try:
    summary_df, dfs_dict = asyncio.run(run_analysis_async())
except Exception as e:
    log_and_show_error(e, "asyncio.run(run_analysis_async)")

# 1. 顯示主表格
st.dataframe(
    summary_df, 
    use_container_width=True, 
    hide_index=True,
    column_config={
        "價格": st.column_config.TextColumn("價格", help="最新 5m 價格"),
        "5m 信號": st.column_config.TextColumn("5m", help="5分鐘級別信號"),
    }
)

st.write(f"最後更新: {time.strftime('%Y-%m-%d %H:%M:%S')} | 模式: 異步安全版 | 信號基準: 收盤確認線")

# --- 佈局：詳細圖表區 ---
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    selected_symbol = st.selectbox('幣種', SYMBOLS)
with col_sel2:
    selected_timeframe = st.selectbox('時框', TIMEFRAMES)

# 2. 通知紀錄視窗 (Notification Log Window)
st.markdown("### 🔔 即時通知紀錄")

# 【修復】: 移除 f-string 內部的縮排，避免被 Markdown 誤判為 Code Block
log_html = """
<div style="height: 200px; overflow-y: auto; background-color: #0e1117; border: 1px solid #303030; border-radius: 5px; padding: 10px; margin-bottom: 20px; font-family: monospace;">
"""

if not st.session_state.notification_log:
    log_html += "<div style='color: #888; text-align: center; padding-top: 20px;'>尚無新通知...</div>"
else:
    for log in st.session_state.notification_log:
        if log['action'] == 'buy':
            bg_color = "rgba(0, 255, 0, 0.1)"
            border_color = "#4CAF50"
            text_color = "#4CAF50"
        else:
            bg_color = "rgba(255, 0, 0, 0.1)"
            border_color = "#FF5252"
            text_color = "#FF5252"
            
        # 注意：這裡使用單行串接或靠左對齊，確保不產生 4 格以上的縮排
        log_html += f"""
<div style="background-color: {bg_color}; border-left: 3px solid {border_color}; margin-bottom: 5px; padding: 5px 10px; border-radius: 3px; font-size: 14px; display: flex; justify-content: space-between; align-items: center;">
    <div>
        <span style="color: #bbb; font-size: 0.9em;">[{log['time']}]</span> 
        <strong style="color: #eee; margin-left: 5px;">{log['symbol']}</strong> 
        <span style="color: #888; font-size: 0.9em;">({log['timeframe']})</span>: 
        <span style="color: {text_color}; font-weight: bold; margin-left: 5px;">{log['emoji']} {log['level']}</span>
    </div>
    <div style="color: #ddd; font-family: monospace;">
        @ {log['price']}
    </div>
</div>
"""

log_html += "</div>"
st.markdown(log_html, unsafe_allow_html=True)

# 3. Candlestick 圖表
if selected_symbol in dfs_dict and selected_timeframe in dfs_dict[selected_symbol]:
    df = dfs_dict[selected_symbol][selected_timeframe]
    if df is not None:
        try:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K線'))
            
            df['st_lower'] = np.nan
            df['st_upper'] = np.nan
            df.loc[df['trend'] == 1, 'st_lower'] = df['up1']
            df.loc[df['trend'] == -1, 'st_upper'] = df['dn1']
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['st_lower'], line=dict(color='lime'), name='多頭支撐'))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['st_upper'], line=dict(color='red'), name='空頭壓力'))
            
            buy_df = df[df['buy_signal'] == 1]
            sell_df = df[df['sell_signal'] == 1]
            fig.add_trace(go.Scatter(x=buy_df['timestamp'], y=buy_df['low'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='買入確認'))
            fig.add_trace(go.Scatter(x=sell_df['timestamp'], y=sell_df['high'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='賣出確認'))

            fig.update_layout(height=600, title=f"{selected_symbol} {selected_timeframe}", xaxis_rangeslider_visible=False, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            log_and_show_error(e, f"繪製圖表 for {selected_symbol} {selected_timeframe}")
    else:
        st.warning(f"⚠️ {selected_symbol} {selected_timeframe} 目前無法顯示 (資料 N/A)，可能是因為 Binance 請求過於頻繁，請稍待下一次刷新。")
else:
    st.warning(f"⚠️ {selected_symbol} {selected_timeframe} 資料暫時無法取得。")