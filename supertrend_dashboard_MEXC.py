import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ccxt.async_support as ccxt_async
import time
import asyncio
from telegram import Bot
import os
from datetime import datetime
import logging

# ==================== 除錯系統設定 ====================
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("debug_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def log_and_show_error(exception, context=""):
    error_msg = f"錯誤發生在 {context}: {str(exception)}"
    logger.error(error_msg)
    st.error(error_msg)

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
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'XRP/USDT:USDT', 'SOL/USDT:USDT',
    'DOGE/USDT:USDT', 'ADA/USDT:USDT', 'BCH/USDT:USDT', 'LINK/USDT:USDT',
    'LTC/USDT:USDT', 'HBAR/USDT:USDT', 'AVAX/USDT:USDT',
    'TRX/USDT:USDT', 'ZEC/USDT:USDT', 'XLM/USDT:USDT', 'XMR/USDT:USDT'
]

# ==================== 狀態管理 ====================
def get_empty_signal_structure():
    return {
        'buy_0': None, 'buy_1': None, 'buy_2': None, 'buy_3': None,
        'sell_0': None, 'sell_1': None, 'sell_2': None, 'sell_3': None,
        'pending_buy': None, 'pending_sell': None,
        'last_buy_ts': None, 'last_sell_ts': None
    }

if 'last_signals' not in st.session_state:
    st.session_state.last_signals = {symbol: {tf: get_empty_signal_structure() for tf in TIMEFRAMES} for symbol in SYMBOLS}
else:
    for symbol in SYMBOLS:
        if symbol not in st.session_state.last_signals:
            st.session_state.last_signals[symbol] = {}
        for tf in TIMEFRAMES:
            if tf not in st.session_state.last_signals[symbol]:
                st.session_state.last_signals[symbol][tf] = get_empty_signal_structure()
            else:
                for k in ['last_buy_ts', 'last_sell_ts']:
                    if k not in st.session_state.last_signals[symbol][tf]:
                        st.session_state.last_signals[symbol][tf][k] = None

last_signals = st.session_state.last_signals

if 'notification_log' not in st.session_state:
    st.session_state.notification_log = []

if 'last_signal_emoji' not in st.session_state:
    st.session_state.last_signal_emoji = {symbol: {tf: None for tf in TIMEFRAMES} for symbol in SYMBOLS}
else:
    for symbol in SYMBOLS:
        if symbol not in st.session_state.last_signal_emoji:
            st.session_state.last_signal_emoji[symbol] = {}
        for tf in TIMEFRAMES:
            if tf not in st.session_state.last_signal_emoji[symbol]:
                st.session_state.last_signal_emoji[symbol][tf] = None

last_signal_emoji = st.session_state.last_signal_emoji

if 'new_signal_detected' not in st.session_state:
    st.session_state.new_signal_detected = False

if 'custom_notifications' not in st.session_state:
    st.session_state.custom_notifications = []
if 'current_prices' not in st.session_state:
    st.session_state.current_prices = {}

# ==================== 核心函式 ====================
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
                    await asyncio.sleep(1)
                else:
                    log_and_show_error(e, f"fetch_ohlcv_async for {symbol} {timeframe}")
                    return None

async def fetch_all_data(symbols, timeframes):
    exchange = ccxt_async.mexc({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap', 'defaultSubType': 'linear'}
    })
    tasks = [fetch_ohlcv_async(exchange, symbol, tf) for symbol in symbols for tf in timeframes]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    await exchange.close()
    data_map = {}
    keys = [(s, t) for s in symbols for t in timeframes]
    for (symbol, tf), result in zip(keys, results):
        data_map[(symbol, tf)] = result if isinstance(result, pd.DataFrame) else None
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
        return df

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

async def check_custom_notifications(price_map):
    try:
        r_levels = [round(0.5 * i, 1) for i in range(2, 41)]
        for custom in st.session_state.custom_notifications[:]:
            if not custom.get('active', True): continue
            symbol = custom['symbol']
            if symbol not in price_map: continue
            curr_price = price_map[symbol]
            entry = custom['entry_price']
            sl = custom['stop_loss_price']
            risk = abs(entry - sl)
            if risk == 0: continue
            is_long = entry > sl
            favorable_dir = 1 if is_long else -1
            current_r = favorable_dir * (curr_price - entry) / risk
            if current_r > custom.get('max_reached_r', 0):
                custom['max_reached_r'] = round(current_r, 2)
            notified = custom.setdefault('notified_levels', set())
            if current_r <= -0.99 and 'SL' not in notified:
                msg = f"🛑 {symbol} 自訂警報 - 已觸及 Stop Loss！\n價格：{curr_price:.4f} (-1R)"
                await send_notification(msg)
                add_to_log(symbol, '自訂', 'sell', '止損', curr_price, datetime.now(), '🛑')
                notified.add('SL')
            for r in r_levels:
                if current_r >= r - 0.01 and r not in notified:
                    msg = f"🎯 {symbol} 自訂警報 - 達到 {r}R！\n價格：{curr_price:.4f}"
                    await send_notification(msg)
                    add_to_log(symbol, '自訂', 'buy', f'{r}R', curr_price, datetime.now(), '🎯')
                    notified.add(r)
    except Exception as e:
        log_and_show_error(e, "check_custom_notifications")

# ==================== 主分析流程（快取優化） ====================
@st.cache_data(ttl=300, show_spinner=False)
def run_analysis_cached():
    return asyncio.run(run_analysis_async())

async def run_analysis_async():
    start_time = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 開始新一輪檢查...")
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
            if buy_signal:
                last_signals[symbol][timeframe]['pending_sell'] = None
                count = buy_smma_count
                if count == 3: key, level, emoji = 'buy_3', '極強買入', '🟢🟢🟢'
                elif count == 2: key, level, emoji = 'buy_2', '很強買入', '🟢🟢'
                elif count == 1: key, level, emoji = 'buy_1', '強買入', '🟢'
                else: key, level, emoji = 'buy_0', '留意買入', '🟡'
                signal_str = emoji
                last_signal_emoji[symbol][timeframe] = emoji
                if (last_signals[symbol][timeframe]['last_buy_ts'] is None or closed_candle['timestamp'] > last_signals[symbol][timeframe]['last_buy_ts']):
                    msg = f"{emoji} {symbol} {timeframe} SuperTrend {level}\n價格：{closed_candle['close']:.4f}"
                    await send_notification(msg)
                    last_signals[symbol][timeframe]['last_buy_ts'] = closed_candle['timestamp']
                    last_signals[symbol][timeframe]['last_sell_ts'] = None
                    st.session_state.new_signal_detected = True
                    st.toast(f"{emoji} {symbol} {timeframe} 買入信號!", icon="🟢")
                    add_to_log(symbol, timeframe, 'buy', level, closed_candle['close'], closed_candle['timestamp'], emoji)
                    if count < 3:
                        last_signals[symbol][timeframe]['pending_buy'] = closed_candle['timestamp']
            elif sell_signal:
                last_signals[symbol][timeframe]['pending_buy'] = None
                count = sell_smma_count
                if count == 3: key, level, emoji = 'sell_3', '極強賣出', '🔴🔴🔴'
                elif count == 2: key, level, emoji = 'sell_2', '很強賣出', '🔴🔴'
                elif count == 1: key, level, emoji = 'sell_1', '強賣出', '🔴'
                else: key, level, emoji = 'sell_0', '留意賣出', '🟡'
                signal_str = emoji
                last_signal_emoji[symbol][timeframe] = emoji
                if (last_signals[symbol][timeframe]['last_sell_ts'] is None or closed_candle['timestamp'] > last_signals[symbol][timeframe]['last_sell_ts']):
                    msg = f"{emoji} {symbol} {timeframe} SuperTrend {level}\n價格：{closed_candle['close']:.4f}"
                    await send_notification(msg)
                    last_signals[symbol][timeframe]['last_sell_ts'] = closed_candle['timestamp']
                    last_signals[symbol][timeframe]['last_buy_ts'] = None
                    st.session_state.new_signal_detected = True
                    st.toast(f"{emoji} {symbol} {timeframe} 賣出信號!", icon="🔴")
                    add_to_log(symbol, timeframe, 'sell', level, closed_candle['close'], closed_candle['timestamp'], emoji)
                    if count < 3:
                        last_signals[symbol][timeframe]['pending_sell'] = closed_candle['timestamp']
            else:
                if last_signal_emoji[symbol][timeframe] is not None:
                    last_sig_time = None
                    for i in range(len(df) - 3, -1, -1):
                        row = df.iloc[i]
                        if row['buy_signal'] or row['sell_signal']:
                            last_sig_time = row['timestamp'].strftime('%m/%d %H:%M')
                            break
                    signal_str = last_signal_emoji[symbol][timeframe] + (f' ({last_sig_time})' if last_sig_time else ' (無時間)')
                else:
                    signal_str = "無歷史信號"
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

    price_map = {symbol: dfs[symbol]['5m'].iloc[-1]['close'] for symbol in SYMBOLS if symbol in dfs and '5m' in dfs[symbol] and len(dfs[symbol]['5m']) > 0}
    st.session_state.current_prices = price_map
    await check_custom_notifications(price_map)
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 檢查完成 (耗時 {time.time()-start_time:.2f}秒)")
    return pd.DataFrame(summary), dfs

# ==================== Streamlit 頁面 ====================
st.set_page_config(page_title="SuperTrend Monitor", layout="wide")
st_autorefresh(interval=300000, key="data_refresh")

st.title('🚀 SuperTrend Pro 監控 - MEXC 版')

summary_df, dfs_dict = run_analysis_cached()

st.dataframe(summary_df, use_container_width=True, hide_index=True,
             column_config={"價格": st.column_config.TextColumn("價格", help="最新 5m 價格")})

st.write(f"最後更新: {time.strftime('%Y-%m-%d %H:%M:%S')} | 模式: MEXC 異步版 | 信號基準: 收盤確認線")

col_sel1, col_sel2 = st.columns(2)
with col_sel1: selected_symbol = st.selectbox('幣種', SYMBOLS)
with col_sel2: selected_timeframe = st.selectbox('時框', TIMEFRAMES)

# ==================== 🔔 即時通知紀錄 ====================
st.markdown("### 🔔 即時通知紀錄")
log_html = """<div style="height: 200px; overflow-y: auto; background-color: #0e1117; border: 1px solid #303030; border-radius: 5px; padding: 10px; margin-bottom: 20px; font-family: monospace;">"""
if not st.session_state.notification_log:
    log_html += "<div style='color: #888; text-align: center; padding-top: 20px;'>尚無新通知...</div>"
else:
    for log in st.session_state.notification_log:
        bg_color = "rgba(0, 255, 0, 0.1)" if log['action'] == 'buy' else "rgba(255, 0, 0, 0.1)"
        border_color = "#4CAF50" if log['action'] == 'buy' else "#FF5252"
        text_color = "#4CAF50" if log['action'] == 'buy' else "#FF5252"
        log_html += f"""
<div style="background-color: {bg_color}; border-left: 3px solid {border_color}; margin-bottom: 5px; padding: 5px 10px; border-radius: 3px; font-size: 14px; display: flex; justify-content: space-between; align-items: center;">
    <div>
        <span style="color: #bbb; font-size: 0.9em;">[{log['time']}]</span> 
        <strong style="color: #eee; margin-left: 5px;">{log['symbol']}</strong> 
        <span style="color: #888; font-size: 0.9em;">({log['timeframe']})</span>: 
        <span style="color: {text_color}; font-weight: bold; margin-left: 5px;">{log['emoji']} {log['level']}</span>
    </div>
    <div style="color: #ddd; font-family: monospace;">@ {log['price']}</div>
</div>"""
log_html += "</div>"
st.markdown(log_html, unsafe_allow_html=True)

# ==================== 📊 自訂交易通知 ====================
st.markdown("### 📊 自訂交易通知")
with st.form("custom_form"):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: custom_symbol = st.selectbox('選擇幣種', SYMBOLS, key='custom_symbol_sel')
    with col2: entry_price = st.number_input('Entry Price', value=0.0, format="%.4f", key='entry_price')
    with col3: sl_price = st.number_input('Stop Loss Price', value=0.0, format="%.4f", key='sl_price')
    if st.form_submit_button('✅ 建立此自訂通知'):
        if entry_price > 0 and sl_price > 0 and abs(entry_price - sl_price) > 0.00001:
            new_custom = {
                'id': int(time.time()*1000), 'symbol': custom_symbol,
                'entry_price': entry_price, 'stop_loss_price': sl_price,
                'active': True, 'max_reached_r': 0.0, 'notified_levels': set()
            }
            st.session_state.custom_notifications.append(new_custom)
            st.success(f'已建立 {custom_symbol} 自訂通知')
            st.rerun()

st.subheader("進行中的自訂通知")
active = [c for c in st.session_state.custom_notifications if c.get('active', True)]
if not active:
    st.info("尚無進行中的自訂通知")
else:
    for custom in active:
        sym = custom['symbol']
        curr_p = st.session_state.current_prices.get(sym)
        curr_p_str = f"{curr_p:.4f}" if isinstance(curr_p, (int, float)) else "N/A"
        entry = custom['entry_price']
        sl = custom['stop_loss_price']
        risk = abs(entry - sl)
        is_long = entry > sl
        fav_dir = 1 if is_long else -1
        curr_r = fav_dir * (curr_p - entry) / risk if isinstance(curr_p, (int, float)) and risk > 0 else 0
        r_text = f"{curr_r:.2f}R"
        col_a, col_b, col_c = st.columns([3,3,1])
        with col_a: st.write(f"**{sym}** | Entry **{custom['entry_price']:.4f}** | SL **{custom['stop_loss_price']:.4f}**")
        with col_b: st.write(f"目前價格 **{curr_p_str}** | **目前 {r_text}**")
        with col_c:
            if st.button("已完成", key=f"done_{custom['id']}"):
                custom['active'] = False
                st.success("已完結此通知")
                st.rerun()
        st.divider()

# ==================== 收益計算模型 ====================
st.markdown("### 💰 收益計算模型")
risk_amount = st.number_input("每單風險金額 (1R) USD", min_value=1.0, value=10.0, step=0.5, format="%.1f")
r_levels = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
data = []
for r in r_levels:
    part1 = 0.5 * 1.5 * risk_amount
    trailing = 0.5 if r == 1.5 else r - 0.5
    part2 = 0.5 * trailing * risk_amount
    total = part1 + part2
    data.append({
        '觸及 R 倍數': f'{r}R',
        '第一部分 (50%)': f'獲利 ${part1:.1f}',
        '第二部分止損': f'{trailing}R (${part2:.1f})',
        '總利潤 (USD)': f'${total:.1f}'
    })
st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
st.caption("公式：Total Profit = (0.5 × 1.5R) + (0.5 × Trailing Stop R)")

# ==================== Candlestick 圖表 ====================
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
        st.warning(f"⚠️ {selected_symbol} {selected_timeframe} 目前無法顯示 (資料 N/A)")
else:
    st.warning(f"⚠️ {selected_symbol} {selected_timeframe} 資料暫時無法取得。")
