# -*- coding: utf-8 -*-
"""
OB + FVG Live Scalping Bot - For Coinex V1 API
"""

import pandas as pd
import requests
import numpy as np
from datetime import datetime
import time
import hashlib
import json

# -------------------------------------------------
# Settings & API Keys
# -------------------------------------------------
CONFIG = {
    "symbol": "BNBUSDT",
    "timeframe": "5min",
    "limit": 200,  # Fetch fewer candles for live analysis
    "risk_percent": 1.0,
    "rr_ratio": 1.5,
    "zone_tolerance": 0.003,
    "min_tests": 2,
    "ob_lookback": 10,
    "fvg_lookback": 5,
    "channel_lookback": 5,
    "channel_min_rising": 3,
    "require_fvg": False,
    # --- IMPORTANT: ADD YOUR API KEYS AND TRADE AMOUNT HERE ---
    "ACCESS_ID": "86AFAD7B71954311A6F5965D3B679DBD",  # Replace with your Coinex Access ID
    "SECRET_KEY": "6375C8048A1A304205ECA376A032722B73C545C8F1AB73E0",  # Replace with your Coinex Secret Key
    "trade_amount_usdt": 6,  # The amount in USDT for each trade
}

# This global variable will store the state of our open position.
position = None


# -------------------------------------------------
# 1. API Authentication & Order Placement
# -------------------------------------------------
def _sign_request(params):
    """Creates the required signature for authenticated API calls for Coinex V1 API."""
    # Add a timestamp (nonce) and sort parameters alphabetically
    params["tonce"] = int(time.time() * 1000)
    sorted_params = sorted(params.items())

    # Create the query string
    query_string = "&".join([f"{k}={v}" for k, v in sorted_params])

    # Append the secret key and create an MD5 hash
    full_string = f"{query_string}&secret_key={CONFIG['SECRET_KEY']}"
    signature = hashlib.md5(full_string.encode("utf-8")).hexdigest().upper()

    return signature


def place_limit_order(symbol, trade_type, amount, price):
    """Places a limit order on Coinex for Spot trading."""
    if (
        CONFIG["ACCESS_ID"] == "YOUR_ACCESS_ID"
        or CONFIG["SECRET_KEY"] == "YOUR_SECRET_KEY"
    ):
        print(
            "\nERROR: Please set your API keys in the CONFIG section before trading.\n"
        )
        return None

    url = "https://api.coinex.com/v1/order/limit"
    params = {
        "access_id": CONFIG["ACCESS_ID"],
        "market": symbol,
        "type": trade_type,  # 'buy' or 'sell'
        "amount": str(round(amount, 5)),  # Round amount to avoid precision issues
        "price": str(price),
    }

    signature = _sign_request(params)
    headers = {"Content-Type": "application/json", "authorization": signature}

    try:
        response = requests.post(url, headers=headers, json=params, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result.get("code") == 0:
            print(
                f"SUCCESS: Placed {trade_type} order for {amount:.5f} {symbol.replace('USDT','')} at {price}."
            )
            return result["data"]
        else:
            print(f"ERROR placing order: {result.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"API request failed when placing order: {e}")
        return None


# -------------------------------------------------
# 2. Data Fetching (No changes needed)
# -------------------------------------------------
def get_coinex_kline(symbol, timeframe, limit):
    """Fetches candlestick data from the Coinex exchange."""
    url = "https://api.coinex.com/v1/market/kline"
    params = {"market": symbol, "type": timeframe, "limit": limit}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json().get("data")
        if not data:
            print("No data received from API.")
            return pd.DataFrame()

        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "close", "high", "low", "volume", "amount"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(
            float, errors="ignore"
        )
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


# -------------------------------------------------
# 3. S/R Zones & Features (Unchanged from backtest)
# -------------------------------------------------
def build_sr_zones(df):
    df["swing_high"] = df["high"].rolling(11, center=True).max() == df["high"]
    df["swing_low"] = df["low"].rolling(11, center=True).min() == df["low"]
    zones = []
    for price in df[df["swing_high"]]["high"].unique():
        z = {
            "center": price,
            "low": price * (1 - CONFIG["zone_tolerance"]),
            "high": price * (1 + CONFIG["zone_tolerance"]),
            "type": "resistance",
        }
        if sum(df["high"] >= z["low"]) >= CONFIG["min_tests"]:
            zones.append(z)
    for price in df[df["swing_low"]]["low"].unique():
        z = {
            "center": price,
            "low": price * (1 - CONFIG["zone_tolerance"]),
            "high": price * (1 + CONFIG["zone_tolerance"]),
            "type": "support",
        }
        if sum(df["low"] <= z["high"]) >= CONFIG["min_tests"]:
            zones.append(z)
    unique_zones = []
    if not zones:
        return []
    for z in sorted(zones, key=lambda x: x["center"]):
        if (
            not unique_zones
            or abs(z["center"] - unique_zones[-1]["center"])
            > CONFIG["zone_tolerance"] * z["center"]
        ):
            unique_zones.append(z)
    return unique_zones[:15]


def is_near_zone(price, zone):
    return zone["low"] <= price <= zone["high"]


def detect_features(df):
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-8))
    df["fvg_bull"] = df["high"].shift(2) < df["low"]
    df["fvg_bear"] = df["low"].shift(2) > df["high"]
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["ob_bear"] = (
        (df["close"] < df["open"])
        & (df["upper_shadow"] / (df["body"] + 1e-8) >= 1.5)
        & (df["high"] > df["high"].shift(1))
    )
    df["ob_bull"] = (
        (df["close"] > df["open"])
        & (df["lower_shadow"] / (df["body"] + 1e-8) >= 1.5)
        & (df["low"] < df["low"].shift(1))
    )
    look, min_rise = CONFIG["channel_lookback"], CONFIG["channel_min_rising"]
    df["up_channel"] = (
        df["low"]
        .rolling(look)
        .apply(lambda x: sum(np.diff(x) > 0) >= min_rise, raw=True)
        .fillna(0)
        .astype(bool)
    )
    df["dn_channel"] = (
        df["high"]
        .rolling(look)
        .apply(lambda x: sum(np.diff(x) < 0) >= min_rise, raw=True)
        .fillna(0)
        .astype(bool)
    )
    return df


# -------------------------------------------------
# 4. Live Trading Logic
# -------------------------------------------------
def check_for_signals_and_trade(df):
    """Analyzes the latest data, manages positions, and places trades."""
    global position

    # --- Position Management ---
    # This is a simplified management logic. A real bot would need to check order status via API.
    if position:
        latest_candle = df.iloc[-2]  # Check against the last fully closed candle

        if position["type"] == "long":
            # Check for Take Profit
            if latest_candle["high"] >= position["tp"]:
                print(
                    f"TP HIT for LONG position. Attempting to close trade at {position['tp']}."
                )
                # --- UNCOMMENT TO GO LIVE ---
                # place_limit_order(CONFIG['symbol'], 'sell', position['size'], position['tp'])
                position = None
            # Check for Stop Loss
            elif latest_candle["low"] <= position["sl"]:
                print(
                    f"SL HIT for LONG position. Attempting to close trade at {position['sl']}."
                )
                # --- UNCOMMENT TO GO LIVE ---
                # place_limit_order(CONFIG['symbol'], 'sell', position['size'], position['sl'])
                position = None

        elif position["type"] == "short":
            # Check for Take Profit
            if latest_candle["low"] <= position["tp"]:
                print(
                    f"TP HIT for SHORT position. Attempting to close trade at {position['tp']}."
                )
                # --- UNCOMMENT TO GO LIVE ---
                # place_limit_order(CONFIG['symbol'], 'buy', position['size'], position['tp'])
                position = None
            # Check for Stop Loss
            elif latest_candle["high"] >= position["sl"]:
                print(
                    f"SL HIT for SHORT position. Attempting to close trade at {position['sl']}."
                )
                # --- UNCOMMENT TO GO LIVE ---
                # place_limit_order(CONFIG['symbol'], 'buy', position['size'], position['sl'])
                position = None

    # If a position is still open, do not look for new signals
    if position:
        print(
            f"Position remains open: {position['type']} at {position['entry']:.4f}. Current SL: {position['sl']:.4f}, TP: {position['tp']:.4f}"
        )
        return

    # --- New Signal Detection ---
    print("No open position. Checking for new trade signals...")
    df_with_features = detect_features(df)
    zones = build_sr_zones(df_with_features)
    if not zones:
        return

    support_zones = [z for z in zones if z["type"] == "support"]
    resistance_zones = [z for z in zones if z["type"] == "resistance"]

    # We only check the most recent, fully closed candle for a signal
    last_closed_candle_index = len(df_with_features) - 2
    ob_window = df_with_features.iloc[
        last_closed_candle_index - CONFIG["ob_lookback"] : last_closed_candle_index + 1
    ]

    # --- Sell Signal ---
    if not ob_window[ob_window["ob_bear"]].empty:
        ob_candle = ob_window[ob_window["ob_bear"]].iloc[-1]
        idx = ob_window[ob_window["ob_bear"]].index[-1]

        if resistance_zones:
            nearest_res = min(
                resistance_zones, key=lambda z: abs(z["center"] - ob_candle["high"])
            )
            if (
                is_near_zone(ob_candle["high"], nearest_res)
                and df_with_features.loc[
                    idx - CONFIG["channel_lookback"] : idx, "up_channel"
                ].any()
                and (
                    not CONFIG["require_fvg"]
                    or df_with_features.loc[
                        idx - CONFIG["fvg_lookback"] : idx, "fvg_bear"
                    ].any()
                )
            ):

                entry_price = ob_candle["close"]
                stop_loss = ob_candle["high"] * 1.002
                take_profit = (
                    entry_price - (stop_loss - entry_price) * CONFIG["rr_ratio"]
                )
                amount_to_trade = CONFIG["trade_amount_usdt"] / entry_price

                print(
                    f"SELL SIGNAL DETECTED at {entry_price:.4f}. SL: {stop_loss:.4f}, TP: {take_profit:.4f}"
                )

                # --- UNCOMMENT TO GO LIVE ---
                # order_result = place_limit_order(CONFIG['symbol'], 'sell', amount_to_trade, entry_price)
                # if order_result:
                #    print(f"Successfully placed SELL order. Order ID: {order_result['id']}")
                #    position = {'type': 'short', 'entry': entry_price, 'sl': stop_loss, 'tp': take_profit, 'size': amount_to_trade, 'id': order_result['id']}

    # --- Buy Signal ---
    if not position and not ob_window[ob_window["ob_bull"]].empty:
        ob_candle = ob_window[ob_window["ob_bull"]].iloc[-1]
        idx = ob_window[ob_window["ob_bull"]].index[-1]

        if support_zones:
            nearest_sup = min(
                support_zones, key=lambda z: abs(z["center"] - ob_candle["low"])
            )
            if (
                is_near_zone(ob_candle["low"], nearest_sup)
                and df_with_features.loc[
                    idx - CONFIG["channel_lookback"] : idx, "dn_channel"
                ].any()
                and (
                    not CONFIG["require_fvg"]
                    or df_with_features.loc[
                        idx - CONFIG["fvg_lookback"] : idx, "fvg_bull"
                    ].any()
                )
            ):

                entry_price = ob_candle["close"]
                stop_loss = ob_candle["low"] * 0.998
                take_profit = (
                    entry_price + (entry_price - stop_loss) * CONFIG["rr_ratio"]
                )
                amount_to_trade = CONFIG["trade_amount_usdt"] / entry_price

                print(
                    f"BUY SIGNAL DETECTED at {entry_price:.4f}. SL: {stop_loss:.4f}, TP: {take_profit:.4f}"
                )

                # --- UNCOMMENT TO GO LIVE ---
                # order_result = place_limit_order(CONFIG['symbol'], 'buy', amount_to_trade, entry_price)
                # if order_result:
                #    print(f"Successfully placed BUY order. Order ID: {order_result['id']}")
                #    position = {'type': 'long', 'entry': entry_price, 'sl': stop_loss, 'tp': take_profit, 'size': amount_to_trade, 'id': order_result['id']}


# -------------------------------------------------
# 5. Main Loop
# -------------------------------------------------
if __name__ == "__main__":
    interval_seconds = int(CONFIG["timeframe"].replace("min", "")) * 60
    while True:
        try:
            print(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
            print(
                f"Fetching latest {CONFIG['limit']} candles for {CONFIG['symbol']}..."
            )

            # Fetch fresh market data
            market_data = get_coinex_kline(
                CONFIG["symbol"], CONFIG["timeframe"], CONFIG["limit"]
            )

            if not market_data.empty and len(market_data) > 50:
                # Check for signals and manage any open trades
                check_for_signals_and_trade(market_data)
            else:
                print("Could not fetch sufficient data. Waiting for next cycle.")

            print(
                f"Waiting for {interval_seconds // 60} minutes until the next check..."
            )
            time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\nBot stopped by user.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            print("Waiting 60 seconds before retrying...")
            time.sleep(60)
