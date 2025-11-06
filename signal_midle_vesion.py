# -*- coding: utf-8 -*-
"""
CoinEx Live Trading Bot - FINAL CORRECTED VERSION

This version provides the definitive fix for the 'invalid argument' error by adding
the mandatory 'market_type' parameter to the POST request body for placing orders,
as required by the official CoinEx V2 documentation.

New Feature:
- Integrated Telegram logging to send real-time updates to a specified channel.
"""

import pandas as pd
import requests
import numpy as np
import logging
import json
import os
import time
import hashlib
import hmac
from datetime import datetime


# ==============================================================================
# 0. TELEGRAM LOGGING HANDLER (NEW)
# ==============================================================================
class TelegramLogHandler(logging.Handler):
    """
    A custom logging handler that sends log records to a Telegram channel.
    It is designed to be robust and not crash the main application if
    sending a message to Telegram fails.
    """

    def __init__(self, token, chat_id):
        super().__init__()
        self.token = token
        self.chat_id = chat_id
        self.url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def emit(self, record):
        """
        Formats the log record and sends it to the Telegram chat.
        """
        log_entry = self.format(record)

        # Add emojis for better visual distinction in the Telegram channel
        if record.levelno >= logging.CRITICAL:
            log_entry = f"💥 CRITICAL 💥\n{log_entry}"
        elif record.levelno >= logging.ERROR:
            log_entry = f"🚨 ERROR 🚨\n{log_entry}"
        elif record.levelno >= logging.WARNING:
            log_entry = f"⚠️ WARNING ⚠️\n{log_entry}"
        elif "successfully" in log_entry.lower():
            log_entry = f"✅ SUCCESS ✅\n{log_entry}"

        payload = {"chat_id": self.chat_id, "text": log_entry, "parse_mode": "Markdown"}
        try:
            # Use a timeout to avoid blocking the main thread for too long
            requests.post(self.url, data=payload, timeout=5)
        except Exception as e:
            # Print the error to the console/file log but do not raise it
            # This ensures the bot continues running even if Telegram is down.
            print(f"CRITICAL: Could not send log message to Telegram. Error: {e}")


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
TEST_MODE = True

ACCESS_ID = "3F631D0E06B946CF9248E8AC248A2A0C"
SECRET_KEY = "0C4201BF0883E02C3970AE6081F879A6CCE5DF07831BC98D"

CONFIG = {
    "symbol": "BTCUSDT",
    "timeframe": "15min",
    "limit": 200,
    "risk_amount_usdt": 0.02,
    "rr_ratio": 3,
    "zone_tolerance": 0.003,
    "min_tests": 3,
    "min_body_ratio": 0.25,
    "min_shadow_ratio": 2.0,
    "max_opposite_shadow_ratio": 0.7,
    "volume_ma_period": 20,
    "volume_multiplier": 1.5,
    "ema_period": 50,
    "atr_period": 14,
    "atr_multiplier": 1.5,
    "swing_lookback": 2,
    "zone_entry_buffer": 0.001,
}

# --- BOT_CONFIG is updated with Telegram settings ---
BOT_CONFIG = {
    "check_interval_seconds": 300,
    "state_file": "bot_state.json",
    "base_asset": "BTC",
    "quote_asset": "USDT",
    "min_order_value_usdt": 5.0,
    # --- NEW: Telegram Logger Configuration ---
    "telegram_enabled": True,  # Set to False to disable Telegram logging
    "telegram_log_level": "INFO",  # Minimum level to send: DEBUG, INFO, WARNING, ERROR, CRITICAL
    "telegram_bot_token": "8429441462:AAE-ZSwKDBK0AN6ek3XwNlN8JnDz9VhAw18",  # PASTE YOUR BOT TOKEN HERE
    "telegram_channel_id": "@tsb1999",  # PASTE YOUR CHANNEL ID HERE (e.g., "@mychannel" or "-100123...")
}

# --- Setup base logging to file and console ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# --- NEW: Attach the custom Telegram handler to the root logger ---
if BOT_CONFIG.get("telegram_enabled", False):
    tg_token = BOT_CONFIG.get("telegram_bot_token")
    tg_chat_id = BOT_CONFIG.get("telegram_channel_id")

    # Check if the token and chat_id are filled and not default values
    if tg_token and tg_chat_id and "YOUR_TELEGRAM" not in tg_token:
        # Create an instance of our custom handler
        telegram_handler = TelegramLogHandler(token=tg_token, chat_id=tg_chat_id)

        # Set the minimum log level that this handler will process
        log_level_str = BOT_CONFIG.get("telegram_log_level", "INFO").upper()
        telegram_handler.setLevel(getattr(logging, log_level_str, logging.INFO))

        # Set a simple format for Telegram messages (level and message)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        telegram_handler.setFormatter(formatter)

        # Add the handler to the root logger
        logging.getLogger().addHandler(telegram_handler)
        logging.info("Telegram logger has been successfully activated.")
    else:
        logging.warning(
            "Telegram logging is enabled, but token/chat_id is not properly configured. Skipping."
        )


# ==============================================================================
# 2. COINEX API CLIENT (POST Body Corrected)
# ==============================================================================
class CoinExAPI:
    BASE_URL_V1 = "https://api.coinex.com/v1"
    BASE_URL_V2 = "https://api.coinex.com"

    def __init__(self, access_id, secret_key):
        self.access_id = access_id.strip()
        self.secret_key = secret_key.strip()

    def _sign(self, sign_str):
        return (
            hmac.new(
                self.secret_key.encode("latin-1"),
                msg=sign_str.encode("latin-1"),
                digestmod=hashlib.sha256,
            )
            .hexdigest()
            .lower()
        )

    def _request(self, method, endpoint_path, params=None, body=None):
        timestamp = str(int(time.time() * 1000))
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-COINEX-KEY": self.access_id,
            "X-COINEX-TIMESTAMP": timestamp,
            "User-Agent": "MyTradingBot/Final",
        }

        url = self.BASE_URL_V2 + "/v2" + endpoint_path

        try:
            string_to_sign = ""
            if method.upper() == "GET":
                full_path_for_sign = "/v2" + endpoint_path
                if params:
                    query_string = "&".join(
                        [f"{k}={v}" for k, v in sorted(params.items())]
                    )
                    full_path_for_sign += "?" + query_string
                string_to_sign = f"GET{full_path_for_sign}{timestamp}"
                headers["X-COINEX-SIGN"] = self._sign(string_to_sign)
                response = requests.get(url, params=params, headers=headers, timeout=15)

            elif method.upper() == "POST":
                body_str = json.dumps(body, separators=(",", ":"))
                string_to_sign = f"POST/v2{endpoint_path}{body_str}{timestamp}"
                headers["X-COINEX-SIGN"] = self._sign(string_to_sign)
                response = requests.post(
                    url, data=body_str, headers=headers, timeout=15
                )
            else:
                raise NotImplementedError()

            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                logging.error(
                    f"API Error at {endpoint_path}: Code {data.get('code')} - {data.get('message')}"
                )
                return None
            return data.get("data")

        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP Request failed: {e}")
            return None

    def check_connection(self):
        return self._request(
            "GET",
            "/spot/pending-order",
            params={"market": "BTCUSDT", "market_type": "SPOT"},
        )

    def create_limit_order(self, symbol, side, amount, price):
        formatted_amount = f"{amount:.8f}"
        formatted_price = f"{price:.2f}"

        # === FINAL FIX: ADDING 'market_type' as required by documentation ===
        body = {
            "market": symbol,
            "market_type": "SPOT",  # This was the missing required parameter
            "side": side.lower(),
            "type": "limit",
            "amount": formatted_amount,
            "price": formatted_price,
            "client_id": f"bot-{int(time.time()*1000)}",  # Optional but good practice
        }
        return self._request("POST", "/spot/order", body=body)

    # ... Public endpoints are unchanged ...
    def get_market_price(self, symbol):
        try:
            response = requests.get(
                self.BASE_URL_V2 + "/v2/spot/ticker",
                params={"market": symbol},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("code") == 0:
                return float(data["data"]["ticker"]["last"])
        except Exception:
            return None

    def get_kline(self, symbol, timeframe, limit):
        try:
            response = requests.get(
                f"https://api.coinex.com/v1/market/kline",
                params={"market": symbol, "type": timeframe, "limit": limit},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("code") != 0:
                return pd.DataFrame()
            df = pd.DataFrame(
                data["data"],
                columns=[
                    "timestamp",
                    "open",
                    "close",
                    "high",
                    "low",
                    "volume",
                    "amount",
                ],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df[df.columns[1:]] = df[df.columns[1:]].astype(float)
            return df.sort_values("timestamp").reset_index(drop=True)
        except Exception:
            return pd.DataFrame()


# ==============================================================================
# 3. STATE & 4. ANALYSIS (Unchanged with bug fixes)
# ==============================================================================
def load_state():
    if os.path.exists(BOT_CONFIG["state_file"]):
        try:
            with open(BOT_CONFIG["state_file"], "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning("State file is corrupted.")
    return {"position_open": False}


def save_state(state):
    with open(BOT_CONFIG["state_file"], "w") as f:
        json.dump(state, f, indent=4)


def find_swing_points(df, lookback):
    df = df.copy()
    high, low = df["high"], df["low"]
    df["is_swing_high"] = False
    df["is_swing_low"] = False
    for i in range(lookback, len(df) - lookback):
        if all(high[i] > high[i - j] for j in range(1, lookback + 1)) and all(
            high[i] > high[i + j] for j in range(1, lookback + 1)
        ):
            df.loc[i, "is_swing_high"] = True
        if all(low[i] < low[i - j] for j in range(1, lookback + 1)) and all(
            low[i] < low[i + j] for j in range(1, lookback + 1)
        ):
            df.loc[i, "is_swing_low"] = True
    return df


def build_sr_zones_from_swings(df, tolerance, min_tests):
    df = find_swing_points(df, CONFIG["swing_lookback"])
    zones = []
    for price in df[df["is_swing_high"]]["high"].values:
        zone_low, zone_high = price * (1 - tolerance), price * (1 + tolerance)
        if (
            sum(1 for h in df["high"] if h >= zone_low)
            + sum(1 for l in df["low"] if l <= zone_high)
            >= min_tests
        ):
            zones.append(
                {
                    "center": price,
                    "type": "resistance",
                    "low": zone_low,
                    "high": zone_high,
                }
            )
    for price in df[df["is_swing_low"]]["low"].values:
        zone_low, zone_high = price * (1 - tolerance), price * (1 + tolerance)
        if (
            sum(1 for l in df["low"] if l <= zone_high)
            + sum(1 for h in df["high"] if h >= zone_low)
            >= min_tests
        ):
            zones.append(
                {"center": price, "type": "support", "low": zone_low, "high": zone_high}
            )
    unique_zones = []
    for zone in sorted(zones, key=lambda x: x["center"]):
        if not unique_zones or abs(zone["center"] - unique_zones[-1]["center"]) > (
            zone["high"] - zone["low"]
        ):
            unique_zones.append(zone)
    return unique_zones[:15]


def price_near_zone(price, zone, buffer=0.001):
    return zone["low"] * (1 - buffer) <= price <= zone["high"] * (1 + buffer)


def detect_strong_patterns(df):
    if len(df) < CONFIG["ema_period"]:
        return df
    df = df.copy().reset_index(drop=True)
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["prev_close"]), abs(df["low"] - df["prev_close"])
        ),
    )
    df["atr"] = df["tr"].rolling(CONFIG["atr_period"]).mean()
    df["ema_50"] = df["close"].ewm(span=CONFIG["ema_period"], adjust=False).mean()
    df["uptrend"] = df["close"] > df["ema_50"]
    df["downtrend"] = df["close"] < df["ema_50"]
    df["volume_ma"] = df["volume"].rolling(CONFIG["volume_ma_period"]).mean()
    df["high_volume"] = df["volume"] >= df["volume_ma"] * CONFIG["volume_multiplier"]
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["lower_shadow"] = np.where(
        df["close"] > df["open"], df["open"] - df["low"], df["close"] - df["low"]
    )
    df["upper_shadow"] = np.where(
        df["close"] > df["open"], df["high"] - df["close"], df["high"] - df["open"]
    )
    df["is_green"], df["is_red"] = df["close"] > df["open"], df["close"] < df["open"]

    for col in [
        "bullish_pinbar",
        "bearish_pinbar",
        "bullish_engulfing",
        "bearish_engulfing",
        "bullish_hammer",
        "bearish_shooting",
    ]:
        df[col] = False

    for i in range(1, len(df)):
        prev, curr = df.iloc[i - 1], df.iloc[i]
        if (
            prev["is_red"]
            and curr["is_green"]
            and curr["open"] < prev["close"]
            and curr["close"] > prev["open"]
        ):
            df.loc[i, "bullish_engulfing"] = True
        if (
            prev["is_green"]
            and curr["is_red"]
            and curr["open"] > prev["close"]
            and curr["close"] < prev["open"]
        ):
            df.loc[i, "bearish_engulfing"] = True

    for i in range(len(df)):
        c = df.iloc[i]
        if c["range"] == 0 or (c["body"] / c["range"]) > 0.8:
            continue
        lr = c["lower_shadow"] / c["body"] if c["body"] > 0 else float("inf")
        ur = c["upper_shadow"] / c["body"] if c["body"] > 0 else 0
        if c["is_green"] and c["high_volume"]:
            if (
                lr >= CONFIG["min_shadow_ratio"]
                and ur <= CONFIG["max_opposite_shadow_ratio"]
            ):
                df.loc[i, "bullish_pinbar"] = True
            if lr >= 2.0 and ur <= 0.7:
                df.loc[i, "bullish_hammer"] = True
        if c["is_red"] and c["high_volume"]:
            if (
                ur >= CONFIG["min_shadow_ratio"]
                and lr <= CONFIG["max_opposite_shadow_ratio"]
            ):
                df.loc[i, "bearish_pinbar"] = True
            if ur >= 2.0 and lr <= 0.7:
                df.loc[i, "bearish_shooting"] = True
    return df


# ==============================================================================
# 5. MAIN BOT LOGIC
# ==============================================================================
def run_bot():
    global TEST_MODE
    api = CoinExAPI(ACCESS_ID, SECRET_KEY)
    logging.info("Bot started. Verifying connection...")
    if api.check_connection() is None:
        logging.critical("Could not connect or authenticate. Exiting.")
        return
    logging.info("Successfully connected and authenticated with CoinEx.")
    while True:
        try:
            state = load_state()
            kline_df = api.get_kline(
                CONFIG["symbol"], CONFIG["timeframe"], CONFIG["limit"]
            )
            if kline_df.empty or len(kline_df) < 100:
                time.sleep(BOT_CONFIG["check_interval_seconds"])
                continue

            analyzed_df = detect_strong_patterns(kline_df)
            previous_candle = analyzed_df.iloc[-2]

            if state.get("position_open"):
                # ... Position management logic ...
                pass
            else:
                logging.info("Searching for new trade signals...")
                atr = previous_candle["atr"]
                if pd.isna(atr):
                    logging.info("ATR not available yet.")
                    continue

                signal_type = None
                if TEST_MODE:
                    logging.warning("!!! TEST MODE IS ACTIVE !!!")
                    signal_type = "long"
                else:
                    zones = build_sr_zones_from_swings(
                        analyzed_df, CONFIG["zone_tolerance"], CONFIG["min_tests"]
                    )
                    entry_price = previous_candle["close"]
                    if not any(
                        price_near_zone(entry_price, z, CONFIG["zone_entry_buffer"])
                        for z in zones
                    ):
                        logging.info(
                            f"Price {entry_price:.2f} is not near any S/R zone."
                        )
                        continue
                    if (
                        previous_candle[
                            ["bullish_pinbar", "bullish_engulfing", "bullish_hammer"]
                        ].any()
                    ) and previous_candle["uptrend"]:
                        signal_type = "long"
                    elif (
                        previous_candle[
                            ["bearish_pinbar", "bearish_engulfing", "bearish_shooting"]
                        ].any()
                    ) and previous_candle["downtrend"]:
                        signal_type = "short"

                if signal_type:
                    entry_price = previous_candle["close"]
                    sl_dist = CONFIG["atr_multiplier"] * atr
                    if sl_dist <= 0:
                        continue

                    position_size = CONFIG["risk_amount_usdt"] / sl_dist
                    order_value = position_size * entry_price
                    min_value = BOT_CONFIG["min_order_value_usdt"]

                    if order_value < min_value:
                        logging.warning(
                            f"Order value ({order_value:.2f} USDT) is below minimum ({min_value} USDT). Skipping."
                        )
                        if TEST_MODE:
                            TEST_MODE = False
                        continue

                    sl = (
                        entry_price - sl_dist
                        if signal_type == "long"
                        else entry_price + sl_dist
                    )
                    tp = (
                        entry_price + sl_dist * CONFIG["rr_ratio"]
                        if signal_type == "long"
                        else entry_price - sl_dist * CONFIG["rr_ratio"]
                    )

                    log_prefix = "!!! LIVE TEST !!!" if TEST_MODE else ""
                    logging.info(
                        f"{log_prefix} Placing {signal_type.upper()} order. Size: {position_size:.8f}, Price: {entry_price:.2f}"
                    )

                    order_result = api.create_limit_order(
                        CONFIG["symbol"],
                        "buy" if signal_type == "long" else "sell",
                        position_size,
                        entry_price,
                    )

                    if order_result:
                        success_msg = (
                            f"LIVE TEST Order placed successfully: {order_result}"
                        )
                        logging.info(f"--- {success_msg} ---")
                        save_state(
                            {
                                "position_open": True,
                                "type": signal_type,
                                "entry_price": entry_price,
                                "amount": position_size,
                                "stop_loss": sl,
                                "take_profit": tp,
                                "order_details": order_result,
                            }
                        )
                        if TEST_MODE:
                            logging.warning(
                                "!!! TEST SUCCESSFUL. NOW SET TEST_MODE to False !!!"
                            )
                            TEST_MODE = False
                    else:
                        error_msg = "LIVE TEST FAILED! Order was not placed."
                        logging.error(f"--- {error_msg} ---")
        except Exception as e:
            logging.error(f"An unexpected error in main loop: {e}", exc_info=True)
        finally:
            logging.info(
                f"Cycle finished. Waiting for {BOT_CONFIG['check_interval_seconds']} seconds..."
            )
            time.sleep(BOT_CONFIG["check_interval_seconds"])


# ==============================================================================
# 6. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    if ACCESS_ID == "YOUR_ACCESS_ID_HERE" or SECRET_KEY == "YOUR_SECRET_KEY_HERE":
        logging.critical(
            "CRITICAL: Please set your ACCESS_ID and SECRET_KEY in the script."
        )
    else:
        run_bot()
