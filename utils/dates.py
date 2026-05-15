from datetime import date, timedelta
from typing import List

import pandas as pd


class AShareCalendar:
    def __init__(self):
        self._holidays: set = set()
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            import akshare as ak
            df = ak.tool_trade_date_hist_sina()
            self._holidays = set(pd.to_datetime(df["trade_date"]).dt.date)
        except Exception:
            self._holidays = set()
        self._loaded = True

    def is_trading_day(self, d: date) -> bool:
        self._ensure_loaded()
        if d.weekday() >= 5:
            return False
        return d not in self._holidays

    def trading_days(self, start: date, end: date) -> List[date]:
        self._ensure_loaded()
        days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days

    def next_trading_day(self, d: date) -> date:
        self._ensure_loaded()
        current = d + timedelta(days=1)
        while not self.is_trading_day(current):
            current += timedelta(days=1)
        return current

    def prev_trading_day(self, d: date) -> date:
        self._ensure_loaded()
        current = d - timedelta(days=1)
        while not self.is_trading_day(current):
            current -= timedelta(days=1)
        return current
