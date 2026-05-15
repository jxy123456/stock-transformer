from abc import ABC, abstractmethod

from execution.order import Order, OrderResult


class LiveTrader(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def query_position(self, symbol: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def query_balance(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError
