import math

class TradeChargesCalculator:
    """
    Calculates all statutory and brokerage charges for an intraday equity trade.
    The calculations are based on standard rates for Indian brokers like Zerodha.
    """

    def __init__(self, quantity: int, buy_price: float, sell_price: float):
        if quantity <= 0 or buy_price <= 0 or sell_price <= 0:
            raise ValueError("Quantity and prices must be positive values.")
            
        self.quantity = quantity
        self.buy_price = buy_price
        self.sell_price = sell_price
        
        self.buy_turnover = self.quantity * self.buy_price
        self.sell_turnover = self.quantity * self.sell_price
        self.total_turnover = self.buy_turnover + self.sell_turnover

    def brokerage(self) -> float:
        """Calculates brokerage per leg (buy and sell). Capped at ₹20 per order."""
        buy_brokerage = min(0.03 / 100 * self.buy_turnover, 20)
        sell_brokerage = min(0.03 / 100 * self.sell_turnover, 20)
        return buy_brokerage + sell_brokerage
        
    def stt_ctt(self) -> float:
        """Calculates STT, which is charged only on the sell side for intraday equity."""
        # Rate is 0.025% on the sell-side turnover
        return (0.025 / 100) * self.sell_turnover
        
    def transaction_charges(self) -> float:
        """Calculates exchange transaction charges (e.g., NSE charges)."""
        # Rate is ~0.00325% on total turnover for NSE
        return (0.00325 / 100) * self.total_turnover
        
    def sebi_charges(self) -> float:
        """Calculates SEBI turnover fees."""
        # Rate is ₹10 per crore (0.0001%) on total turnover
        return (10 / 10**7) * self.total_turnover

    def stamp_charges(self) -> float:
        """Calculates stamp duty, charged only on the buy side."""
        # Rate is 0.003% on the buy-side turnover
        return (0.003 / 100) * self.buy_turnover
        
    def gst(self) -> float:
        """Calculates GST (18%) on brokerage and transaction charges."""
        # GST is not applicable on STT, Stamp Duty, or SEBI charges
        gst_base = self.brokerage() + self.transaction_charges()
        return (18 / 100) * gst_base

    def total_charges(self) -> float:
        """Calculates the sum of all charges for the round trip."""
        total = (
            self.brokerage() +
            self.stt_ctt() +
            self.transaction_charges() +
            self.sebi_charges() +
            self.stamp_charges() +
            self.gst()
        )
        return total

    def get_charges_breakdown(self) -> dict:
        """Returns a dictionary with a breakdown of all charges."""
        return {
            "brokerage": self.brokerage(),
            "stt": self.stt_ctt(),
            "transaction_charges": self.transaction_charges(),
            "sebi_charges": self.sebi_charges(),
            "stamp_charges": self.stamp_charges(),
            "gst": self.gst(),
            "total_charges": self.total_charges()
        }