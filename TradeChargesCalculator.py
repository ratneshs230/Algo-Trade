import math

# Centralized Configuration Dictionary for Zerodha Intraday Equity Charges
# All rates are based on Zerodha's official charge sheet for Equity Intraday.
# Source: https://zerodha.com/brokerage-calculator/
CHARGE_CONFIG = {
    "brokerage_rate": 0.0003,       # 0.03% or ₹20 whichever is lower (per order)
    "brokerage_flat_fee": 20.0,     # Flat fee per order
    "stt_rate": 0.00025,            # 0.025% on sell turnover (Equity Delivery is 0.1% on both)
    "transaction_charge_nse_equity": 0.0000325, # 0.00325% on turnover (NSE Equity)
    "gst_rate": 0.18,               # 18% on (Brokerage + Transaction Charges + SEBI Fees)
    "sebi_turnover_fee": 10 / 1_00_00_000, # ₹10 per Crore on turnover
    "stamp_duty_rate": 0.00003      # 0.003% on buy turnover (Equity Intraday)
}

class TradeChargesCalculator:
    """
    Calculates total charges for a complete round-trip (Buy + Sell) intraday equity trade
    based on Zerodha's official charge sheet.
    """

    def __init__(self, quantity: int, buy_price: float, sell_price: float):
        """
        Initializes the calculator with trade details.
        
        Args:
            quantity (int): Number of shares traded.
            buy_price (float): Price at which shares were bought.
            sell_price (float): Price at which shares were sold.
        """
        self.quantity = quantity
        self.buy_price = buy_price
        self.sell_price = sell_price
        
        self.buy_turnover = self.quantity * self.buy_price
        self.sell_turnover = self.quantity * self.sell_price
        self.total_turnover = self.buy_turnover + self.sell_turnover

    def _calculate_brokerage(self) -> float:
        """
        Calculates brokerage for both buy and sell legs.
        Brokerage is min(0.03% of turnover, ₹20) per order.
        """
        buy_brokerage = min(self.buy_turnover * CHARGE_CONFIG["brokerage_rate"], CHARGE_CONFIG["brokerage_flat_fee"])
        sell_brokerage = min(self.sell_turnover * CHARGE_CONFIG["brokerage_rate"], CHARGE_CONFIG["brokerage_flat_fee"])
        return buy_brokerage + sell_brokerage
        
    def _calculate_stt(self) -> float:
        """
        Calculates Securities Transaction Tax (STT).
        For intraday equity, STT is 0.025% on sell turnover.
        """
        return self.sell_turnover * CHARGE_CONFIG["stt_rate"]
        
    def _calculate_transaction_charges(self) -> float:
        """
        Calculates exchange transaction charges (NSE Equity).
        0.00325% on total turnover.
        """
        return self.total_turnover * CHARGE_CONFIG["transaction_charge_nse_equity"]
        
    def _calculate_sebi_fees(self) -> float:
        """
        Calculates SEBI turnover fees.
        ₹10 per Crore on total turnover.
        """
        return self.total_turnover * CHARGE_CONFIG["sebi_turnover_fee"]

    def _calculate_stamp_duty(self) -> float:
        """
        Calculates stamp duty.
        For intraday equity, 0.003% on buy turnover.
        """
        return self.buy_turnover * CHARGE_CONFIG["stamp_duty_rate"]
        
    def _calculate_gst(self, brokerage: float, transaction_charges: float, sebi_fees: float) -> float:
        """
        Calculates Goods and Services Tax (GST).
        18% on (Brokerage + Transaction Charges + SEBI Fees).
        """
        gst_base = brokerage + transaction_charges + sebi_fees
        return gst_base * CHARGE_CONFIG["gst_rate"]

    def calculate(self) -> float:
        """
        Orchestrates the calculation of all charges for a round-trip trade
        and returns the total cost.
        """
        brokerage = self._calculate_brokerage()
        stt = self._calculate_stt()
        transaction_charges = self._calculate_transaction_charges()
        sebi_fees = self._calculate_sebi_fees()
        stamp_duty = self._calculate_stamp_duty()
        
        gst = self._calculate_gst(brokerage, transaction_charges, sebi_fees)
        
        total_charges = (
            brokerage +
            stt +
            transaction_charges +
            sebi_fees +
            stamp_duty +
            gst
        )
        return total_charges
