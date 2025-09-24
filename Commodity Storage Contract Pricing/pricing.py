import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

def price_storage_contract(injection_dates, withdrawal_dates, prices, injection_rate, withdrawal_rate, max_volume, storage_costs):
    """
    Prices a commodity storage contract using a simplified model.
    
    Parameters:
    - injection_dates: List of dates when injection is allowed.
    - withdrawal_dates: List of dates when withdrawal is allowed.
    - prices: List of commodity prices corresponding to the dates.
    - injection_rate: Maximum rate of injection (units per day).
    - withdrawal_rate: Maximum rate of withdrawal (units per day).
    - max_volume: Maximum storage volume (units).
    - storage_costs: Cost of storing the commodity (cost per unit per day).
    
    Returns:
    - total_value: The total value of the storage contract.
    """
    
    volume = 0    
    total_value = 0
    current_date = min(injection_dates[0], withdrawal_dates[0])
    end_date = max(injection_dates[-1], withdrawal_dates[-1])
    
    date_to_price = {date: price for date, price in zip(injection_dates + withdrawal_dates, prices)}
    
    while current_date <= end_date:
        if current_date in injection_dates and volume < max_volume:
            injectable_volume = min(injection_rate, max_volume - volume)
            volume += injectable_volume
            total_value -= injectable_volume * date_to_price[current_date]
        
        if current_date in withdrawal_dates and volume > 0:
            withdrawable_volume = min(withdrawal_rate, volume)
            volume -= withdrawable_volume
            total_value += withdrawable_volume * date_to_price[current_date]
        
        total_value -= volume * storage_costs
        current_date += timedelta(days=1)
    
    return total_value