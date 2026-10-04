import numpy as np
def simulate_inventory_costs(y_true, order_levels, cu=12.00, co=1.33):
    stockouts = np.maximum(y_true - order_levels, 0)
    overstock = np.maximum(order_levels - y_true, 0)
    stockout_cost = np.sum(stockouts * cu)
    holding_cost = np.sum(overstock * co)
    total_cost = stockout_cost + holding_cost
    service_level = np.mean(order_levels >= y_true)
    return {
        "Total Cost ($)": round(float(total_cost), 2),
        "Stockout Cost ($)": round(float(stockout_cost), 2),
        "Holding Cost ($)": round(float(holding_cost), 2),
        "Fill Rate / Service Level": round(float(service_level), 4)
    }