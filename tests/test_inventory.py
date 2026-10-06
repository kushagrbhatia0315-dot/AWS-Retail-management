import numpy as np
from src.inventory import simulate_inventory_costs
def test_inventory_cost_math():
    y_true = np.array([20, 30])
    order_levels = np.array([10, 35])
    res = simulate_inventory_costs(y_true, order_levels, cu=12.00, co=1.33)
    assert np.isclose(res["Stockout Cost (₹)"], 120.00)
    assert np.isclose(res["Holding Cost (₹)"], 6.65)
    assert np.isclose(res["Total Cost (₹)"], 126.65)