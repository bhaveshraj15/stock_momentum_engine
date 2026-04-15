"""
filters/
--------
All filter classes. Import from here for clean usage:

    from filters import TrendFilter, High52wFilter, MinReturnFilter, UpDaysFilter
"""

from filters.base_filter       import BaseFilter
from filters.trend_filter      import TrendFilter
from filters.high52w_filter    import High52wFilter
from filters.min_return_filter import MinReturnFilter
from filters.up_days_filter    import UpDaysFilter
from filters.momentum_filter    import MomentumFilter
from filters.correlation_filter import CorrelationFilter
from filters.diversification_filter  import DiversificationFilter

__all__ = [
    "BaseFilter",
    "TrendFilter",
    "High52wFilter",
    "MinReturnFilter",
    "UpDaysFilter",
    "MomentumFilter",
    "CorrelationFilter",
    "DiversificationFilter",
]