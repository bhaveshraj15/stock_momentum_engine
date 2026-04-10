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

__all__ = [
    "BaseFilter",
    "TrendFilter",
    "High52wFilter",
    "MinReturnFilter",
    "UpDaysFilter",
]