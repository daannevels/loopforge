"""Sample library: cached slice index + nearest-neighbour search."""

from loopforge.library.index import (
    IndexReport,
    SliceTable,
    StaleIndexError,
    default_cache_path,
    index_is_current,
    load_slice_table,
    update_index,
)
from loopforge.library.search import FeatureSpace, NearestNeighbors

__all__ = [
    "FeatureSpace",
    "IndexReport",
    "NearestNeighbors",
    "SliceTable",
    "StaleIndexError",
    "index_is_current",
    "default_cache_path",
    "load_slice_table",
    "update_index",
]
