#!/usr/bin/env python3
"""
DeLoRA-Shield Tier 3: Coordinate-Wise Trimmed Mean Aggregation
(Backward-compatibility module forwarding to aggregator.py)
"""

from aggregator import CoordinateWiseTrimmedMeanAggregator, CoordinateWiseTrimmedMean

__all__ = ["CoordinateWiseTrimmedMeanAggregator", "CoordinateWiseTrimmedMean"]
