"""
Data Processor Module
--------------------
"""

import statistics
from typing import List, Optional


class DataProcessor:
    """
    A class for processing numerical data with various statistical operations.
    
    Attributes:
        data (List[float]): The data to be processed
    """
    
    def __init__(self, data: List[float]):
        """
        Initialize DataProcessor with a list of numerical data.
        
        Args:
            data (List[float]): Input data list
            
        Raises:
            TypeError: If data is not a list
        """
        if not isinstance(data, list):
            raise TypeError("Data must be a list")
        self.data = data
    
    def get_mean(self) -> float:
        """
        Calculate the arithmetic mean of the data.
        
        Returns:
            float: Mean value
            
        Raises:
            ValueError: If data list is empty
        """
        if not self.data:
            raise ValueError("Cannot calculate mean of empty list")
        return statistics.mean(self.data)
    
    def get_median(self) -> float:
        """
        Calculate the median of the data.
        
        Returns:
            float: Median value
            
        Raises:
            ValueError: If data list is empty
        """
        if not self.data:
            raise ValueError("Cannot calculate median of empty list")
        return statistics.median(self.data)
    
    def get_std_dev(self) -> float:
        """
        Calculate the standard deviation of the data.
        
        Returns:
            float: Standard deviation
            
        Raises:
            ValueError: If data has less than 2 values
        """
        if len(self.data) < 2:
            raise ValueError("Need at least 2 values for standard deviation")
        return statistics.stdev(self.data)
    
    def normalize(self) -> List[float]:
        """
        Normalize data to 0-1 range using min-max normalization.
        
        Returns:
            List[float]: Normalized data
        """
        if not self.data:
            return []
        
        min_val = min(self.data)
        max_val = max(self.data)
        
        # Handle case where all values are the same
        if max_val == min_val:
            return [0.5] * len(self.data)
        
        return [(x - min_val) / (max_val - min_val) for x in self.data]
    
    def remove_outliers(self, z_threshold: float = 2.0) -> List[float]:
        """
        Remove outliers using z-score method.
        
        Args:
            z_threshold (float): Z-score threshold for outlier detection
            
        Returns:
            List[float]: Data with outliers removed
        """
        if len(self.data) < 3:
            return self.data.copy()
        
        mean = self.get_mean()
        std_dev = self.get_std_dev()
        
        if std_dev == 0:
            return self.data.copy()
        
        filtered_data = []
        for x in self.data:
            z_score = abs((x - mean) / std_dev)
            if z_score <= z_threshold:
                filtered_data.append(x)
        
        return filtered_data
    
    def get_summary_stats(self) -> dict:
        """
        Get summary statistics of the data.
        
        Returns:
            dict: Dictionary containing mean, median, min, max, and count
        """
        if not self.data:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "min": None,
                "max": None
            }
        
        return {
            "count": len(self.data),
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": min(self.data),
            "max": max(self.data)
        }