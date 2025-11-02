"""
Pytest Test Suite
-----------------
"""

import pytest
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from calculator import (add, subtract, multiply, divide, power, 
                        sqrt_sum, percentage, compound_interest)
from data_processor import DataProcessor


# ==================== Calculator Module Tests ====================

class TestCalculatorOperations:
    """Test suite for calculator operations"""
    
    def test_add(self):
        """Test addition operation"""
        assert add(2, 3) == 5
        assert add(-1, 1) == 0
        assert add(0, 0) == 0
        assert add(0.5, 0.5) == 1.0
        assert add(-5, -3) == -8
    
    def test_subtract(self):
        """Test subtraction operation"""
        assert subtract(5, 3) == 2
        assert subtract(0, 5) == -5
        assert subtract(-5, -3) == -2
        assert subtract(10.5, 0.5) == 10.0
    
    def test_multiply(self):
        """Test multiplication operation"""
        assert multiply(3, 4) == 12
        assert multiply(-2, 3) == -6
        assert multiply(0, 100) == 0
        assert multiply(0.5, 4) == 2.0
        assert multiply(-3, -3) == 9
    
    def test_divide(self):
        """Test division operation"""
        assert divide(10, 2) == 5
        assert divide(7, 2) == 3.5
        assert divide(-10, 2) == -5
        assert divide(0, 5) == 0
    
    def test_divide_by_zero(self):
        """Test division by zero raises ValueError"""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10, 0)
    
    @pytest.mark.parametrize("base,exp,expected", [
        (2, 3, 8),
        (5, 0, 1),
        (10, 2, 100),
        (-2, 2, 4),
        (2, -1, 0.5),
        (1, 100, 1)
    ])
    def test_power_parametrized(self, base, exp, expected):
        """Test power function with multiple test cases"""
        assert power(base, exp) == expected
    
    def test_sqrt_sum(self):
        """Test square root of sum"""
        assert sqrt_sum(3, 6) == 3.0
        assert sqrt_sum(0, 4) == 2.0
        assert sqrt_sum(5, 20) == 5.0
    
    def test_sqrt_sum_negative(self):
        """Test sqrt_sum with negative sum raises ValueError"""
        with pytest.raises(ValueError, match="Cannot calculate square root of negative number"):
            sqrt_sum(-5, 2)
    
    def test_percentage(self):
        """Test percentage calculation"""
        assert percentage(25, 100) == 25.0
        assert percentage(50, 200) == 25.0
        assert percentage(75, 150) == 50.0
    
    def test_percentage_zero_total(self):
        """Test percentage with zero total raises ValueError"""
        with pytest.raises(ValueError, match="Total cannot be zero"):
            percentage(10, 0)
    
    @pytest.mark.parametrize("principal,rate,time,expected", [
        (1000, 10, 1, 1100),
        (1000, 10, 2, 1210),
        (5000, 5, 3, 5788.125),
        (10000, 0, 5, 10000)
    ])
    def test_compound_interest(self, principal, rate, time, expected):
        """Test compound interest calculation"""
        result = compound_interest(principal, rate, time)
        assert result == pytest.approx(expected, rel=1e-9)


# ==================== DataProcessor Class Tests ====================

class TestDataProcessor:
    """Test suite for DataProcessor class"""
    
    def test_initialization(self):
        """Test DataProcessor initialization"""
        processor = DataProcessor([1, 2, 3, 4, 5])
        assert processor.data == [1, 2, 3, 4, 5]
    
    def test_invalid_initialization(self):
        """Test DataProcessor with invalid input"""
        with pytest.raises(TypeError, match="Data must be a list"):
            DataProcessor("not a list")
        
        with pytest.raises(TypeError):
            DataProcessor(123)
    
    def test_mean_calculation(self):
        """Test mean calculation"""
        processor = DataProcessor([1, 2, 3, 4, 5])
        assert processor.get_mean() == 3.0
        
        processor2 = DataProcessor([10, 20, 30])
        assert processor2.get_mean() == 20.0
        
        processor3 = DataProcessor([2.5, 2.5, 2.5, 2.5])
        assert processor3.get_mean() == 2.5
    
    def test_median_calculation(self):
        """Test median calculation"""
        processor1 = DataProcessor([1, 2, 3, 4, 5])
        assert processor1.get_median() == 3
        
        processor2 = DataProcessor([1, 2, 3, 4])
        assert processor2.get_median() == 2.5
        
        processor3 = DataProcessor([5, 1, 3, 2, 4])
        assert processor3.get_median() == 3
    
    def test_normalization(self):
        """Test data normalization"""
        processor = DataProcessor([1, 2, 3, 4, 5])
        normalized = processor.normalize()
        assert normalized[0] == 0.0
        assert normalized[-1] == 1.0
        assert len(normalized) == 5
        assert all(0 <= x <= 1 for x in normalized)
    
    def test_normalization_same_values(self):
        """Test normalization when all values are same"""
        processor = DataProcessor([5, 5, 5, 5])
        normalized = processor.normalize()
        assert all(x == 0.5 for x in normalized)
    
    def test_empty_data_operations(self):
        """Test operations on empty data"""
        processor = DataProcessor([])
        
        with pytest.raises(ValueError, match="Cannot calculate mean of empty list"):
            processor.get_mean()
        
        with pytest.raises(ValueError, match="Cannot calculate median of empty list"):
            processor.get_median()
        
        assert processor.normalize() == []
    
    def test_outlier_removal(self):
        """Test outlier removal"""
        processor = DataProcessor([1, 2, 3, 4, 5, 100])
        filtered = processor.remove_outliers(z_threshold=2)
        assert 100 not in filtered
        assert len(filtered) == 5
    
    def test_summary_statistics(self):
        """Test summary statistics"""
        processor = DataProcessor([1, 2, 3, 4, 5])
        stats = processor.get_summary_stats()
        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["median"] == 3
        assert stats["min"] == 1
        assert stats["max"] == 5
    
    def test_standard_deviation(self):
        """Test standard deviation calculation"""
        processor = DataProcessor([2, 4, 4, 4, 5, 5, 7, 9])
        std_dev = processor.get_std_dev()
        assert std_dev > 0
        
        # Test with less than 2 values
        processor2 = DataProcessor([5])
        with pytest.raises(ValueError):
            processor2.get_std_dev()