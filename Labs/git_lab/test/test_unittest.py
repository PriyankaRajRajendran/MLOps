"""
Unittest Test Suite
------------------
"""

import unittest
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from calculator import add, subtract, multiply, divide, power, sqrt_sum
from data_processor import DataProcessor


class TestCalculator(unittest.TestCase):
    """Unit tests for calculator module functions"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.x = 10
        self.y = 5
        self.zero = 0
    
    def test_addition(self):
        """Test add function with various inputs"""
        self.assertEqual(add(self.x, self.y), 15)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)
        self.assertEqual(add(-5, -5), -10)
        self.assertAlmostEqual(add(0.1, 0.2), 0.3, places=1)
    
    def test_subtraction(self):
        """Test subtract function with various inputs"""
        self.assertEqual(subtract(self.x, self.y), 5)
        self.assertEqual(subtract(0, 5), -5)
        self.assertEqual(subtract(-5, -3), -2)
        self.assertEqual(subtract(100, 50), 50)
    
    def test_multiplication(self):
        """Test multiply function with various inputs"""
        self.assertEqual(multiply(3, 4), 12)
        self.assertEqual(multiply(-2, 3), -6)
        self.assertEqual(multiply(0, 100), 0)
        self.assertEqual(multiply(self.x, self.zero), 0)
        self.assertEqual(multiply(-5, -5), 25)
    
    def test_division(self):
        """Test divide function with various inputs"""
        self.assertEqual(divide(self.x, self.y), 2.0)
        self.assertAlmostEqual(divide(7, 2), 3.5)
        self.assertEqual(divide(-10, 2), -5)
        self.assertEqual(divide(0, 5), 0)
    
    def test_division_by_zero(self):
        """Test that dividing by zero raises ValueError"""
        with self.assertRaises(ValueError) as context:
            divide(self.x, self.zero)
        self.assertIn("Cannot divide by zero", str(context.exception))
    
    def test_power_function(self):
        """Test power function with various inputs"""
        self.assertEqual(power(2, 3), 8)
        self.assertEqual(power(5, 0), 1)
        self.assertEqual(power(10, 2), 100)
        self.assertEqual(power(2, -1), 0.5)
    
    def test_sqrt_sum_function(self):
        """Test sqrt_sum function"""
        self.assertEqual(sqrt_sum(3, 6), 3.0)
        self.assertEqual(sqrt_sum(0, 4), 2.0)
        self.assertAlmostEqual(sqrt_sum(2, 2), 2.0)
    
    def tearDown(self):
        """Clean up after each test method"""
        pass


class TestDataProcessor(unittest.TestCase):
    """Unit tests for DataProcessor class"""
    
    def setUp(self):
        """Set up test data before each test"""
        self.sample_data = [1, 2, 3, 4, 5]
        self.processor = DataProcessor(self.sample_data)
        self.empty_processor = DataProcessor([])
    
    def test_initialization(self):
        """Test DataProcessor initialization"""
        self.assertEqual(self.processor.data, self.sample_data)
        self.assertEqual(self.empty_processor.data, [])
    
    def test_invalid_initialization(self):
        """Test DataProcessor with invalid input raises TypeError"""
        with self.assertRaises(TypeError):
            DataProcessor("invalid")
        
        with self.assertRaises(TypeError):
            DataProcessor(123)
        
        with self.assertRaises(TypeError):
            DataProcessor(None)
    
    def test_mean_calculation(self):
        """Test mean calculation"""
        self.assertEqual(self.processor.get_mean(), 3.0)
        
        processor2 = DataProcessor([10, 20, 30])
        self.assertEqual(processor2.get_mean(), 20.0)
    
    def test_median_calculation(self):
        """Test median calculation"""
        self.assertEqual(self.processor.get_median(), 3)
        
        processor2 = DataProcessor([1, 2, 3, 4])
        self.assertEqual(processor2.get_median(), 2.5)
    
    def test_normalization(self):
        """Test data normalization"""
        normalized = self.processor.normalize()
        self.assertEqual(normalized[0], 0.0)
        self.assertEqual(normalized[-1], 1.0)
        self.assertEqual(len(normalized), 5)
        
        # Check all values are between 0 and 1
        for value in normalized:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_empty_data_handling(self):
        """Test operations on empty data raise appropriate errors"""
        with self.assertRaises(ValueError):
            self.empty_processor.get_mean()
        
        with self.assertRaises(ValueError):
            self.empty_processor.get_median()
        
        self.assertEqual(self.empty_processor.normalize(), [])
    
    def test_outlier_removal(self):
        """Test outlier removal functionality"""
        processor = DataProcessor([1, 2, 3, 4, 5, 100])
        filtered = processor.remove_outliers(z_threshold=2)
        self.assertNotIn(100, filtered)
        self.assertEqual(len(filtered), 5)
    
    def test_summary_statistics(self):
        """Test summary statistics generation"""
        stats = self.processor.get_summary_stats()
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["mean"], 3.0)
        self.assertEqual(stats["median"], 3)
        self.assertEqual(stats["min"], 1)
        self.assertEqual(stats["max"], 5)
    
    def tearDown(self):
        """Clean up after each test"""
        self.processor = None
        self.empty_processor = None


def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCalculator))
    suite.addTest(unittest.makeSuite(TestDataProcessor))
    return suite


if __name__ == '__main__':
    # Run all tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())