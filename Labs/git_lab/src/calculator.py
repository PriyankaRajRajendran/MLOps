"""
Calculator Module
-----------------
"""

import math


def add(x, y):
    """
    Add two numbers.
    
    Args:
        x (float): First number
        y (float): Second number
    
    Returns:
        float: Sum of x and y
    
    Examples:
        >>> add(2, 3)
        5
        >>> add(-1, 1)
        0
    """
    return x + y


def subtract(x, y):
    """
    Subtract y from x.
    
    Args:
        x (float): First number
        y (float): Second number to subtract
    
    Returns:
        float: Difference of x and y
    """
    return x - y


def multiply(x, y):
    """
    Multiply two numbers.
    
    Args:
        x (float): First number
        y (float): Second number
    
    Returns:
        float: Product of x and y
    """
    return x * y


def divide(x, y):
    """
    Divide x by y with zero-check.
    
    Args:
        x (float): Numerator
        y (float): Denominator
    
    Returns:
        float: Quotient of x and y
    
    Raises:
        ValueError: If y is zero
    """
    if y == 0:
        raise ValueError("Cannot divide by zero!")
    return x / y


def power(x, n):
    """
    Calculate x raised to power n.
    
    Args:
        x (float): Base number
        n (float): Exponent
    
    Returns:
        float: x raised to power n
    
    Examples:
        >>> power(2, 3)
        8
        >>> power(5, 0)
        1
    """
    return x ** n


def sqrt_sum(x, y):
    """
    Calculate square root of the sum of two numbers.
    
    Args:
        x (float): First number
        y (float): Second number
    
    Returns:
        float: Square root of (x + y)
    
    Raises:
        ValueError: If sum is negative
    """
    if x + y < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(x + y)


def percentage(value, total):
    """
    Calculate percentage.
    
    Args:
        value (float): Part value
        total (float): Total value
    
    Returns:
        float: Percentage (0-100)
    
    Raises:
        ValueError: If total is zero
    """
    if total == 0:
        raise ValueError("Total cannot be zero for percentage calculation")
    return (value / total) * 100


def compound_interest(principal, rate, time):
    """
    Calculate compound interest.
    
    Args:
        principal (float): Initial amount
        rate (float): Interest rate (as percentage)
        time (float): Time period
    
    Returns:
        float: Final amount after compound interest
    """
    return principal * (1 + rate/100) ** time