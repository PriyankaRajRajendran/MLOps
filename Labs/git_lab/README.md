# MLOps Lab 4 - CI/CD Pipeline with GitHub Actions(git_lab1)

## Project Overview
This lab demonstrates MLOps best practices by implementing automated testing and CI/CD pipelines using GitHub Actions, Pytest, and Unittest frameworks.

## Project Structure
```
git_lab/
├── .github/
│   └── workflows/        # GitHub Actions CI/CD pipelines
├── src/                  # Source code modules
│   ├── calculator.py     # Mathematical operations
│   └── data_processor.py # Data processing utilities
├── test/                 # Test suites
│   ├── test_pytest.py    # Pytest test cases
│   └── test_unittest.py  # Unittest test cases
├── data/                 # Data directory (empty for now)
└── requirements.txt      # Python dependencies
```

## Features
- Automated testing with Pytest and Unittest
- GitHub Actions CI/CD pipeline
- Code coverage reporting
- Multi-version Python testing (3.8, 3.9)
- Modular code architecture
- Comprehensive error handling

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mlops.git
cd mlops/Labs/git_lab
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Tests Locally
```bash
# Run pytest
pytest test/test_pytest.py -v

# Run unittest
python -m unittest test.test_unittest -v

# Run with coverage
pytest test/ --cov=src --cov-report=term
```

## Modules Description

### Calculator Module
Mathematical operations module with the following functions:
- `add(x, y)`: Addition of two numbers
- `subtract(x, y)`: Subtraction operation
- `multiply(x, y)`: Multiplication operation
- `divide(x, y)`: Division with zero-check validation
- `power(x, n)`: Exponential calculation
- `sqrt_sum(x, y)`: Square root of sum with negative validation
- `percentage(value, total)`: Percentage calculation
- `compound_interest(principal, rate, time)`: Compound interest formula

### DataProcessor Class
Data processing utilities class with methods for:
- `get_mean()`: Calculate arithmetic mean
- `get_median()`: Calculate median value
- `get_std_dev()`: Calculate standard deviation
- `normalize()`: Min-max normalization to [0,1] range
- `remove_outliers()`: Z-score based outlier removal
- `get_summary_stats()`: Generate statistical summary

## CI/CD Pipeline

### GitHub Actions Workflows

#### 1. Pytest CI Pipeline (`pytest_action.yml`)
- Triggers on push to main/develop branches
- Tests on Python 3.8, 3.9, and 3.10
- Generates test artifacts and coverage reports
- Caches dependencies for faster builds

#### 2. Unittest CI Pipeline (`unittest_action.yml`)
- Triggers on push to main branch
- Runs unittest suite with XML reporting
- Uploads test results as artifacts

## Test Coverage Summary

| Module | Statements | Missed | Coverage |
|--------|------------|--------|----------|
| `src/__init__.py` | 0 | 0 | 100% |
| `src/calculator.py` | 23 | 0 | 100% |
| `src/data_processor.py` | 44 | 3 | 93% |
| **Total** | **67** | **3** | **96%** |

## Learning Outcomes

This lab demonstrates:
- Setting up Python virtual environments
- Creating modular, testable code
- Writing comprehensive test suites
- Implementing CI/CD with GitHub Actions
- Achieving high code coverage
- Following MLOps best practices

## Workflow Status

| Workflow | Status |
|----------|--------|
| Pytest CI | Passing |
| Unittest CI | Passing |
| Code Coverage | 96% |
| Total Tests | 44 |

## Technologies Used

- **Python 3.8+**: Core programming language
- **Pytest**: Modern testing framework
- **Unittest**: Python standard testing framework
- **GitHub Actions**: CI/CD automation
- **Coverage.py**: Code coverage measurement
- **Virtual Environment**: Dependency isolation


## Author
**Priyanka Raj Rajendran**
- Master's in Data Analytics Engineering
- Course: IE-7374 (MLOps)
---
