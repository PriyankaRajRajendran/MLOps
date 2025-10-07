# Logging Lab - Custom JSON Logger

This lab demonstrates advanced logging practices in **Python** with both console and JSON file logging — essential for **MLOps pipelines** and **production systems**. It implements a **Dual Handler System** with two outputs: a **Console Handler** for human-readable logs during development and debugging, and a **File Handler** for JSON-formatted logs designed for machine processing and integration with log aggregation tools such as **ELK Stack**, **Splunk**, and **CloudWatch**. The custom JSON formatter converts standard Python log records into structured JSON format, enabling easy search, filtering, and analysis.

### Example JSON Output:
```json
{
  "timestamp": "2025-01-15T10:30:45.123456",
  "level": "INFO",
  "logger": "app",
  "message": "Application started",
  "module": "log_demo",
  "function": "main",
  "line": 42
}
```

Key Features include multiple log levels (DEBUG, INFO, WARNING, ERROR), automatic exception tracking with full stack traces, and structured logging that supports filtering by timestamp, level, function, or module. It is production-ready, separating human-readable console logs from machine-parseable JSON file logs for analytics and monitoring systems.

Project Structure:
```
log_lab
├── log_demo.py          # Main logging implementation
├── logs/
│   ├── .gitkeep        # Keeps logs/ directory in git
│   └── app.log         # JSON formatted logs (gitignored)
├── log_lab2_env/       # Virtual environment
└── README.md           # This file
```
Example Outputs:
```
Console Output (Human-Readable)
2025-01-15 10:30:45,123 - app - INFO - Application started
2025-01-15 10:30:45,124 - app - WARNING - Memory usage high
2025-01-15 10:30:45,125 - app - ERROR - Failed to connect
```
File Output (JSON Format)
```
{"timestamp": "2025-01-15T10:30:45.123456", "level": "INFO", "message": "Application started", "module": "log_demo", "function": "main", "line": 43}
{"timestamp": "2025-01-15T10:30:45.124789", "level": "WARNING", "message": "Memory usage high", "module": "log_demo", "function": "main", "line": 44}
```
Relevance to MLOps:
This logging setup is crucial for model monitoring, tracking predictions, performance metrics, and model versions. It aids in debugging data pipelines, ensuring traceability for compliance and audit trails, and enables alerting by allowing automated monitoring tools to parse machine-readable logs.

Logging Levels Explained:
```
DEBUG: Detailed diagnostic information (e.g., variable values, function calls)
INFO: General operational events such as “pipeline started” or “model saved”
WARNING: Potential issues like high memory usage or deprecated features
ERROR: Serious problems such as API failures or missing files
```

Author: Priyanka Raj

Course: IE 7374 – MLOps
