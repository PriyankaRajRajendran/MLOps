# Logging Lab - Custom JSON Logger

## Overview
This lab demonstrates advanced logging practices in Python with both console and JSON file logging.

## Features
•⁠  ⁠*Dual Handler System*: Console (human-readable) and File (JSON format)
•⁠  ⁠*Custom JSON Formatter*: Structured logging with timestamps, levels, and metadata
•⁠  ⁠*Exception Tracking*: Automatic exception capture with full stack traces
•⁠  ⁠*Multiple Log Levels*: DEBUG, INFO, WARNING, ERROR demonstration

## Structure
log_lab/
├── log_demo.py          # Main logging implementation
├── logs/
│   ├── .gitkeep
│   └── app.log         # JSON formatted logs (gitignored)
└── README.md

## Running the Lab
```bash
# Activate virtual environment
source log_lab2_env/bin/activate

# Run the logging demo
python3 log_demo.py

## Output
- **Console**: Color-coded, human-readable logs
- **File (logs/app.log)**: JSON structured logs for machine processing

## Author
Created by PriyankaRaj for IE 7374 MLOps Course
