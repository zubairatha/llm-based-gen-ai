#!/usr/bin/env python3
"""
HTTP MCP Server for file analysis tools
Alternative to stdio server for web-based access
"""

import pandas as pd
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize MCP server with HTTP transport
mcp = FastMCP("file_analyzer_server")

# Base directory for data files
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Import all the same tools from main.py
from main import (
    list_data_files,
    summarize_csv_file, 
    summarize_parquet_file,
    analyze_csv_data,
    create_sample_data,
    get_data_schema
)

# Register all tools with the HTTP server
mcp.tool()(list_data_files)
mcp.tool()(summarize_csv_file)
mcp.tool()(summarize_parquet_file) 
mcp.tool()(analyze_csv_data)
mcp.tool()(create_sample_data)
mcp.resource("data://schema")(get_data_schema)

if __name__ == "__main__":
    # Create sample data if it doesn't exist
    sample_csv = DATA_DIR / "sample.csv"
    if not sample_csv.exists():
        sample_data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice Johnson', 'Bob Smith', 'Carol Lee', 'David Wu', 'Eva Brown'],
            'email': ['alice@example.com', 'bob@example.com', 'carol@example.com', 'david@example.com', 'eva@example.com'],
            'signup_date': ['2023-01-15', '2023-02-22', '2023-03-10', '2023-04-18', '2023-05-30']
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(sample_csv, index=False)
        df.to_parquet(DATA_DIR / "sample.parquet", index=False)
        print(f"Created sample data files in {DATA_DIR}")
    
    print("Starting HTTP MCP File Analyzer Server on http://localhost:8000")
    print("Available endpoints:")
    print("  - http://localhost:8000/docs - Interactive API documentation")
    print("  - http://localhost:8000/tools - List available tools")
    print("  - http://localhost:8000/health - Health check")
    
    # Run HTTP server
    mcp.run(transport="http", host="localhost", port=8000)