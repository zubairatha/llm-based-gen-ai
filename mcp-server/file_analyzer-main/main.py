#!/usr/bin/env python3
"""
MCP Server for file analysis tools
Provides CSV and Parquet file reading capabilities to AI assistants
"""

import pandas as pd
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("file_analyzer_server")

# Base directory for data files
DATA_DIR = Path(__file__).resolve().parent / "data"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Utility functions
def read_csv_summary(filename: str) -> str:
    """Read a CSV file and return a summary."""
    file_path = DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file '{filename}' not found in data directory")
    
    try:
        df = pd.read_csv(file_path)
        return f"CSV file '{filename}' has {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns.tolist())}"
    except Exception as e:
        raise ValueError(f"Error reading CSV file '{filename}': {str(e)}")

def read_parquet_summary(filename: str) -> str:
    """Read a Parquet file and return a summary."""
    file_path = DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file '{filename}' not found in data directory")
    
    try:
        df = pd.read_parquet(file_path)
        return f"Parquet file '{filename}' has {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns.tolist())}"
    except Exception as e:
        raise ValueError(f"Error reading Parquet file '{filename}': {str(e)}")

# MCP Tools
@mcp.tool()
def list_data_files() -> str:
    """
    List all available data files in the data directory.
    Returns:
        A string listing all available data files.
    """
    try:
        csv_files = sorted([f.name for f in DATA_DIR.glob("*.csv")])
        parquet_files = sorted([f.name for f in DATA_DIR.glob("*.parquet")])
        
        all_files = csv_files + parquet_files
        
        if not all_files:
            return "No data files found in the data directory."
        
        file_list = ", ".join(all_files)
        return f"Available data files: {file_list}"
    except Exception as e:
        return f"Error listing data files: {str(e)}"

@mcp.tool()
def summarize_csv_file(filename: str) -> str:
    """
    Summarize a CSV file by reporting its number of rows and columns.
    Args:
        filename: Name of the CSV file in the /data directory (e.g., 'sample.csv')
    Returns:
        A string describing the file's dimensions and columns.
    """
    try:
        return read_csv_summary(filename)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def summarize_parquet_file(filename: str) -> str:
    """
    Summarize a Parquet file by reporting its number of rows and columns.
    Args:
        filename: Name of the Parquet file in the /data directory (e.g., 'sample.parquet')
    Returns:
        A string describing the file's dimensions and columns.
    """
    try:
        return read_parquet_summary(filename)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def analyze_csv_data(filename: str, operation: str = "describe") -> str:
    """
    Perform advanced analysis on a CSV file.
    Args:
        filename: Name of the CSV file (e.g., 'sample.csv')
        operation: Type of analysis ('describe', 'head', 'info', 'columns')
    Returns:
        A string with the analysis results.
    """
    file_path = DATA_DIR / filename
    if not file_path.exists():
        return f"Error: CSV file '{filename}' not found in data directory"
    
    try:
        df = pd.read_csv(file_path)
        
        if operation == "describe":
            # Statistical summary for numeric columns
            desc = df.describe()
            return f"Statistical summary for '{filename}':\n{desc.to_string()}"
        
        elif operation == "head":
            # First 5 rows
            head_data = df.head()
            return f"First 5 rows of '{filename}':\n{head_data.to_string(index=False)}"
        
        elif operation == "info":
            # Data types and non-null counts
            info_lines = []
            info_lines.append(f"Data types and non-null counts for '{filename}':")
            info_lines.append(f"Total rows: {len(df)}")
            info_lines.append(f"Total columns: {len(df.columns)}")
            info_lines.append("\nColumn details:")
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null = df[col].notna().sum()
                null_count = df[col].isna().sum()
                info_lines.append(f"  {col}: {dtype} (non-null: {non_null}, null: {null_count})")
            return "\n".join(info_lines)
        
        elif operation == "columns":
            # List all columns with their data types
            col_info = []
            col_info.append(f"Columns in '{filename}':")
            for col in df.columns:
                dtype = str(df[col].dtype)
                col_info.append(f"  - {col} ({dtype})")
            return "\n".join(col_info)
        
        else:
            return f"Error: Unknown operation '{operation}'. Supported operations: 'describe', 'head', 'info', 'columns'"
    
    except Exception as e:
        return f"Error analyzing CSV file '{filename}': {str(e)}"

@mcp.tool()
def create_sample_data(filename: str, rows: int = 10) -> str:
    """
    Create a new sample dataset.
    Args:
        filename: Name for the new file (e.g., 'new_data.csv')
        rows: Number of rows to generate
    Returns:
        A confirmation message.
    """
    if rows < 1:
        return "Error: Number of rows must be at least 1"
    
    if rows > 10000:
        return "Error: Number of rows cannot exceed 10000"
    
    try:
        import random
        from datetime import datetime, timedelta
        
        # Generate sample data
        names = ['Alice', 'Bob', 'Carol', 'David', 'Eva', 'Frank', 'Grace', 'Henry', 'Iris', 'Jack']
        domains = ['example.com', 'test.com', 'demo.com', 'sample.org']
        
        data = {
            'id': list(range(1, rows + 1)),
            'name': [random.choice(names) + ' ' + random.choice(['Johnson', 'Smith', 'Lee', 'Wu', 'Brown', 'Davis', 'Miller', 'Wilson']) 
                     for _ in range(rows)],
            'email': [f"user{i}@{random.choice(domains)}" for i in range(1, rows + 1)],
            'signup_date': [(datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d') 
                           for _ in range(rows)],
            'score': [random.randint(0, 100) for _ in range(rows)]
        }
        
        df = pd.DataFrame(data)
        file_path = DATA_DIR / filename
        
        # Save based on file extension
        if filename.endswith('.csv'):
            df.to_csv(file_path, index=False)
            return f"Successfully created CSV file '{filename}' with {rows} rows in the data directory."
        elif filename.endswith('.parquet'):
            df.to_parquet(file_path, index=False)
            return f"Successfully created Parquet file '{filename}' with {rows} rows in the data directory."
        else:
            # Default to CSV
            csv_filename = filename if filename.endswith('.csv') else filename + '.csv'
            csv_path = DATA_DIR / csv_filename
            df.to_csv(csv_path, index=False)
            return f"Successfully created CSV file '{csv_filename}' with {rows} rows in the data directory. (Added .csv extension)"
    
    except Exception as e:
        return f"Error creating sample data file '{filename}': {str(e)}"

# MCP Resources
@mcp.resource("data://schema")
def get_data_schema() -> str:
    """Provide schema information for available datasets."""
    schema_info = {
        "description": "Schema information for data files",
        "supported_formats": ["CSV", "Parquet"],
        "sample_structure": {
            "id": "integer - unique identifier",
            "name": "string - user name", 
            "email": "string - email address",
            "signup_date": "date - registration date"
        }
    }
    return json.dumps(schema_info, indent=2)

if __name__ == "__main__":
    import sys
    try:
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
            # Print to stderr to avoid interfering with MCP stdio protocol
            print(f"Created sample data files in {DATA_DIR}", file=sys.stderr)
        
        # Print to stderr to avoid interfering with MCP stdio protocol
        print("Starting MCP File Analyzer Server...", file=sys.stderr)
        # Run the MCP server (this blocks and waits for stdin)
        mcp.run()
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        print("Shutting down MCP server...", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        # Print errors to stderr
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)