# MCP File Analyzer Server

A Model Context Protocol (MCP) server implementation that enables AI assistants like Claude to interact with local data files through natural language queries. This project demonstrates building a complete MCP server with tools for analyzing CSV and Parquet files, integrating with Claude Desktop, and providing programmatic access through a client interface.

## Overview

This project implements a fully functional MCP server that bridges the gap between AI assistants and local data files. The server provides a set of tools that allow Claude to read, analyze, and create data files, making it possible to perform data analysis tasks through natural language conversations.

The implementation follows the MCP specification and demonstrates best practices for building production-ready MCP servers, including proper error handling, type hints, comprehensive documentation, and cross-platform compatibility.

## Key Achievements

### MCP Server Implementation

**Core Server Architecture**
- Built a robust MCP server using the FastMCP framework with stdio transport
- Implemented proper server initialization and lifecycle management
- Created automatic data directory setup and sample data generation
- Ensured all tools follow MCP specification with proper type hints and docstrings

**File Analysis Tools**
The server provides five comprehensive tools for data file operations:

1. **`list_data_files()`** - Discovers and lists all available CSV and Parquet files in the data directory
2. **`summarize_csv_file(filename)`** - Provides quick overviews of CSV file structure, dimensions, and column information
3. **`summarize_parquet_file(filename)`** - Summarizes Parquet file structure and metadata
4. **`analyze_csv_data(filename, operation)`** - Performs advanced analysis with multiple operation modes:
   - `describe`: Statistical summary for numeric columns (mean, median, std dev, etc.)
   - `head`: Preview first 5 rows of data
   - `info`: Data types and non-null counts for all columns
   - `columns`: List all columns with their data types
5. **`create_sample_data(filename, rows)`** - Generates synthetic datasets with customizable row counts (1-10,000 rows)

**Resource Management**
- Implemented MCP resources for schema information (`data://schema`)
- Provides structured metadata about supported file formats and data structures
- Enables Claude to understand available data formats and schemas

**Error Handling & Validation**
- Comprehensive error handling for file operations (missing files, invalid formats, etc.)
- Input validation for tool parameters (row counts, file names, operations)
- Clear, actionable error messages for debugging
- Graceful handling of edge cases (empty directories, malformed files, etc.)

### Client Development

**Async MCP Client**
- Developed a fully-featured asynchronous client using `ClientSession` and stdio transport
- Implemented proper connection lifecycle management with context managers
- Created robust error handling and cleanup procedures
- Built connection retry logic and graceful disconnection

**Testing Capabilities**
- **Demo Mode**: Automated test suite that exercises all server tools
- **Interactive Mode**: Command-line interface for manual testing and exploration
- Tool discovery and introspection capabilities
- Resource retrieval and validation
- Comprehensive test coverage for all tool operations

**Client Features**
- Lists available tools and resources from the server
- Calls tools with proper argument handling
- Retrieves and displays resources
- Parses and formats tool responses
- Provides clear feedback and error reporting

### Claude Desktop Integration

**Cross-platform Launcher**
- Created launcher scripts (`run_mcp_server.sh`) that work across macOS, Windows, and Linux
- Proper virtual environment activation in launcher scripts
- Absolute path resolution for reliable execution
- Unbuffered output for real-time logging

**Configuration Management**
- Implemented Claude Desktop configuration with proper JSON structure
- Cross-platform path handling for configuration files
- Support for both script-based and direct Python execution
- Clear documentation for configuration setup

**Integration Testing**
- Successfully integrated with Claude Desktop application
- Verified tool discovery and invocation through natural language
- Tested complex multi-step analysis workflows
- Validated error handling and edge cases through Claude interface

### Technical Highlights

**Protocol Compliance**
- Full adherence to MCP specification
- Proper use of MCP tool decorators and resource registration
- Correct stdio transport implementation
- Standardized error response formats

**Data Processing**
- Efficient CSV file reading using pandas
- Parquet file support via pyarrow
- Automatic data type inference
- Memory-efficient processing for large files

**Code Quality**
- Comprehensive type hints throughout
- Detailed docstrings for all functions and tools
- Modular design with clear separation of concerns
- Clean code structure following Python best practices

**Architecture**
- Server implementation (`main.py`) - Core MCP server with stdio transport
- Client implementation (`client.py`) - Testing and integration client
- HTTP server (`http_server.py`) - Alternative HTTP-based server for web access
- Clear separation between server, client, and HTTP implementations

## Project Structure

```
mcp-server/
├── file_analyzer-main/          # Main implementation directory
│   ├── main.py                  # MCP server implementation (stdio)
│   ├── client.py                # MCP client for testing
│   ├── http_server.py           # HTTP alternative server
│   ├── requirements.txt         # Python dependencies
│   ├── run_mcp_server.sh        # Claude Desktop launcher script
│   ├── activate_env.sh          # Environment activation helper
│   ├── claude_desktop_config.json  # Claude Desktop configuration template
│   ├── data/                    # Data files directory (auto-created)
│   │   ├── sample.csv           # Sample CSV data (auto-generated)
│   │   ├── sample.parquet      # Sample Parquet data (auto-generated)
│   │   └── ...                  # User data files
│   ├── images/                  # Documentation screenshots
│   └── README.md                # Detailed setup and usage guide
└── README.md                    # This file (project overview)
```

## Features

### Data File Operations

**File Discovery**
- Automatic scanning of data directory for CSV and Parquet files
- Sorted file listing for consistent results
- Support for multiple file formats in single directory

**File Summarization**
- Quick overviews of file structure and dimensions
- Column name extraction and listing
- Row and column count reporting
- Format-specific metadata extraction

**Advanced Analysis**
- **Statistical Analysis**: Mean, median, standard deviation, min/max for numeric columns
- **Data Preview**: First N rows display with proper formatting
- **Type Inspection**: Data type information for all columns
- **Quality Checks**: Non-null counts and null value detection

**Data Generation**
- Synthetic dataset creation with realistic data
- Customizable row counts (1-10,000 rows)
- Multiple data types (integers, strings, dates, scores)
- Automatic file format detection (CSV/Parquet)

### Integration Capabilities

**Claude Desktop**
- Seamless natural language interaction
- Tool discovery and invocation through conversation
- Multi-step analysis workflows
- Error handling and recovery

**Programmatic Access**
- Full-featured client for automated testing
- Tool introspection and discovery
- Resource retrieval and validation
- Integration with other Python applications

**HTTP Server**
- Alternative HTTP-based server for web access
- RESTful API for tool invocation
- Web-based debugging and testing
- Integration with web applications

### Developer Experience

**Documentation**
- Comprehensive setup guides
- Usage examples for all tools
- Troubleshooting section with common issues
- Extension examples for adding new tools

**Testing Tools**
- Automated demo mode for quick verification
- Interactive mode for manual exploration
- Clear output formatting and error messages
- Test coverage for all operations

**Extensibility**
- Easy-to-follow patterns for adding new tools
- Resource registration examples
- Database integration examples
- Clear extension points documented

## Technology Stack

- **Python 3.8+**: Core implementation language
- **MCP SDK**: Model Context Protocol Python SDK (`mcp>=1.0.0`)
- **Pandas**: Data manipulation and analysis (`pandas>=2.0.0`)
- **PyArrow**: Parquet file support (`pyarrow>=10.0.0`)
- **FastMCP**: Framework for building MCP servers
- **asyncio**: Asynchronous client implementation
- **pathlib**: Modern file path handling

## Use Cases

**Data Exploration**
- Quickly explore and understand data files through natural language
- Get instant summaries of file structure and contents
- Discover data types and quality issues

**Statistical Analysis**
- Get statistical summaries and insights about datasets
- Perform exploratory data analysis through conversation
- Identify patterns and anomalies in data

**Data Validation**
- Check file structure, data types, and data quality
- Verify data integrity and completeness
- Identify missing or malformed data

**Prototyping & Testing**
- Generate sample datasets for testing and development
- Create mock data for application development
- Test data processing pipelines

**AI-Assisted Analysis**
- Leverage Claude's capabilities for complex data analysis tasks
- Perform multi-step analysis workflows
- Get insights and recommendations about data

## Implementation Details

### Server Architecture

The server uses FastMCP framework which provides:
- Automatic tool registration via decorators
- Resource management capabilities
- Stdio transport for Claude Desktop integration
- Type validation and error handling

### Tool Implementation Pattern

All tools follow a consistent pattern:
1. Input validation (file existence, parameter ranges)
2. Data loading with error handling
3. Processing with pandas operations
4. Formatted output generation
5. Error propagation with clear messages

### Client Architecture

The client implements:
- Async/await pattern for non-blocking operations
- Context managers for proper resource cleanup
- Session management for connection lifecycle
- Response parsing and formatting

### Sample Data Generation

The server automatically creates sample data on first run:
- User data with IDs, names, emails, signup dates
- Realistic data distributions
- Multiple data types for testing
- Both CSV and Parquet formats

## Getting Started

For detailed setup instructions, installation steps, usage examples, troubleshooting, and extension guides, see the comprehensive documentation in [`file_analyzer-main/README.md`](file_analyzer-main/README.md).

The detailed guide includes:
- Step-by-step installation instructions
- Virtual environment setup
- Claude Desktop integration walkthrough
- Usage examples with Claude
- Testing and verification procedures
- Troubleshooting common issues
- Examples for extending the server

## Key Learnings & Insights

This project demonstrates several important concepts:

**MCP Protocol Mastery**
- Understanding the Model Context Protocol specification
- Implementing proper server and client patterns
- Managing stdio transport for AI assistant integration
- Resource and tool registration best practices

**AI Integration Patterns**
- Building tools that work well with natural language interfaces
- Designing APIs for both programmatic and conversational access
- Error handling that provides useful feedback to AI assistants
- Tool descriptions that enable effective AI tool selection

**Production-Ready Development**
- Comprehensive error handling and validation
- Cross-platform compatibility considerations
- Virtual environment and dependency management
- Configuration management for different deployment scenarios

**Testing & Quality Assurance**
- Building test clients for MCP servers
- Interactive testing modes for exploration
- Automated test suites for verification
- Edge case handling and validation

**Documentation & Usability**
- Creating comprehensive setup guides
- Providing clear usage examples
- Troubleshooting documentation
- Extension examples for developers

## Future Enhancements

Potential areas for extension:
- Excel file support (.xlsx, .xls)
- JSON file analysis
- Data visualization tools (chart generation)
- Database integration (SQL queries)
- Web API connections
- Machine learning analysis tools
- File monitoring and change detection
- Data transformation operations
- Export capabilities (to various formats)
