#!/usr/bin/env python3
"""
MCP Client for testing the file analyzer server
Demonstrates how to programmatically interact with MCP tools
"""

import asyncio
import json
import sys
from pathlib import Path
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

class MCPFileAnalyzerClient:
    """A client for interacting with the MCP file analyzer server."""
    
    def __init__(self, server_script_path: str = "main.py"):
        self.server_script_path = server_script_path
        self.session = None
        self._stdio_transport = None
        self._read_stream = None
        self._write_stream = None
    
    async def connect(self):
        """Connect to the MCP server."""
        try:
            # Get absolute path to server script
            server_path = Path(self.server_script_path).resolve()
            if not server_path.exists():
                print(f"Error: Server script not found at {server_path}")
                return False
            
            # Create server parameters
            server_params = StdioServerParameters(
                command="python",
                args=[str(server_path)]
            )
            
            # Create stdio client transport - enter the context manager
            self._stdio_transport = stdio_client(server_params)
            self._read_stream, self._write_stream = await self._stdio_transport.__aenter__()
            
            # Create client session - enter the context manager
            session_context = ClientSession(self._read_stream, self._write_stream)
            self.session = await session_context.__aenter__()
            self._session_context = session_context
            
            # Initialize the session (required MCP handshake)
            await self.session.initialize()
            
            print("Connected to MCP server successfully!")
            return True
            
        except Exception as e:
            print(f"Error connecting to MCP server: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        try:
            if hasattr(self, '_session_context') and self._session_context:
                await self._session_context.__aexit__(None, None, None)
                self._session_context = None
                self.session = None
            
            if self._stdio_transport:
                await self._stdio_transport.__aexit__(None, None, None)
                self._stdio_transport = None
                self._read_stream = None
                self._write_stream = None
            
            print("Disconnected from MCP server")
        except Exception as e:
            print(f"Error during disconnect: {e}")
    
    async def list_tools(self):
        """List all available tools from the server."""
        if not self.session:
            print("Not connected to server")
            return None
        
        try:
            tools_result = await self.session.list_tools()
            tools = tools_result.tools
            
            print(f"\nAvailable tools ({len(tools)}):")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description or 'No description'}")
            
            return tools
        except Exception as e:
            print(f"Error listing tools: {e}")
            return None
    
    async def list_resources(self):
        """List all available resources from the server."""
        if not self.session:
            print("Not connected to server")
            return None
        
        try:
            resources_result = await self.session.list_resources()
            resources = resources_result.resources
            
            print(f"\nAvailable resources ({len(resources)}):")
            for resource in resources:
                print(f"  - {resource.uri}: {resource.name or 'No name'}")
            
            return resources
        except Exception as e:
            print(f"Error listing resources: {e}")
            return None
    
    async def call_tool(self, tool_name: str, arguments: dict = None):
        """Call a specific tool with arguments."""
        if not self.session:
            print("Not connected to server")
            return None
        
        try:
            if arguments is None:
                arguments = {}
            
            result = await self.session.call_tool(tool_name, arguments)
            
            if result.content:
                # Extract text from content
                content_text = ""
                for item in result.content:
                    if hasattr(item, 'text'):
                        content_text += item.text
                    elif isinstance(item, str):
                        content_text += item
                
                print(f"Result: {content_text}")
                return content_text
            else:
                print(f"Result: {result}")
                return str(result)
                
        except Exception as e:
            print(f"Error calling tool '{tool_name}': {e}")
            return None
    
    async def get_resource(self, uri: str):
        """Get a resource from the server."""
        if not self.session:
            print("Not connected to server")
            return None
        
        try:
            result = await self.session.read_resource(uri)
            
            if result.contents:
                content_text = ""
                for item in result.contents:
                    if hasattr(item, 'text'):
                        content_text += item.text
                    elif isinstance(item, str):
                        content_text += item
                
                print(f"Resource '{uri}':\n{content_text}")
                return content_text
            else:
                print(f"Resource '{uri}': {result}")
                return str(result)
                
        except Exception as e:
            print(f"Error getting resource '{uri}': {e}")
            return None

async def interactive_demo():
    """Run an interactive demo of the MCP client."""
    client = MCPFileAnalyzerClient()
    
    print("Starting MCP File Analyzer Client Demo")
    print("=" * 50)
    
    # Connect to server
    if not await client.connect():
        return
    
    try:
        # List available tools
        print("\n1. Discovering available tools...")
        tools = await client.list_tools()
        
        # List available resources
        print("\n2. Discovering available resources...")
        resources = await client.list_resources()
        
        # Test basic functionality
        print("\n3. Testing basic functionality...")
        
        # List data files
        print("\nListing data files:")
        await client.call_tool("list_data_files")
        
        # Summarize CSV file
        print("\nSummarizing CSV file:")
        await client.call_tool("summarize_csv_file", {"filename": "sample.csv"})
        
        # Analyze CSV data
        print("\nAnalyzing CSV data (info):")
        await client.call_tool("analyze_csv_data", {"filename": "sample.csv", "operation": "info"})
        
        # Show first few rows
        print("\nShowing first 5 rows:")
        await client.call_tool("analyze_csv_data", {"filename": "sample.csv", "operation": "head"})
        
        # Create sample data
        print("\nCreating new sample data:")
        await client.call_tool("create_sample_data", {"filename": "demo_data.csv", "rows": 15})
        
        # Get schema resource
        print("\nGetting data schema:")
        await client.get_resource("data://schema")
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.disconnect()

async def run_custom_commands():
    """Allow users to run custom commands."""
    client = MCPFileAnalyzerClient()
    
    if not await client.connect():
        return
    
    try:
        print("\nInteractive MCP Client")
        print("Available commands:")
        print("  - list_tools: Show available tools")
        print("  - list_files: List data files")
        print("  - summarize <filename>: Summarize a file")
        print("  - analyze <filename> <operation>: Analyze data")
        print("  - create <filename> <rows>: Create sample data")
        print("  - quit: Exit")
        
        while True:
            try:
                command = input("\n🤖 Enter command: ").strip()
                
                if command == "quit":
                    break
                elif command == "list_tools":
                    await client.list_tools()
                elif command == "list_files":
                    await client.call_tool("list_data_files")
                elif command.startswith("summarize"):
                    parts = command.split()
                    if len(parts) >= 2:
                        filename = parts[1]
                        if filename.endswith('.csv'):
                            await client.call_tool("summarize_csv_file", {"filename": filename})
                        elif filename.endswith('.parquet'):
                            await client.call_tool("summarize_parquet_file", {"filename": filename})
                        else:
                            print("Please specify .csv or .parquet extension")
                    else:
                        print("Usage: summarize <filename>")
                elif command.startswith("analyze"):
                    parts = command.split()
                    if len(parts) >= 3:
                        filename = parts[1]
                        operation = parts[2]
                        await client.call_tool("analyze_csv_data", {"filename": filename, "operation": operation})
                    else:
                        print("Usage: analyze <filename> <operation>")
                elif command.startswith("create"):
                    parts = command.split()
                    if len(parts) >= 3:
                        filename = parts[1]
                        rows = int(parts[2])
                        await client.call_tool("create_sample_data", {"filename": filename, "rows": rows})
                    else:
                        print("Usage: create <filename> <rows>")
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
    finally:
        await client.disconnect()

def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
       asyncio.run(run_custom_commands())
    else:
       asyncio.run(interactive_demo())

if __name__ == "__main__":
    main()