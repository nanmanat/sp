#!/usr/bin/env python3
"""
Launcher script for the ML Training Trigger Listener.

This script starts the WebSocket server that listens for training triggers
from remote servers.
"""

import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the trigger listener
from code.trigger_listener import main
import asyncio

if __name__ == "__main__":
    print("🚀 Starting ML Training Trigger Listener...")
    print("📡 This will start a WebSocket server on port 8765")
    print("🔧 Remote servers can connect and trigger training runs")
    print("📝 Logs will be saved to ./code/logs/trigger_listener.log")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Trigger listener stopped by user")
    except Exception as e:
        print(f"❌ Error starting trigger listener: {str(e)}")
        sys.exit(1)
