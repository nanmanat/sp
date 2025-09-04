#!/usr/bin/env python3
"""
Launcher script for the ML Training Trigger Publisher.

This script starts a WebSocket server that manages training clients
and publishes training commands to them.
"""

import os
import sys
import argparse

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the trigger publisher
from code.trigger_publisher import main
import asyncio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start ML Training Trigger Publisher Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind server (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8765, help='Port to bind server (default: 8765)')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode after server starts')
    
    args = parser.parse_args()
    
    print("🚀 Starting ML Training Trigger Publisher...")
    print(f"📡 Starting server on {args.host}:{args.port}")
    print("🔔 Training clients can subscribe to receive triggers")
    
    if args.interactive:
        print("🎮 Interactive mode will start after server initialization")
    
    print("📝 Logs will be saved to ./trigger_publisher.log")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    # Set up sys.argv for the publisher main function
    sys.argv = ['trigger_publisher.py']
    if args.host != '0.0.0.0':
        sys.argv.extend(['--host', args.host])
    if args.port != 8765:
        sys.argv.extend(['--port', str(args.port)])
    if args.interactive:
        sys.argv.append('--interactive')
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Trigger publisher stopped by user")
    except Exception as e:
        print(f"❌ Error starting trigger publisher: {str(e)}")
        sys.exit(1)
