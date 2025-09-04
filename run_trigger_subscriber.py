#!/usr/bin/env python3
"""
Launcher script for the ML Training Trigger Subscriber.

This script starts a WebSocket client that subscribes to a remote trigger server
and listens for training commands.
"""

import os
import sys
import argparse

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the trigger subscriber
from code.trigger_subscriber import main
import asyncio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start ML Training Trigger Subscriber')
    parser.add_argument('server_host', help='Host address of the trigger server to subscribe to')
    parser.add_argument('--port', type=int, default=8765, help='Port of the trigger server (default: 8765)')
    parser.add_argument('--client-id', help='Unique client identifier (default: auto-generated)')
    
    args = parser.parse_args()
    
    print("🚀 Starting ML Training Trigger Subscriber...")
    print(f"📡 Subscribing to trigger server at {args.server_host}:{args.port}")
    print("🔔 This client will receive training triggers from the server")
    print("📝 Logs will be saved to ./code/logs/trigger_subscriber.log")
    print("\nPress Ctrl+C to stop the subscriber")
    print("=" * 60)
    
    # Set up sys.argv for the subscriber main function
    sys.argv = ['trigger_subscriber.py', args.server_host]
    if args.port != 8765:
        sys.argv.extend(['--port', str(args.port)])
    if args.client_id:
        sys.argv.extend(['--client-id', args.client_id])
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Trigger subscriber stopped by user")
    except Exception as e:
        print(f"❌ Error starting trigger subscriber: {str(e)}")
        sys.exit(1)
