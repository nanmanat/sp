#!/usr/bin/env python3
"""
Test script to verify that the trigger subscriber can connect to an existing server.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the subscriber
from code.trigger_subscriber import TrainingTriggerSubscriber

async def test_connection(server_host, server_port=8765, client_id=None):
    """Test connecting to a server."""
    logger.info(f"Testing connection to server at {server_host}:{server_port}")
    
    # Create subscriber
    subscriber = TrainingTriggerSubscriber(
        server_host=server_host,
        server_port=server_port,
        client_id=client_id or f"test_client_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    
    # Set a shorter reconnect delay for testing
    subscriber.reconnect_delay = 2
    
    # Set a timeout for the test
    connection_timeout = 10  # seconds
    
    try:
        # Create a task for the connection
        connection_task = asyncio.create_task(subscriber.connect_and_subscribe())
        
        # Wait for the connection to be established or timeout
        start_time = asyncio.get_event_loop().time()
        while not subscriber.is_connected:
            await asyncio.sleep(0.5)
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > connection_timeout:
                logger.error(f"Connection timeout after {connection_timeout} seconds")
                break
        
        # Check if connection was successful
        if subscriber.is_connected:
            logger.info("✅ Successfully connected to the server!")
            logger.info(f"Client ID: {subscriber.client_id}")
            logger.info("Waiting for 5 seconds to receive any messages...")
            await asyncio.sleep(5)
        else:
            logger.error("❌ Failed to connect to the server")
        
        # Disconnect
        logger.info("Disconnecting from server...")
        await subscriber.disconnect()
        
        # Cancel the connection task
        connection_task.cancel()
        try:
            await connection_task
        except asyncio.CancelledError:
            pass
        
    except Exception as e:
        logger.error(f"Error during connection test: {str(e)}")
    
    return subscriber.is_connected

async def main():
    """Main function to run the test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test connection to trigger server')
    parser.add_argument('server_host', help='Host address of the trigger server')
    parser.add_argument('--port', type=int, default=8765, help='Port of the trigger server (default: 8765)')
    parser.add_argument('--client-id', help='Unique client identifier (default: auto-generated)')
    
    args = parser.parse_args()
    
    print("🧪 Testing Trigger Subscriber Connection")
    print(f"📡 Target server: {args.server_host}:{args.port}")
    print("=" * 50)
    
    success = await test_connection(
        server_host=args.server_host,
        server_port=args.port,
        client_id=args.client_id
    )
    
    if success:
        print("\n✅ Connection test PASSED")
        return 0
    else:
        print("\n❌ Connection test FAILED")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)