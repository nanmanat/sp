#!/usr/bin/env python3
"""
Test script to verify that the trigger subscriber can connect to a publisher server.
This script starts both a publisher server and a subscriber client.
"""

import asyncio
import sys
import os
import logging
import threading
import time
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

# Import the publisher and subscriber
from code.trigger_publisher import TrainingTriggerPublisher
from code.trigger_subscriber import TrainingTriggerSubscriber

# Global flag to indicate if the server is ready
server_ready = False

async def run_publisher_server(host="localhost", port=8765):
    """Run the publisher server."""
    global server_ready
    
    logger.info(f"Starting publisher server on {host}:{port}")
    
    # Create publisher
    publisher = TrainingTriggerPublisher(host=host, port=port)
    
    # Set the server_ready flag to True
    server_ready = True
    logger.info("Publisher server is ready to accept connections")
    
    # Start the server
    try:
        await publisher.start_server()
    except Exception as e:
        logger.error(f"Error running publisher server: {str(e)}")
    finally:
        publisher.stop_server()
        logger.info("Publisher server stopped")

async def run_subscriber_client(server_host="localhost", server_port=8765, client_id=None):
    """Run the subscriber client."""
    global server_ready
    
    # Wait for the server to be ready
    while not server_ready:
        logger.info("Waiting for publisher server to be ready...")
        await asyncio.sleep(1)
    
    logger.info(f"Starting subscriber client connecting to {server_host}:{server_port}")
    
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
    
    parser = argparse.ArgumentParser(description='Test publisher-subscriber connection')
    parser.add_argument('--host', default='localhost', help='Host to bind server (default: localhost)')
    parser.add_argument('--port', type=int, default=8765, help='Port to bind server (default: 8765)')
    parser.add_argument('--client-id', help='Unique client identifier (default: auto-generated)')
    
    args = parser.parse_args()
    
    print("🧪 Testing Publisher-Subscriber Connection")
    print(f"📡 Server: {args.host}:{args.port}")
    print("=" * 50)
    
    # Start the publisher server and subscriber client
    server_task = asyncio.create_task(run_publisher_server(host=args.host, port=args.port))
    
    # Give the server a moment to start
    await asyncio.sleep(2)
    
    # Run the subscriber client
    success = await run_subscriber_client(
        server_host=args.host,
        server_port=args.port,
        client_id=args.client_id
    )
    
    # Stop the server
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass
    
    if success:
        print("\n✅ Connection test PASSED")
        return 0
    else:
        print("\n❌ Connection test FAILED")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)