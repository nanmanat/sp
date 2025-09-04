#!/usr/bin/env python3
"""
Example script demonstrating the WebSocket subscription model
for remote ML training management.
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from code.trigger_publisher import TrainingTriggerPublisher

async def subscription_model_demo():
    """Demonstrate the subscription model with a mock scenario."""
    
    print("🚀 WebSocket Subscription Model Demo")
    print("=" * 60)
    print("This demo shows how the subscription model works:")
    print("1. Publisher server manages multiple training clients")
    print("2. Clients subscribe to receive training commands")
    print("3. Server can send commands to specific clients or broadcast to all")
    print()
    
    # Create publisher instance
    publisher = TrainingTriggerPublisher(host="localhost", port=8766)  # Use different port for demo
    
    print("📡 Starting publisher server on localhost:8766...")
    
    # Start server in background
    server_task = asyncio.create_task(publisher.start_server())
    
    # Give server time to start
    await asyncio.sleep(1)
    
    print("✅ Publisher server started!")
    print()
    print("🔔 In a real scenario, training clients would now connect:")
    print("   python3 run_trigger_subscriber.py localhost --port 8766 --client-id gpu-server-1")
    print("   python3 run_trigger_subscriber.py localhost --port 8766 --client-id gpu-server-2")
    print()
    
    # Simulate some operations
    print("📊 Publisher Operations Demo:")
    print()
    
    # Show empty client list
    client_list = publisher.get_client_list()
    print(f"👥 Connected clients: {client_list['total_clients']}")
    
    if client_list['total_clients'] == 0:
        print("📭 No clients connected (this is expected in the demo)")
        print()
        print("🎯 To see the full system in action:")
        print("1. Start the publisher server:")
        print("   python3 run_trigger_publisher.py --port 8766 --interactive")
        print()
        print("2. In separate terminals, start subscriber clients:")
        print("   python3 run_trigger_subscriber.py localhost --port 8766 --client-id training-node-1")
        print("   python3 run_trigger_subscriber.py localhost --port 8766 --client-id training-node-2")
        print()
        print("3. Use publisher interactive commands:")
        print("   Publisher> list                    # See connected clients")
        print("   Publisher> start training-node-1 resnet50")
        print("   Publisher> start_all efficientnet_v2_s")
        print("   Publisher> status_all")
        print()
    
    # Demonstrate command structure
    print("📋 Example Commands Structure:")
    
    # Example training command
    training_command = {
        "type": "start_training",
        "model": "resnet50",
        "folds": ["0", "1", "2"],
        "batch_size": 64,
        "lr": 0.001,
        "timestamp": datetime.now().isoformat()
    }
    
    print("🚀 Start Training Command:")
    print(json.dumps(training_command, indent=2))
    print()
    
    # Example status command
    status_command = {
        "type": "get_status",
        "timestamp": datetime.now().isoformat()
    }
    
    print("📊 Status Query Command:")
    print(json.dumps(status_command, indent=2))
    print()
    
    # Example broadcast command
    broadcast_command = {
        "type": "broadcast",
        "message": "Maintenance window starting in 10 minutes",
        "timestamp": datetime.now().isoformat()
    }
    
    print("📡 Broadcast Command:")
    print(json.dumps(broadcast_command, indent=2))
    print()
    
    print("🎉 Demo completed!")
    print()
    print("💡 Key Benefits of Subscription Model:")
    print("   ✅ Centralized management of multiple training machines")
    print("   ✅ Real-time monitoring of all connected clients")
    print("   ✅ Broadcast capabilities for coordinated operations")
    print("   ✅ Network-friendly (clients connect outbound only)")
    print("   ✅ Scalable to unlimited number of training machines")
    
    # Stop the server
    publisher.stop_server()
    server_task.cancel()
    
    try:
        await server_task
    except asyncio.CancelledError:
        pass

async def interactive_publisher_demo():
    """Start an interactive publisher for hands-on testing."""
    print("🎮 Interactive Publisher Demo")
    print("=" * 40)
    print("Starting interactive publisher server...")
    print("Connect clients with:")
    print("  python3 run_trigger_subscriber.py localhost --client-id test-client")
    print()
    
    # Import and use the CLI
    from code.trigger_publisher import PublisherCLI
    
    publisher = TrainingTriggerPublisher(host="localhost", port=8765)
    
    # Start server in background
    server_task = asyncio.create_task(publisher.start_server())
    
    # Give server time to start
    await asyncio.sleep(1)
    
    # Start interactive CLI
    cli = PublisherCLI(publisher)
    await cli.interactive_mode()
    
    # Stop server
    publisher.stop_server()
    server_task.cancel()
    
    try:
        await server_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    print("Choose a demo to run:")
    print("1. Subscription model overview (recommended)")
    print("2. Interactive publisher (requires manual client connections)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            asyncio.run(subscription_model_demo())
        elif choice == "2":
            asyncio.run(interactive_publisher_demo())
        else:
            print("Invalid choice. Running overview demo...")
            asyncio.run(subscription_model_demo())
    
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error running demo: {str(e)}")
