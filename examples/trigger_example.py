#!/usr/bin/env python3
"""
Example script demonstrating how to use the WebSocket trigger system
to remotely start ML training runs.
"""

import asyncio
import sys
import os

# Add parent directory to path to import trigger_sender
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.trigger_sender import TrainingTriggerSender

async def example_training_workflow():
    """Example workflow showing how to trigger training remotely."""
    
    # Replace with your training machine's IP address
    # Use "localhost" if running on the same machine
    client_ip = "localhost"  # Change this to your client's IP
    
    print("🚀 WebSocket Trigger System Example")
    print("=" * 50)
    
    # Create sender instance
    sender = TrainingTriggerSender(client_ip)
    print(f"🔗 Connecting to training server at {sender.uri}")
    
    try:
        # 1. Check if server is available and get status
        print("\n📊 Step 1: Checking server status...")
        status_response = await sender.get_status()
        print(f"Status: {status_response}")
        
        # 2. List available models
        print("\n🤖 Step 2: Getting available models...")
        models_response = await sender.list_models()
        if models_response.get("status") == "success":
            print("Available models:")
            for model in models_response.get("models", []):
                print(f"  • {model}")
        
        # 3. Start a training run
        print("\n🚀 Step 3: Starting training...")
        training_response = await sender.start_training(
            model="resnet50",
            folds=["0"],  # Just fold 0 for quick testing
            batch_size=32,
            lr=0.001,
            num_workers=4  # Reduced for testing
        )
        print(f"Training start response: {training_response}")
        
        if training_response.get("status") == "success":
            print("✅ Training started successfully!")
            
            # 4. Monitor training status
            print("\n🔄 Step 4: Monitoring training status...")
            for i in range(3):  # Check status 3 times
                await asyncio.sleep(5)  # Wait 5 seconds between checks
                status = await sender.get_status()
                is_training = status.get("is_training", False)
                print(f"Check {i+1}: Training active = {is_training}")
                
                if not is_training:
                    print("Training completed or stopped.")
                    break
        else:
            print(f"❌ Failed to start training: {training_response.get('message')}")
    
    except Exception as e:
        print(f"❌ Error in example workflow: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure the trigger listener is running:")
        print("   python3 run_trigger_listener.py")
        print("2. Check the client IP address in this script")
        print("3. Ensure port 8765 is not blocked by firewall")

async def simple_trigger_example():
    """Simple example - just trigger training and exit."""
    client_ip = "localhost"  # Change this to your client's IP
    
    sender = TrainingTriggerSender(client_ip)
    
    print("🎯 Simple Training Trigger Example")
    print(f"Sending training trigger to {client_ip}...")
    
    response = await sender.start_training(
        model="resnet50",
        folds=["0"]
    )
    
    if response.get("status") == "success":
        print("✅ Training triggered successfully!")
        print(f"Message: {response.get('message')}")
    else:
        print(f"❌ Failed to trigger training: {response.get('message')}")

if __name__ == "__main__":
    print("Choose an example to run:")
    print("1. Full workflow example (recommended)")
    print("2. Simple trigger example")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            asyncio.run(example_training_workflow())
        elif choice == "2":
            asyncio.run(simple_trigger_example())
        else:
            print("Invalid choice. Running full workflow example...")
            asyncio.run(example_training_workflow())
    
    except KeyboardInterrupt:
        print("\n👋 Example interrupted by user")
    except Exception as e:
        print(f"❌ Error running example: {str(e)}")
