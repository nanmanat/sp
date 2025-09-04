#!/usr/bin/env python3
"""
WebSocket Trigger Sender for ML Training System

This module provides functionality to send training triggers to remote
ML training clients via WebSocket connections.
"""

import asyncio
import websockets
import json
import argparse
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

class TrainingTriggerSender:
    """Client for sending training triggers to remote ML training servers."""
    
    def __init__(self, host: str, port: int = 8765):
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
    
    async def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command to the training server and return the response."""
        try:
            async with websockets.connect(self.uri, ping_interval=20, ping_timeout=10) as websocket:
                # Send command
                command_json = json.dumps(command)
                await websocket.send(command_json)
                print(f"📤 Sent command to {self.uri}: {command_json}")
                
                # Wait for response
                response = await websocket.recv()
                print(f"📥 Received response: {response}")
                
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    return {"status": "success", "message": response}
                    
        except (ConnectionRefusedError, OSError) as e:
            return {
                "status": "error",
                "message": f"Connection refused to {self.uri}. Is the training server running? ({str(e)})"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection error: {str(e)}"
            }
    
    async def send_text_command(self, message: str) -> str:
        """Send a plain text command (for backward compatibility)."""
        try:
            async with websockets.connect(self.uri, ping_interval=20, ping_timeout=10) as websocket:
                await websocket.send(message)
                print(f"📤 Sent text command to {self.uri}: {message}")
                
                response = await websocket.recv()
                print(f"📥 Received response: {response}")
                return response
                
        except (ConnectionRefusedError, OSError) as e:
            return f"❌ Connection refused to {self.uri}. Is the training server running? ({str(e)})"
        except Exception as e:
            return f"❌ Connection error: {str(e)}"
    
    async def start_training(self, model: str, folds: List[str] = None, 
                           batch_size: Optional[int] = None, lr: Optional[float] = None,
                           num_workers: Optional[int] = None, data_path: Optional[str] = None,
                           early_stop_patience: Optional[int] = None, 
                           improvement_threshold: float = 0.015) -> Dict[str, Any]:
        """Start a training run on the remote server."""
        if folds is None:
            folds = ["0"]
        
        command = {
            "type": "start_training",
            "model": model,
            "folds": folds,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add optional parameters if provided
        if batch_size is not None:
            command["batch_size"] = batch_size
        if lr is not None:
            command["lr"] = lr
        if num_workers is not None:
            command["num_workers"] = num_workers
        if data_path is not None:
            command["data_path"] = data_path
        if early_stop_patience is not None:
            command["early_stop_patience"] = early_stop_patience
        if improvement_threshold != 0.015:
            command["improvement_threshold"] = improvement_threshold
        
        return await self.send_command(command)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get the current training status from the remote server."""
        command = {
            "type": "get_status",
            "timestamp": datetime.now().isoformat()
        }
        return await self.send_command(command)
    
    async def stop_training(self) -> Dict[str, Any]:
        """Request to stop the current training on the remote server."""
        command = {
            "type": "stop_training",
            "timestamp": datetime.now().isoformat()
        }
        return await self.send_command(command)
    
    async def list_models(self) -> Dict[str, Any]:
        """Get the list of available models from the remote server."""
        command = {
            "type": "list_models",
            "timestamp": datetime.now().isoformat()
        }
        return await self.send_command(command)

def print_response(response: Dict[str, Any]):
    """Pretty print a response from the server."""
    status = response.get("status", "unknown")
    message = response.get("message", "")
    
    if status == "success":
        print(f"✅ {message}")
    elif status == "error":
        print(f"❌ {message}")
    elif status == "info":
        print(f"ℹ️  {message}")
    else:
        print(f"📋 {message}")
    
    # Print additional details if available
    if "training_config" in response:
        print("\n📊 Training Configuration:")
        config = response["training_config"]
        for key, value in config.items():
            if value is not None:
                print(f"   {key}: {value}")
    
    if "current_training" in response and response["current_training"]:
        print("\n🔄 Current Training:")
        training = response["current_training"]
        for key, value in training.items():
            if value is not None:
                print(f"   {key}: {value}")
    
    if "models" in response:
        print("\n🤖 Available Models:")
        for model in response["models"]:
            print(f"   • {model}")
    
    if "is_training" in response:
        status_emoji = "🔄" if response["is_training"] else "⏸️"
        print(f"\n{status_emoji} Training Status: {'Running' if response['is_training'] else 'Idle'}")

async def interactive_mode(sender: TrainingTriggerSender):
    """Run in interactive mode for testing."""
    print("\n🎮 Interactive Mode - Available Commands:")
    print("  start <model> [folds] - Start training (e.g., 'start resnet50 0,1,2')")
    print("  status                - Get current status")
    print("  stop                  - Stop current training")
    print("  models                - List available models")
    print("  quit                  - Exit interactive mode")
    print()
    
    while True:
        try:
            command = input("🔧 Enter command: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            if command.startswith('start '):
                parts = command.split()
                if len(parts) < 2:
                    print("❌ Usage: start <model> [folds]")
                    continue
                
                model = parts[1]
                folds = parts[2].split(',') if len(parts) > 2 else ['0']
                
                print(f"🚀 Starting training for {model} on folds {folds}...")
                response = await sender.start_training(model, folds)
                print_response(response)
            
            elif command == 'status':
                print("📊 Getting status...")
                response = await sender.get_status()
                print_response(response)
            
            elif command == 'stop':
                print("⏹️  Requesting training stop...")
                response = await sender.stop_training()
                print_response(response)
            
            elif command == 'models':
                print("📋 Getting available models...")
                response = await sender.list_models()
                print_response(response)
            
            else:
                print("❌ Unknown command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

async def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Send training triggers to ML training server')
    parser.add_argument('host', help='Host address of the training server')
    parser.add_argument('--port', type=int, default=8765, help='Port of the training server (default: 8765)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start training command
    start_parser = subparsers.add_parser('start', help='Start training')
    start_parser.add_argument('model', help='Model name to train')
    start_parser.add_argument('--folds', default='0', help='Comma-separated fold indices (default: 0)')
    start_parser.add_argument('--batch-size', type=int, help='Batch size')
    start_parser.add_argument('--lr', type=float, help='Learning rate')
    start_parser.add_argument('--workers', type=int, help='Number of workers')
    start_parser.add_argument('--data-path', help='Path to data directory')
    start_parser.add_argument('--early-stop', type=int, help='Early stopping patience')
    start_parser.add_argument('--threshold', type=float, default=0.015, help='Improvement threshold')
    
    # Other commands
    subparsers.add_parser('status', help='Get training status')
    subparsers.add_parser('stop', help='Stop current training')
    subparsers.add_parser('models', help='List available models')
    subparsers.add_parser('interactive', help='Enter interactive mode')
    
    # Text command
    text_parser = subparsers.add_parser('text', help='Send raw text command')
    text_parser.add_argument('message', help='Text message to send')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    sender = TrainingTriggerSender(args.host, args.port)
    print(f"🔗 Connecting to training server at {sender.uri}")
    
    try:
        if args.command == 'start':
            folds = args.folds.split(',')
            response = await sender.start_training(
                model=args.model,
                folds=folds,
                batch_size=args.batch_size,
                lr=args.lr,
                num_workers=args.workers,
                data_path=args.data_path,
                early_stop_patience=args.early_stop,
                improvement_threshold=args.threshold
            )
            print_response(response)
        
        elif args.command == 'status':
            response = await sender.get_status()
            print_response(response)
        
        elif args.command == 'stop':
            response = await sender.stop_training()
            print_response(response)
        
        elif args.command == 'models':
            response = await sender.list_models()
            print_response(response)
        
        elif args.command == 'text':
            response = await sender.send_text_command(args.message)
            print(f"📥 Response: {response}")
        
        elif args.command == 'interactive':
            await interactive_mode(sender)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🚀 ML Training Trigger Sender")
    print("=" * 50)
    
    if len(sys.argv) == 1:
        print("Usage examples:")
        print("  python trigger_sender.py <host> start resnet50 --folds 0,1,2")
        print("  python trigger_sender.py <host> status")
        print("  python trigger_sender.py <host> interactive")
        print("  python trigger_sender.py localhost start resnet50")
        sys.exit(1)
    
    asyncio.run(main())
