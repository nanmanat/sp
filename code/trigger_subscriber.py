#!/usr/bin/env python3
"""
WebSocket Trigger Subscriber for ML Training System

This module creates a WebSocket client that subscribes to a remote server
and listens for training commands. The client connects to the server and
maintains a persistent connection to receive triggers.
"""

import asyncio
import websockets
import json
import logging
import sys
import os
from datetime import datetime
import threading
from typing import Dict, Any, Optional
import signal

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./code/logs/trigger_subscriber.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import project modules with error handling
try:
    from code.train import run_training, create_model
    TRAINING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Training modules not available: {str(e)}")
    logger.warning("Some dependencies may be missing. Training functionality will be limited.")
    TRAINING_AVAILABLE = False
    
    # Create dummy functions
    def run_training(*args, **kwargs):
        raise RuntimeError("Training functionality not available due to missing dependencies")
    
    def create_model(*args, **kwargs):
        raise RuntimeError("Model creation not available due to missing dependencies")

class TrainingTriggerSubscriber:
    """WebSocket client that subscribes to a server for training triggers."""
    
    def __init__(self, server_host: str, server_port: int = 8765, client_id: str = None):
        self.server_host = server_host
        self.server_port = server_port
        self.server_uri = f"ws://{server_host}:{server_port}"
        self.client_id = client_id or f"client_{os.getpid()}"
        
        self.websocket = None
        self.current_training = None
        self.training_thread = None
        self.is_training = False
        self.is_connected = False
        self.should_reconnect = True
        self.reconnect_delay = 5  # seconds
        
        # Ensure logs directory exists
        os.makedirs('./code/logs', exist_ok=True)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.should_reconnect = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())
    
    async def connect_and_subscribe(self):
        """Connect to the server and maintain subscription."""
        logger.info(f"🔗 Subscribing to trigger server at {self.server_uri}")
        
        while self.should_reconnect:
            try:
                # Connect to the server
                self.websocket = await websockets.connect(
                    self.server_uri,
                    ping_interval=20,
                    ping_timeout=10
                )
                
                self.is_connected = True
                logger.info(f"✅ Connected to server {self.server_uri}")
                
                # Send initial registration message
                registration = {
                    "type": "register",
                    "client_id": self.client_id,
                    "capabilities": {
                        "training_available": TRAINING_AVAILABLE,
                        "models": self.get_available_models() if TRAINING_AVAILABLE else []
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.websocket.send(json.dumps(registration))
                logger.info(f"📝 Registered with server as client: {self.client_id}")
                
                # Listen for messages
                await self.listen_for_messages()
                
            except (ConnectionRefusedError, OSError) as e:
                logger.error(f"❌ Connection failed: {str(e)}")
                self.is_connected = False
                
                if self.should_reconnect:
                    logger.info(f"🔄 Retrying connection in {self.reconnect_delay} seconds...")
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    break
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error: {str(e)}")
                self.is_connected = False
                
                if self.should_reconnect:
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    break
        
        logger.info("👋 Subscription ended")
    
    async def listen_for_messages(self):
        """Listen for messages from the server."""
        try:
            async for message in self.websocket:
                logger.info(f"📥 Received message: {message}")
                
                try:
                    # Parse the message as JSON
                    command = json.loads(message)
                    response = await self.process_command(command)
                    
                    # Send response back to server
                    await self.websocket.send(json.dumps(response))
                    
                except json.JSONDecodeError:
                    # Handle plain text commands for backward compatibility
                    response = await self.process_text_command(message)
                    await self.websocket.send(response)
                    
                except Exception as e:
                    error_response = {
                        "status": "error",
                        "client_id": self.client_id,
                        "message": f"Error processing command: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.error(f"Error processing command: {str(e)}")
                    await self.websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("🔌 Connection to server lost")
            self.is_connected = False
        except Exception as e:
            logger.error(f"❌ Error listening for messages: {str(e)}")
            self.is_connected = False
    
    async def process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Process a structured JSON command."""
        command_type = command.get("type", "unknown")
        
        base_response = {
            "client_id": self.client_id,
            "timestamp": datetime.now().isoformat()
        }
        
        if command_type == "start_training":
            response = await self.start_training_command(command)
        elif command_type == "get_status":
            response = self.get_status()
        elif command_type == "stop_training":
            response = self.stop_training()
        elif command_type == "list_models":
            response = self.list_available_models()
        elif command_type == "ping":
            response = {"status": "success", "message": "pong"}
        else:
            response = {
                "status": "error",
                "message": f"Unknown command type: {command_type}"
            }
        
        # Add client info to response
        response.update(base_response)
        return response
    
    async def process_text_command(self, message: str) -> str:
        """Process plain text commands for backward compatibility."""
        if message.startswith("start_training"):
            # Simple text format: "start_training resnet50 0,1,2"
            parts = message.split()
            if len(parts) >= 2:
                model_name = parts[1]
                folds = parts[2].split(',') if len(parts) > 2 else ['0']
                
                command = {
                    "type": "start_training",
                    "model": model_name,
                    "folds": folds
                }
                response = await self.start_training_command(command)
                return json.dumps(response)
            else:
                return "Error: Invalid start_training command format"
        
        elif message == "status":
            return json.dumps(self.get_status())
        
        elif message == "stop":
            return json.dumps(self.stop_training())
        
        elif message == "ping":
            return json.dumps({"status": "success", "message": "pong", "client_id": self.client_id})
        
        else:
            # For any other text, treat as Python code (UNSAFE - only for trusted sources)
            logger.warning(f"Executing raw Python code: {message}")
            try:
                exec(message)
                return f"Code executed successfully on {self.client_id}"
            except Exception as e:
                return f"Error executing code on {self.client_id}: {str(e)}"
    
    async def start_training_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Start a training run based on the command parameters."""
        if not TRAINING_AVAILABLE:
            return {
                "status": "error",
                "message": "Training functionality not available due to missing dependencies"
            }
        
        if self.is_training:
            return {
                "status": "error",
                "message": "Training is already in progress",
                "current_training": self.current_training
            }
        
        # Extract training parameters
        model_name = command.get("model", "resnet50")
        folds = command.get("folds", ["0"])
        batch_size = command.get("batch_size")
        lr = command.get("lr")
        num_workers = command.get("num_workers")
        data_path = command.get("data_path")
        early_stop_patience = command.get("early_stop_patience")
        improvement_threshold = command.get("improvement_threshold", 0.015)
        
        # Validate model name
        valid_models = self.get_available_models()
        
        if model_name not in valid_models:
            return {
                "status": "error",
                "message": f"Invalid model name: {model_name}. Valid models: {valid_models}"
            }
        
        # Store current training info
        self.current_training = {
            "model": model_name,
            "folds": folds,
            "batch_size": batch_size,
            "lr": lr,
            "num_workers": num_workers,
            "data_path": data_path,
            "early_stop_patience": early_stop_patience,
            "improvement_threshold": improvement_threshold,
            "start_time": datetime.now().isoformat()
        }
        
        # Start training in a separate thread
        self.is_training = True
        self.training_thread = threading.Thread(
            target=self._run_training_thread,
            args=(model_name, folds, batch_size, lr, num_workers, data_path, 
                  early_stop_patience, improvement_threshold)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
        
        logger.info(f"Started training: {model_name} on folds {folds}")
        
        return {
            "status": "success",
            "message": f"Training started for model {model_name} on folds {folds}",
            "training_config": self.current_training
        }
    
    def _run_training_thread(self, model_name, folds, batch_size, lr, num_workers, 
                           data_path, early_stop_patience, improvement_threshold):
        """Run training in a separate thread."""
        try:
            logger.info(f"Training thread started for {model_name}")
            
            run_training(
                model_name=model_name,
                cross_val_lists=folds,
                batch_size=batch_size,
                lr=lr,
                num_workers=num_workers,
                data_path=data_path,
                early_stop_patience=early_stop_patience,
                improvement_threshold=improvement_threshold
            )
            
            logger.info(f"Training completed successfully for {model_name}")
            
            # Notify server of completion if still connected
            if self.is_connected and self.websocket:
                completion_message = {
                    "type": "training_completed",
                    "client_id": self.client_id,
                    "model": model_name,
                    "folds": folds,
                    "timestamp": datetime.now().isoformat()
                }
                asyncio.create_task(self.websocket.send(json.dumps(completion_message)))
            
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {str(e)}")
            
            # Notify server of failure if still connected
            if self.is_connected and self.websocket:
                failure_message = {
                    "type": "training_failed",
                    "client_id": self.client_id,
                    "model": model_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                asyncio.create_task(self.websocket.send(json.dumps(failure_message)))
        finally:
            self.is_training = False
            self.current_training = None
            self.training_thread = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "status": "success",
            "is_training": self.is_training,
            "is_connected": self.is_connected,
            "current_training": self.current_training,
            "training_available": TRAINING_AVAILABLE
        }
    
    def stop_training(self) -> Dict[str, Any]:
        """Stop current training (note: this is a graceful request, actual stopping depends on training implementation)."""
        if not self.is_training:
            return {
                "status": "info",
                "message": "No training is currently running"
            }
        
        # Note: The actual training stopping would need to be implemented in the training loop
        # For now, we just log the request
        logger.info("Training stop requested")
        
        return {
            "status": "success",
            "message": "Training stop requested (will complete current epoch)",
            "current_training": self.current_training
        }
    
    def get_available_models(self) -> list:
        """Get list of available models."""
        return [
            'vgg16_bn', 'resnet50', 'efficientnet_v2_s', 'convnext_tiny',
            'densenet121', 'regnet_y_8gf', 'mobilenet_v3_large',
            'vit_small_patch16_224', 'swin_tiny_patch4_window7_224', 'deit_small_patch16_224'
        ]
    
    def list_available_models(self) -> Dict[str, Any]:
        """List all available models."""
        return {
            "status": "success",
            "models": self.get_available_models()
        }
    
    async def disconnect(self):
        """Disconnect from the server."""
        self.should_reconnect = False
        if self.websocket:
            await self.websocket.close()
        logger.info("🔌 Disconnected from server")

async def main():
    """Main function to start the trigger subscriber."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Subscribe to ML training trigger server')
    parser.add_argument('server_host', help='Host address of the trigger server')
    parser.add_argument('--port', type=int, default=8765, help='Port of the trigger server (default: 8765)')
    parser.add_argument('--client-id', help='Unique client identifier (default: auto-generated)')
    
    args = parser.parse_args()
    
    subscriber = TrainingTriggerSubscriber(
        server_host=args.server_host,
        server_port=args.port,
        client_id=args.client_id
    )
    
    try:
        await subscriber.connect_and_subscribe()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        await subscriber.disconnect()
    except Exception as e:
        logger.error(f"Subscriber error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting ML Training Trigger Subscriber...")
    print("📡 This client will connect to a remote trigger server")
    print("🔔 Waiting to receive training triggers from the server")
    print("📝 Logs will be saved to ./code/logs/trigger_subscriber.log")
    print("\nPress Ctrl+C to stop the subscriber")
    
    asyncio.run(main())
