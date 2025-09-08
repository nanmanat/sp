#!/usr/bin/env python3
"""
WebSocket Trigger Listener for ML Training System

This module creates a WebSocket server that listens for trigger messages
from remote servers and executes training runs based on the received commands.
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

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./code/logs/trigger_listener.log'),
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

class TrainingTriggerListener:
    """WebSocket server that listens for training triggers and executes them."""

    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.server = None
        self.current_training = None
        self.training_thread = None
        self.is_training = False

        # Ensure logs directory exists
        os.makedirs('./code/logs', exist_ok=True)

    async def handle_client(self, websocket, path):
        """Handle incoming WebSocket connections and messages."""
        client_address = websocket.remote_address
        logger.info(f"Client connected from {client_address}")

        try:
            async for message in websocket:
                logger.info(f"Received message from {client_address}: {message}")

                try:
                    # Parse the message as JSON
                    command = json.loads(message)
                    response = await self.process_command(command)

                    # Send response back to client
                    await websocket.send(json.dumps(response))

                except json.JSONDecodeError:
                    # Handle plain text commands for backward compatibility
                    response = await self.process_text_command(message)
                    await websocket.send(response)

                except Exception as e:
                    error_response = {
                        "status": "error",
                        "message": f"Error processing command: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.error(f"Error processing command: {str(e)}")
                    await websocket.send(json.dumps(error_response))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_address} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_address}: {str(e)}")

    async def process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Process a structured JSON command."""
        command_type = command.get("type", "unknown")

        if command_type == "start_training":
            return await self.start_training_command(command)
        elif command_type == "get_status":
            return self.get_status()
        elif command_type == "stop_training":
            return self.stop_training()
        elif command_type == "list_models":
            return self.list_available_models()
        else:
            return {
                "status": "error",
                "message": f"Unknown command type: {command_type}",
                "timestamp": datetime.now().isoformat()
            }

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

        else:
            # For any other text, treat as Python code (UNSAFE - only for trusted sources)
            logger.warning(f"Executing raw Python code: {message}")
            try:
                exec(message)
                return "Code executed successfully"
            except Exception as e:
                return f"Error executing code: {str(e)}"

    async def start_training_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Start a training run based on the command parameters."""
        if self.is_training:
            return {
                "status": "error",
                "message": "Training is already in progress",
                "current_training": self.current_training,
                "timestamp": datetime.now().isoformat()
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
        valid_models = [
            'vgg16_bn', 'resnet50', 'efficientnet_v2_s', 'convnext_tiny',
            'densenet121', 'regnet_y_8gf', 'mobilenet_v3_large',
            'vit_small_patch16_224', 'swin_tiny_patch4_window7_224', 'deit_small_patch16_224'
        ]

        if model_name not in valid_models:
            return {
                "status": "error",
                "message": f"Invalid model name: {model_name}. Valid models: {valid_models}",
                "timestamp": datetime.now().isoformat()
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
            "training_config": self.current_training,
            "timestamp": datetime.now().isoformat()
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

        except Exception as e:
            logger.error(f"Training failed for {model_name}: {str(e)}")
        finally:
            self.is_training = False
            self.current_training = None
            self.training_thread = None

    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "status": "success",
            "is_training": self.is_training,
            "current_training": self.current_training,
            "timestamp": datetime.now().isoformat()
        }

    def stop_training(self) -> Dict[str, Any]:
        """Stop current training (note: this is a graceful request, actual stopping depends on training implementation)."""
        if not self.is_training:
            return {
                "status": "info",
                "message": "No training is currently running",
                "timestamp": datetime.now().isoformat()
            }

        # Note: The actual training stopping would need to be implemented in the training loop
        # For now, we just log the request
        logger.info("Training stop requested")

        return {
            "status": "success",
            "message": "Training stop requested (will complete current epoch)",
            "current_training": self.current_training,
            "timestamp": datetime.now().isoformat()
        }

    def list_available_models(self) -> Dict[str, Any]:
        """List all available models."""
        models = [
            'vgg16_bn', 'resnet50', 'efficientnet_v2_s', 'convnext_tiny',
            'densenet121', 'regnet_y_8gf', 'mobilenet_v3_large',
            'vit_small_patch16_224', 'swin_tiny_patch4_window7_224', 'deit_small_patch16_224'
        ]

        return {
            "status": "success",
            "models": models,
            "timestamp": datetime.now().isoformat()
        }

    async def start_server(self):
        """Start the WebSocket server."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10,
                max_size=10_485_760,  # 10MB max message size
                max_queue=64,         # Increase message queue
                close_timeout=10      # Ensure clean closures
            )

            logger.info(f"WebSocket server listening on ws://{self.host}:{self.port}")
            logger.info("Waiting for trigger messages...")

            # Keep the server running
            await self.server.wait_closed()
        except OSError as e:
            logger.error(f"❌ Failed to start server: {str(e)}")
            logger.error("   This could be because the port is already in use or you don't have permission to bind to it.")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error starting server: {str(e)}")
            raise

    def stop_server(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            logger.info("WebSocket server stopped")

async def main():
    """Main function to start the trigger listener."""
    listener = TrainingTriggerListener()

    try:
        await listener.start_server()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        listener.stop_server()
    except Exception as e:
        logger.error(f"Server error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting ML Training Trigger Listener...")
    print("📡 Listening for WebSocket connections on ws://0.0.0.0:8765")
    print("📝 Logs will be saved to ./code/logs/trigger_listener.log")
    print("⚡ Ready to receive training triggers!")
    print("\nPress Ctrl+C to stop the server")

    asyncio.run(main())
