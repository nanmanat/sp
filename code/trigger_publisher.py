#!/usr/bin/env python3
"""
WebSocket Trigger Publisher for ML Training System

This module creates a WebSocket server that manages multiple training clients
and publishes training commands to them. Clients subscribe to this server
and receive training triggers.
"""

import asyncio
import websockets
import json
import logging
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Set, Optional
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./trigger_publisher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingTriggerPublisher:
    """WebSocket server that manages training clients and publishes commands."""

    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.clients: Dict[str, Dict[str, Any]] = {}  # client_id -> client_info
        self.websockets: Dict[str, websockets.WebSocketServerProtocol] = {}  # client_id -> websocket
        self.server = None

    async def register_client(self, websocket: websockets.WebSocketServerProtocol, 
                            client_info: Dict[str, Any]) -> str:
        """Register a new client."""
        client_id = client_info.get("client_id", str(uuid.uuid4()))

        self.clients[client_id] = {
            "websocket": websocket,
            "info": client_info,
            "connected_at": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "address": websocket.remote_address
        }
        self.websockets[client_id] = websocket

        logger.info(f"📝 Client registered: {client_id} from {websocket.remote_address}")
        logger.info(f"👥 Total clients: {len(self.clients)}")

        return client_id

    async def unregister_client(self, client_id: str):
        """Unregister a client."""
        if client_id in self.clients:
            del self.clients[client_id]
        if client_id in self.websockets:
            del self.websockets[client_id]

        logger.info(f"👋 Client unregistered: {client_id}")
        logger.info(f"👥 Total clients: {len(self.clients)}")

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str = ""):
        """Handle incoming WebSocket connections from clients."""
        client_address = websocket.remote_address
        client_id = None

        logger.info(f"🔗 New client connection from {client_address}")

        try:
            # Wait for registration message
            registration_message = await websocket.recv()
            registration = json.loads(registration_message)

            if registration.get("type") == "register":
                client_id = await self.register_client(websocket, registration)

                # Send registration confirmation
                confirmation = {
                    "type": "registration_confirmed",
                    "client_id": client_id,
                    "server_time": datetime.now().isoformat(),
                    "message": "Successfully registered with trigger server"
                }
                await websocket.send(json.dumps(confirmation))

                # Listen for responses from client
                async for message in websocket:
                    try:
                        response = json.loads(message)
                        await self.handle_client_response(client_id, response)

                        # Update last seen
                        if client_id in self.clients:
                            self.clients[client_id]["last_seen"] = datetime.now().isoformat()

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from client {client_id}: {message}")
                    except Exception as e:
                        logger.error(f"Error handling response from {client_id}: {str(e)}")
            else:
                logger.warning(f"Client {client_address} did not send registration message")
                await websocket.close()

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id or client_address} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id or client_address}: {str(e)}")
        finally:
            if client_id:
                await self.unregister_client(client_id)

    async def handle_client_response(self, client_id: str, response: Dict[str, Any]):
        """Handle responses from clients."""
        response_type = response.get("type", "unknown")

        if response_type == "training_completed":
            logger.info(f"✅ Training completed on client {client_id}: {response.get('model')}")
        elif response_type == "training_failed":
            logger.error(f"❌ Training failed on client {client_id}: {response.get('error')}")
        else:
            logger.info(f"📥 Response from {client_id}: {response}")

    async def send_command_to_client(self, client_id: str, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command to a specific client and wait for response."""
        if client_id not in self.websockets:
            return {
                "status": "error",
                "message": f"Client {client_id} not found or not connected"
            }

        websocket = self.websockets[client_id]

        try:
            # Send command
            command_json = json.dumps(command)
            await websocket.send(command_json)
            logger.info(f"📤 Sent command to {client_id}: {command.get('type', 'unknown')}")

            # Wait for response (with timeout)
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            return json.loads(response)

        except asyncio.TimeoutError:
            return {
                "status": "error",
                "message": f"Timeout waiting for response from client {client_id}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error communicating with client {client_id}: {str(e)}"
            }

    async def send_command_to_all_clients(self, command: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Send a command to all connected clients."""
        results = {}

        for client_id in list(self.websockets.keys()):
            try:
                result = await self.send_command_to_client(client_id, command)
                results[client_id] = result
            except Exception as e:
                results[client_id] = {
                    "status": "error",
                    "message": f"Error sending to {client_id}: {str(e)}"
                }

        return results

    async def broadcast_command(self, command: Dict[str, Any]):
        """Broadcast a command to all clients without waiting for responses."""
        command_json = json.dumps(command)
        disconnected_clients = []

        for client_id, websocket in self.websockets.items():
            try:
                await websocket.send(command_json)
                logger.info(f"📡 Broadcasted to {client_id}: {command.get('type', 'unknown')}")
            except Exception as e:
                logger.error(f"Failed to broadcast to {client_id}: {str(e)}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.unregister_client(client_id)

    def get_client_list(self) -> Dict[str, Any]:
        """Get information about all connected clients."""
        client_info = {}

        for client_id, client_data in self.clients.items():
            client_info[client_id] = {
                "connected_at": client_data["connected_at"],
                "last_seen": client_data["last_seen"],
                "address": client_data["address"],
                "capabilities": client_data["info"].get("capabilities", {}),
                "training_available": client_data["info"].get("capabilities", {}).get("training_available", False)
            }

        return {
            "status": "success",
            "total_clients": len(self.clients),
            "clients": client_info,
            "timestamp": datetime.now().isoformat()
        }

    async def start_server(self):
        """Start the WebSocket server."""
        logger.info(f"🚀 Starting trigger publisher server on {self.host}:{self.port}")

        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )

        logger.info(f"📡 Trigger publisher listening on ws://{self.host}:{self.port}")
        logger.info("🔔 Waiting for training clients to subscribe...")

        # Keep the server running
        await self.server.wait_closed()

    def stop_server(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            logger.info("🛑 Trigger publisher server stopped")

class PublisherCLI:
    """Command-line interface for the trigger publisher."""

    def __init__(self, publisher: TrainingTriggerPublisher):
        self.publisher = publisher

    async def interactive_mode(self):
        """Run interactive command mode."""
        print("\n🎮 Publisher Interactive Mode - Available Commands:")
        print("  list                          - List connected clients")
        print("  start <client_id> <model>     - Start training on specific client")
        print("  start_all <model>             - Start training on all clients")
        print("  status <client_id>            - Get status from specific client")
        print("  status_all                    - Get status from all clients")
        print("  broadcast <message>           - Broadcast message to all clients")
        print("  ping <client_id>              - Ping specific client")
        print("  ping_all                      - Ping all clients")
        print("  quit                          - Exit interactive mode")
        print()

        while True:
            try:
                command = input("🔧 Publisher> ").strip()

                if command.lower() in ['quit', 'exit', 'q']:
                    break

                await self.process_command(command)

            except KeyboardInterrupt:
                print("\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    async def process_command(self, command: str):
        """Process a command from the CLI."""
        parts = command.split()

        if not parts:
            return

        cmd = parts[0].lower()

        if cmd == "list":
            client_list = self.publisher.get_client_list()
            self.print_client_list(client_list)

        elif cmd == "start" and len(parts) >= 3:
            client_id = parts[1]
            model = parts[2]
            folds = parts[3].split(',') if len(parts) > 3 else ['0']

            command_obj = {
                "type": "start_training",
                "model": model,
                "folds": folds,
                "timestamp": datetime.now().isoformat()
            }

            response = await self.publisher.send_command_to_client(client_id, command_obj)
            self.print_response(f"Start training on {client_id}", response)

        elif cmd == "start_all" and len(parts) >= 2:
            model = parts[1]
            folds = parts[2].split(',') if len(parts) > 2 else ['0']

            command_obj = {
                "type": "start_training",
                "model": model,
                "folds": folds,
                "timestamp": datetime.now().isoformat()
            }

            responses = await self.publisher.send_command_to_all_clients(command_obj)
            for client_id, response in responses.items():
                self.print_response(f"Start training on {client_id}", response)

        elif cmd == "status" and len(parts) >= 2:
            client_id = parts[1]
            command_obj = {"type": "get_status", "timestamp": datetime.now().isoformat()}

            response = await self.publisher.send_command_to_client(client_id, command_obj)
            self.print_response(f"Status from {client_id}", response)

        elif cmd == "status_all":
            command_obj = {"type": "get_status", "timestamp": datetime.now().isoformat()}

            responses = await self.publisher.send_command_to_all_clients(command_obj)
            for client_id, response in responses.items():
                self.print_response(f"Status from {client_id}", response)

        elif cmd == "ping" and len(parts) >= 2:
            client_id = parts[1]
            command_obj = {"type": "ping", "timestamp": datetime.now().isoformat()}

            response = await self.publisher.send_command_to_client(client_id, command_obj)
            self.print_response(f"Ping {client_id}", response)

        elif cmd == "ping_all":
            command_obj = {"type": "ping", "timestamp": datetime.now().isoformat()}

            responses = await self.publisher.send_command_to_all_clients(command_obj)
            for client_id, response in responses.items():
                self.print_response(f"Ping {client_id}", response)

        elif cmd == "broadcast" and len(parts) >= 2:
            message = " ".join(parts[1:])
            command_obj = {
                "type": "broadcast",
                "message": message,
                "timestamp": datetime.now().isoformat()
            }

            await self.publisher.broadcast_command(command_obj)
            print(f"📡 Broadcasted message to all clients: {message}")

        else:
            print("❌ Unknown command or missing parameters")

    def print_client_list(self, client_list: Dict[str, Any]):
        """Print the client list in a formatted way."""
        if client_list["total_clients"] == 0:
            print("📭 No clients connected")
            return

        print(f"\n👥 Connected Clients ({client_list['total_clients']}):")
        print("-" * 80)

        for client_id, info in client_list["clients"].items():
            training_status = "✅ Available" if info["training_available"] else "❌ Unavailable"
            print(f"🔹 {client_id}")
            print(f"   Address: {info['address']}")
            print(f"   Connected: {info['connected_at']}")
            print(f"   Last Seen: {info['last_seen']}")
            print(f"   Training: {training_status}")
            print()

    def print_response(self, title: str, response: Dict[str, Any]):
        """Print a response in a formatted way."""
        status = response.get("status", "unknown")
        message = response.get("message", "")

        if status == "success":
            print(f"✅ {title}: {message}")
        elif status == "error":
            print(f"❌ {title}: {message}")
        else:
            print(f"📋 {title}: {message}")

        # Print additional details
        if "is_training" in response:
            training_emoji = "🔄" if response["is_training"] else "⏸️"
            print(f"   {training_emoji} Training: {'Active' if response['is_training'] else 'Idle'}")

        if "current_training" in response and response["current_training"]:
            training = response["current_training"]
            print(f"   📊 Model: {training.get('model', 'Unknown')}")
            print(f"   📁 Folds: {training.get('folds', 'Unknown')}")

async def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='ML Training Trigger Publisher Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind server (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8765, help='Port to bind server (default: 8765)')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode after server starts')

    args = parser.parse_args()

    publisher = TrainingTriggerPublisher(args.host, args.port)

    if args.interactive:
        # Start server in background and run interactive mode
        server_task = asyncio.create_task(publisher.start_server())

        # Give server time to start
        await asyncio.sleep(1)

        cli = PublisherCLI(publisher)
        await cli.interactive_mode()

        # Stop server
        publisher.stop_server()
        await server_task
    else:
        # Just run the server
        try:
            await publisher.start_server()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            publisher.stop_server()

if __name__ == "__main__":
    print("🚀 ML Training Trigger Publisher")
    print("=" * 50)
    print("📡 This server manages training clients and publishes commands")
    print("🔔 Clients will subscribe to receive training triggers")
    print("\nPress Ctrl+C to stop the server")

    asyncio.run(main())
