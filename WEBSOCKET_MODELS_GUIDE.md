# 🚀 WebSocket Trigger System - Two Models

This system provides **two different connection models** for remote ML training triggers:

## 🔄 Model Comparison

| Feature | **Server Model** | **Subscription Model** |
|---------|------------------|------------------------|
| **Connection Direction** | Server → Client | Client → Server |
| **Client Role** | Hosts WebSocket server | Subscribes to server |
| **Server Role** | Connects to send commands | Manages multiple clients |
| **Network Requirements** | Client needs open port | Server needs open port |
| **Scalability** | One-to-one | One-to-many |
| **Use Case** | Direct control | Centralized management |

---

## 📡 Model 1: Server Model (Original)

**Client hosts server, remote machine connects to send commands**

### Architecture
```
┌─────────────────┐         ┌─────────────────┐
│   Remote        │ ──────► │   Training      │
│   (sender)      │ WebSocket  │   (listener)    │
│ Sends commands  │         │ Hosts server    │
└─────────────────┘         └─────────────────┘
```

### Files
- `code/trigger_listener.py` - WebSocket server (client-side)
- `code/trigger_sender.py` - WebSocket client (server-side)
- `run_trigger_listener.py` - Start listener

### Usage
```bash
# Start client (training machine)
python3 run_trigger_listener.py

# Send commands (remote machine)
python3 code/trigger_sender.py <client_ip> start resnet50
```

---

## 🔔 Model 2: Subscription Model (New)

**Client subscribes to server, server publishes commands to multiple clients**

### Architecture
```
┌─────────────────┐         ┌─────────────────┐
│   Central       │ ◄────── │   Training      │
│   (publisher)   │ WebSocket  │ (subscriber)    │
│ Manages clients │         │ Subscribes      │
└─────────────────┘         └─────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│   Training      │         │   Training      │
│ (subscriber)    │         │ (subscriber)    │
│ Client 2        │         │ Client 3        │
└─────────────────┘         └─────────────────┘
```

### Files
- `code/trigger_publisher.py` - WebSocket server (central management)
- `code/trigger_subscriber.py` - WebSocket client (training machines)
- `run_trigger_publisher.py` - Start publisher server
- `run_trigger_subscriber.py` - Start subscriber client

### Usage
```bash
# Start central server
python3 run_trigger_publisher.py --interactive

# Subscribe clients (on each training machine)
python3 run_trigger_subscriber.py <server_ip>

# Use interactive commands on server
Publisher> list                    # List connected clients
Publisher> start client_123 resnet50  # Start training on specific client
Publisher> start_all resnet50      # Start training on all clients
```

---

## 🎯 When to Use Each Model

### 🔧 Use Server Model When:
- ✅ **Direct Control**: You want to directly control one training machine
- ✅ **Simple Setup**: One-to-one relationship is sufficient
- ✅ **Firewall Friendly**: Training machine can open ports
- ✅ **Script Integration**: Easy to integrate with existing scripts

### 🏢 Use Subscription Model When:
- ✅ **Multiple Clients**: You need to manage multiple training machines
- ✅ **Centralized Control**: You want a central command center
- ✅ **Scalability**: You plan to add more training machines
- ✅ **Network Restrictions**: Training machines can't host servers
- ✅ **Monitoring**: You want to see all clients in one place

---

## 🚀 Quick Start Guide

### Server Model (Original)
```bash
# Training Machine
python3 run_trigger_listener.py

# Remote Machine  
python3 code/trigger_sender.py 192.168.1.100 start resnet50
```

### Subscription Model (New)
```bash
# Central Server
python3 run_trigger_publisher.py --interactive

# Training Machine 1
python3 run_trigger_subscriber.py 192.168.1.50

# Training Machine 2  
python3 run_trigger_subscriber.py 192.168.1.50 --client-id gpu-server-2

# Use publisher interactive mode to control all clients
```

---

## 📊 Detailed Usage Examples

### Subscription Model Interactive Commands

```bash
# Start the publisher with interactive mode
python3 run_trigger_publisher.py --interactive

# Available commands in interactive mode:
Publisher> list                           # Show all connected clients
Publisher> start client_123 resnet50      # Start training on specific client
Publisher> start client_123 resnet50 0,1,2  # With specific folds
Publisher> start_all efficientnet_v2_s   # Start same model on all clients
Publisher> status client_123              # Get status from specific client
Publisher> status_all                     # Get status from all clients
Publisher> ping client_123                # Test connection to client
Publisher> ping_all                       # Test all connections
Publisher> broadcast "Hello all clients"  # Send message to all
Publisher> quit                           # Exit interactive mode
```

### Programmatic Usage (Subscription Model)

```python
# Publisher side
from code.trigger_publisher import TrainingTriggerPublisher

async def manage_training():
    publisher = TrainingTriggerPublisher()
    
    # Start training on specific client
    response = await publisher.send_command_to_client("client_123", {
        "type": "start_training",
        "model": "resnet50",
        "folds": ["0", "1", "2"]
    })
    
    # Start training on all clients
    responses = await publisher.send_command_to_all_clients({
        "type": "start_training", 
        "model": "efficientnet_v2_s"
    })
```

---

## 🔒 Security Considerations

### Server Model
- Training machine exposes port 8765
- Direct connection from remote machines
- Consider firewall rules and VPN access

### Subscription Model  
- Central server exposes port 8765
- Training machines connect outbound only
- Better for restrictive network environments
- Central point for access control

---

## 🎉 Benefits of Each Model

### Server Model Benefits
- ✅ **Simple**: Direct point-to-point communication
- ✅ **Low Latency**: No intermediate server
- ✅ **Independent**: Each client operates independently
- ✅ **Existing Scripts**: Easy to integrate with current workflows

### Subscription Model Benefits
- ✅ **Scalable**: Manage unlimited training machines
- ✅ **Centralized**: Single point of control and monitoring
- ✅ **Network Friendly**: Clients make outbound connections only
- ✅ **Coordinated**: Synchronize training across multiple machines
- ✅ **Monitoring**: Real-time view of all training activities
- ✅ **Broadcasting**: Send commands to multiple clients simultaneously

---

## 🔧 Configuration Options

### Subscription Model Advanced Usage

```bash
# Custom client ID and server details
python3 run_trigger_subscriber.py 192.168.1.50 --port 9000 --client-id gpu-cluster-node-1

# Publisher with custom binding
python3 run_trigger_publisher.py --host 0.0.0.0 --port 9000 --interactive

# Non-interactive publisher (for production)
python3 run_trigger_publisher.py --host 192.168.1.50 --port 8765
```

---

## 🎯 Choose Your Model

Both models are fully implemented and ready to use. Choose based on your specific needs:

- **Start with Server Model** if you have one training machine and want simple control
- **Upgrade to Subscription Model** when you need to manage multiple training machines or want centralized control

Both models can coexist and you can migrate between them as your needs evolve!
