# 🚀 WebSocket Trigger System for ML Training

This system allows remote servers to trigger machine learning training runs on your client machine via WebSocket connections.

## 📋 Overview

- **Client (Training Machine)**: Runs a WebSocket server, waits for triggers, executes training
- **Server (Remote Machine)**: Connects to client, sends training commands

## 🛠️ Setup

### Prerequisites
- Python 3.x
- `websockets` library (already installed)

### Files Created
- `code/trigger_listener.py` - WebSocket server for the client
- `code/trigger_sender.py` - WebSocket client for sending triggers
- `run_trigger_listener.py` - Launcher script for the listener

## 🎯 Usage

### 1. Start the Client (Training Machine)

```bash
# Start the WebSocket listener
python3 run_trigger_listener.py
```

This will:
- Start a WebSocket server on `ws://0.0.0.0:8765`
- Listen for incoming training triggers
- Log activities to `./code/logs/trigger_listener.log`

### 2. Send Triggers from Server (Remote Machine)

#### Option A: Command Line Interface

```bash
# Start training with ResNet50 on fold 0
python3 code/trigger_sender.py <client_ip> start resnet50

# Start training with multiple folds and custom parameters
python3 code/trigger_sender.py <client_ip> start resnet50 --folds 0,1,2 --batch-size 64 --lr 0.001

# Get current training status
python3 code/trigger_sender.py <client_ip> status

# Stop current training
python3 code/trigger_sender.py <client_ip> stop

# List available models
python3 code/trigger_sender.py <client_ip> models
```

#### Option B: Interactive Mode

```bash
python3 code/trigger_sender.py <client_ip> interactive
```

Then use commands like:
- `start resnet50 0,1,2`
- `status`
- `stop`
- `models`
- `quit`

#### Option C: Programmatic Usage

```python
import asyncio
from code.trigger_sender import TrainingTriggerSender

async def trigger_training():
    sender = TrainingTriggerSender("192.168.1.100")  # Client IP
    
    # Start training
    response = await sender.start_training(
        model="resnet50",
        folds=["0", "1", "2"],
        batch_size=64,
        lr=0.001
    )
    print(response)
    
    # Check status
    status = await sender.get_status()
    print(status)

asyncio.run(trigger_training())
```

## 📊 Available Models

- `vgg16_bn`
- `resnet50`
- `efficientnet_v2_s`
- `convnext_tiny`
- `densenet121`
- `regnet_y_8gf`
- `mobilenet_v3_large`
- `vit_small_patch16_224`
- `swin_tiny_patch4_window7_224`
- `deit_small_patch16_224`

## 🔧 Command Format

### JSON Command Structure

```json
{
  "type": "start_training",
  "model": "resnet50",
  "folds": ["0", "1", "2"],
  "batch_size": 64,
  "lr": 0.001,
  "num_workers": 12,
  "data_path": "./code/Classification/JPEGImages",
  "early_stop_patience": 10,
  "improvement_threshold": 0.015
}
```

### Response Format

```json
{
  "status": "success",
  "message": "Training started for model resnet50 on folds ['0', '1', '2']",
  "training_config": {
    "model": "resnet50",
    "folds": ["0", "1", "2"],
    "start_time": "2025-01-27T10:30:00"
  },
  "timestamp": "2025-01-27T10:30:00"
}
```

## 🔒 Security Notes

⚠️ **Important Security Considerations:**

1. **Network Security**: The WebSocket server listens on `0.0.0.0:8765` (all interfaces)
2. **Firewall**: Consider restricting access to trusted IPs only
3. **Authentication**: Current implementation has no authentication
4. **Code Execution**: Avoid using the raw text command mode in production

### Production Recommendations

1. **Use a reverse proxy** (nginx) with SSL/TLS
2. **Implement authentication** (API keys, JWT tokens)
3. **Restrict network access** (VPN, firewall rules)
4. **Monitor logs** for suspicious activity

## 📝 Logs and Monitoring

- **Trigger Listener Logs**: `./code/logs/trigger_listener.log`
- **Training Logs**: `./code/logs/` (CSV files per training run)
- **Console Output**: Real-time status updates

## 🐛 Troubleshooting

### Connection Issues

```bash
# Check if the listener is running
netstat -an | grep 8765

# Test connection locally
python3 code/trigger_sender.py localhost status
```

### Common Errors

1. **"Connection refused"**: Listener not running or wrong IP/port
2. **"Invalid model name"**: Use `models` command to see available models
3. **"Training already in progress"**: Wait for current training to finish or stop it

## 🔄 Integration with Existing GUI

The trigger system works alongside your existing GUI (`run_gui.py`). You can:

1. Run the GUI for local training management
2. Run the trigger listener for remote training triggers
3. Both can operate simultaneously

## 📈 Example Workflow

1. **Start the listener** on your training machine:
   ```bash
   python3 run_trigger_listener.py
   ```

2. **From a remote server**, trigger training:
   ```bash
   python3 code/trigger_sender.py 192.168.1.100 start resnet50 --folds 0,1,2,3,4
   ```

3. **Monitor progress** remotely:
   ```bash
   python3 code/trigger_sender.py 192.168.1.100 status
   ```

4. **Training completes** automatically with logs saved to CSV files

## 🎉 Benefits

- ✅ **Remote Training**: Trigger training from anywhere on the network
- ✅ **Automation**: Integrate with scripts, cron jobs, or CI/CD pipelines
- ✅ **Monitoring**: Real-time status updates and comprehensive logging
- ✅ **Flexibility**: Support for all training parameters and models
- ✅ **Compatibility**: Works alongside existing GUI and training scripts
