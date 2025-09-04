# 🌐 Server Port Configuration Guide

## 📡 Exposing Server Ports in WebSocket Trigger System

Both models support flexible port configuration. Here's how to expose and configure server ports for different scenarios:

---

## 🔧 Model 1: Server Model (Original)

### Client Exposes Port (Training Machine)
```bash
# Default port 8765 on all interfaces
python3 run_trigger_listener.py

# Custom port
python3 code/trigger_listener.py --port 9000

# Specific interface (more secure)
python3 code/trigger_listener.py --host 192.168.1.100 --port 8765

# Localhost only (most secure)
python3 code/trigger_listener.py --host 127.0.0.1 --port 8765
```

### Remote Machine Connects
```bash
# Connect to custom port
python3 code/trigger_sender.py 192.168.1.100:9000 start resnet50

# Standard connection
python3 code/trigger_sender.py 192.168.1.100 start resnet50
```

---

## 🔔 Model 2: Subscription Model (New)

### Publisher Server Exposes Port (Central Server)
```bash
# Default: All interfaces, port 8765
python3 run_trigger_publisher.py

# Custom port
python3 run_trigger_publisher.py --port 9000

# Specific interface
python3 run_trigger_publisher.py --host 192.168.1.50 --port 8765

# Public interface (accessible from internet)
python3 run_trigger_publisher.py --host 0.0.0.0 --port 8765

# Interactive mode with custom settings
python3 run_trigger_publisher.py --host 192.168.1.50 --port 9000 --interactive
```

### Subscribers Connect to Custom Port
```bash
# Connect to custom port
python3 run_trigger_subscriber.py 192.168.1.50 --port 9000

# With client ID
python3 run_trigger_subscriber.py 192.168.1.50 --port 9000 --client-id gpu-node-1
```

---

## 🌍 Network Configuration Scenarios

### 1. **Local Network Setup**
```bash
# Publisher on local network
python3 run_trigger_publisher.py --host 192.168.1.50 --port 8765

# Subscribers from other machines on same network
python3 run_trigger_subscriber.py 192.168.1.50 --port 8765 --client-id office-gpu-1
python3 run_trigger_subscriber.py 192.168.1.50 --port 8765 --client-id office-gpu-2
```

### 2. **Internet/Cloud Setup**
```bash
# Publisher on cloud server (public IP)
python3 run_trigger_publisher.py --host 0.0.0.0 --port 8765

# Subscribers from anywhere
python3 run_trigger_subscriber.py your-server.com --port 8765 --client-id home-gpu
python3 run_trigger_subscriber.py 203.0.113.1 --port 8765 --client-id cloud-gpu
```

### 3. **Multiple Environments**
```bash
# Development environment
python3 run_trigger_publisher.py --host localhost --port 8766

# Staging environment  
python3 run_trigger_publisher.py --host 0.0.0.0 --port 8767

# Production environment
python3 run_trigger_publisher.py --host 0.0.0.0 --port 8765
```

### 4. **Behind Reverse Proxy**
```bash
# Publisher behind nginx/apache
python3 run_trigger_publisher.py --host 127.0.0.1 --port 8080

# Clients connect through proxy
python3 run_trigger_subscriber.py your-domain.com --port 443 --client-id secure-client
```

---

## 🔒 Security Considerations by Interface

### 🏠 **Localhost (127.0.0.1)**
- **Security**: Highest - only local connections
- **Use Case**: Development, testing, same-machine communication
```bash
python3 run_trigger_publisher.py --host 127.0.0.1 --port 8765
```

### 🏢 **Private Network (192.168.x.x)**
- **Security**: Medium - local network only
- **Use Case**: Office networks, private cloud
```bash
python3 run_trigger_publisher.py --host 192.168.1.50 --port 8765
```

### 🌍 **All Interfaces (0.0.0.0)**
- **Security**: Lowest - accessible from anywhere
- **Use Case**: Public cloud, internet-accessible services
- **⚠️ Requires**: Firewall rules, authentication, SSL/TLS
```bash
python3 run_trigger_publisher.py --host 0.0.0.0 --port 8765
```

---

## 🛠️ Advanced Port Configuration

### Custom Port Ranges
```bash
# Use non-standard ports to avoid conflicts
python3 run_trigger_publisher.py --port 9001  # Custom port
python3 run_trigger_publisher.py --port 8080  # HTTP alternative
python3 run_trigger_publisher.py --port 3000  # Development port
```

### Multiple Publishers
```bash
# Run multiple publishers on different ports
python3 run_trigger_publisher.py --port 8765 &  # Production
python3 run_trigger_publisher.py --port 8766 &  # Staging  
python3 run_trigger_publisher.py --port 8767 &  # Development
```

### Port Forwarding Setup
```bash
# SSH tunnel for secure remote access
ssh -L 8765:localhost:8765 user@remote-server

# Then connect locally
python3 run_trigger_subscriber.py localhost --port 8765
```

---

## 🔧 Firewall Configuration

### Linux (iptables)
```bash
# Allow specific port
sudo iptables -A INPUT -p tcp --dport 8765 -j ACCEPT

# Allow from specific network
sudo iptables -A INPUT -p tcp -s 192.168.1.0/24 --dport 8765 -j ACCEPT
```

### Linux (ufw)
```bash
# Simple port opening
sudo ufw allow 8765

# From specific IP
sudo ufw allow from 192.168.1.0/24 to any port 8765
```

### macOS
```bash
# Check if port is blocked
sudo pfctl -sr | grep 8765

# Add rule to allow port (requires pf configuration)
```

### Windows
```powershell
# Allow port through Windows Firewall
netsh advfirewall firewall add rule name="WebSocket Trigger" dir=in action=allow protocol=TCP localport=8765
```

---

## 📊 Monitoring and Diagnostics

### Check Port Status
```bash
# Check if port is listening
netstat -an | grep 8765
lsof -i :8765

# Test connection
telnet localhost 8765
nc -zv localhost 8765
```

### Network Diagnostics
```bash
# Test from remote machine
ping 192.168.1.50
telnet 192.168.1.50 8765

# Check routing
traceroute 192.168.1.50
```

---

## 🎯 Configuration Examples

### Example 1: Office Network
```bash
# Central server (office server)
python3 run_trigger_publisher.py --host 192.168.1.100 --port 8765 --interactive

# Training machines (office workstations)
python3 run_trigger_subscriber.py 192.168.1.100 --client-id workstation-1
python3 run_trigger_subscriber.py 192.168.1.100 --client-id workstation-2
```

### Example 2: Cloud Deployment
```bash
# Cloud server (AWS/GCP/Azure)
python3 run_trigger_publisher.py --host 0.0.0.0 --port 8765

# Remote clients (home/office)
python3 run_trigger_subscriber.py your-cloud-server.com --client-id home-gpu
python3 run_trigger_subscriber.py your-cloud-server.com --client-id office-cluster
```

### Example 3: Hybrid Setup
```bash
# Publisher on cloud
python3 run_trigger_publisher.py --host 0.0.0.0 --port 8765

# Some clients on local network
python3 run_trigger_subscriber.py cloud-server.com --client-id local-gpu-1

# Some clients on other networks
python3 run_trigger_subscriber.py cloud-server.com --client-id remote-gpu-1
```

---

## 🚀 Quick Setup Commands

### Development Setup
```bash
# Start publisher (development)
python3 run_trigger_publisher.py --host localhost --port 8766 --interactive

# Connect subscriber (same machine)
python3 run_trigger_subscriber.py localhost --port 8766 --client-id dev-client
```

### Production Setup
```bash
# Start publisher (production)
python3 run_trigger_publisher.py --host 0.0.0.0 --port 8765

# Connect subscribers (remote machines)
python3 run_trigger_subscriber.py production-server.com --client-id prod-gpu-1
```

---

## 💡 Best Practices

### 🔒 **Security**
1. Use specific host IPs instead of 0.0.0.0 when possible
2. Implement firewall rules to restrict access
3. Consider VPN for internet-accessible deployments
4. Use non-standard ports to reduce attack surface

### 📊 **Performance**
1. Use local network addresses for better performance
2. Avoid unnecessary port forwarding
3. Monitor network latency for remote connections

### 🛠️ **Management**
1. Document your port assignments
2. Use consistent port numbers across environments
3. Test connectivity before deploying clients
4. Monitor port usage and conflicts

The system is designed to be flexible - you can expose ports on any interface and use any available port number to fit your specific network requirements!
