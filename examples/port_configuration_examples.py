#!/usr/bin/env python3
"""
Examples demonstrating different server port configurations
for the WebSocket trigger system.
"""

import asyncio
import sys
import os
import subprocess
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def show_network_info():
    """Show current network configuration."""
    print("🌐 Network Configuration Examples")
    print("=" * 50)
    
    # Get local IP addresses
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"🏠 Hostname: {hostname}")
        print(f"📍 Local IP: {local_ip}")
    except:
        print("🏠 Hostname: Unable to determine")
        print("📍 Local IP: Unable to determine")
    
    print(f"🔗 Localhost: 127.0.0.1")
    print()

def show_configuration_examples():
    """Show different server configuration examples."""
    print("🔧 Server Port Configuration Examples")
    print("=" * 50)
    
    examples = [
        {
            "name": "Development (Localhost Only)",
            "description": "Most secure - only local connections allowed",
            "publisher": "python3 run_trigger_publisher.py --host 127.0.0.1 --port 8765",
            "subscriber": "python3 run_trigger_subscriber.py 127.0.0.1 --port 8765",
            "security": "🔒 High",
            "use_case": "Development, testing, same-machine"
        },
        {
            "name": "Local Network",
            "description": "Medium security - local network access",
            "publisher": "python3 run_trigger_publisher.py --host 192.168.1.100 --port 8765",
            "subscriber": "python3 run_trigger_subscriber.py 192.168.1.100 --port 8765",
            "security": "🔐 Medium", 
            "use_case": "Office networks, private cloud"
        },
        {
            "name": "Public Access",
            "description": "Accessible from internet - requires security measures",
            "publisher": "python3 run_trigger_publisher.py --host 0.0.0.0 --port 8765",
            "subscriber": "python3 run_trigger_subscriber.py your-server.com --port 8765",
            "security": "⚠️ Low (needs firewall/VPN)",
            "use_case": "Cloud deployments, remote access"
        },
        {
            "name": "Custom Port",
            "description": "Non-standard port to avoid conflicts",
            "publisher": "python3 run_trigger_publisher.py --host 0.0.0.0 --port 9000",
            "subscriber": "python3 run_trigger_subscriber.py server-ip --port 9000",
            "security": "🔐 Configurable",
            "use_case": "Avoid port conflicts, multiple environments"
        },
        {
            "name": "Interactive Mode",
            "description": "Publisher with interactive control interface",
            "publisher": "python3 run_trigger_publisher.py --host 0.0.0.0 --port 8765 --interactive",
            "subscriber": "python3 run_trigger_subscriber.py server-ip --client-id gpu-node-1",
            "security": "🔐 Configurable",
            "use_case": "Manual control, testing, monitoring"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"📋 Example {i}: {example['name']}")
        print(f"   Description: {example['description']}")
        print(f"   Security: {example['security']}")
        print(f"   Use Case: {example['use_case']}")
        print(f"   Publisher: {example['publisher']}")
        print(f"   Subscriber: {example['subscriber']}")
        print()

def show_firewall_commands():
    """Show firewall configuration commands."""
    print("🛡️ Firewall Configuration")
    print("=" * 30)
    
    print("🐧 Linux (ufw):")
    print("   sudo ufw allow 8765")
    print("   sudo ufw allow from 192.168.1.0/24 to any port 8765")
    print()
    
    print("🐧 Linux (iptables):")
    print("   sudo iptables -A INPUT -p tcp --dport 8765 -j ACCEPT")
    print("   sudo iptables -A INPUT -p tcp -s 192.168.1.0/24 --dport 8765 -j ACCEPT")
    print()
    
    print("🪟 Windows:")
    print('   netsh advfirewall firewall add rule name="ML Trigger" dir=in action=allow protocol=TCP localport=8765')
    print()
    
    print("🍎 macOS:")
    print("   # Use System Preferences > Security & Privacy > Firewall")
    print("   # Or configure pf rules in /etc/pf.conf")
    print()

def show_testing_commands():
    """Show commands for testing connectivity."""
    print("🧪 Testing Connectivity")
    print("=" * 25)
    
    print("📡 Check if port is listening:")
    print("   netstat -an | grep 8765")
    print("   lsof -i :8765")
    print()
    
    print("🔌 Test connection:")
    print("   telnet localhost 8765")
    print("   nc -zv localhost 8765")
    print()
    
    print("🌐 Test from remote machine:")
    print("   ping 192.168.1.100")
    print("   telnet 192.168.1.100 8765")
    print("   nc -zv 192.168.1.100 8765")
    print()

def demonstrate_port_check():
    """Demonstrate checking if a port is available."""
    print("🔍 Port Availability Check")
    print("=" * 30)
    
    import socket
    
    def check_port(host, port):
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((host, port))
                return result != 0  # 0 means connection successful (port in use)
        except:
            return False
    
    test_ports = [8765, 8766, 9000, 3000, 8080]
    
    for port in test_ports:
        available = check_port('localhost', port)
        status = "✅ Available" if available else "❌ In Use"
        print(f"   Port {port}: {status}")
    
    print()

async def run_quick_demo():
    """Run a quick demonstration of port configuration."""
    print("🚀 Quick Port Configuration Demo")
    print("=" * 40)
    
    # Import the publisher
    from code.trigger_publisher import TrainingTriggerPublisher
    
    # Test different port configurations
    configs = [
        {"host": "localhost", "port": 8766, "name": "Localhost"},
        {"host": "127.0.0.1", "port": 8767, "name": "Loopback"},
    ]
    
    for config in configs:
        print(f"🔧 Testing {config['name']} configuration...")
        print(f"   Host: {config['host']}, Port: {config['port']}")
        
        try:
            publisher = TrainingTriggerPublisher(config['host'], config['port'])
            
            # Start server briefly
            server_task = asyncio.create_task(publisher.start_server())
            await asyncio.sleep(0.5)  # Let it start
            
            print(f"   ✅ Server started successfully on {config['host']}:{config['port']}")
            
            # Stop server
            publisher.stop_server()
            server_task.cancel()
            
            try:
                await server_task
            except asyncio.CancelledError:
                pass
                
        except Exception as e:
            print(f"   ❌ Failed to start server: {str(e)}")
        
        print()

def main():
    """Main function to run the port configuration examples."""
    print("🌐 WebSocket Server Port Configuration Guide")
    print("=" * 60)
    print()
    
    show_network_info()
    show_configuration_examples()
    show_firewall_commands()
    show_testing_commands()
    demonstrate_port_check()
    
    print("🎯 Quick Demo")
    print("=" * 15)
    print("Running quick server configuration test...")
    print()
    
    try:
        asyncio.run(run_quick_demo())
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
    
    print("🎉 Port Configuration Guide Complete!")
    print()
    print("💡 Key Takeaways:")
    print("   ✅ Use --host and --port to configure server binding")
    print("   ✅ localhost (127.0.0.1) for development/testing")
    print("   ✅ Private IP (192.168.x.x) for local network")
    print("   ✅ 0.0.0.0 for public access (with proper security)")
    print("   ✅ Custom ports to avoid conflicts")
    print("   ✅ Configure firewall rules as needed")

if __name__ == "__main__":
    main()
