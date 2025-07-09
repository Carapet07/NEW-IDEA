"""
Unity-Python Communication Bridge
Enhanced socket-based communication between Unity and Python with robust error handling
"""

import socket
import json
import threading
import numpy as np
from typing import Dict, Any, List, Optional
import time
import logging
from contextlib import contextmanager


class UnityBridge:
    """
    Enhanced socket-based communication bridge between Unity and Python ML training.
    
    Features:
    - Robust error handling and connection recovery
    - Configurable timeouts and retry mechanisms
    - Comprehensive logging
    - Connection health monitoring
    - Graceful shutdown handling
    """
    
    def __init__(self, host: str = 'localhost', port: int = 9999, 
                 max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Unity bridge with enhanced configuration.
        
        Args:
            host: Host address for socket server
            port: Port number for socket communication
            max_retries: Maximum number of retry attempts for operations
            retry_delay: Delay between retry attempts in seconds
        """
        self.host: str = host
        self.port: int = port
        self.max_retries: int = max_retries
        self.retry_delay: float = retry_delay
        
        # Connection state
        self.socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.is_connected: bool = False
        self.is_server_running: bool = False
        
        # Connection monitoring
        self.connection_start_time: Optional[float] = None
        self.last_successful_communication: Optional[float] = None
        self.failed_communications: int = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def start_server(self) -> bool:
        """
        Start the Python server to accept Unity connections with enhanced error handling.
        
        Returns:
            True if server started successfully, False otherwise
        """
        try:
            # Clean up any existing socket
            self._cleanup_socket()
            
            # Create new socket with reuse address option
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Set socket options for better performance
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Bind and listen
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.is_server_running = True
            
            self.logger.info(f"🚀 Python server started on {self.host}:{self.port}")
            self.logger.info("⏳ Waiting for Unity connection...")
            
            return True
            
        except socket.error as e:
            if e.errno == 48:  # Address already in use
                self.logger.error(f"❌ Port {self.port} is already in use. "
                                "Try a different port or check if another instance is running.")
            else:
                self.logger.error(f"❌ Socket error starting server: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error starting server: {e}")
            return False
    
    def wait_for_unity(self, timeout: float = 30.0) -> bool:
        """
        Wait for Unity to connect with enhanced timeout and error handling.
        
        Args:
            timeout: Maximum time to wait for connection in seconds
            
        Returns:
            True if Unity connected successfully, False otherwise
        """
        if not self.is_server_running or not self.socket:
            self.logger.error("❌ Server not running. Call start_server() first.")
            return False
        
        try:
            self.socket.settimeout(timeout)
            self.client_socket, addr = self.socket.accept()
            
            # Configure client socket
            self.client_socket.settimeout(5.0)  # 5 second timeout for operations
            self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            self.is_connected = True
            self.connection_start_time = time.time()
            self.last_successful_communication = time.time()
            self.failed_communications = 0
            
            self.logger.info(f"✅ Unity connected from {addr}")
            self.logger.info(f"🔗 Connection established at {time.strftime('%H:%M:%S')}")
            
            # Test the connection with a ping
            if self._test_connection():
                return True
            else:
                self.logger.warning("⚠️ Connection test failed, but proceeding...")
                return True
                
        except socket.timeout:
            self.logger.error(f"⏰ Timeout waiting for Unity connection ({timeout}s)")
            self.logger.info("💡 Make sure Unity is running and the escape cage scene is loaded")
            return False
        except socket.error as e:
            self.logger.error(f"❌ Socket error during connection: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected connection error: {e}")
            return False
    
    def _test_connection(self) -> bool:
        """Test the connection by sending a ping message."""
        try:
            test_message = "ping\n"
            self.client_socket.send(test_message.encode('utf-8'))
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Connection test failed: {e}")
            return False
    
    def send_action(self, action: int) -> bool:
        """
        Send action to Unity with retry mechanism and error handling.
        
        Args:
            action: Action number to send (0-3 for movement directions)
            
        Returns:
            True if action sent successfully, False otherwise
        """
        if not self._check_connection():
            return False
        
        for attempt in range(self.max_retries):
            try:
                # Validate action
                if not isinstance(action, int) or action < 0 or action > 3:
                    self.logger.warning(f"⚠️ Invalid action: {action}. Using 0 instead.")
                    action = 0
                
                # Send action with newline delimiter
                message = f"{action}\n"
                self.client_socket.send(message.encode('utf-8'))
                
                # Update success tracking
                self.last_successful_communication = time.time()
                if self.failed_communications > 0:
                    self.failed_communications = 0
                    self.logger.info("✅ Communication restored")
                
                return True
                
            except socket.timeout:
                self.logger.warning(f"⏰ Timeout sending action (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
            except socket.error as e:
                self.logger.warning(f"❌ Socket error sending action: {e} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    if self._try_reconnect():
                        continue
                    time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"❌ Unexpected error sending action: {e}")
                break
        
        # All attempts failed
        self.failed_communications += 1
        self.logger.error(f"❌ Failed to send action after {self.max_retries} attempts")
        
        # Mark as disconnected if too many failures
        if self.failed_communications >= 5:
            self.logger.error("❌ Too many communication failures. Marking as disconnected.")
            self.is_connected = False
        
        return False
    
    def receive_observation(self) -> Dict[str, Any]:
        """
        Receive observation from Unity with enhanced parsing and error handling.
        
        Returns:
            Dictionary containing observation data, empty dict if failed
        """
        if not self._check_connection():
            return {}
        
        for attempt in range(self.max_retries):
            try:
                # Receive data with timeout
                data = self.client_socket.recv(4096).decode('utf-8')
                
                if not data:
                    self.logger.warning("⚠️ Received empty data from Unity")
                    self.is_connected = False
                    return {}
                
                # Parse observation data
                observation = self._parse_observation_data(data)
                
                if observation:
                    # Update success tracking
                    self.last_successful_communication = time.time()
                    if self.failed_communications > 0:
                        self.failed_communications = 0
                        self.logger.info("✅ Communication restored")
                    return observation
                else:
                    self.logger.warning(f"⚠️ Failed to parse observation data: {data[:100]}...")
                    
            except socket.timeout:
                # Timeout is common during normal operation, don't spam logs
                if attempt == 0:  # Only log on first timeout
                    self.logger.debug("⏰ Timeout receiving observation")
                if attempt < self.max_retries - 1:
                    continue
            except socket.error as e:
                self.logger.warning(f"❌ Socket error receiving observation: {e} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    if self._try_reconnect():
                        continue
                    time.sleep(self.retry_delay * 0.5)  # Shorter delay for observations
            except Exception as e:
                self.logger.error(f"❌ Unexpected error receiving observation: {e}")
                break
        
        # All attempts failed
        self.failed_communications += 1
        return {}
    
    def _parse_observation_data(self, data: str) -> Dict[str, Any]:
        """
        Parse observation data from Unity with enhanced format support.
        
        Args:
            data: Raw data string from Unity
            
        Returns:
            Parsed observation dictionary or empty dict if parsing failed
        """
        try:
            # Handle multiple messages in one receive
            lines = data.strip().split('\n')
            
            for line in lines:
                if not line:
                    continue
                
                # Try JSON format first
                if line.startswith('{'):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        pass
                
                # Try simple pipe-delimited format: "observation|x|y|haskey|escaped|keypickedup"
                if '|' in line and 'observation' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        try:
                            return {
                                'player_x': float(parts[1]),
                                'player_y': float(parts[2]),
                                'has_key': parts[3] == '1' or parts[3].lower() == 'true',
                                'escaped': parts[4] == '1' if len(parts) > 4 else False,
                                'key_picked_up': parts[5] == '1' if len(parts) > 5 else False,
                                'key_x': float(parts[6]) if len(parts) > 6 else 5.0,
                                'key_y': float(parts[7]) if len(parts) > 7 else 5.0,
                                'exit_x': float(parts[8]) if len(parts) > 8 else 8.0,
                                'exit_y': float(parts[9]) if len(parts) > 9 else 8.0
                            }
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"Error parsing pipe format: {e}")
                            continue
            
            return {}
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing observation data: {e}")
            return {}
    
    def reset_environment(self) -> bool:
        """
        Send reset command to Unity with retry mechanism.
        
        Returns:
            True if reset command sent successfully, False otherwise
        """
        if not self._check_connection():
            return False
        
        for attempt in range(self.max_retries):
            try:
                message = 'reset\n'
                self.client_socket.send(message.encode('utf-8'))
                
                # Update success tracking
                self.last_successful_communication = time.time()
                return True
                
            except socket.timeout:
                self.logger.warning(f"⏰ Timeout sending reset (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            except socket.error as e:
                self.logger.warning(f"❌ Socket error sending reset: {e}")
                if attempt < self.max_retries - 1:
                    if self._try_reconnect():
                        continue
                    time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"❌ Unexpected error sending reset: {e}")
                break
        
        self.logger.error(f"❌ Failed to send reset after {self.max_retries} attempts")
        return False
    
    def _check_connection(self) -> bool:
        """Check if the connection is still valid."""
        if not self.is_connected or not self.client_socket:
            self.logger.debug("❌ Not connected to Unity")
            return False
        
        # Check if connection has been idle for too long
        if self.last_successful_communication:
            idle_time = time.time() - self.last_successful_communication
            if idle_time > 300:  # 5 minutes of inactivity
                self.logger.warning(f"⚠️ Connection has been idle for {idle_time:.1f} seconds")
        
        return True
    
    def _try_reconnect(self) -> bool:
        """Attempt to reconnect to Unity."""
        self.logger.info("🔄 Attempting to reconnect to Unity...")
        
        # Close current connection
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        
        self.is_connected = False
        
        # Try to accept a new connection (non-blocking)
        if self.socket:
            try:
                self.socket.settimeout(5.0)  # Short timeout for reconnection
                self.client_socket, addr = self.socket.accept()
                self.client_socket.settimeout(5.0)
                self.is_connected = True
                self.logger.info(f"✅ Reconnected to Unity from {addr}")
                return True
            except socket.timeout:
                self.logger.debug("⏰ Reconnection timeout")
            except Exception as e:
                self.logger.debug(f"❌ Reconnection failed: {e}")
        
        return False
    
    def _cleanup_socket(self):
        """Clean up existing socket connections."""
        try:
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None
            if self.socket:
                self.socket.close()
                self.socket = None
        except Exception as e:
            self.logger.debug(f"Error during socket cleanup: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics and health information.
        
        Returns:
            Dictionary containing connection statistics
        """
        stats = {
            'is_connected': self.is_connected,
            'is_server_running': self.is_server_running,
            'failed_communications': self.failed_communications,
            'connection_start_time': self.connection_start_time,
            'last_successful_communication': self.last_successful_communication
        }
        
        if self.connection_start_time:
            stats['connection_duration'] = time.time() - self.connection_start_time
        
        if self.last_successful_communication:
            stats['time_since_last_communication'] = time.time() - self.last_successful_communication
        
        return stats
    
    def close(self):
        """Close all connections and clean up resources."""
        self.logger.info("🔒 Closing Unity bridge...")
        
        self.is_connected = False
        self.is_server_running = False
        
        try:
            # Send a shutdown message if possible
            if self.client_socket and self.is_connected:
                try:
                    self.client_socket.send("shutdown\n".encode('utf-8'))
                    time.sleep(0.1)  # Give Unity time to process
                except:
                    pass
            
            self._cleanup_socket()
            self.logger.info("🔒 Unity bridge closed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Enhanced testing function
def test_unity_bridge(episodes: int = 5, detailed_logging: bool = False):
    """
    Enhanced test function for the Unity bridge with comprehensive testing.
    
    Args:
        episodes: Number of test episodes to run
        detailed_logging: Whether to enable detailed debug logging
    """
    # Configure logging level
    logger = logging.getLogger(__name__)
    if detailed_logging:
        logger.setLevel(logging.DEBUG)
    
    print("🧪 Starting Unity Bridge Test")
    print(f"📊 Running {episodes} test episodes")
    
    with UnityBridge() as bridge:
        if not bridge.start_server():
            print("❌ Failed to start server")
            return
        
        if not bridge.wait_for_unity():
            print("❌ Failed to connect to Unity")
            return
        
        print("🎮 Unity connected! Starting communication test...")
        
        # Test basic communication
        success_count = 0
        total_messages = 0
        
        for episode in range(episodes):
            print(f"\n🎯 Test Episode {episode + 1}")
            
            # Reset environment
            if bridge.reset_environment():
                print("✅ Environment reset successful")
            else:
                print("❌ Environment reset failed")
                continue
            
            time.sleep(0.1)
            
            # Test action sending and observation receiving
            episode_successes = 0
            for step in range(10):
                # Send random action
                action = np.random.randint(0, 4)
                
                if bridge.send_action(action):
                    total_messages += 1
                    
                    # Receive observation
                    obs = bridge.receive_observation()
                    if obs:
                        episode_successes += 1
                        if detailed_logging:
                            print(f"   Step {step}: Action={action}, Obs keys: {list(obs.keys())}")
                    else:
                        print(f"   Step {step}: Action sent but no observation received")
                else:
                    print(f"   Step {step}: Failed to send action")
                
                time.sleep(0.1)
            
            success_rate = (episode_successes / 10) * 100
            print(f"   Episode success rate: {success_rate:.1f}%")
            
            if episode_successes >= 8:  # 80% success threshold
                success_count += 1
        
        # Display final results
        overall_success_rate = (success_count / episodes) * 100
        
        print(f"\n📊 TEST RESULTS:")
        print(f"🎯 Episodes Successful: {success_count}/{episodes}")
        print(f"📈 Overall Success Rate: {overall_success_rate:.1f}%")
        print(f"📡 Total Messages: {total_messages}")
        
        # Display connection stats
        stats = bridge.get_connection_stats()
        print(f"\n🔗 CONNECTION STATISTICS:")
        for key, value in stats.items():
            if isinstance(value, float) and key.endswith('_time'):
                if value > 0:
                    print(f"   {key}: {time.strftime('%H:%M:%S', time.localtime(value))}")
            else:
                print(f"   {key}: {value}")
        
        # Performance assessment
        if overall_success_rate >= 90:
            print("\n🌟 EXCELLENT! Unity bridge is working perfectly.")
        elif overall_success_rate >= 70:
            print("\n👍 GOOD! Unity bridge is working well with minor issues.")
        elif overall_success_rate >= 50:
            print("\n⚠️ MODERATE! Unity bridge has some communication problems.")
        else:
            print("\n❌ POOR! Unity bridge has significant issues.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Unity Bridge Communication')
    parser.add_argument('--episodes', '-e', type=int, default=5,
                       help='Number of test episodes (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable detailed logging')
    
    args = parser.parse_args()
    test_unity_bridge(args.episodes, args.verbose) 