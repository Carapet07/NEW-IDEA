"""
Communication Tests

Test suite for Unity-Python communication bridge functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import socket
import json
import sys
import os

# Add communication to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'communication'))

try:
    from unity_bridge import UnityBridge
except ImportError:
    UnityBridge = None


class MockSocket:
    """Mock socket for testing without network operations."""
    
    def __init__(self):
        self.connected = False
        self.data_buffer = []
        self.timeout = None
    
    def bind(self, address):
        pass
    
    def listen(self, backlog):
        pass
    
    def accept(self):
        mock_client = MockSocket()
        mock_client.connected = True
        return mock_client, ('127.0.0.1', 12345)
    
    def settimeout(self, timeout):
        self.timeout = timeout
    
    def send(self, data):
        self.data_buffer.append(data)
        return len(data)
    
    def recv(self, bufsize):
        if self.data_buffer:
            return self.data_buffer.pop(0)
        return b'observation|1.0|2.0|1\n'
    
    def close(self):
        self.connected = False
    
    def setsockopt(self, level, optname, value):
        pass


@unittest.skipIf(UnityBridge is None, "UnityBridge not available")
class TestUnityBridge(unittest.TestCase):
    """Test cases for UnityBridge class."""
    
    def setUp(self):
        """Set up test environment with mocked socket."""
        self.mock_socket_patcher = patch('socket.socket')
        self.mock_socket = self.mock_socket_patcher.start()
        self.mock_socket.return_value = MockSocket()
        
        self.bridge = UnityBridge()
    
    def tearDown(self):
        """Clean up after tests."""
        self.mock_socket_patcher.stop()
        if hasattr(self, 'bridge'):
            self.bridge.close()
    
    def test_bridge_initialization(self):
        """Test that bridge initializes with correct defaults."""
        self.assertEqual(self.bridge.host, 'localhost')
        self.assertEqual(self.bridge.port, 9999)
        self.assertFalse(self.bridge.is_connected)
        self.assertFalse(self.bridge.is_server_running)
    
    def test_server_startup(self):
        """Test server startup functionality."""
        result = self.bridge.start_server()
        self.assertTrue(result)
        self.assertTrue(self.bridge.is_server_running)
    
    def test_unity_connection(self):
        """Test Unity connection establishment."""
        self.bridge.start_server()
        result = self.bridge.wait_for_unity(timeout=1.0)
        self.assertTrue(result)
        self.assertTrue(self.bridge.is_connected)
    
    def test_action_sending(self):
        """Test sending actions to Unity."""
        self.bridge.start_server()
        self.bridge.wait_for_unity()
        
        result = self.bridge.send_action(2)
        self.assertTrue(result)
    
    def test_observation_receiving(self):
        """Test receiving observations from Unity."""
        self.bridge.start_server()
        self.bridge.wait_for_unity()
        
        obs = self.bridge.receive_observation()
        self.assertIsInstance(obs, dict)
        
        # Check for expected keys
        if obs:
            expected_keys = ['player_x', 'player_y', 'has_key']
            for key in expected_keys:
                self.assertIn(key, obs)
    
    def test_environment_reset(self):
        """Test environment reset command."""
        self.bridge.start_server()
        self.bridge.wait_for_unity()
        
        result = self.bridge.reset_environment()
        self.assertTrue(result)
    
    def test_connection_failure_handling(self):
        """Test handling of connection failures."""
        with patch.object(self.bridge, 'socket', None):
            result = self.bridge.send_action(0)
            self.assertFalse(result)
            
            obs = self.bridge.receive_observation()
            self.assertEqual(obs, {})
    
    def test_multiple_connections(self):
        """Test that bridge can handle multiple connection cycles."""
        for _ in range(3):
            self.bridge.start_server()
            self.bridge.wait_for_unity()
            self.assertTrue(self.bridge.is_connected)
            self.bridge.close()
            self.assertFalse(self.bridge.is_connected)


class TestUnityBridgePerformance(unittest.TestCase):
    """Performance tests for Unity communication."""
    
    def setUp(self):
        if UnityBridge:
            with patch('socket.socket', return_value=MockSocket()):
                self.bridge = UnityBridge()
                self.bridge.start_server()
                self.bridge.wait_for_unity()
    
    def tearDown(self):
        if hasattr(self, 'bridge'):
            self.bridge.close()
    
    @unittest.skipIf(UnityBridge is None, "UnityBridge not available")
    def test_action_sending_performance(self):
        """Test that action sending is fast enough for real-time use."""
        import time
        
        start_time = time.time()
        for i in range(100):
            self.bridge.send_action(i % 4)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        
        # Each action should take less than 1ms
        self.assertLess(avg_time, 0.001, "Action sending is too slow")
    
    @unittest.skipIf(UnityBridge is None, "UnityBridge not available")
    def test_observation_receiving_performance(self):
        """Test that observation receiving is fast enough."""
        import time
        
        start_time = time.time()
        for _ in range(100):
            self.bridge.receive_observation()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        
        # Each observation should take less than 5ms
        self.assertLess(avg_time, 0.005, "Observation receiving is too slow")


def run_communication_tests():
    """Run all communication tests and return results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    unittest.main(verbosity=2) 