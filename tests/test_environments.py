"""
🧪 Environment Tests
Unit tests for escape cage environments to ensure reliability and correctness.

Test coverage:
- Environment initialization
- Action/observation space validation
- Reward calculation logic
- Episode termination conditions
- Unity communication (mocked)
- Performance benchmarks
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os

# Add ml_training to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_training'))

try:
    from base_environment import BaseEscapeCageEnv, StandardEscapeCageEnv, FastEscapeCageEnv
except ImportError:
    # Handle case where base_environment doesn't exist yet
    BaseEscapeCageEnv = None
    StandardEscapeCageEnv = None
    FastEscapeCageEnv = None


class MockUnityBridge:
    """Mock Unity bridge for testing without Unity dependency."""
    
    def __init__(self):
        self.is_connected = True
        self.step_count = 0
        self.has_key = False
        self.escaped = False
    
    def start_server(self):
        return True
    
    def wait_for_unity(self, timeout=60):
        return True
    
    def reset_environment(self):
        self.step_count = 0
        self.has_key = False
        self.escaped = False
    
    def send_action(self, action):
        self.step_count += 1
        # Simulate key pickup after 10 steps
        if self.step_count >= 10:
            self.has_key = True
        # Simulate escape after 20 steps
        if self.step_count >= 20 and self.has_key:
            self.escaped = True
    
    def receive_observation(self):
        return {
            'player_x': float(self.step_count * 0.1),
            'player_y': float(self.step_count * 0.1),
            'has_key': self.has_key,
            'key_x': 1.0,
            'key_y': 1.0,
            'exit_x': 2.0,
            'exit_y': 2.0,
            'escaped': self.escaped,
            'key_picked_up': self.has_key and self.step_count == 10
        }
    
    def close(self):
        pass


@unittest.skipIf(BaseEscapeCageEnv is None, "BaseEscapeCageEnv not available")
class TestBaseEscapeCageEnv(unittest.TestCase):
    """Test cases for BaseEscapeCageEnv class."""
    
    def setUp(self):
        """Set up test environment with mocked Unity bridge."""
        with patch('base_environment.UnityBridge', MockUnityBridge):
            self.env = BaseEscapeCageEnv(
                max_episode_steps=50,
                time_penalty=-0.01,
                unity_timeout=1.0,
                communication_delay=0.0  # No delay for tests
            )
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'env'):
            self.env.close()
    
    def test_environment_initialization(self):
        """Test that environment initializes correctly."""
        self.assertIsInstance(self.env.action_space, spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 4)
        
        self.assertIsInstance(self.env.observation_space, spaces.Box)
        self.assertEqual(self.env.observation_space.shape, (7,))
        
        self.assertEqual(self.env.max_episode_steps, 50)
        self.assertEqual(self.env.time_penalty, -0.01)
    
    def test_observation_space_bounds(self):
        """Test observation space has correct bounds."""
        obs_space = self.env.observation_space
        np.testing.assert_array_equal(obs_space.low, np.full(7, -10.0))
        np.testing.assert_array_equal(obs_space.high, np.full(7, 10.0))
        self.assertEqual(obs_space.dtype, np.float32)
    
    def test_reset_functionality(self):
        """Test environment reset works correctly."""
        observation, info = self.env.reset()
        
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (7,))
        self.assertIsInstance(info, dict)
        
        # Check tracking variables are reset
        self.assertEqual(self.env.steps_in_episode, 0)
        self.assertEqual(self.env.episode_reward, 0.0)
        self.assertIsNone(self.env.last_distance_to_key)
        self.assertIsNone(self.env.last_distance_to_exit)
    
    def test_step_functionality(self):
        """Test environment step function."""
        self.env.reset()
        
        # Take a step
        action = 0  # Up
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Check return values
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (7,))
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        
        # Check info contains expected keys
        expected_keys = {'success', 'steps', 'episode_reward', 'key_found_step', 'escape_step'}
        self.assertTrue(expected_keys.issubset(info.keys()))
        
        # Check step counter incremented
        self.assertEqual(self.env.steps_in_episode, 1)
    
    def test_reward_calculation(self):
        """Test basic reward calculation."""
        self.env.reset()
        
        # Normal step should give time penalty
        obs, reward, _, _, _ = self.env.step(0)
        self.assertEqual(reward, self.env.time_penalty)
        
        # Take steps until key pickup
        for _ in range(10):
            obs, reward, terminated, truncated, info = self.env.step(0)
            if info.get('key_found_step') is not None:
                # Key pickup should give positive reward
                self.assertGreater(reward, 0)
                break
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        self.env.reset()
        
        # Test truncation after max steps
        for step in range(self.env.max_episode_steps + 1):
            obs, reward, terminated, truncated, info = self.env.step(0)
            if step < self.env.max_episode_steps - 1:
                self.assertFalse(terminated)
                self.assertFalse(truncated)
            else:
                # Should be truncated at max steps
                self.assertTrue(truncated or terminated)
                break
    
    def test_observation_format(self):
        """Test observation data format and content."""
        obs, _ = self.env.reset()
        
        # Check observation contains expected data
        player_x, player_y, has_key, key_x, key_y, exit_x, exit_y = obs
        
        self.assertIsInstance(player_x, (int, float, np.number))
        self.assertIsInstance(player_y, (int, float, np.number))
        self.assertIn(has_key, [0.0, 1.0])  # Boolean converted to float
        self.assertIsInstance(key_x, (int, float, np.number))
        self.assertIsInstance(key_y, (int, float, np.number))
        self.assertIsInstance(exit_x, (int, float, np.number))
        self.assertIsInstance(exit_y, (int, float, np.number))
    
    def test_multiple_resets(self):
        """Test that environment can be reset multiple times."""
        for i in range(3):
            obs, info = self.env.reset()
            self.assertEqual(self.env.steps_in_episode, 0)
            self.assertEqual(obs.shape, (7,))
            
            # Take a few steps
            for _ in range(5):
                self.env.step(0)
            
            self.assertGreater(self.env.steps_in_episode, 0)


@unittest.skipIf(StandardEscapeCageEnv is None, "StandardEscapeCageEnv not available")
class TestStandardEscapeCageEnv(unittest.TestCase):
    """Test cases for StandardEscapeCageEnv."""
    
    def setUp(self):
        with patch('base_environment.UnityBridge', MockUnityBridge):
            self.env = StandardEscapeCageEnv()
    
    def tearDown(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    def test_standard_configuration(self):
        """Test standard environment has correct configuration."""
        self.assertEqual(self.env.max_episode_steps, 500)
        self.assertEqual(self.env.time_penalty, -0.01)
        self.assertEqual(self.env.communication_delay, 0.1)


@unittest.skipIf(FastEscapeCageEnv is None, "FastEscapeCageEnv not available")
class TestFastEscapeCageEnv(unittest.TestCase):
    """Test cases for FastEscapeCageEnv."""
    
    def setUp(self):
        with patch('base_environment.UnityBridge', MockUnityBridge):
            self.env = FastEscapeCageEnv()
    
    def tearDown(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    def test_fast_configuration(self):
        """Test fast environment has aggressive configuration."""
        self.assertEqual(self.env.max_episode_steps, 200)
        self.assertEqual(self.env.time_penalty, -0.1)
        self.assertEqual(self.env.communication_delay, 0.05)
        
        # Check aggressive reward scaling
        self.assertEqual(self.env.reward_scaling['escape'], 200.0)
        self.assertEqual(self.env.reward_scaling['key_pickup'], 50.0)
        self.assertGreater(self.env.reward_scaling['progress'], 1.0)


class TestEnvironmentWithoutUnity(unittest.TestCase):
    """Test environment behavior without Unity connection."""
    
    def test_unity_connection_failure(self):
        """Test graceful handling of Unity connection failure."""
        if BaseEscapeCageEnv is None:
            self.skipTest("BaseEscapeCageEnv not available")
            
        with patch('base_environment.UnityBridge') as mock_bridge:
            mock_instance = mock_bridge.return_value
            mock_instance.start_server.return_value = False
            
            with self.assertRaises(ConnectionError):
                BaseEscapeCageEnv()
    
    def test_unity_timeout(self):
        """Test handling of Unity connection timeout."""
        if BaseEscapeCageEnv is None:
            self.skipTest("BaseEscapeCageEnv not available")
            
        with patch('base_environment.UnityBridge') as mock_bridge:
            mock_instance = mock_bridge.return_value
            mock_instance.start_server.return_value = True
            mock_instance.wait_for_unity.return_value = False
            
            with self.assertRaises(ConnectionError):
                BaseEscapeCageEnv()


class TestRewardCalculation(unittest.TestCase):
    """Detailed tests for reward calculation logic."""
    
    def setUp(self):
        if BaseEscapeCageEnv:
            with patch('base_environment.UnityBridge', MockUnityBridge):
                self.env = BaseEscapeCageEnv()
    
    def tearDown(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    @unittest.skipIf(BaseEscapeCageEnv is None, "BaseEscapeCageEnv not available")
    def test_distance_reward_calculation(self):
        """Test distance-based reward calculation."""
        self.env.reset()
        
        # Create test observation
        observation = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 5.0, 5.0], dtype=np.float32)
        
        # Calculate distance reward
        distance_reward = self.env._calculate_distance_reward(observation)
        
        # Should be a float
        self.assertIsInstance(distance_reward, (int, float))


class TestPerformance(unittest.TestCase):
    """Performance tests for environment operations."""
    
    def setUp(self):
        if BaseEscapeCageEnv:
            with patch('base_environment.UnityBridge', MockUnityBridge):
                self.env = BaseEscapeCageEnv(communication_delay=0.0)
    
    def tearDown(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    @unittest.skipIf(BaseEscapeCageEnv is None, "BaseEscapeCageEnv not available")
    def test_step_performance(self):
        """Test that step operations are reasonably fast."""
        import time
        
        self.env.reset()
        
        start_time = time.time()
        for _ in range(100):
            self.env.step(0)
        end_time = time.time()
        
        avg_step_time = (end_time - start_time) / 100
        
        # Each step should take less than 10ms (without Unity communication delay)
        self.assertLess(avg_step_time, 0.01, "Step operations are too slow")
    
    @unittest.skipIf(BaseEscapeCageEnv is None, "BaseEscapeCageEnv not available")
    def test_reset_performance(self):
        """Test that reset operations are reasonably fast."""
        import time
        
        start_time = time.time()
        for _ in range(10):
            self.env.reset()
        end_time = time.time()
        
        avg_reset_time = (end_time - start_time) / 10
        
        # Each reset should take less than 100ms
        self.assertLess(avg_reset_time, 0.1, "Reset operations are too slow")


def run_environment_tests():
    """Run all environment tests and return results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    unittest.main(verbosity=2) 