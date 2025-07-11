"""
Base Environment Classes

Shared environment implementations for the AI escape cage training system.
Provides standardized interfaces and Unity integration.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List

# Import Unity bridge from communication module
from communication.unity_bridge import UnityBridge


class BaseEscapeCageEnv(gym.Env, ABC):
    """
    Base environment class for escape cage training.
    
    Contains shared functionality for Unity connection, basic observation/action spaces,
    and common environment operations. Subclasses should implement reward calculation
    and any environment-specific behavior.
    
    Args:
        step_delay (float): Delay between steps for Unity communication
        max_episode_steps (int): Maximum steps before episode ends
        connection_timeout (int): Timeout for Unity connection
    """
    
    def __init__(self, step_delay: float = 0.1, max_episode_steps: int = 500, 
                 connection_timeout: int = 60):
        super().__init__()
        
        # Environment configuration
        self.step_delay = step_delay
        self.max_episode_steps = max_episode_steps
        self.connection_timeout = connection_timeout
        
        # Standard action and observation spaces
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(7,), dtype=np.float32
        )
        
        # Episode tracking
        self.steps_in_episode = 0
        self.episode_count = 0
        
        # Initialize Unity connection
        self.unity = UnityBridge()
        self._setup_unity_connection()
        
        # Initialize environment-specific variables
        self._init_environment_variables()
    
    def _setup_unity_connection(self) -> None:
        """
        Establish connection to Unity environment.
        
        Raises:
            SystemExit: If Unity connection fails
        """
        print("Starting AI system...")
        print("Waiting for Unity connection...")
        
        try:
            if self.unity.start_server():
                if self.unity.wait_for_unity(timeout=self.connection_timeout):
                    print("Unity connected! Training will start!")
                else:
                    print(f"Unity didn't connect within {self.connection_timeout}s. "
                          "Make sure Unity is running.")
                    raise SystemExit(1)
            else:
                print("Failed to start server.")
                raise SystemExit(1)
        except Exception as e:
            print(f"Unity connection failed: {e}")
            raise SystemExit(1)
    
    @abstractmethod
    def _init_environment_variables(self) -> None:
        """
        Initialize environment-specific variables.
        Subclasses should override this to set up their specific tracking variables.
        """
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.steps_in_episode = 0
        self.episode_count += 1
        
        # Reset environment-specific variables
        self._reset_environment_variables()
        
        # Tell Unity to reset
        try:
            self.unity.reset_environment()
            time.sleep(self.step_delay)
        except Exception as e:
            print(f"Warning: Failed to reset Unity environment: {e}")
        
        # Get initial state
        obs_data = self._safe_receive_observation()
        observation = self._make_observation(obs_data)
        
        return observation, {}
    
    @abstractmethod
    def _reset_environment_variables(self) -> None:
        """
        Reset environment-specific variables at the start of each episode.
        Subclasses should override this to reset their tracking variables.
        """
        pass
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action to take (0-3 for up, down, left, right)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Send action to Unity with error handling
        try:
            self.unity.send_action(action)
            time.sleep(self.step_delay)
        except Exception as e:
            print(f"Warning: Failed to send action to Unity: {e}")
        
        # Get new state from Unity
        obs_data = self._safe_receive_observation()
        observation = self._make_observation(obs_data)
        
        # Calculate reward using subclass implementation
        reward = self._calculate_reward(obs_data, observation)
        
        # Check if episode is done
        escaped = obs_data.get('escaped', False) if obs_data else False
        self.steps_in_episode += 1
        
        # Episode termination conditions
        terminated = escaped
        truncated = self.steps_in_episode >= self.max_episode_steps
        done = terminated or truncated
        
        # Prepare info dictionary
        info = {
            'success': escaped,
            'steps': self.steps_in_episode,
            'episode': self.episode_count,
            'truncated': truncated,
            'terminated': terminated
        }
        
        return observation, reward, terminated, truncated, info
    
    def _safe_receive_observation(self) -> Dict[str, Any]:
        """
        Safely receive observation from Unity with error handling.
        
        Returns:
            Dictionary containing observation data, empty dict if failed
        """
        try:
            return self.unity.receive_observation()
        except Exception as e:
            print(f"Warning: Failed to receive observation from Unity: {e}")
            return {}
    
    def _make_observation(self, obs_data: Dict[str, Any]) -> np.ndarray:
        """
        Convert Unity observation data to standardized numpy array.
        
        Args:
            obs_data: Raw observation data from Unity
            
        Returns:
            Standardized observation array [player_x, player_y, has_key, key_x, key_y, exit_x, exit_y]
        """
        if not obs_data:
            return np.zeros(7, dtype=np.float32)
        
        return np.array([
            obs_data.get('player_x', 0.0),
            obs_data.get('player_y', 0.0), 
            float(obs_data.get('has_key', False)),
            obs_data.get('key_x', 5.0),
            obs_data.get('key_y', 5.0),
            obs_data.get('exit_x', 8.0),
            obs_data.get('exit_y', 8.0)
        ], dtype=np.float32)
    
    @abstractmethod
    def _calculate_reward(self, obs_data: Dict[str, Any], observation: np.ndarray) -> float:
        """
        Calculate reward for the current step.
        
        Args:
            obs_data: Raw observation data from Unity
            observation: Processed observation array
            
        Returns:
            Reward value for this step
        """
        pass
    
    def close(self) -> None:
        """Close Unity connection and clean up resources."""
        try:
            self.unity.close()
            print("Environment closed successfully")
        except Exception as e:
            print(f"Warning: Error closing environment: {e}")


class SimpleEscapeCageEnv(BaseEscapeCageEnv):
    """
    Simple escape cage environment with basic reward structure.
    
    This is the standard training environment with straightforward rewards:
    - Large reward for escaping
    - Medium reward for getting the key
    - Small time penalty each step
    """
    
    def _init_environment_variables(self) -> None:
        """Initialize variables for simple environment."""
        pass  # No additional variables needed for simple environment
    
    def _reset_environment_variables(self) -> None:
        """Reset variables for simple environment."""
        pass  # No additional variables to reset
    
    def _calculate_reward(self, obs_data: Dict[str, Any], observation: np.ndarray) -> float:
        """
        Calculate simple reward structure.
        
        Args:
            obs_data: Raw observation data from Unity
            observation: Processed observation array
            
        Returns:
            Reward value: +100 for escape, +10 for key, -0.01 time penalty
        """
        if not obs_data:
            return -0.01
        
        reward = -0.01  # Small time penalty
        
        if obs_data.get('escaped', False):
            reward += 100
            print("AI ESCAPED! +100 points")
        
        if obs_data.get('key_picked_up', False):
            reward += 10
            print("AI found the key! +10 points")
        
        return reward


class FastEscapeCageEnv(BaseEscapeCageEnv):
    """
    Fast training environment with enhanced reward structure and progress tracking.
    
    Features:
    - Progressive rewards for moving toward objectives
    - Distance-based reward shaping
    - Aggressive reward structure for faster learning
    - Shorter episodes to prevent endless wandering
    """
    
    def __init__(self):
        super().__init__(
            step_delay=0.05,  # Faster communication
            max_episode_steps=200,  # Shorter episodes
            connection_timeout=60
        )
    
    def _init_environment_variables(self) -> None:
        """Initialize tracking variables for enhanced reward calculation."""
        self.last_distance_to_key: Optional[float] = None
        self.last_distance_to_exit: Optional[float] = None
    
    def _reset_environment_variables(self) -> None:
        """Reset tracking variables at episode start."""
        self.last_distance_to_key = None
        self.last_distance_to_exit = None
    
    def _calculate_reward(self, obs_data: Dict[str, Any], observation: np.ndarray) -> float:
        """
        Calculate enhanced reward with aggressive distance-based shaping for fast learning.
        
        This reward function implements a sophisticated multi-phase reward system:
        
        Phase 1 (No Key): Agent learns to navigate toward and collect the key
        - Dense reward shaping based on distance reduction to key
        - Proximity bonus when very close to key
        - Penalty for moving away from key to discourage wandering
        
        Phase 2 (Has Key): Agent learns to navigate to the exit
        - Higher rewards for exit progress (double the key phase rewards)
        - Strong proximity bonus when close to exit with key
        - Larger penalties for moving away from exit
        
        Reward Structure:
        - Base time penalty: -0.1 (encourages efficiency)
        - Key pickup: +50 (major milestone achievement)
        - Successful escape: +200 (ultimate goal achievement)
        - Progress rewards: +1.0 to +2.0 (dense reward shaping)
        - Proximity bonuses: +5.0 to +10.0 (encourage final approach)
        
        Args:
            obs_data: Raw observation data from Unity containing game state
            observation: Processed observation array [x, y, has_key, key_x, key_y, exit_x, exit_y]
            
        Returns:
            Enhanced reward value with progress-based bonuses for rapid learning
        """
        # Handle edge case where Unity communication fails
        if not obs_data:
            return -0.1  # Default penalty to encourage reconnection
        
        # Start with base time penalty to encourage efficient behavior
        reward = -0.1
        
        # Unpack observation array for clear position tracking
        player_x, player_y, has_key, key_x, key_y, exit_x, exit_y = observation
        
        # === TERMINAL REWARDS (Episode-ending achievements) ===
        
        # Ultimate success: Agent has successfully escaped
        if obs_data.get('escaped', False):
            reward += 200  # Maximum reward for completing the task
            print("AI ESCAPED! +200 points")
            return reward  # Early return since episode is complete
        
        # Major milestone: Agent has collected the key
        if obs_data.get('key_picked_up', False):
            reward += 50   # Large reward for reaching first major goal
            print("AI found the key! +50 points")
            return reward  # Early return to avoid double-counting
        
        # === PROGRESSIVE REWARDS (Dense reward shaping for continuous learning) ===
        
        # Phase 1: Key Collection Behavior (when agent doesn't have key)
        if not has_key:
            # Calculate Euclidean distance to key for progress measurement
            distance_to_key = np.sqrt((player_x - key_x)**2 + (player_y - key_y)**2)
            
            # Dense reward shaping: compare with previous step's distance
            if self.last_distance_to_key is not None:
                if distance_to_key < self.last_distance_to_key:
                    # Agent moved closer to key - positive reinforcement
                    reward += 1.0
                else:
                    # Agent moved away from key - negative feedback
                    reward -= 0.5  # Smaller penalty to avoid over-penalizing exploration
            
            # Update distance tracking for next step comparison
            self.last_distance_to_key = distance_to_key
            
            # Proximity bonus: Extra reward when very close to key
            # This helps the agent learn the final approach behavior
            if distance_to_key < 1.5:
                reward += 5.0  # Significant bonus for being in pickup range
        
        # Phase 2: Exit Seeking Behavior (when agent has the key)
        else:
            # Calculate Euclidean distance to exit for progress measurement
            distance_to_exit = np.sqrt((player_x - exit_x)**2 + (player_y - exit_y)**2)
            
            # Enhanced reward shaping: higher rewards since exit is the final objective
            if self.last_distance_to_exit is not None:
                if distance_to_exit < self.last_distance_to_exit:
                    # Agent moved closer to exit - double reward vs key phase
                    reward += 2.0
                else:
                    # Agent moved away from exit - larger penalty since behavior is more critical
                    reward -= 1.0
            
            # Update distance tracking for next step comparison
            self.last_distance_to_exit = distance_to_exit
            
            # Enhanced proximity bonus: Much larger reward when close to exit with key
            # This strongly reinforces the final escape behavior
            if distance_to_exit < 1.5:
                reward += 10.0  # Maximum proximity bonus for being in escape range
        
        return reward


class TestEscapeCageEnv(BaseEscapeCageEnv):
    """
    Testing environment optimized for model evaluation and visualization.
    
    Features:
    - Slower step timing for better observation
    - Enhanced feedback and logging
    - Detailed episode statistics tracking
    """
    
    def __init__(self):
        super().__init__(
            step_delay=0.2,  # Slower for better viewing
            max_episode_steps=500,  # Longer episodes for thorough testing
            connection_timeout=60
        )
    
    def _init_environment_variables(self) -> None:
        """Initialize variables for test environment."""
        self.total_reward = 0.0
        self.actions_taken = []
    
    def _reset_environment_variables(self) -> None:
        """Reset variables for test environment."""
        self.total_reward = 0.0
        self.actions_taken = []
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Enhanced step method with action tracking for testing.
        
        Args:
            action: Action to take
            
        Returns:
            Standard gym step return with enhanced info
        """
        # Track actions for analysis
        self.actions_taken.append(action)
        
        # Call parent step method
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Track total reward
        self.total_reward += reward
        
        # Add testing-specific info
        info.update({
            'total_reward': self.total_reward,
            'action_history': self.actions_taken.copy(),
            'actions_count': len(self.actions_taken)
        })
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, obs_data: Dict[str, Any], observation: np.ndarray) -> float:
        """
        Calculate reward with enhanced feedback for testing.
        
        Args:
            obs_data: Raw observation data from Unity
            observation: Processed observation array
            
        Returns:
            Reward value with enhanced logging
        """
        if not obs_data:
            return -0.01
        
        reward = -0.01  # Time penalty
        
        if obs_data.get('escaped', False):
            reward += 100
            print("AI ESCAPED! Smart AI!")
        
        if obs_data.get('key_picked_up', False):
            reward += 10
            print("AI grabbed the key! Strategic thinking!")
        
        return reward 