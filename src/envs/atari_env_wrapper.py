from typing import Any, Optional, Tuple

import cv2
import gym
import gym.spaces
import numpy as np
from gym.spaces.box import Box
from gym.utils import seeding

from action_tokenizer import ATARI_NUM_ACTIONS, FULL_ACTION_TO_LIMITED_ACTION


class AtariEnvWrapper():
    """Environment wrapper with a unified API."""

    def __init__(self, game_name: str, full_action_set: Optional[bool] = True, seed: Optional[int] = None):
        # Disable randomized sticky actions to reduce variance in evaluation.
        self._env = None
        self.game_name = game_name
        self.full_action_set = full_action_set

    def create_env(self):
        self._env = create_atari_environment(
            self.game_name, 
            sticky_actions=False, 
        )
        return self

    @property
    def observation_space(self) -> gym.Space:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        if self.full_action_set:
            return gym.spaces.Discrete(ATARI_NUM_ACTIONS)
        return self._env.action_space

    def reset(self) -> np.ndarray:
        """Reset environment and return observation."""
        return self._env.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Any]:
        """Step environment and return observation, reward, done, info."""
        if self.full_action_set:
          # atari_py library expects limited action set, so convert to limited.
          action_ = FULL_ACTION_TO_LIMITED_ACTION[self.game_name][action]
          assert action_ != 0 or (action_ == 0 and action == 0), f"Invalid global action {action} for game {self.game_name}."
          action = action_
        obs, rew, done, info = self._env.step(action)
        return obs, rew, done, info
    
    def seed(self, seed: int):
        self._env.seed(seed)


class AtariPreprocessing(object):
  """A class implementing image preprocessing for Atari 2600 agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).
    * Grayscale and max-pooling of the last two frames.
    * Downsample the screen to a square image (defaults to 84x84).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  """

  def __init__(
      self,
      environment,
      frame_skip=4,
      terminal_on_life_loss=False,
      screen_size=84,
  ):
    """Constructor for an Atari 2600 preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.
      frame_skip: int, the frequency at which the agent experiences the game.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.

    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
    if frame_skip <= 0:
      raise ValueError(
          'Frame skip should be strictly positive, got {}'.format(frame_skip)
      )
    if screen_size <= 0:
      raise ValueError(
          'Target screen size should be strictly positive, got {}'.format(
              screen_size
          )
      )

    self.environment = environment
    self.terminal_on_life_loss = terminal_on_life_loss
    self.frame_skip = frame_skip
    self.screen_size = screen_size

    obs_dims = self.environment.observation_space
    # Stores temporary observations used for pooling over two successive
    # frames.
    self.screen_buffer = [
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
    ]

    self.game_over = False
    self.lives = 0  # Will need to be set by reset().

  @property
  def observation_space(self):
    # Return the observation space adjusted to match the shape of the processed
    # observations.
    return Box(
        low=0,
        high=255,
        shape=(self.screen_size, self.screen_size, 1),
        dtype=np.uint8,
    )

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  def close(self):
    return self.environment.close()

  def reset(self):
    """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
    self.environment.reset()
    self.lives = self.environment.ale.lives()
    self._fetch_grayscale_observation(self.screen_buffer[0])
    self.screen_buffer[1].fill(0)
    return self._pool_and_resize()

  def render(self, mode):
    """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method. Valid values
        (str) are: 'rgb_array': returns the raw ALE image. 'human': renders to
        display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
    return self.environment.render(mode)

  def step(self, action):
    """Applies the given action in the environment.

    Remarks:

      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
    accumulated_reward = 0.0

    for time_step in range(self.frame_skip):
      # We bypass the Gym observation altogether and directly fetch the
      # grayscale image from the ALE. This is a little faster.
      _, reward, game_over, info = self.environment.step(action)
      accumulated_reward += reward

      if self.terminal_on_life_loss:
        new_lives = self.environment.ale.lives()
        is_terminal = game_over or new_lives < self.lives
        self.lives = new_lives
      else:
        is_terminal = game_over

      # We max-pool over the last two frames, in grayscale.
      if time_step >= self.frame_skip - 2:
        t = time_step - (self.frame_skip - 2)
        self._fetch_grayscale_observation(self.screen_buffer[t])

      if is_terminal:
        break

    # Pool the last two observations.
    observation = self._pool_and_resize()

    self.game_over = game_over
    return observation, accumulated_reward, is_terminal, info

  def _fetch_grayscale_observation(self, output):
    """Returns the current observation in grayscale.

    The returned observation is stored in 'output'.

    Args:
      output: numpy array, screen buffer to hold the returned observation.

    Returns:
      observation: numpy array, the current observation in grayscale.
    """
    self.environment.ale.getScreenGrayscale(output)
    return output

  def _pool_and_resize(self):
    """Transforms two frames into a Nature DQN observation.

    For efficiency, the transformation is done in-place in self.screen_buffer.

    Returns:
      transformed_screen: numpy array, pooled, resized screen.
    """
    # Pool if there are enough screens to do so.
    if self.frame_skip > 1:
      np.maximum(
          self.screen_buffer[0],
          self.screen_buffer[1],
          out=self.screen_buffer[0],
      )

    transformed_image = cv2.resize(
        self.screen_buffer[0],
        (self.screen_size, self.screen_size),
        interpolation=cv2.INTER_AREA,
    )
    int_image = np.asarray(transformed_image, dtype=np.uint8)
    return np.expand_dims(int_image, axis=2)
  
  def seed(self, seed: int):
    _, seed1 = seeding.np_random(seed)
    seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
    # TODO: Empirically, we need to seed before loading the ROM.
    self.environment.ale.setInt(b"random_seed", seed2)


def create_atari_environment(game_name, sticky_actions=True):
    """Wraps an Atari 2600 Gym environment with some basic preprocessing.

    This preprocessing matches the guidelines proposed in Machado et al. (2017),
    "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
    Problems for General Agents".

    The created environment is the Gym wrapper around the Arcade Learning
    Environment.

    The main choice available to the user is whether to use sticky actions or not.
    Sticky actions, as prescribed by Machado et al., cause actions to persist
    with some probability (0.25) when a new command is sent to the ALE. This
    can be viewed as introducing a mild form of stochasticity in the environment.
    We use them by default.

    Args:
        game_name: str, the name of the Atari 2600 domain.
        sticky_actions: bool, whether to use sticky_actions as per Machado et al.

    Returns:
        An Atari 2600 environment with some standard preprocessing.
    """
    game_version = 'v0' if sticky_actions else 'v4'
    full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
    env = gym.make(full_game_name)

    # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
    # handle this time limit internally instead, which lets us cap at 108k frames
    # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
    # restoring states.
    env = env.env
    env = AtariPreprocessing(env)
    return env