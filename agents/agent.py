import numpy as np

from task import Task

from models.actor import Policy
from models.critic import Value
from models.noise import OUNoise
from models.replayBuffer import ReplayBuffer

""" Reinforcement learning agent using
    Deep Deterministic Policy Gradient(DDPG) """
class Agent():

    def __init__(self, task):
        self.task = task

        self.state_size = task.state_size
        self.action_size = task.action_size

        self.action_dim_min = task.action_low
        self.action_dim_max = task.action_high

        self.action_range = task.action_high - task.action_low

        ## Need two copies of each model - 'local' and 'target' (as an extension
        ## of "Fixed Q Targets" from Deep Q-Learning) to decouple parameters
        ## being updated from the ones that are producing target values.

        # Actor model (policy)
        self.local_actor = Policy(self.state_size, self.action_size,
                                  self.action_dim_min, self.action_dim_max)
        self.target_actor = Policy(self.state_size, self.action_size,
                                   self.action_dim_min, self.action_dim_max)

        # Critic model (value)
        self.local_critic = Value(self.state_size, self.action_size)
        self.target_critic = Value(self.state_size, self.action_size)

        # Initializing target model parameters with local model parameters
        self.target_critic.model.set_weights(self.local_critic.model.get_weights())
        self.target_actor.model.set_weights(self.local_actor.model.get_weights())

        # Add some noise components and process
        self.exploration_mu = 0.1
        self.exploration_theta = 0.25
        self.exploration_sigma = 0.3

        self.noise = OUNoise(self.action_size, self.exploration_mu,
                             self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.9 # discount rate
        self.tau = 0.01 # used when soft-updating target parameters

        # Score tracker and learning parameters
        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        self.reset_episode() # reset episode variables

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0.0
        self.score = -np.inf

        self.noise.reset()
        state = self.task.reset()
        self.last_state = state

        return state

    def step(self, next_state, action, reward, done):
        # Save experience, reward
        self.memory.add(self.last_state, action, reward, next_state, done)
        self.total_reward += reward
        self.count += 1.0

        # Learn if enough samples available in memory
        if self.memory.__len__() > self.batch_size:
            experiences = self.memory.memory_sample()
            self.learn(experiences)

        # Update to last state and action
        self.last_state = next_state

    def act(self, state):
        """Take actions for given state as per current policy"""
        state = np.reshape(state, [-1, self.state_size])
        action = self.local_actor.model.predict(state)[0]

        # Add noise for more exploration
        return list(action + self.noise.noise_sample())

    def learn(self, experiences):
        """ Update policy and value parameters
            using given batch of experience tuples"""

        # Convert experience tuples to separate arrays
        states = np.vstack([exp.state for exp in experiences if exp is not None])
        actions = np.array([exp.action for exp in experiences if exp is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([exp.reward for exp in experiences if exp is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([exp.done for exp in experiences if exp is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([exp.next_state for exp in experiences if exp is not None])

        # Predict next-state actions and Q values from target models
        next_actions = self.target_actor.model.predict_on_batch(next_states)
        next_Q_targets = self.target_critic.model.predict_on_batch([next_states, next_actions])

        # Calculate Q targets for current states and train local critic model
        Q_targets = rewards + self.gamma * next_Q_targets * (1 - dones)
        self.local_critic.model.train_on_batch(x = [states, actions], y = Q_targets)

        # Train local actor model
        actions_gradients = np.reshape(self.local_critic.get_action_gradients([states, actions, 0]),
                                       (-1, self.action_size))
        self.local_actor.train_func([states, actions_gradients, 1])

        # Perform soft-update on target models
        self.soft_update(self.local_actor.model, self.target_actor.model)
        self.soft_update(self.local_critic.model, self.target_critic.model)

        # Learn using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0

        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)

        self.w = self.w + self.noise_scale * np.random.normal(size = self.w.shape)  # equal noise in all directions

    def soft_update(self, local_model, target_model):
        """Update for model parameters"""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        error_message = "Local and Target model parameters must have same size"
        assert len(local_weights) == len(target_weights), error_message

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
