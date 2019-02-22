import numpy as np
from task import Task

""" Reinforcement learning agent using
    Deep Deterministic Policy Gradient(DDPG) """
class Agent():

    def __init__(self, task):
        self.task = task

        self.state_size = task.state_size
        self.action_size = task.action_size

        self.action_dim_min = task.action_dim_min
        self.action_dim_max = task.action_dim_max

        ## Need two copies of each model - 'local' and 'target' (as an extension
        ## of "Fixed Q Targets" from Deep Q-Learning) to decouple parameters
        ## being updated from the ones that are producing target values.

        # Actor model (policy)
        self.local_actor = Actor(self.state_size, self.action_size,
                                 self.action_dim_min, self.action_dim_max)
        self.target_actor = Actor(self.state_size, self.action_size,
                                  self.action_dim_min, self.action_dim_max)

        # Critic model (value)
        self.local_critic = Critic(self.state_size, self.action_size)
        self.target_critic = Critic(self.state_size, self.action_size)

        # Initializing target model parameters with local model parameters
        self.target_critic.model
            .set_weights(self.local_critic.model.get_weights())
        self.target_actor.model
            .set_weights(self.local_actor.model.get_weights())

        # Add some noise components and process
        self.exploration_mu = 0.0
        self.exploration_theta = 0.2
        self.exploration_sigma = 0.2

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
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        self.reset_episode() # reset episode variables

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0.0

        self.noise.reset()
        state = self.task.reset()
        self.last_state = state

        return state

    def step(self, next_state, action, reward, done):
        # Save experience, reward
        self.memory.add(self.last_state, action, reward, next_state, done)
        self.total_reward += reward
        self.count += 1

        # Learn if enough samples available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
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
        actions = np.array([exp.action for exp in experiences if exp is not None])
            .astype(np.float32)
            .reshape(-1, self.action_size)
        rewards = np.array([exp.reward for exp in experiences if exp is not None])
            .astype(np.float32)
            .reshape(-1, 1)
        dones = np.array([exp.done for exp in experiences if exp is not None])

        # Predict next-state actions and Q values from target models
        next_actions = self.target_actor.model.predict_on_batch(next_state)
        next_Q_targets = self.target.critic.model
            .predict_on_batch([next_states, next_actions])

        # Calculate Q targets for current states and train local critic model
        Q_targets = rewards + self.gamma * next_Q_targets * (1 - dones)
        self.local_critic.model.train_on_batch(x = [states, actions],
                                               y = Q_targets)

        # Train local actor model
        actions_gradients = np.reshape(self.local_critic.get_action_gradients([states, actions, 0]),
                                       (-1, self.action_size))
        self.local_actor.train_func([states, actions_gradients, 1])

        # Perform soft-update on target models
        self.soft_update(self.local_actor.model, self.target_actor.model)
        self.soft_update(self.local_critic.model, self.target_critic.model)

    def soft_update(self, local_model, target_model):
        """Update for model parameters"""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        error_message = "Local and Target model parameters must have same size"
        assert len(local_weights) == len(target_weights), error_message

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
