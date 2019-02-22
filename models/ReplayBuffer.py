import random
from collections import namedtuple, deque

"""Fixed-size buffer to persist experience tuples."""
class ReplayBuffer:
	
	def __init__(self, buffer_size, batch_size):
		"""Initialize a ReplayBuffer object.
		Params
		======
			buffer_size: maximum size of buffer
			batch_size: size of each training batch
		"""
		self.memory = deque(maxlen = buffer_size)
		self.batch_size = batch_size

		names = ["state", "action", "reward", "next_state", "done"]
		self.experience = namedtuple("Experience", field_names = names)

	def add(self, state, action, reward, next_state, done):
		"""Add new experience to memory"""
		exp = self.experience(state, action, reward, next_state, done)
		self.memory.append(exp)
		
	def memory_sample(self, batch_size = 64):
		"""Returns a random sample batch of experiences from memory"""
		return random.sample(self.memory, k = self.batch_size)

	def __len__(self):
		"""Returns current size of internal memory"""
		return len(self.memory)