from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import Input, Dense, Lambda


"""Policy model"""
class Policy:

	def __init__(self, state_size, action_size, action_dim_min, action_dim_max):
		"""Initialize parameters and build model
		Params
		======
			state_size (int): Dimension of each state
			action_size (int): Dimension of each action
			action_dim_min (int): Min value for each action dimension
			action_dim_max (int): Max value for each action dimension
		"""
		self.state_size = state_size
		self.action_size = action_size
		self.action_dim_min = action_dim_min
		self.action_dim_max = action_dim_max
		self.action_range = self.action_dim_max - self.action_dim_min

		self.build_model()

	def build_model(self):
		"""Build an Actor (Policy) network that maps states -> actions"""

		# Input layer (states)
		states = Input((self.state_size,), name = 'states')

		# Hidden layers
		net = Dense(32, activation = 'relu')(states)
		net = Dense(64, activation = 'relu')(net)
		net = Dense(32, activation = 'relu')(net)

		# Final output layer with sigmoide layer
		raw_actions = Dense(self.action_size, activation = 'sigmoid', name = 'raw_actions')(net)

		# Scale [0, 1] output for each action dimension to proper range
		actions = Lambda(lambda action: action * self.action_range + self.action_dim_min, 
						 name = 'actions')(raw_actions)

		# Create our Keras model
		self.model = models.Model(inputs = states, outputs = actions)

		# Define loss function using action value (Q value) gradients
		action_gradients = Input((self.action_size,))
		loss = K.mean(-action_gradients * actions)

		# More losses here (e.g. from regulizers)
		# ...


		# Define optimizers and training function
		optimizer = optimizers.Adam()
		updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)

		self.train_func = K.function(
			inputs = [self.model.input, action_gradients, K.learning_phase()],
			outputs = [],
			updates = updates_op
		)






