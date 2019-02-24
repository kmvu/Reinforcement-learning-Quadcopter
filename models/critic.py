from keras import layers, models, optimizers
from keras.layers import Input, Dense, Add, Activation
from keras import backend as K


"""Value model"""
class Value:
	
	def __init__(self, state_size, action_size):
		"""Initialize parameters and build model
		params
		======
			state_size (int): dimension of each state
			action_size (int): dimension of each action
		"""
		self.state_size = state_size
		self.action_size = action_size

		self.build_model()

	def build_model(self):
		"""Build a Critic (Value) network that maps (state, action) pairs -> Q-values"""
		
		# Input layers
		states = Input((self.state_size,), name = 'states')
		actions = Input((self.action_size,), name = 'actions')

		# Hidden layers for state pathway
		net_states = Dense(64, activation = 'relu')(states)
		net_states = Dense(128, activation = 'relu')(net_states)

		# Hidden layers for action pathway
		net_actions = Dense(64, activation = 'relu')(actions)
		net_actions = Dense(128, activation = 'relu')(net_actions)

		# Combine state, action pathway
		net = Add()([net_states, net_actions])
		net = Activation('relu')(net)

		# More layers here
		net = Dense(128, activation = 'relu')(net)
		net = Dense(64, activation = 'relu')(net)

		# Add final output layer to produce action values (Q values)
		Q_values = Dense(1, name = 'q_values')(net)

		# Create our Keras model
		self.model = models.Model(inputs = [states, actions], outputs = Q_values)

		# Optimizing and compiling model with built-in mean squared error loss function
		optimizer = optimizers.Adam()
		self.model.compile(optimizer, loss = 'mse')

		# Compute action gradients (derivative of Q values with respect to actions)
		action_gradients = K.gradients(Q_values, actions)

		# Custom function to fetch action gradients, so it can be used by Actor model
		self.get_action_gradients = K.function(
			inputs = [*self.model.input, K.learning_phase()],
			outputs = action_gradients
		)