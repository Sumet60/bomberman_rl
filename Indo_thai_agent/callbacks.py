import numpy as np
import random
from random import shuffle
from time import time, sleep
from matplotlib import pyplot
from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense, PReLU
from keras.optimizers import Adam
from keras.models import load_model
from settings import s,e
import logging

scores=[]
rewards_episodes_list =[]

def setup(self):
	"""Called once before a set of games to initialize data structures etc.

	The 'self' object passed to this method will be the same in all other
	callback methods. You can assign new properties (like bomb_history below)
	here or later on and they will be persistent even across multiple games.
	You can also use the self.logger object at any time to write to the log
	file for debugging (see https://docs.python.org/3.7/library/logging.html).
	"""
	self.logger.debug('Successfully entered setup code')
	np.random.seed()
	# Keeping fixed length FIFO queues to avoid repeating the same actions
	self.bomb_history = deque([], 5)
	self.coordinate_history = deque([], 20)
	#timer for agent to hunt/attack opponents
	self.ignore_others_timer = 0

	# Initialize the size of the state
	self.state_size = s.cols*s.rows

	#defining state with some real values
	self.state = np.zeros(self.state_size)

	# defining next_state with real values
	self.next_state = np.zeros(self.state_size)


	# Init the 6 possible actions
	# 0. UP ->    (x  , y-1)
	# 1. DOWN ->  (x  , y+1)
	# 2. LEFT ->  (x-1, y  )
	# 3. RIGHT -> (x+1, y  )
	# 4. WAIT ->  (x  , y  )
	# 5. BOMB ->    ?
	self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

	#mapping the actions to integers
	self.map_actions = {
		'UP': 0,
		'DOWN': 1 ,
		'LEFT': 2,
		'RIGHT': 3,
		'WAIT': 4,
		'BOMB': 5
	}

	# Set to 6
	self.actions_size = 6

	# action index (initialize to the default value (WAIT))
	self.indx_action = 4

	# Initialize gamma = 0.95 for no good reason
	self.gamma = 0.95

	# Initialize the learning rate
	self.learning_rate = 0.001

	# List for the previous experiences
	# It's a deque because once the maxlen is reached, the oldest
	# experiences are forgotten (the newest experiences is what matters)
	# (state, action, reward, next_action, done)
	self.memory = deque(maxlen=20000)

	# Exploration rate
	self.epsilon = 0.6
	# This hyperparameter used to decrease the number of explorations
	self.epsilon_decay = 0.995

	# Minimum value of epsilon reacheable
	self.epsilon_min = 0.1

	# Neural network model for training
	#self.model = build_model(self)

	
    #training loss count
	self.loss =[]

	# loading the trained model for playing
	self.model = load_model('user_nn_10k.h5')

	# Counting the total reward
	self.total_reward = 0

	# Reward at time step
	self.reward = 0

	# Various Counters
	self.random = 0
	self.simple_agent = 0
	self.NN_agent = 0
	self.coins_collected = 0
	self.invalid_actions = 0

	self.episodes = 0

	# Reward List for events
	self.reward_dict = {
			'KILLED_OPPONENT' : 500,
			'KILLED_SELF' : -100,
			'CRATE_DESTROYED' : 100,
			'BOMB_DROPPED' : 50,
			'SURVIVED_ROUND' : 500,

			'COIN_COLLECTED':  200,
			'EMPTY_CELL' : -100/(s.rows*s.cols),
			'INVALID_ACTION' : -4.0,
			'MAX_REWARD' : 500
	}

	# values for objects such as crates, walls, coins, agent position in the game
	self.values = {
		'FREE' : 0.0,
		'CRATE' : 1.0,
		'WALL' : -1.0,
		'COIN' : 0.75,
		'SELF' : 0.5,
	}


	

#returning the position index
def pos_xy(x, y):
	indx = (x-1) + (y-1)*(s.rows-2)
	indx -= ((y-1)//2)*(s.rows//2 - 1)
	if (y%2 == 0):
		indx -= x//2
	return indx

#state matrix needs to be updated as the state changes after one coin is being collected by the agent
def state_update(self):


	state = np.zeros(self.state_size)
	coins = self.game_state['coins']

	# set the values for coins
	for (x0,y0) in coins:
		indx = pos_xy(x0,y0)
		state[indx] = self.values['COIN']

	x, y, _, _, _ = self.game_state['self']
	indx = pos_xy(x,y)
	state[indx] = self.values['SELF']

	#others = self.game_state['others']

	# for (x_o,y_o, _, _,) in others:
	#   indx=pos_xy(x_o, y_o)
	#   state[indx] = self.values['others']

	return state






def look_for_targets(free_space, start, targets, logger=None):
	"""Find direction of closest target that can be reached via free tiles.

	Performs a breadth-first search of the reachable free tiles until a target is encountered.
	If no target can be reached, the path that takes the agent closest to any target is chosen.

	Args:
		free_space: Boolean numpy array. True for free tiles and False for obstacles.
		start: the coordinate from which to begin the search.
		targets: list or array holding the coordinates of all target tiles.
		logger: optional logger object for debugging.
	Returns:
		coordinate of first step towards closest target or towards tile closest to any target.
	"""
	if len(targets) == 0: return None

	frontier = [start]
	parent_dict = {start: start}
	dist_so_far = {start: 0}
	best = start
	best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

	while len(frontier) > 0:
		current = frontier.pop(0)
		# Find distance from current position to all targets, track closest
		d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
		if d + dist_so_far[current] <= best_dist:
			best = current
			best_dist = d + dist_so_far[current]
		if d == 0:
			# Found path to a target's exact position, mission accomplished!
			best = current
			break
		# Add unexplored free neighboring tiles to the queue in a random order
		x, y = current
		neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
		shuffle(neighbors)
		for neighbor in neighbors:
			if neighbor not in parent_dict:
				frontier.append(neighbor)
				parent_dict[neighbor] = current
				dist_so_far[neighbor] = dist_so_far[current] + 1
	if logger: logger.debug(f'Suitable target found at {best}')
	# Determine the first step towards the best found target tile
	current = best
	while True:
		if parent_dict[current] == start: return current
		current = parent_dict[current]




def build_model(self):
	# Buiding neural network model for deep-Q learning
	model = Sequential()
	model.add(Dense(self.state_size, input_dim=self.state_size, activation='relu'))
	model.add(Dense(self.state_size, activation='relu'))
	model.add(Dense(self.actions_size, activation='linear'))
	model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
	#model.add(Dense(self.state_size, input_dim=self.state_size))
	#model.add(PReLU())
	#model.add(Dense(self.state_size))
	#model.add(PReLU())
	#model.add(Dense(self.state_size))
	#model.add(PReLU())
	#model.add(Dense(self.actions_size))
	#model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

	return model

def remember(self, state, action, reward, next_state, done):
	self.memory.append((state, action, reward, next_state, done))


# def act(self, state):
#   if np.random.rand() <= self.epsilon:
#       return random.randrange(self.action_size)

#     act_values = self.model.predict(state)

#     return np.argmax(act_values[0])  # returns action


#A method that trains the neural net with experiences in the memory, implementing the concept of experience replay as discussed in the report 
def replay(self, batch_size):
	
		data_size=min(len(self.memory),batch_size)
		indx=np.random.choice(len(self.memory), batch_size)
		minibatch= np.array(self.memory)[indx]
		
		#minibatch = random.sample(self.memory, data_size )

		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
			  target = reward + self.gamma * \
					   np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target

			history=self.model.fit(state, target_f, epochs=1, verbose=1)
			#self.loss.append(history.history['loss'])


			#pyplot.plot(self.history.history['val_loss'][0])
			#history=self.model.fit(state, target_f, validation_split=0.2, epochs=1, verbose=0)
			#print(history.history.keys())
		#pyplot.plot(history.history['loss'])
		self.loss.append(history.history['loss'])
		#pyplot.show()
			


		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay




def act(self):
	"""Called each game step to determine the agent's next action.

	You can find out about the state of the game environment via self.game_state,
	which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
	what it contains.

	Set the action you wish to perform by assigning the relevant string to
	self.next_action. You can assign to this variable multiple times during
	your computations. If this method takes longer than the time limit specified
	in settings.py, execution is interrupted by the game and the current value
	of self.next_action will be used. The default value is 'WAIT'.
	"""


	self.logger.info(f'STATE SIZE: {self.state_size}')
	# Gather information about the game state
	self.state = state_update(self)

	arena = self.game_state['arena']
	x, y, _, bombs_left, score = self.game_state['self']
	bombs = self.game_state['bombs']
	bomb_xys = [(x,y) for (x,y,t) in bombs]
	others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
	coins = self.game_state['coins']
	self.logger.info(f'COINS: {coins}')
	bomb_map = np.ones(arena.shape) * 5
	for xb,yb,t in bombs:
		for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
			if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
				bomb_map[i,j] = min(bomb_map[i,j], t)

	# If agent has been in the same location three times recently, it's a loop
	if self.coordinate_history.count((x,y)) > 2:
		self.ignore_others_timer = 5
	else:
		self.ignore_others_timer -= 1
	self.coordinate_history.append((x,y))

	# Check which moves make sense at all
	directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)]
	valid_tiles, valid_actions = [], []
	for d in directions:
		if ((arena[d] == 0) and
			(self.game_state['explosions'][d] <= 1) and
			(bomb_map[d] > 0) and
			(not d in others) and
			(not d in bomb_xys)):
			valid_tiles.append(d)
	if (x-1,y) in valid_tiles: valid_actions.append('LEFT')
	if (x+1,y) in valid_tiles: valid_actions.append('RIGHT')
	if (x,y-1) in valid_tiles: valid_actions.append('UP')
	if (x,y+1) in valid_tiles: valid_actions.append('DOWN')
	if (x,y)   in valid_tiles: valid_actions.append('WAIT')
	# Disallow the BOMB action if agent dropped a bomb in the same spot recently
	if (bombs_left > 0) and (x,y) not in self.bomb_history: valid_actions.append('BOMB')
	self.logger.debug(f'Valid actions: {valid_actions}')

	# Collect basic action proposals in a queue
	# Later on, the last added action that is also valid will be chosen
	action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
	shuffle(action_ideas)

	# Compile a list of 'targets' the agent should head towards
	dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
					and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
	crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
	targets = coins + dead_ends + crates
	# Add other agents as targets if in hunting mode or no crates/coins left
	if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
		targets.extend(others)

	# Exclude targets that are currently occupied by a bomb
	targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

	# Take a step towards the most immediately interesting target
	free_space = arena == 0
	if self.ignore_others_timer > 0:
		for o in others:
			free_space[o] = False
	d = look_for_targets(free_space, (x,y), targets, self.logger)
	if d == (x,y-1): action_ideas.append('UP')
	if d == (x,y+1): action_ideas.append('DOWN')
	if d == (x-1,y): action_ideas.append('LEFT')
	if d == (x+1,y): action_ideas.append('RIGHT')
	if d is None:
		self.logger.debug('All targets gone, nothing to do anymore')
		action_ideas.append('WAIT')

	# Add proposal to drop a bomb if at dead end
	if (x,y) in dead_ends:
		action_ideas.append('BOMB')
	# Add proposal to drop a bomb if touching an opponent
	if len(others) > 0:
		if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
			action_ideas.append('BOMB')
	# Add proposal to drop a bomb if arrived at target and touching crate
	if d == (x,y) and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(1) > 0):
		action_ideas.append('BOMB')

	# Add proposal to run away from any nearby bomb about to blow
	for xb,yb,t in bombs:
		if (xb == x) and (abs(yb-y) < 4):
			# Run away
			if (yb > y): action_ideas.append('UP')
			if (yb < y): action_ideas.append('DOWN')
			# If possible, turn a corner
			action_ideas.append('LEFT')
			action_ideas.append('RIGHT')
		if (yb == y) and (abs(xb-x) < 4):
			# Run away
			if (xb > x): action_ideas.append('LEFT')
			if (xb < x): action_ideas.append('RIGHT')
			# If possible, turn a corner
			action_ideas.append('UP')
			action_ideas.append('DOWN')
	# Try random direction if directly on top of a bomb
	for xb,yb,t in bombs:
		if xb == x and yb == y:
			action_ideas.extend(action_ideas[:4])

	#self.logger.debug(f'STATE: {self.state}')

	#Exploitation step
	if np.random.rand() >= self.epsilon:
		act_values = self.model.predict(self.state.reshape(1, -1))
		self.indx_action = np.argmax(act_values[0])
		self.next_action = self.actions[self.indx_action]
		self.NN_agent += 1
		if self.next_action in valid_actions:
			self.logger.info('Picking action according to TRAINING')
			return

	#Exploration step
	# rand_indx= np.random.choice(self.actions_size, 1)
	# self.indx_action = rand_indx
	# if self.actions[rand_indx] in valid_actions:
	#   self.next_action = self.actions[rand_indx]
	# else:
	#   self.next_action = 'WAIT'

	# Pick last action added to the proposals list that is also valid
	while len(action_ideas) > 0:
		a = action_ideas.pop()
		if a in valid_actions:
			self.next_action = a
			break
	

	# Keep track of chosen action for cycle detection
	if self.next_action == 'BOMB':
		self.bomb_history.append((x,y))


	
	self.simple_agent += 1
	self.indx_action = self.map_actions[self.next_action]
	self.logger.info('Picking action according to RULE SET')
	#self.next_action =  self.actions[indx_random]

	

def reward_update(self):

	#print("I am inside reward_update")
	#updating rewards for different events during training

	self.logger.debug(f'Encountered {len(self.events)} game event(s)')


	self.next_state = state_update(self)

	# The agent took a invalid action and it should be punished and counting the invalid actions
	if e.INVALID_ACTION in self.events:
		self.reward += self.reward_dict['INVALID_ACTION']
		self.logger.debug("INVALID ACTION")
		self.invalid_actions += 1
    # Killing the opponent
	if e.KILLED_OPPONENT in self.events:
		self.reward += self.reward_dict['KILLED_OPPONENT'] 
		self.logger.debug("KILLED OPPONENT")
    # dropping bomb
	if e.BOMB_DROPPED in self.events:
		self.reward += self.reward_dict['BOMB_DROPPED']
		self.logger.debug("BOMB DROPPED")
    # surviving the round
	if e.SURVIVED_ROUND in self.events:
		self.reward += self.reward_dict['SURVIVED_ROUND']
		self.logger.debug("SURVIVED ROUND")
    # getting killed
	if e.KILLED_SELF in self.events:
		self.reward += self.reward_dict['KILLED_SELF']
		self.logger.debug("KILLED SELF")
    # crates being destroyed
	if e.CRATE_DESTROYED in self.events:
		self.reward += self.reward_dict['CRATE_DESTROYED']
		self.logger.debug("CRATE DESTROYED")

	# A coin was found, therefore the agent receives a reward for that
	if e.COIN_COLLECTED in self.events:
		self.reward += self.reward_dict['COIN_COLLECTED']
		self.coins_collected += 1
		self.logger.debug("COIN COLLECTED")
	else:
		self.reward += self.reward_dict['EMPTY_CELL']

	self.total_reward += self.reward

	remember(self, self.state.reshape(1, -1), self.indx_action, self.reward, self.next_state.reshape(1, -1), False)


def end_of_episode(self):

	#scores=[]

	#print("I am inside end_of_episode")
	
	

	self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')

	self.next_state = state_update(self)

	
	# The agent took a invalid action and it should be punished
	if e.INVALID_ACTION in self.events:
		self.reward += self.reward_dict['INVALID_ACTION']
		self.logger.debug("INVALID ACTION")
		self.invalid_actions += 1

	if e.KILLED_OPPONENT in self.events:
		self.reward += self.reward_dict['KILLED_OPPONENT'] 
		self.logger.debug("KILLED OPPONENT")

	if e.BOMB_DROPPED in self.events:
		self.reward += self.reward_dict['BOMB_DROPPED']
		self.logger.debug("BOMB DROPPED")

	if e.SURVIVED_ROUND in self.events:
		self.reward += self.reward_dict['SURVIVED_ROUND']
		self.logger.debug("SURVIVED ROUND")

	if e.KILLED_SELF in self.events:
		self.reward += self.reward_dict['KILLED_SELF']
		self.logger.debug("KILLED SELF")

	if e.CRATE_DESTROYED in self.events:
		self.reward += self.reward_dict['CRATE_DESTROYED']
		self.logger.debug("CRATE DESTROYED")


	# A coin was found, therefore the agent receives a reward for that
	if e.COIN_COLLECTED in self.events:
		self.reward += self.reward_dict['COIN_COLLECTED']
		self.coins_collected += 1
		self.logger.debug("COIN COLLECTED")
	else:
		self.reward += self.reward_dict['EMPTY_CELL']

	self.total_reward += self.reward
	reward_episode = self.reward
	rewards_episodes_list.append(self.reward)

	remember(self, self.state.reshape(1, -1), self.indx_action, self.reward, self.next_state.reshape(1, -1), True)

	scores.append(self.coins_collected)


	#calling the replay function wth batch_size 100
	replay(self,100)
	self.episodes += 1
	
	
    #logging for debugging
	self.logger.debug(f'Episode No: {self.episodes}')
	self.logger.debug(f'Coins collected: {self.coins_collected}')
	self.logger.debug(f'Value of Epsilon: {self.epsilon}')
	self.logger.debug(f'Count of Simple agent: {self.simple_agent}')
	self.logger.debug(f'Count of Neural agent: {self.NN_agent}')
	self.logger.debug(f'Count of Invalid actions: {self.invalid_actions}')
	self.logger.debug(f'Reward per episode: {reward_episode}')


	logging.basicConfig(filename='app-nn.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
	logging.debug('This will get logged to a file')
	logging.debug(f'Episode No: {self.episodes}')
	logging.debug(f'Coins collected: {self.coins_collected}')
	logging.debug(f'Value of Epsilon: {self.epsilon}')
	logging.debug(f'Count of Simple agent: {self.simple_agent}')
	logging.debug(f'Count of Neural agent: {self.NN_agent}')
	logging.debug(f'Count of Invalid actions: {self.invalid_actions}')
	logging.debug(f'Reward per episode: {reward_episode}')

    #reinitializing the counters for next episode
	self.coins_collected = 0
	self.NN_agent = 0
	self.simple_agent = 0
	self.invalid_actions = 0
	reward_episode = 0
	

	if (self.episodes == s.n_rounds):
		self.logger.debug('Saving the model')
		self.model.save('user_nn_10k.h5')
		print ("Coins collected", scores)
		#potting the total number of coins collected during training over all rounds
		pyplot.plot(np.arange(self.episodes),scores)
		pyplot.xlabel('Round Number')
		pyplot.ylabel('Coins collected')
		pyplot.show()

		print(len(self.loss))
        #plotting the training loss over all the rounds
		pyplot.plot(self.loss)
		pyplot.xlabel('Rounds')
		pyplot.ylabel('Loss')
		pyplot.legend(['train', 'test'], loc='upper left')
		pyplot.show()
        #plotting the rewards over rounds
		pyplot.plot(np.arange(self.episodes), rewards_episodes_list)
		pyplot.xlabel('Rounds')
		pyplot.ylabel('Rewards')
		pyplot.show()


	# At the end of the game, the agent is being rewarded
	# with the number of coins that it has collected
	self.reward = self.reward_dict['COIN_COLLECTED'] * self.coins_collected





def learn(agent):

	pass
