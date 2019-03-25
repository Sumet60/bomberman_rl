import numpy as np
from random import shuffle
from time import sleep
from settings import e 
from settings import s
from collections import deque


###################
#Hyperparameters
###################
#exploration
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005
#learning rate and discounting rate
alpha = 0.8
gamma = 0.95
#rar = 0.5
#counter

        
def setup(self):
    """Called once before a set of games to initialize data structures etc."""
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    
    #our code
    self.count_step = 1
    self.count_episode = 1
    
    self.numer_actions = [0,1,2,3,4,5] #up down l r WAIT bomb
    self.num_actions = 6
    
    #state
    self.numer_state = [0,1,2,3,4,5] #bomb coin killself killop wall crate
    self.num_states = 15*15 #17*17
    self.s = None
    self.s1 = None
    
    #action
    self.all_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB'] #agent.actions  
    self.action_q = None
    
    
    # Rewards
    
    self.reward = 0.0
    self.total_reward = 0.0
    
    #latest values
    self.last_state = None
    self.last_action = None
    self.last_reward = None
    self.targets = None
    
    self.reward_dict = {
            'KILLED_OPPONENT' : 500, #
            'KILLED_SELF' : -100, #
            'GOT_KILLED' : -50,
            'CRATE_DESTROYED': 200, #
            'COIN_COLLECTED':  500, #
            'INVALID_ACTION' : -4, #-4.0/s.rows #
            'EMPTY_CELL' : -4, #-1.0/(s.rows*s.cols) #
            'MAX_REWARD' : 500 #
    }
    #initialize the q_table
    self.reset = True
    if self.reset:
        self.table_Q = np.zeros(shape=(self.num_states, self.num_actions))
        self.reward_table = np.zeros(shape=(self.num_states, self.num_actions))
        self.Rt = [] #np.zeros(100)
    else:
        #2. load tables for q, r, total r
        self.table_Q = np.load('q_table.npy')
        self.reward_table = np.load('reward_table.npy')
        self.Rt = [] #np.zeros(100)
    
    
def act(self):
    """Called each game step to determine the agent's next action.
    """
    self.logger.info('Picking action according to rule set')
    
    #count steps in each episode
    self.count_step += 1

    # Gather information about the game state
    #used to check wall, crate, free tile
    arena = self.game_state['arena']
    #position of my agent and other agents
    x, y, _, bombs_left, score = self.game_state['self']
    #self.s = 15*(int(x)-1)+(int(y)-1)  #current state for q learning
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    #bomb
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)
    #coin
    coins = self.game_state['coins']
    
    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x,y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x,y))
   
    #find valid actions 
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
    
    if len(others)==0:
        valid_actions.remove('BOMB')
    
    #collecting ideas to take action
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)
    
    #q learning
    #1 choose action 
    tradeoff = np.random.uniform(0,1)
    if tradeoff > epsilon:
        action_q = np.argmax(table_Q[self.s,:])
    else:
        action_q = np.random.randint(0,self.num_actions)
 
    if action_q == 0:
        action_ideas.append('UP')
        #self.s1 = 15*(int(x)-1)+(int(y)-1-1)
    if action_q == 1:
        action_ideas.append('DOWN')
        #self.s1 = 15*(int(x)-1)+(int(y)-1+1)
    if action_q == 2:
        action_ideas.append('LEFT')
        #self.s1 = 15*(int(x)-1-1)+(int(y)-1)
    if action_q == 3:
        action_ideas.append('RIGHT')
        #self.s1 = 15*(int(x)-1+1)+(int(y)-1)
    if action_q == 4:
        action_ideas.append('WAIT')
        #self.s1 = 15*(int(x)-1)+(int(y)-1)
    if action_q == 5:
        action_ideas.append('BOMB')
        
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            self.next_action = a
            break
    self.logger.debug(f'Action ideas : {action_ideas}')
    
    #2 find next_state s1 and reward=score
    x0, y0, _, bombs_left, score = self.game_state['self']
    self.s1 = 15*(int(x0)-1)+(int(y0)-1)
    #reward = score
    reward_update(self)
    #3 compute q-value
    max_Q = max([self.table_Q[self.s1, a] for a in self.numer_actions]) 
    #use if better than for
    old_Q = self.table_Q[self.s, action_q]
    self.table_Q[self.s, action_q] = old_Q + alpha*((self.reward + gamma*max_Q) - old_Q)
    self.logger.debug(f'table Q : {self.table_Q}')

    if self.next_action == 'BOMB':
        self.bomb_history.append((x,y)) 
    
    #memo latest state, latest action, lastes reward for next round
    self.s = self.s1
    self.last_action = self.next_action
    self.last_reward = self.reward

def reward_update(self):

    self.logger.debug(f'Encountered {len(self.events)} game event(s)')

    self.reward += -10
    #self.next_state = state_update(self)


    if e.INVALID_ACTION in self.events: #self.events is in environment.py -> agents.py
        self.reward = self.reward_dict['INVALID_ACTION']
        #self.logger.debug('INVALID ACTION')

    if e.KILLED_OPPONENT in self.events:
        self.reward = self.reward_dict['KILLED_OPPONENT'] 
        #self.logger.debug('KILLED_OPPONENT')

    if e.KILLED_SELF in self.events:
        self.reward = self.reward_dict['KILLED_SELF']
        #self.logger.debug('KILLED_SELF')

    if e.COIN_COLLECTED in self.events:
        self.reward = self.reward_dict['COIN_COLLECTED']
        self.coins_collected += 1
        #self.logger.debug('COIN_COLLECTED')
        
    if e.CRATE_DESTROYED in self.events:
        self.reward = self.reward_dict['CRATE_DESTROYED']
        #self.logger.debug('CRATE_DESTROYED')
    if e.GOT_KILLED in self.events:
        self.reward = self.reward_dict['GOT_KILLED']
    else:
        self.reward = self.reward_dict['EMPTY_CELL']
        #self.logger.debug('EMPTY_CELL')
    self.total_reward += self.reward
    #print(self.Rt)
   
    
def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.
    """
    #tutor's code
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
    
    #our code
    epsilon = min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay_rate*self.count_episode)
    self.count_episode += 1
    self.logger.debug(f'Epsilon: {epsilon}')
    
    #total rewards at each episode
    self.Rt.append(self.total_reward)
    np.save('total_rewards.npy', self.Rt)
    
    #rewards
    np.save('reward_table.npy', self.reward_table)
    
    #save q-table
    np.save('q_table.npy', self.table_Q)
    

   
