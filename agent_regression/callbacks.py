from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

import numpy as np
from random import shuffle
from time import sleep
from settings import e 
from settings import s
from collections import deque

#exploration
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005
#learning rate and discounting rate
alpha = 0.8
gamma = 0.95

#--------------------------------------------------------------------------------------------------------------------------------------#
                               #Helping Function (by Prof. Kothe and Tutor), 
                        #only this part that I copied. Please do not count for points
#--------------------------------------------------------------------------------------------------------------------------------------#

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.
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
    return best

#--------------------------------------------------------------------------------------------------------------------------------------#
                                                #Set up everything important
#--------------------------------------------------------------------------------------------------------------------------------------#     
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
    self.count_episode = 1  
    self.numer_actions = [0,1,2,3,4,5] #up down l r WAIT bomb
    self.num_actions = 6
    
    #state    
    self.features = 6
    self.s = None
    self.s1 = None
       
    #action
    self.all_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB'] #agent.actions  
    self.action_q = None
    
    #Rewards
    self.reward = 0.0
    self.total_reward = 0.0
    
    #latest values
    self.last_state = None
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
    self.reset = False #If False, load the existing weights and use regression fitting
    
    #regression
    self.regr = MultiOutputRegressor(LGBMRegressor(zero_as_missing=True, use_missing=False))
    if self.reset:
        #1. declare tables for q, r, total r
        self.last_actions = np.zeros((0,1), dtype=np.int8)
        self.table_Q = np.zeros(shape=(2, self.features))
        self.observ_table = np.zeros(shape=(2, self.num_act))
        self.reward_table = np.zeros(shape=(2, self.num_actions))
        self.regr.fit(self.observ_table, self.table_Q) 
        self.Rt = [] #np.zeros(100)
    else:
        #2. load tables for q, r, total r
        self.last_actions = np.load('last_actions.npy')
        self.table_Q = np.load('q_table.npy')
        self.observ_table = np.load('observ_table.npy')
        self.reward_table = np.load('reward_table.npy')
        self.regr.fit(self.observ_table, self.table_Q) 
        self.Rt = [] #np.zeros(100)
        self.logger.info('Model is trained')
        
    #states
    self.s,_ = np.shape(self.table_Q) 
    self.s = self.s-2
    self.s1,_ = np.shape(self.table_Q)
    self.s1 = self.s1-1
    
#--------------------------------------------------------------------------------------------------------------------------------------#
                                                #Run in time step of each episode
#--------------------------------------------------------------------------------------------------------------------------------------#     
def act(self):
    """Called each game step to determine the agent's next action.
    """
    self.logger.info('Picking action according to rule set')

    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    #coin
    coins = self.game_state['coins']
    #find targets
    free_space = arena == 0
    targets = []
    tar = look_for_targets(free_space, (x,y), coins)
    targets.append(tar)
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    #bomb
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)  
    #observations
    top = arena[x, y-1]
    bottom = arena[x, y+1]
    left = arena[x-1, y]
    right = arena[x+1, y]
    observ = [left, right, top, bottom]
    for tar in targets:
        if tar !=None:
            d = (tar[0]-x, tar[1]-y)
        else:
            d = (0,0)
        observ.append(d[0])
        observ.append(d[1])
    self.observ_table = np.vstack((self.observ_table, observ))

    #q learning
    #1 choose action 
    tradeoff = np.random.uniform(0,1)
    if tradeoff > epsilon:
        action_q = np.argmax(table_Q[self.s,:])
        self.last_actions = np.vstack((self.last_actions, action_q))
    else:
        action_q = np.random.randint(0,self.num_actions)
        self.last_actions = np.vstack((self.last_actions, action_q))
    self.next_action = self.all_actions[self.last_actions[-1][0]] 

    #2. find reward r and state s'
    #reward_update(self)
    x0, y0, _, bombs_left, score = self.game_state['self']
    get_reward = np.zeros((1, self.num_actions))
    get_reward[0][self.last_actions[-1]] = self.reward
    self.reward_table = np.vstack((self.reward_table, get_reward))
    #3 compute q-value
    if self.reset:
        max_Q = max([self.table_Q[self.s1][a] for a in self.numer_actions]) 
    else:
        arr_maxQ = np.argmax(self.regr.predict(self.observ_table), axis=1)
        max_Q = arr_maxQ[np.random.randint(len(arr_maxQ), size=1)] 
 
    old_Q = self.table_Q[self.s][action_q]
    old_Q = old_Q + alpha*((self.reward + gamma*max_Q) - old_Q)
    Q=np.zeros((1,6))
    Q[0,action_q]= old_Q
    self.table_Q = np.vstack((self.table_Q, Q))
    #self.logger.debug(f'table Q : {self.table_Q}')

    if self.next_action == 'BOMB':
        self.bomb_history.append((x,y)) 
    
    #memo latest state, latest action, lastes reward for next round
    self.s=self.s1
    self.s1 += 1
    self.last_action = self.next_action
    self.last_reward = self.reward
    #print(self.reward)
    #print(self.total_reward)
    
#--------------------------------------------------------------------------------------------------------------------------------------#
                                                #For training
#--------------------------------------------------------------------------------------------------------------------------------------#

def reward_update(self):
    '''Called once per step to allow intermediate rewards based on game events.
    '''
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    self.reward += -10
    if e.INVALID_ACTION in self.events: 
        self.reward = self.reward_dict['INVALID_ACTION']

    if e.KILLED_OPPONENT in self.events:
        self.reward = self.reward_dict['KILLED_OPPONENT'] 

    if e.KILLED_SELF in self.events:
        self.reward = self.reward_dict['KILLED_SELF']

    if e.COIN_COLLECTED in self.events:
        self.reward = self.reward_dict['COIN_COLLECTED']
        #self.coins_collected += 1
        
    if e.CRATE_DESTROYED in self.events:
        self.reward = self.reward_dict['CRATE_DESTROYED']
        
    if e.GOT_KILLED in self.events:
        self.reward = self.reward_dict['GOT_KILLED']
        
    else:
        self.reward = self.reward_dict['EMPTY_CELL']       
    self.total_reward += self.reward

    
def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
    #our code
    epsilon = min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay_rate*self.count_episode)
    self.count_episode += 1
    self.logger.debug(f'Epsilon: {epsilon}')   
    #total rewards at each episode
    self.Rt.append(self.total_reward)
    #fit with the regression model
    self.regr.fit(self.observ_table, self.table_Q)
    np.save('last_actions.npy', self.last_actions) 
    np.save('total_rewards.npy', self.Rt)  
    #rewards
    np.save('reward_table.npy', self.reward_table) 
    #save q-table
    np.save('q_table.npy', self.table_Q)
    #save observation
    np.save('observ_table.npy', self.observ_table)
    
    #print(self.Rt)
   
