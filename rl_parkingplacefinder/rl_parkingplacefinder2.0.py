
# %%
"""
    parkingplacefinder is an OpenSource python package for the reinforcement learning
    of an agent that searches for the most optimal free parking spot in a parking lot

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
"""
#%%
import numpy as np
import collections
import matplotlib.pylab as plt
import networkx as nx
from PIL import Image
import cv2
import scipy.misc

# from rl_parkingplacefinder.Parking_lot import Parking_Lot as Parking_Lot
parking_lot = nx.read_gpickle('/Users/pascal/Coding/DRL_ParkingPlaceFinderRobot/parking_lot.gpl')


# %%

PARK_CRASH_REWARD = -50
WALL_CRASH_REWARD = -50
TIME_REWARD = -0.25
BACKWARD_REWARD = -40
STUCK_REWARD = -30
EPISODES = 60000


#%%
"""
TODO:
    - EWhat is the best option to combine optimization for 1) reward of parking lot 2) walking distance 3) parking distance
    - Enable rendering with agent on the terminal state for better understanding
    - After maximizing reward (very early) agent somehow still goes for smaller rewards and sticks to close parking slots in the end
    - Allow agent only to drive straight into parking lot




"""
# %%
class Park_Finder_Agent():
    def __init__(self, parking_lot):
        self.parking_lot = parking_lot
        self.m = self.get_parking_lot_width()
        self.n = self.get_parking_lot_length()
        self.actionSpace = {'U': -self.m, 'D': self.m, 'L': -1, 'R': 1}
        self.possibleActions = ['U', 'D', 'L', 'R']
        self.taken_list = []
        self.vacant_list = []
        self.drive_list = []
        for i in range(len(self.parking_lot.nodes)):
            if len(parking_lot.nodes[i]) == 2:
                if parking_lot.nodes[i]['occupation'] == 'taken':
                    self.taken_list.append(i)
                elif parking_lot.nodes[i]['occupation'] == 'vacant':
                    self.vacant_list.append(i)
            else:
                self.drive_list.append(i)
        self.stateSpacePlus = self.drive_list #+ self.vacant_list + self.taken_list
        self.stateSpace = self.vacant_list + self.taken_list
        self.agentPosition = 0
        self.grid = self.parkingLotToArray()
        
        
    def get_parking_lot_width(self):
        # gives back the number of rows of the complete parking lot including driveways.
        connection_list = []
        for pair in self.parking_lot.edges:
            connection_list.append(pair[0])
        single_connection_list = [item for item, count in collections.Counter(connection_list).items() if count == 1]
        return single_connection_list[1]-single_connection_list[0]

    def get_parking_lot_length(self):
        return int(len(self.parking_lot.nodes)/self.get_parking_lot_width())
     
    def parkingLotToArray(self):
        parking_lot_indices = np.array_split(self.parking_lot.nodes, self.m)
        grid = np.zeros((self.m,self.n))
        for i in range(0,self.m):
            for k in range(0,self.n):
                if self.parking_lot.node[parking_lot_indices[i][k]]['slot_type'] == 'drive':
                    grid[i][k] = 1
                if self.parking_lot.node[parking_lot_indices[i][k]]['slot_type'] == 'park' and self.parking_lot.node[parking_lot_indices[i][k]]['occupation'] == 'vacant':
                    grid[i][k] = 2
                if self.parking_lot.node[parking_lot_indices[i][k]]['slot_type'] == 'park' and self.parking_lot.node[parking_lot_indices[i][k]]['occupation'] == 'taken':
                    grid[i][k] = 0
        return grid
                
                
    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace
    
    def getElementRowAndColumn(self, position):
        x = position // self.m
        y = position % self.n
        return x, y

    def getAgentRowAndColumn(self):
        x = self.agentPosition // self.m
        y = self.agentPosition % self.n
        return x, y
    
    
    def setState(self, state):
        # where agent was, make it driveway again
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 1
        # where agent is, make square
        self.agentPosition = state
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 3
        
    def getReward(self, actualState, resultingState):
        # reward of -1 for wasting time and driving around
        if resultingState in self.drive_list:
            if resultingState < actualState:
                return BACKWARD_REWARD
            if resultingState == actualState:
                return STUCK_REWARD
            else:
                return TIME_REWARD
        # reward of -300 of crashing in a parked car
        if resultingState in self.taken_list:
            return PARK_CRASH_REWARD
        # reward for a parking lot. If the distance to the exit is close, the reward is nearly 25. If the distance is far, reward gets smaller
        if resultingState in self.vacant_list and resultingState != max(self.vacant_list):
            return - (nx.shortest_path_length(parking_lot,source=self.agentPosition,target=max(self.parking_lot.nodes)))/40
        if resultingState == max(self.vacant_list):
            return 1
        
        else:
            # reward of -400 for hitting the wall on the side of the parking lot
            return WALL_CRASH_REWARD
        
        
    def step(self, action):
        # agentX, agentY = self.getAgentRowAndColumn()
        resultingState = self.agentPosition + self.actionSpace[action]

        reward = self.getReward(self.agentPosition,resultingState)

        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)
            # self.agentPosition = resultingState
            return resultingState, reward, self.isTerminalState(resultingState), None
        else:
            return self.agentPosition, reward, self.isTerminalState(self.agentPosition), None

    def offGridMove(self, newState, oldState):
        # if we move into a row not in the grid
        if newState not in self.stateSpacePlus:
            return True
        # if we're trying to wrap around to next row
        elif oldState % self.m == 0 and newState  % self.m == self.m - 1:
            return True
        elif oldState % self.m == self.m - 1 and newState % self.m == 0:
            return True
        else:
            return False
        
    def render(self):
        print('Entrance-----------------------------------------')
        for row in self.grid:
            for col in row:
                if col == 1:
                    print('.', end='\t')
                elif col == 2:
                    print('O', end='\t')
                elif col == 0:
                    print('X', end='\t')
                elif col == 3:
                    print('█', end='\t')
            print('\n')
        print('---------------------------------------------Exit')
        
    def renderToFile(self):
        H = self.grid
        plt.imshow(H)
        plt.show()


    def reset(self):
        self.agentPosition = 0
        self.grid = self.parkingLotToArray()
        return self.agentPosition
    
    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)
    

    
#%%
        


def print_frames(frames):
    for i, frame in enumerate(frames):
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Resulting state: {frame['resulting state']}")
        print(f"previous state(s): {frame['state history']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        print(f"Found parking: {frame['new start']}")
        print(f"Walking distance: {frame['walk distance']}")
        print(f"Driving distance: {frame['drive distance']}")
        print('-------------')
        

    
def maxAction(Q, state, actions):
    values = np.array([Q[state,a] for a in actions])
    # if all the values are 0 in the Q table, pick random an action (otherwise it would always chose the first one)
    if sum(values)==0:
        print("picking random")
        action = np.random.randint(0,4)
    else:
        # if there is already something learned, pick the one which has the highest reward attached to it
        action = np.argmax(values)
    return actions[action]

if __name__ == '__main__':


    env = Park_Finder_Agent(parking_lot)
    # model hyperparameters
    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 0.9
    frames = [] # for information
    fig = plt.figure()

    Q = {}
    for state in env.stateSpacePlus:
        for action in env.possibleActions:
            Q[state, action] = 0

    numEpisodes = EPISODES
    totalRewards = np.zeros(numEpisodes)
    for i in range(numEpisodes):
        observation = env.agentPosition
        # Every xth episode we reset the car to position 0 and start again 
        if i % (EPISODES/100) == 0:
            # print('starting episode ', i)
            observation = env.reset()
        done = False
        show = True
        
        epRewards = 0
        history = [0]
        
        while not done:
            finish = False

            rand = np.random.random()
            
            action = maxAction(Q,observation, env.possibleActions) if rand > EPS else env.actionSpaceSample()
            
            
            observation_, reward, done, info = env.step(action)
            
            if show:
                img = scipy.misc.toimage(env.grid)
                img = img.resize((750, 750))  # resizing so we can see our agent in all its glory.
                
                cv2.imshow("Parking Agent", np.array(img))
            
            # making sure that when parking lot was reached or a crash with a parked car occures, we terminate this episode (just experimental, should be handled in the getReward function)
            resulting_state = observation+env.actionSpace[action]
            # print(env.actionSpace[action])
            
            if reward >= 1 or reward == PARK_CRASH_REWARD or reward == WALL_CRASH_REWARD:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0x7F == ord('q'):
                    break
            # time.sleep(0.001)
                
            if observation_+env.actionSpace[action] in env.vacant_list:
                finish = True
            #     reward = 25
            if observation_+env.actionSpace[action] in env.taken_list:
                finish = True
            #     reward = -300
            epRewards += reward
            # print(epRewards)

            action_ = maxAction(Q, observation_, env.possibleActions)
            Q[observation,action] = Q[observation,action] + ALPHA*(reward + GAMMA*Q[observation_,action_] - Q[observation,action])
            history.append(observation_)
            observation = observation_
            if finish:
                # print("start a new episode")
                done = True
                env.reset()
                history.append(resulting_state)
            if resulting_state in env.vacant_list+env.taken_list+env.drive_list:
                walk_distance = nx.shortest_path_length(parking_lot,source=resulting_state,target=max(env.parking_lot.nodes))
                drive_distance = nx.shortest_path_length(parking_lot,source=0,target=resulting_state)
            else:
                walk_distance = False
                drive_distance = False
                
            frames.append({'state': observation,'resulting state': resulting_state,'state history':history, 'action': action,'reward': reward, 'new start': done,'walk distance': walk_distance, 'drive distance':drive_distance})
        if EPS - 2 / numEpisodes > 0:
            EPS -= 2 / numEpisodes
        else:
            EPS = 0
        totalRewards[i] = epRewards
        # if i % 100 ==0:
        #     env.render()
        
        
    print(print_frames(frames))
    plt.plot(totalRewards)
    plt.show()

        
                    
#%%

    # def find_parking(self, parking_lot :  Parking_Lot):
    #     # if True: let an agent run through the created parking lot and park in the first available parking spot

    #     curr_pos = 0 # stores current position of agent, currently always starts on node 0
    #     status = 'moving' # movement status of agent
    #     step = 0 # counts number of steps agent takes until it parks
    #     print("History:")

    #     while status == 'moving':
    #         step += 1

    #         # get all neighboring nodes ("possible ways to drive")
    #         options = list(parking_lot.g.neighbors(curr_pos))
    #         print(f"Step {step}: Currently on node {curr_pos}")

    #         # check if parking spot is vacant and park on it if yes
    #         for spot in options:
    #             if parking_lot.g.nodes[spot]['slot_type'] == 'park' and parking_lot.g.nodes[spot]['occupation'] == 'vacant':
    #                 curr_pos = spot
    #                 status = 'parked'
    #                 print(f"We park in Spot {curr_pos} after {step} steps")
    #                 break

    #         # update the status of the parking slot
    #         if (status == 'parked'):
    #             parking_lot.g.nodes[curr_pos]['occupation'] = 'taken'
    #             parking_lot.node_color_map[curr_pos]='red'
    #             # unelegant return but for efficiency
    #             return curr_pos

    #         # if no parking spot is vacant: restrict set of options to driveway nodes and randomly continue
    #         options = [spot for spot in options if parking_lot.g.nodes[spot]['slot_type'] == 'drive']
    #         curr_pos = rnd.choice(options)







#%%
