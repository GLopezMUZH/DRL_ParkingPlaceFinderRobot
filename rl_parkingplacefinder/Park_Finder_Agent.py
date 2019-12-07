#%%
import numpy as np
import collections
import networkx as nx
import matplotlib.pylab as plt

import os
os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import Parking_lot
from Parking_lot import Parking_Lot


#%%
class Reward_Parameters():
    def __init__(self):
        self.PARK_CRASH_REWARD = -200
        self.WALL_CRASH_REWARD = -200
        self.TIME_REWARD = -1
        self.BACKWARD_REWARD = -40
        self.STUCK_REWARD = -30
        self.PARKING_REWARD = 100
        self.DRIVEWAY_PARKING_REWARD = -100



# %%
class Park_Finder_Agent():
    def __init__(self, reward_parameters: Reward_Parameters, parking_environment: Parking_Lot):
        self.parking_lot = parking_environment.get_env()
        print(self.parking_lot)
        self.m = self.get_parking_lot_width()
        self.n = self.get_parking_lot_length()
        # 1 = UP, 2 = Down, 3 = Left, 4 = Right, 5 = Park
        self.actionSpace = {1: -self.m, 2: self.m, 3: -1, 4: 1, 5: 0}
        self.possibleActions = [1, 2, 3, 4, 5]
        self.taken_list = []
        self.vacant_list = []
        self.drive_list = []
        for i in range(len(self.parking_lot.nodes)):
            if len(self.parking_lot.nodes[i]) == 2:
                if self.parking_lot.nodes[i]['occupation'] == 'taken':
                    self.taken_list.append(i)
                elif self.parking_lot.nodes[i]['occupation'] == 'vacant':
                    self.vacant_list.append(i)
            else:
                self.drive_list.append(i)
        self.stateSpace = self.drive_list + self.vacant_list
        self.stateSpacePlus = self.drive_list + self.vacant_list + self.taken_list
        self.agentPosition = 0
        self.grid = self.parkingLotToArray()
        self.reward_parameters = reward_parameters
        
        
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
                if self.parking_lot.nodes[parking_lot_indices[i][k]]['slot_type'] == 'drive':
                    grid[i][k] = 1
                if self.parking_lot.nodes[parking_lot_indices[i][k]]['slot_type'] == 'park' and self.parking_lot.nodes[parking_lot_indices[i][k]]['occupation'] == 'vacant':
                    grid[i][k] = 2
                if self.parking_lot.nodes[parking_lot_indices[i][k]]['slot_type'] == 'park' and self.parking_lot.nodes[parking_lot_indices[i][k]]['occupation'] == 'taken':
                    grid[i][k] = 0
        return grid
                
                
    def isTerminalState(self, state, action):
        return state in self.vacant_list and action == 5
        # return state in self.stateSpacePlus and state not in self.stateSpace
    
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
        
    def getReward(self, actualState, resultingState, action):
        # reward of -1 for wasting time and driving around
        if resultingState in self.drive_list:
            if resultingState < actualState and action in [1,2,3,4]:
                return self.reward_parameters.BACKWARD_REWARD
            if resultingState == actualState and action == 5:
                return self.reward_parameters.DRIVEWAY_PARKING_REWARD
            else:
                return self.reward_parameters.TIME_REWARD


        # reward of -300 of crashing in a parked car
        if resultingState in self.taken_list:
            return self.reward_parameters.PARK_CRASH_REWARD
        if actualState in self.taken_list:
            if action == 5:
                return 10 * self.reward_parameters.PARK_CRASH_REWARD
            else:
                return 5 * self.reward_parameters.PARK_CRASH_REWARD

        if actualState in self.vacant_list:
            if actualState == resultingState:
                if resultingState != max(self.vacant_list) and action == 5:
                    disc_reward = self.reward_parameters.PARKING_REWARD / ((nx.shortest_path_length(self.parking_lot,
                                                                                             source=self.agentPosition,
                                                                                             target=max(self.drive_list))) * 2)
                    return disc_reward
                if resultingState == max(self.vacant_list) and action == 5:
                    return self.reward_parameters.PARKING_REWARD
                # when driving over an empty parking slot to reach a better empty parking slot
            if resultingState != actualState and resultingState in self.vacant_list:
                disc_reward = self.reward_parameters.PARKING_REWARD / ((nx.shortest_path_length(self.parking_lot,
                                                                                                source=self.agentPosition,
                                                                                                target=max(self.drive_list))) * 2)
                return disc_reward
            else:
                return self.reward_parameters.TIME_REWARD

        if actualState in self.drive_list and resultingState in self.vacant_list:
            return self.reward_parameters.TIME_REWARD

        if self.offGridMove(resultingState,actualState):
            return self.reward_parameters.WALL_CRASH_REWARD

        else:
            return 0

        
        
    def step(self, action):
        # agentX, agentY = self.getAgentRowAndColumn()
        resultingState = self.agentPosition + self.actionSpace[action]
        reward = self.getReward(self.agentPosition,resultingState, action)
        if reward == 0:
            print("something went wrong with the reward.. {} -> {}, action: {}".format(self.agentPosition,resultingState,action))
        else:
            reward = round(reward, 3)
        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)
            # self.agentPosition = resultingState
            return resultingState, reward, self.isTerminalState(resultingState, action), None
        else:
            return self.agentPosition, reward, self.isTerminalState(self.agentPosition, action), None


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
    

# %%
