
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
import numpy.random as rnd
import collections

import matplotlib.pylab as plt

import time
from datetime import datetime
import math

import networkx as nx
import csv

# from rl_parkingplacefinder.Parking_lot import Parking_Lot as Parking_Lot
parking_lot = nx.read_gpickle('/Users/pascal/Coding/DRL_ParkingPlaceFinderRobot/parking_lot.gpl')


# %%
"""
class ParkingPlaceFinderModel():
    This is the general class to run a reinforcement learning parking place finder experiment
    Parameters:
    def __init__(self):
        self.id_name = ""
        self.rl_strategy = ""
"""

# %%
class Park_Finder_Agent():
    def __init__(self, parking_lot):
        self.parking_lot = parking_lot
        # self.row = agent//self.nr_parking_lanes
        # self.col = agent%self.nr_parking_lanes
        self.id_name = 'hola'
        self.m = self.get_parking_lot_width()
        self.n = self.get_parking_lot_length()
                
        self.actionSpace = {'U': -self.m, 'D': self.m,
                            'L': -1, 'R': 1}
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
        self.stateSpacePlus = self.drive_list
        self.stateSpace = self.vacant_list + self.taken_list
        # self.grid = np.zeros((self.m,self.n))
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
        parking_lot_indices = np.array_split(self.parking_lot.nodes, 7)
        grid = np.zeros((self.m,self.n))
        for i in range(0,self.m):
            for k in range(0,self.n):
                if self.parking_lot.node[parking_lot_indices[i][k]]['slot_type'] == 'drive':
                    grid[i][k] = 1
                if self.parking_lot.node[parking_lot_indices[i][k]]['slot_type'] == 'park' and self.parking_lot.node[parking_lot_indices[i][k]]['occupation'] == 'vacant':
                    grid[i][k] = 0
                if self.parking_lot.node[parking_lot_indices[i][k]]['slot_type'] == 'park' and self.parking_lot.node[parking_lot_indices[i][k]]['occupation'] == 'taken':
                    grid[i][k] = 2
        return grid
                
                
    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace

    def getAgentRowAndColumn(self):
        x = self.agentPosition // self.m
        y = self.agentPosition % self.n
        return x, y
    
    def setState(self, state):
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 1
        self.agentPosition = state
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 3
        
    def getReward(self, resultingState):
        if not self.isTerminalState(resultingState):
            return -1
        if resultingState in self.taken_list:
            return -5
        if resultingState in self.vacant_list:
            return  5 / nx.shortest_path_length(parking_lot,source=self.agentPosition,target=max(self.parking_lot.nodes))
        else:
            return 0
        
    def step(self, action):
        agentX, agentY = self.getAgentRowAndColumn()
        resultingState = self.agentPosition + self.actionSpace[action]

        reward = self.getReward(resultingState)
        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)
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
                elif col == 0:
                    print('O', end='\t')
                elif col == 2:
                    print('X', end='\t')
                elif col == 3:
                    print('█', end='\t')
            print('\n')
        print('---------------------------------------------Exit')


    def reset(self):
        self.agentPosition = 0
        return self.agentPosition
    
    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)
    
#%%    
    
    
from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    for i, frame in enumerate(frames):
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['observation']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        print('-----------------------------')
        

    
def maxAction(Q, state, actions):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]

if __name__ == '__main__':
    # map magic squares to their connecting square
#    magicSquares = {18: 54, 63: 14}
    env = Park_Finder_Agent(parking_lot)
    # model hyperparameters
    ALPHA = 0.01
    GAMMA = 0.6
    EPS = 0.4
    frames = [] # for animation

    Q = {}
    for state in env.stateSpacePlus:
        for action in env.possibleActions:
            Q[state, action] = 0

    numGames = 30
    totalRewards = np.zeros(numGames)
    for i in range(numGames):
        if i % 30 == 0:
            print('starting game ', i)
        done = False
        epRewards = 0
        observation = env.reset()
        while not done:
            
            rand = np.random.random()
            action = maxAction(Q,observation, env.possibleActions) if rand < (1-EPS) \
                                                    else env.actionSpaceSample()
            observation_, reward, done, info = env.step(action)
            epRewards += reward

            action_ = maxAction(Q, observation_, env.possibleActions)
            Q[observation,action] = Q[observation,action] + ALPHA*(reward + \
                        GAMMA*Q[observation_,action_] - Q[observation,action])
            observation = observation_
            
        
        frames.append({'observation': observation,'action': action,'reward': reward})
        
        if EPS - 2 / numGames > 0:
            EPS -= 2 / numGames
        else:
            EPS = 0
        totalRewards[i] = epRewards
        env.render()
        
        
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
