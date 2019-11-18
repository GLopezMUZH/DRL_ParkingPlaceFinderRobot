
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
        self.stateSpacePlus = self.drive_list
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
        # where agent was, make it driveway again
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 1
        # where agent is, make square
        self.agentPosition = state
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 3
        
    def getReward(self, resultingState):
        # reward of -1 for wasting time and driving around
        if resultingState in self.drive_list:
            return -1
        # reward of -300 of crashing in a parked car
        if resultingState in self.taken_list:
            return -300
        # reward for a parking lot. If the distance to the exit is close, the reward is nearly 25. If the distance is far, reward gets smaller
        if resultingState in self.vacant_list:
            return  25 / nx.shortest_path_length(parking_lot,source=self.agentPosition,target=max(self.parking_lot.nodes))
        else:
            # reward of -400 for hitting the wall on the side of the parking lot
            return -400
        
        
    def step(self, action):
        # agentX, agentY = self.getAgentRowAndColumn()
        resultingState = self.agentPosition + self.actionSpace[action]
        print(resultingState)

        reward = self.getReward(resultingState)
        if resultingState == self.agentPosition:
            reward = -200
        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)
            self.agentPosition = resultingState
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
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
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

    Q = {}
    for state in env.stateSpacePlus:
        for action in env.possibleActions:
            Q[state, action] = 0

    numEpisodes = 10
    totalRewards = np.zeros(numEpisodes)
    for i in range(numEpisodes):
        # Every xth episode we reset the car to position 0 and start again 
        if i % 5 == 0:
            print('starting episode ', i)
        done = False
        finish = False
        epRewards = 0
        observation = env.reset()
        while not done:

            rand = np.random.random()
            
            action = maxAction(Q,observation, env.possibleActions) if rand > EPS else env.actionSpaceSample()
            
            
            observation_, reward, done, info = env.step(action)
            
            # making sure that when parking lot was reached or a crash with a parked car occures, we terminate this episode (just experimental, should be handled in the getReward function)
            resulting_state = observation+env.actionSpace[action]
            print(env.actionSpace[action])
            if observation_+env.actionSpace[action] in env.vacant_list:
                finish = True
                reward = 25
            if observation_+env.actionSpace[action] in env.taken_list:
                finish = True
                reward = -300
            epRewards += reward
            # print(epRewards)

            action_ = maxAction(Q, observation_, env.possibleActions)
            Q[observation,action] = Q[observation,action] + ALPHA*(reward + GAMMA*Q[observation_,action_] - Q[observation,action])
            observation = observation_
            if finish:
                print("start a new episode")
                done = True
            frames.append({'state': observation,'resulting state': resulting_state, 'action': action,'reward': reward})

            
        
        
        
        if EPS - 2 / numEpisodes > 0:
            EPS -= 2 / numEpisodes
        else:
            EPS = 0
        totalRewards[i] = epRewards
        if i % 100 ==0:
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
