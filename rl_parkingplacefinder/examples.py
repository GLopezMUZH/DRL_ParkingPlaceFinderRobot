#%%
#import rl_parkingplacefinder
#from rl_parkingplacefinder import Slot as Slot
#from rl_parkingplacefinder import ParkingLot as ParkingLot

import networkx as nx

# E D D D D D D  7 Entry 
#   D P P P P D  6
#   D D D D D D  5
#   D P P P P D  4
#   D D D D D D  3
#   D P P P P D  2
# E D D D D D D  1
# x 1 2 3 4 5 6 
# parking w=4, l=
#%%
nr_lanes = 7
nr_parking_lanes = 3
nr_places_in_lane = 6

#%%
# each place is a node, attribute type and attribute state

g_grid = nx.grid_2d_graph(nr_lanes, nr_places_in_lane)
g_grid = nx.convert_node_labels_to_integers(g_grid)

node_color_map = []

#only for single depth
slot_nr = -1
for lane in range(nr_lanes):
    lane_nr = lane+1
    for nr_slot_in_lane in range(nr_places_in_lane):
        slot_nr += 1
        # the parking lanes
        if(lane_nr%2==0):
            if((slot_nr%nr_places_in_lane == 0) or (slot_nr%nr_places_in_lane == nr_places_in_lane-1)):
                g_grid.nodes[slot_nr]['slot_type'] = 'drive'
                node_color_map.append('grey')
            else:
                g_grid.nodes[slot_nr]['slot_type'] = 'park'
                node_color_map.append('green')
        else:
                g_grid.nodes[slot_nr]['slot_type'] = 'drive'
                node_color_map.append('grey')

g_grid.nodes.data()
pos = nx.spring_layout(g_grid,iterations=500)
nx.draw(g_grid, pos=pos,  node_color = node_color_map, with_labels=True)



#%%
#slot = Slot('A1','P',0)


#%%
