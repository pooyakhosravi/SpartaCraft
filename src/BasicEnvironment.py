from src.ActionSpace import *
from src.ObservationSpace import BasicObservationSpace
from src.environment import MalmoEnvironment
import src.constants as c
try:
    import src.MalmoPython as MalmoPython
except ImportError:
    import malmo.MalmoPython as MalmoPython

import time, json, sys, os
import numpy as np

def startMission(ms_per_tick, max_retries = 20, debug=False):
    if debug:
        print("Creating Agent Host")
    agent_host = MalmoPython.AgentHost()
    environment = MalmoEnvironment(tickrate=ms_per_tick)
    my_mission = MalmoPython.MissionSpec(environment.getMissionXML(), True)
    my_mission_record = MalmoPython.MissionRecordSpec()

#     my_mission.requestVideo(1200,720)
#     my_mission_record.recordMP4(30, 2000000)

    # Attempt to start a mission:
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                sys.exit(-1)
            else:
                time.sleep(2)
    
    # Loop until mission starts:
    if debug:
        print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(10 * ms_per_tick / 1000)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    if debug:
        print()
        print("Mission running ", end=' ')
        print("Starting AGENT!!")
    agent_host.sendCommand("chat /give @p diamond_sword 1 0 {ench:[{id:16,lvl:1000}]}")
    agent_host.sendCommand(f"chat Here we go again!")
    agent_host.sendCommand("hotbar.1 1")
    agent_host.sendCommand("hotbar.1 0")
    return agent_host


def wait_for_observation(agent_host, ms_per_tick, max_retries=20):
    for retry in range(max_retries):
        # print(world_state)
        time.sleep(ms_per_tick * c.AGENT_TICK_RATE_MULTIPLIER / 1000)
        world_state = agent_host.getWorldState() 
        for error in world_state.errors:
            print("Error:",error.text)
        if world_state.number_of_observations_since_last_state > 0:
            break
        if not world_state.is_mission_running:
            return world_state, None
        if retry == max_retries - 1:
            print(f"Error: Did not receive observation after {max_retries} times")
            return None, None

    msg = world_state.observations[-1].text
    ob = json.loads(msg)
    return world_state, ob

class BasicEnvironment():
    def __init__(self, ms_per_tick = c.MINECRAFT_DEFAULT_MS_PER_TICK, debug=False):
        self.scale_factor = 2
        self.action_space = SpartosActionSpace() # BasicDiscreteActionSpace() 
        self.observation_space = BasicObservationSpace(4, (10,10))
        self.debug = debug
        self.ms_per_tick = ms_per_tick
        self.goal_count = -1

    def reset(self):
        self.agent_host = startMission(self.ms_per_tick, debug=self.debug, max_retries=100)
        world_state = self.agent_host.getWorldState()
        self.world_state, observed = wait_for_observation(self.agent_host, self.ms_per_tick, max_retries=100)
        return self.get_state(observed)

    def step(self, a):
        action = self.action_space.actions[a]
        if self.debug:
            print(f"Taking action: {a}, {action}")
        moveactions = {"forward 1": "back 0", "left 1": "right 0", "right 1": "left 0", "back 1": "forward 0"}
        if action in moveactions.keys():
            self.agent_host.sendCommand(moveactions[action])
        self.agent_host.sendCommand(action)
        self.agent_host.sendCommand("attack 1")
        
        self.world_state, observed = wait_for_observation(self.agent_host, self.ms_per_tick)
        state = self.get_state(observed)

        reward = self.get_reward(observed)
        done = self.get_done(observed)

        if self.debug:
            print(f"state: {state}, reward: {reward}, done: {done}")
        return state, reward, done, None
    
    
    def get_state(self, observed=None):
        grid_world = np.zeros(self.observation_space.shape)
        #print(f"grid_world.shape: {grid_world.shape}")
        player_state = np.zeros(self.observation_space.n)
        enemy_locations = []
        if observed:
            for entity in observed["entities"]:
                entity_location = np.round((entity['x'], entity['z']))
                if entity["name"] == c.PLAYER_NAME:
                    player_location = entity_location
                    player_state[0] = entity['motionX']
                    player_state[1] = entity['motionZ']
                    player_state[2] = np.cos(entity['yaw'])
                    player_state[3] = np.sin(entity['yaw'])
                else:
                    enemy_locations.append(entity_location)

            if self.goal_count == -1:
                self.goal_count = len([ent for ent in observed["entities"] if ent["name"] == "ZOMBIE"])

            max_x_dist = np.round(self.observation_space.shape[0] // 2)
            max_z_dist = np.round(self.observation_space.shape[1] // 2)
            #print(f"max_dist: ({max_x_dist}, {max_z_dist})")
            for enemy_location in enemy_locations:
                manhattan_dist = np.array(player_location - enemy_location, dtype=np.int32)
                if np.abs(manhattan_dist[0]) < max_x_dist and np.abs(manhattan_dist[1]) < max_z_dist:
                    manhattan_dist[0] = manhattan_dist[0] + max_x_dist
                    manhattan_dist[1] = manhattan_dist[1] + max_z_dist
                    #print(f"dist: {manhattan_dist}")
                    grid_world[manhattan_dist[0], manhattan_dist[1]] += 1
        return (grid_world, player_state)


    def get_reward(self, ob):
        if not self.world_state:
            return 0.0

        reward = 0
        rem_goals = 0
        if ob:
            for entity in ob["entities"]:
                if entity["name"] == "ZOMBIE":
                    rem_goals += 1

            if rem_goals < self.goal_count:
                reward += (self.goal_count - rem_goals)*c.ENTITY_KILLED_REWARD["ZOMBIE"]
                self.goal_count = rem_goals
        
        if self.world_state.number_of_rewards_since_last_state > 0:
            reward = self.world_state.rewards[-1].getValue()
            return reward
        return -1.0

    def get_done(self, ob):
        if ob == None:
            return True
        
        entity_count = 0
        for entity in ob["entities"]:
            if entity["name"] in c.ENTITIES_SPAWN:
                entity_count += 1

        if entity_count == 0 or not self.world_state.is_mission_running or not ob["IsAlive"]:
            self.agent_host.sendCommand("quit")
            return True
        return False

    def get_additional_info(self, ob):
        info = {}
        info["Failure"] = ob is None