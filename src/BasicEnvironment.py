from src.ActionSpace import BasicActionSpace
from src.ObservationSpace import BasicObservationSpace
import src.environment as environment
import src.constants as c
try:
    import MalmoPython
except ImportError:
    import malmo.MalmoPython as MalmoPython

import time, json, sys, os
import numpy as np

def startMission(max_retries = 20):
    print("Creating Agent Host")
    agent_host = MalmoPython.AgentHost()
    my_mission = MalmoPython.MissionSpec(environment.getMissionXML(), True)
    my_mission_record = MalmoPython.MissionRecordSpec()

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
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission running ", end=' ')
    
    print("Starting AGENT!!")
    agent_host.sendCommand("chat /give @p diamond_sword 1 0 {ench:[{id:16,lvl:1000},{id:17,lvl:500},{id:19,lvl:300}]}")
    agent_host.sendCommand("chat hello!")
    agent_host.sendCommand("hotbar.1 1")   
    agent_host.sendCommand("moveMouse 0 -150")
    return agent_host

def wait_for_observation(agent_host):
    do_while_emulator = True
    while True:
        time.sleep(c.AGENT_TICK_RATE / 1000) 
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
        if world_state.number_of_observations_since_last_state > 0:
            break
    msg = world_state.observations[-1].text
    ob = json.loads(msg)
    return world_state, ob

class BasicEnvironment():
    def __init__(self):
        self.action_space = BasicActionSpace()
        self.observation_space = BasicObservationSpace(c.ARENA_WIDTH, c.ARENA_BREADTH)

    def reset(self):
        self.agent_host = startMission()
        self.world_state, observed = wait_for_observation(self.agent_host)
        return self.get_state(observed)

    def step(self, a):
        action = self.action_space.actions[a]
        #print(f"Taking action: {a}, {action}")
        moveactions = {"forward 1": "back 0", "left 1": "right 0", "right 1": "left 0", "back 1": "forward 0"}
        if action in moveactions.keys():
            self.agent_host.sendCommand(moveactions[action])
        self.agent_host.sendCommand(action)

        self.world_state, observed = wait_for_observation(self.agent_host)
        state = self.get_state(observed)
        reward = self.get_reward()
        done = self.get_done(observed)

        #print(f"state: {state}, reward: {reward}, done: {done}")
        return state, reward, done, None

    def get_state(self, observed):
        life = observed["Life"]
        xpos = observed["XPos"]
        ypos = observed["YPos"]
        zpos = observed["ZPos"]
        pitch = observed["Pitch"]
        yaw = observed["Yaw"]

        x_range = c.ARENA_WIDTH + .4
        z_range = c.ARENA_BREADTH + .4
        x_min = .3 - c.ARENA_WIDTH // 2
        z_min = .3 - c.ARENA_BREADTH // 2
        
        x_index = int((xpos - x_min) / x_range * c.ARENA_WIDTH)
        z_index = int((zpos - z_min) / z_range * c.ARENA_BREADTH)
        state = x_index * c.ARENA_WIDTH + z_index
        if (state < 0 or state > (c.ARENA_BREADTH * c.ARENA_WIDTH)):
            print(f"State: {state} with indices {x_index} {z_index} is invalid. xpos {xpos}, zpos {zpos}")
        return state

    def get_reward(self):
        if self.world_state.number_of_rewards_since_last_state > 0:
            #print(f"dmg_dealt: {dmg_dealt}, dmg_taken: {dmg_taken}, mobs_killed: {mobs_killed}")
            reward = self.world_state.rewards[-1].getValue()
            #print(f"Got reward: {reward}")
            return reward
        return -1.0

    def get_done(self, ob):
        entity_count = 0
        for entity in ob["entities"]:
            if entity["name"] in c.ENTITIES_SPAWN:
                entity_count += 1

        if entity_count == 0 or not self.world_state.is_mission_running or not ob["IsAlive"]:
            self.agent_host.sendCommand("quit")
            return True
        return False

    