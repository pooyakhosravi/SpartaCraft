from src.ActionSpace import BasicActionSpace, BasicDiscreteActionSpace
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
    agent_host.sendCommand("moveMouse 0 -250")
    # agent_host.sendCommand("attack 1")
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
        self.action_space = BasicActionSpace() # BasicDiscreteActionSpace() 
        self.observation_space = BasicObservationSpace(len(self.get_state(None)))
        self.debug = debug
        self.ms_per_tick = ms_per_tick

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
        
        self.world_state, observed = wait_for_observation(self.agent_host, self.ms_per_tick)
        state = self.get_state(observed)
        reward = self.get_reward()
        done = self.get_done(observed)

        if self.debug:
            print(f"state: {state}, reward: {reward}, done: {done}")
        return state, reward, done, None
    
    
    def get_state(self, observed):
        xpos = None
        zpos = None
        if observed:
            life = observed["Life"]
            xpos = observed["XPos"]
            ypos = observed["YPos"]
            zpos = observed["ZPos"]
            pitch = observed["Pitch"]
            yaw = observed["Yaw"]
        return (xpos, zpos)


    def get_reward(self):
        if self.world_state.number_of_rewards_since_last_state > 0:
            #print(f"dmg_dealt: {dmg_dealt}, dmg_taken: {dmg_taken}, mobs_killed: {mobs_killed}")
            reward = self.world_state.rewards[-1].getValue()
            # print(f"Got reward: {reward}")
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