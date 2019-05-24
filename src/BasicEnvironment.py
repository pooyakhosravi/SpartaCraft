from src.ActionSpace import *
import src.environment as environment
try:
    import MalmoPython
except ImportError:
    import malmo.MalmoPython as MalmoPython

import time

def startMission(max_retries = 20):
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
    
    agent_host = startMission()

    print("Starting AGENT!!")

    agent_host.sendCommand("chat /give @p diamond_sword 1 0 {ench:[{id:16,lvl:1000},{id:17,lvl:500},{id:19,lvl:300}]}")
    agent_host.sendCommand("chat hello!")
    agent_host.sendCommand("hotbar.2 1")
    agent_host.sendCommand("hotbar.2 0")
    agent_host.sendCommand("moveMouse 0 -150")
    return agent_host

def get_observation(world_state):
    msg = world_state.observations[-1].text
    ob = json.loads(msg)
    return ob

class BasicEnvironment():
    def __init__(self):
        self.action_space = BasicActionSpace()
        self.reset()

    def reset(self):
        self.agent_host = startMission()
        self.world_state = self.agent_host.getWorldState()
        ob = get_observation(self.world_state)
        self.damage_dealt = observed["DamageDealt"]
        self.damage_taken = observed["DamageTaken"]
        self.mobs_killed  = observed["MobsKilled"]
        return self.get_state(ob)

    def step(self, action):
        agent_host.sendCommand(action)
        
        do_while_emulator = False
        while do_while_emulator and self.world_state.number_of_observations_since_last_state < 0:
            time.sleep(c.AGENT_TICK_RATE / 1000) 
            self.world_state = agent_host.getWorldState()
            for error in self.world_state.errors:
                print("Error:",error.text)
            do_while_emulator = True
        
        ob = get_observation(self.world_state)
        state = self.get_state(ob)
        reward = self.get_reward(ob)
        done = not self.world_state.is_mission_running or not observed["IsAlive"]
            
        return state, reward, done, None

    def get_state(self, observed):
        life = observed["Life"]
        xpos = observed["XPos"]
        ypos = observed["YPos"]
        zpos = observed["ZPos"]
        pitch = observed["Pitch"]
        yaw = observed["Yaw"]

        return (life, xpos, ypos, zpos, pitch, yaw)

    def get_reward(self, observed):
        reward = -1.0
        damage_dealt = observed["DamageDealt"]
        damage_taken = observed["DamageTaken"]
        mobs_killed = observed["MobsKilled"]

        # Award Based on Deltas
        reward += (damage_dealt - self.damage_dealt) * 2.0
        reward += (damage_taken - self.damage_taken) * -5.0
        reward += (mobs_killed - self.mobs_killed) * 100.0

        # Update
        self.damage_dealt = damage_dealt
        self.damage_taken = damage_taken
        self.mobs_killed = mobs_killed
        return reward

    