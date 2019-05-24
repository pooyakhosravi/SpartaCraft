from src.ActionSpace import BasicActionSpace
from src.ObservationSpace import BasicObservationSpace
import src.environment as environment
import src.constants as c
try:
    import MalmoPython
except ImportError:
    import malmo.MalmoPython as MalmoPython

import time, json

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
    
    print("Starting AGENT!!")
    agent_host.sendCommand("chat /give @p diamond_sword 1 0 {ench:[{id:16,lvl:1000},{id:17,lvl:500},{id:19,lvl:300}]}")
    agent_host.sendCommand("chat hello!")
    agent_host.sendCommand("hotbar.2 1")
    agent_host.sendCommand("hotbar.2 0")
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
        self.reset()

    def reset(self):
        self.agent_host = startMission()
        self.world_state, ob = wait_for_observation(self.agent_host)
        ob = get_observation(self.world_state)
        self.damage_dealt = observed["DamageDealt"]
        self.damage_taken = observed["DamageTaken"]
        self.mobs_killed  = observed["MobsKilled"]
        return self.get_state(ob)

    def step(self, a):
        action = self.action_space.actions[a]
        print(f"Taking action: {a}, {action}")
        agent_host.sendCommand(action)
        self.world_state, ob = wait_for_observation(self.agent_host)
        state = self.get_state(ob)
        reward = self.get_reward(ob)
        done = not self.world_state.is_mission_running or not observed["IsAlive"]
        print(f"Step returned state, reward, done: {state}, {reward}, {done}")
        return state, reward, done, None

    def get_state(self, observed):
        life = observed["Life"]
        xpos = observed["XPos"]
        ypos = observed["YPos"]
        zpos = observed["ZPos"]
        pitch = observed["Pitch"]
        yaw = observed["Yaw"]

        linearized_x = np.round(xpos + c.ARENA_WIDTH // 2)
        linearized_z = np.round(zpos + c.ARENA_BREADTH // 2)

        state = linearized_x * c.ARENA_WIDTH + linearized_z
        if (state < 0 or state > (c.ARENA_BREADTH * c.ARENA_WIDTH)):
            print(f"State: {state} is invalid. Brennan Sucks at coding")
        return state

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

    