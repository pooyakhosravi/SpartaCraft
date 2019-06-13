# This file is the main file to be run. 

from tqdm import tqdm
import time
import os
import sys
import time
import json
import src.constants as c
import random
import tkinter as tk
from src.canvas import Canvas
from src.environment import MalmoEnvironment
from src.ActionSpace import ActionSpace


from past.utils import old_div
import math


try:
    import src.MalmoPython as MalmoPython
except ImportError:
    import malmo.MalmoPython as MalmoPython


# Agent parameters:
agent_stepsize = -.85
agent_search_resolution = 60 # Smaller values make computation faster, which seems to offset any benefit from the higher resolution.
agent_goal_weight = 1500
agent_edge_weight = 0
agent_mob_weight = 0
agent_turn_weight = 0 # Negative values to penalise turning, positive to encourage.

# Task parameters:
NUM_GOALS = 20
GOAL_TYPE = "Zombie"
GOAL_REWARD = 100
MOB_TYPE = "Cow"


def findUs(entities):
    for ent in entities:
        if ent["name"] == MOB_TYPE:
            continue
        elif ent["name"] == GOAL_TYPE:
            continue
        elif ent["name"] == c.PLAYER_NAME:
            return ent


def getBestAngle(entities, current_yaw, current_health):
    '''Scan through 360 degrees, looking for the best direction in which to take the next step.'''
    us = findUs(entities)

    scores=[]
    # Normalise current yaw:
    while current_yaw < 0:
        current_yaw += 360
    while current_yaw > 360:
        current_yaw -= 360

    # Look for best option
    for i in range(agent_search_resolution):
        # Calculate cost of turning:
        ang = 2 * math.pi * (old_div(i, float(agent_search_resolution)))
        yaw = i * 360.0 / float(agent_search_resolution)
        yawdist = min(abs(yaw-current_yaw), 360-abs(yaw-current_yaw))
        turncost = agent_turn_weight * yawdist
        score = turncost

        # Calculate entity proximity cost for new (x,z):
        x = us["x"] + 1 - math.sin(ang)
        z = us["z"] + 1 * math.cos(ang)
        for ent in entities:
            dist = (ent["x"] - x)*(ent["x"] - x) + (ent["z"] - z)*(ent["z"] - z)
            weight = 0.0
            if ent["name"] == MOB_TYPE:
                weight = agent_mob_weight
                
            elif ent["name"] == GOAL_TYPE:
                dist -= 1   # assume mobs are moving towards us
                if dist <= 0:
                    dist = 0.1
                weight = agent_goal_weight # * current_health / 20.0
            score += old_div(weight, float(dist**2 + 1e-3))

        # Calculate cost of proximity to edges:
        distRight = (2+old_div(c.ARENA_WIDTH,2)) - x
        distLeft = (-2-old_div(c.ARENA_WIDTH,2)) - x
        distTop = (2+old_div(c.ARENA_BREADTH,2)) - z
        distBottom = (-2-old_div(c.ARENA_BREADTH,2)) - z
        score += old_div(agent_edge_weight, float(distRight * distRight * distRight * distRight))
        score += old_div(agent_edge_weight, float(distLeft * distLeft * distLeft * distLeft))
        score += old_div(agent_edge_weight, float(distTop * distTop * distTop * distTop))
        score += old_div(agent_edge_weight, float(distBottom * distBottom * distBottom * distBottom))
        scores.append(score)

    # Find best score:
    i = scores.index(max(scores))
    # Return as an angle in degrees:
    return i * 360.0 / float(agent_search_resolution)



def getPlayerState(observed):
    is_alive = observed["IsAlive"]
    life = observed["Life"]
    xpos = observed["XPos"]
    ypos = observed["YPos"]
    zpos = observed["ZPos"]
    pitch = observed["Pitch"]
    yaw = observed["Yaw"]

    return (is_alive, life, xpos, ypos, zpos, pitch, yaw)


def getState(observed):
    damage_dealth = observed["DamageDealt"]
    damage_taken = observed["DamageTaken"]
    mobs_killed = observed["MobsKilled"]

    return (mobs_killed, damage_dealth, damage_taken)



# canvas = Canvas(mobsize=5).init_canvas()


# if sys.version_info[0] == 2:
#     sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
# else:
#     import functools
#     print = functools.partial(print, flush=True)


def run(current_life, current_yaw, best_yaw):

    # Create default Malmo objects:
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    environment = MalmoEnvironment(tickrate= c.MINECRAFT_DEFAULT_MS_PER_TICK, movement_commands='manual')
    my_mission = MalmoPython.MissionSpec(environment.getMissionXML(), True)
    my_mission_record = MalmoPython.MissionRecordSpec()


    # my_clients = MalmoPython.ClientPool()
    # my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))

    # my_mission.requestVideo(800,600)
    # my_mission_record.recordMP4(30, 2000000)

    # Attempt to start a mission:
    max_retries = 5
    for retry in range(max_retries):
        try:
            # agent_host.startMission( my_mission, my_clients, my_mission_record, 0, "Hunter")
            agent_host.startMission(my_mission, my_mission_record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(1)


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


    #moves = ["forward 1",  "attack 1", "attack 0",  "back 1", "moveMouse 0 100", "left 1", "right 1", "moveMouse 100 0", "moveMouse -100 0", "moveMouse 0 -100"]
    moves = ["forward 1", "back 1", "left 1", "right 1", "attack 1", "attack 0", "moveMouse 50 0", "moveMouse -50 0"]
    moveactions = {"forward 1": "back 0", "left 1": "right 0", "right 1": "left 0", "back 1": "forward 0"}
    action_space = ActionSpace(moves)

    print("Starting AGENT!!")

    agent_host.sendCommand("chat /give @p diamond_sword 1 0 {ench:[{id:16,lvl:1000}]}")
    # agent_host.sendCommand(f"chat Hello! This is Episode {i}")
    agent_host.sendCommand("hotbar.1 1")
    agent_host.sendCommand("hotbar.1 0")
    agent_host.sendCommand("moveMouse 0 -150")
    # agent_host.sendCommand("pitch 0.1")
    # time.sleep(4*environment.AGENT_TICK_RATE / 1000)
    # agent_host.sendCommand("pitch 0")
    is_start = True



    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()

    
    # main loop:
    total_reward = 0
    total_commands = 0
    total_killed = 0
    goal_count = -1
    flash = False
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        if random.random() < .85:
            if random.random() < .75:
                agent_host.sendCommand(f"move {agent_stepsize}")
            else:
                agent_host.sendCommand(f"move {1}")
        if random.random() < .70:
            if random.random() < .55:
                agent_host.sendCommand("strafe 1")
            else:
                agent_host.sendCommand("strafe -1")
        agent_host.sendCommand("attack 1")
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            ob = json.loads(msg)
            if "Yaw" in ob:
                current_yaw = ob[u'Yaw']
            if "Life" in ob:
                life = ob[u'Life']
                if life < current_life:
                    # agent_host.sendCommand("chat aaaaaaaaargh!!!")
                    flash = True
                    total_reward += (c.PLAYER_DAMAGE_TAKEN_REWARD) * (current_life - life)
                current_life = life
            if "entities" in ob:
                entities = ob["entities"]

                if goal_count == -1:
                    goal_count = len([ent for ent in entities if ent["name"] == GOAL_TYPE])

                rem_goals = 0
                for entity in ob["entities"]:
                    if entity["name"] == GOAL_TYPE:
                        rem_goals += 1

                if rem_goals < goal_count:
                    total_reward += (goal_count - rem_goals)*c.ENTITY_KILLED_REWARD[GOAL_TYPE]
                    goal_count = rem_goals

                # canvas.drawMobs(entities, True)
                best_yaw = getBestAngle(entities, current_yaw, current_life)
                difference = best_yaw - current_yaw
                while difference < -180:
                    difference += 360
                while difference > 180:
                    difference -= 360
                difference /= 180.0

                # print(f'Best Yaw {best_yaw}, current {current_yaw}')
                agent_host.sendCommand("turn " + str(difference))
                total_commands += 1
                entity_count = 0

                doquit = True
                for entity in ob["entities"]:
                    if entity["name"] == GOAL_TYPE:
                        doquit = False

                if doquit:
                    agent_host.sendCommand("quit")
                    break
        if world_state.number_of_rewards_since_last_state > 0:
            # A reward signal has come in - see what it is:
            total_reward += world_state.rewards[-1].getValue()
        time.sleep(environment.AGENT_TICK_RATE / 1000)
        flash = False

    # mission has ended.
    for error in world_state.errors:
        print("Error:",error.text)
    if world_state.number_of_rewards_since_last_state > 0:
        # A reward signal has come in - see what it is:
        total_reward += world_state.rewards[-1].getValue()

    print("We stayed alive for " + str(total_commands) + " commands, and scored " + str(total_reward))
    time.sleep(1) # Give the mod a little time to prepare for the next mission.

    return total_commands, total_reward




if __name__ == '__main__':
    num_episodes = 100

    current_yaw = 0
    best_yaw = 0
    current_life = 0

    rewards = []
    steps = []

    
    for ep in tqdm(range(num_episodes)):
        print(f"Running episode {ep}::")
        s, r  = run(current_life, current_yaw, best_yaw)
        steps.append(s)
        rewards.append(r)
        print()
        print(f"::End episode {ep}.")
        # Mission has ended.

    print(rewards)
