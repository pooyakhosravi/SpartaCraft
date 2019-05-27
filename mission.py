# This file is the main file to be run. 

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

try:
    import src.MalmoPython as MalmoPython
except ImportError:
    import malmo.MalmoPython as MalmoPython



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



canvas = Canvas().init_canvas()


# if sys.version_info[0] == 2:
#     sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
# else:
#     import functools
#     print = functools.partial(print, flush=True)


def run():

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

    environment = MalmoEnvironment()
    my_mission = MalmoPython.MissionSpec(environment.getMissionXML(), True)
    my_mission_record = MalmoPython.MissionRecordSpec(c.RECORD_FILENAME)


    print(c.RECORD_FILENAME)
    # my_clients = MalmoPython.ClientPool()
    # my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))

    # my_mission.requestVideo(800,600)
    # my_mission_record.recordMP4(30, 2000000)

    # Attempt to start a mission:
    max_retries = 20
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


    #moves = ["forward 1",  "attack 1", "attack 0",  "back 1", "moveMouse 0 100", "left 1", "right 1", "moveMouse 100 0", "moveMouse -100 0", "moveMouse 0 -100"]
    moves = ["forward 1", "back 1", "left 1", "right 1", "attack 1", "attack 0", "moveMouse 50 0", "moveMouse -50 0"]
    moveactions = {"forward 1": "back 0", "left 1": "right 0", "right 1": "left 0", "back 1": "forward 0"}
    action_space = ActionSpace(moves)

    print("Starting AGENT!!")

    agent_host.sendCommand("chat /give @p diamond_sword 1 0 {ench:[{id:16,lvl:1000}]}")
    agent_host.sendCommand(f"chat Hello! This is Episode {i}")
    agent_host.sendCommand("hotbar.1 1")
    agent_host.sendCommand("hotbar.1 0")
    agent_host.sendCommand("moveMouse 0 -150")

    is_start = True


    total_reward = 0
    num_steps = 0

    # Loop until mission ends:
    while world_state.is_mission_running:
        # print(".", end="")
        time.sleep(c.AGENT_TICK_RATE / 1000)
        world_state = agent_host.getWorldState()
        


        if True:
            a = random.choice(moves)
            if a in moveactions:
                agent_host.sendCommand(moveactions[a])
            agent_host.sendCommand(a)
            num_steps += 1

        if world_state.number_of_observations_since_last_state > 0:

            msg = world_state.observations[-1].text
            ob = json.loads(msg)

            if is_start:
                damage_dealth = ob["DamageDealt"]
                damage_taken = ob["DamageTaken"]
                mobs_killed = ob["MobsKilled"]
                print(getState(ob))
                is_start = False

            damage_dealth = ob["DamageDealt"] - damage_dealth
            damage_taken = ob["DamageTaken"] - damage_taken
            mobs_killed = ob["MobsKilled"] - mobs_killed

            entity_count = 0
            for entity in ob["entities"]:
                if entity["name"] in c.ENTITIES_SPAWN:
                    entity_count += 1

            if entity_count == 0:
                agent_host.sendCommand("quit")
            

            print(f"dmg_dealth: {damage_dealth}, dmg_taken: {damage_taken}, mobs_killed: {mobs_killed}")
            if "entities" in ob:
                entities = ob["entities"]
                canvas.drawMobs(entities, True)

        if world_state.number_of_rewards_since_last_state > 0:

            total_reward += world_state.rewards[-1].getValue()
            
        else:
            total_reward += -1.0

        print(f"  total reward: {total_reward}")
        for error in world_state.errors:
            print("Error:",error.text)

    return num_steps, total_reward


if __name__ == '__main__':
    num_repeats = 100

    for i in range(num_repeats):
        print(f"Running episode {i}::")
        run()
        print()
        print(f"::End episode {i}.")
        # Mission has ended.
