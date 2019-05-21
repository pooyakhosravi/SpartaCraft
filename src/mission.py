# This file is the main file to be run. 

import time
import os
import sys
import time
import json
import constants as c
import random
import tkinter as tk
import canvas as cnv
import environment 
from ActionSpace import ActionSpace

try:
    import MalmoPython
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
    is_alive = observed["IsAlive"]
    life = observed["Life"]
    xpos = observed["XPos"]
    ypos = observed["YPos"]
    zpos = observed["ZPos"]
    pitch = observed["Pitch"]
    yaw = observed["Yaw"]

    

    return observed



root, canvas = cnv.init_canvas()


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

    my_mission = MalmoPython.MissionSpec(environment.getMissionXML(), True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    # Attempt to start a mission:
    max_retries = 20
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
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
    moves = ["forward 1", "back 1", "left 1", "right 1", "attack 1", "attack 0", "moveMouse 100 0", "moveMouse -100 0"]
    moveactions = {"forward 1": "back 0", "left 1": "right 0", "right 1": "left 0", "back 1": "forward 0"}
    action_space = ActionSpace(moves)

    print("Starting AGENT!!")

    agent_host.sendCommand("chat /give @p diamond_sword 1 0 {ench:[{id:16,lvl:1000},{id:17,lvl:500},{id:19,lvl:300}]}")
    agent_host.sendCommand("chat hello!")
    agent_host.sendCommand("hotbar.2 1")
    agent_host.sendCommand("hotbar.2 0")
    agent_host.sendCommand("moveMouse 0 -150")

    # Loop until mission ends:
    while world_state.is_mission_running:
        print(".", end="")
        time.sleep(c.AGENT_TICK_RATE / 1000)
        world_state = agent_host.getWorldState()

        if True:
            a = action_space.random()
            if a in moveactions:
                agent_host.sendCommand(moveactions[a])
            agent_host.sendCommand(a)


        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text

            ob = json.loads(msg)
            # print(getState(ob))
            if "entities" in ob:
                entities = ob["entities"]
                cnv.drawMobs(root, canvas, entities, True)



        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission ended")
    # Mission has ended.


if __name__ == '__main__':
    num_repeats = 100

    for i in range(num_repeats):
        run()
