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

try:
    import MalmoPython
except ImportError:
    import malmo.MalmoPython as MalmoPython





root, canvas = cnv.init_canvas()


# if sys.version_info[0] == 2:
#     sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
# else:
#     import functools
#     print = functools.partial(print, flush=True)

if __name__ == '__main__':

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



    moves = ["forward 1", "forward 0", "left 1", "left 0"]

    # agent_host.sendCommand("forward 1")

    # Loop until mission ends:
    while world_state.is_mission_running:
        print(".", end="")
        time.sleep(c.AGENT_TICK_RATE / 1000)
        world_state = agent_host.getWorldState()
        if moves:
            agent_host.sendCommand(moves.pop(0))
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            # print(msg)

            ob = json.loads(msg)
            if "entities" in ob:
                entities = ob["entities"]
                cnv.drawMobs(root, canvas, entities, True)



        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission ended")
    # Mission has ended.