"""
This file contains all the constants needed for the mission and other parts of project.
"""

import os


MOB_TYPE = "Pig"

MINECRAFT_DEFAULT_MS_PER_TICK = 30
TRAINING_DEFAULT_MS_PER_TICK = 5
AGENT_TICK_RATE_MULTIPLIER = 3

MANUAL_MOVEMENT_COMMANDS = "<ContinuousMovementCommands turnSpeedDegs=\"480\"/><HumanLevelCommands/><DiscreteMovementCommands/>"
AUTOMATIC_MOVEMENT_COMMANDS = "<HumanLevelCommands/><DiscreteMovementCommands/>"

RECORD_FILENAME = os.getcwd() + '/Recordings/record.tgz'

PLAYER_NAME = "Spartos"


ENTITIES_SPAWN = {"Cow": 0, "Zombie": 12} # {"Pig": 20, "Cow": 20, "Zombie":5}
ITEMS_SPAWN = {} #{"carrot": 30, "apple": 10}


PLAYER_DAMAGE_TAKEN_REWARD = -5
ENTITY_KILLED_REWARD = {"Zombie": 400}

DAMAGE_ENTITY_REWARDS = {"Cow": 1, "Zombie": 1}

COLORS = {"Pig": "#FFDAB9", "Cow": "#A52A2A","Zombie":"#800080", PLAYER_NAME: "#0000FF"}



ARENA_WIDTH = 30
ARENA_BREADTH = 25


PLAYER_X = 0.5
PLAYER_Y = 218.0
PLAYER_Z = .5 - ARENA_BREADTH/2

GENERATOR_STRING = f"3;7,{int(PLAYER_Y-7)}*1,12*minecraft:sea_lantern;3;,biome_1" #"1;7,2x3,2,89,95:8;1"

PLAYER_SPAWN = f'<Placement x="{PLAYER_X}" y="{PLAYER_Y}" z="{PLAYER_Z}"/>'

TIME_LIMIT = int(1000 * 60 * 1)

# Display parameters:
CANVAS_BORDER = 20
CANVAS_WIDTH = 400
CANVAS_HEIGHT = CANVAS_BORDER + ((CANVAS_WIDTH - CANVAS_BORDER) * ARENA_BREADTH / ARENA_WIDTH)
CANVAS_SCALEX = (CANVAS_WIDTH-CANVAS_BORDER) // ARENA_WIDTH
CANVAS_SCALEY = (CANVAS_HEIGHT-CANVAS_BORDER) // ARENA_BREADTH
CANVAS_ORGX = -ARENA_WIDTH // CANVAS_SCALEX
CANVAS_ORGY = -ARENA_BREADTH // CANVAS_SCALEY

def getCorner(index,top,left,expand=0,y=PLAYER_Y):
    ''' Return part of the XML string that defines the requested corner'''
    x = str(-(expand+ARENA_WIDTH//2)) if left else str(expand+ARENA_WIDTH//2)
    z = str(-(expand+ARENA_BREADTH//2)) if top else str(expand+ARENA_BREADTH//2)
    return 'x'+index+'="'+x+'" y'+index+'="' +str(y)+'" z'+index+'="'+z+'"'


def getCornerXYZ(top,left,expand=0,y=PLAYER_Y):
    ''' Return part of the XML string that defines the requested corner'''
    x = -(expand+ARENA_WIDTH//2) if left else expand+ARENA_WIDTH//2
    z = -(expand+ARENA_BREADTH//2) if top else expand+ARENA_BREADTH//2
    return (x, y, z)


ENTITIES_SPAWN_WITH_POSITION = {"Zombie" :[getCornerXYZ(False, False, -.5)]}#, getCornerXYZ(False, True, -.5)]}
