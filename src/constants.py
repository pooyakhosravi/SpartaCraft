"""
This file contains all the constants needed for the mission and other parts of project.
"""

import os


MOB_TYPE = "Pig"

MS_PER_TICK = 2
AGENT_TICK_RATE = int(2.5 * MS_PER_TICK)


RECORD_FILENAME = os.getcwd() + '/Recordings/record.tgz'

PLAYER_NAME = "Hunter"


ENTITIES_SPAWN = {"Cow": 2, "Zombie": 0} # {"Pig": 20, "Cow": 20, "Zombie":5}
ITEMS_SPAWN = {} #{"carrot": 30, "apple": 10}



DAMAGE_ENTITY_REWARDS = {"Cow": 1, "Zombie": 1}

COLORS = {"Pig": "#FFDAB9", "Cow": "#A52A2A","Zombie":"#800080", PLAYER_NAME: "#0000FF"}

GENERATOR_STRING = "3;7,220*1,5*3,2;3;,biome_1" #"1;7,2x3,2,89,95:8;1"

ARENA_WIDTH = 6
ARENA_BREADTH = 6


PLAYER_X = 0.5
PLAYER_Y = 207.0
PLAYER_Z = .5 - ARENA_BREADTH/2

PLAYER_SPAWN = f'<Placement x="{PLAYER_X}" y="{PLAYER_Y}" z="{PLAYER_Z}"/>'

TIME_LIMIT = int(1000 * 60 * .5)

# Display parameters:
CANVAS_BORDER = 20
CANVAS_WIDTH = 400
CANVAS_HEIGHT = CANVAS_BORDER + ((CANVAS_WIDTH - CANVAS_BORDER) * ARENA_BREADTH / ARENA_WIDTH)
CANVAS_SCALEX = (CANVAS_WIDTH-CANVAS_BORDER) // ARENA_WIDTH
CANVAS_SCALEY = (CANVAS_HEIGHT-CANVAS_BORDER) // ARENA_BREADTH
CANVAS_ORGX = -ARENA_WIDTH // CANVAS_SCALEX
CANVAS_ORGY = -ARENA_BREADTH // CANVAS_SCALEY

def getCorner(index,top,left,expand=0,y=206):
    ''' Return part of the XML string that defines the requested corner'''
    x = str(-(expand+ARENA_WIDTH//2)) if left else str(expand+ARENA_WIDTH//2)
    z = str(-(expand+ARENA_BREADTH//2)) if top else str(expand+ARENA_BREADTH//2)
    return 'x'+index+'="'+x+'" y'+index+'="' +str(y)+'" z'+index+'="'+z+'"'


def getCornerXYZ(top,left,expand=0,y=207):
    ''' Return part of the XML string that defines the requested corner'''
    x = -(expand+ARENA_WIDTH//2) if left else expand+ARENA_WIDTH//2
    z = -(expand+ARENA_BREADTH//2) if top else expand+ARENA_BREADTH//2
    return (x, y, z)


ENTITIES_SPAWN_WITH_POSITION = {"Cow" :[getCornerXYZ(False, False, -1), getCornerXYZ(False, True, -1)]}
