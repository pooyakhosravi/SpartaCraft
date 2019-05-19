"""
This file contains all the constants needed for the mission and other parts of project.
"""


MOB_TYPE = "Pig"

MS_PER_TICK = 30
AGENT_TICK_RATE = 4 * MS_PER_TICK

PLAYER_NAME = "Hunter"

PLAYER_SPAWN = '<Placement x="0.5" y="207.0" z="0.5"/>'

ENTITIES_SPAWN = {"Pig": 20, "Cow": 20}
ITEMS_SPAWN = {"carrot": 30, "apple": 10}


COLORS = {"Pig": "#FFDAB9", "Cow": "#A52A2A", PLAYER_NAME: "#0000FF"}

GENERATOR_STRING = "3;7,220*1,5*3,2;3;,biome_1" #"1;7,2x3,2,89,95:8;1"

ARENA_WIDTH = 30
ARENA_BREADTH = 30

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
