import random
import constants as c

def getRandomMultiItemXML(num_item, item_type):
    ''' Build an XML string that contains some randomly positioned goal items'''
    xml=""
    for item in range(num_item):
        x = str(random.randint(-c.ARENA_WIDTH // 2, c.ARENA_WIDTH // 2))
        z = str(random.randint(-c.ARENA_BREADTH // 2,c.ARENA_BREADTH // 2))
        xml += '''<DrawItem x="''' + x + '''" y="210" z="''' + z + '''" type="''' + item_type + '''"/>'''
    return xml


def getRandomMultiEntityXML(num_entity, mob_type):
    ''' Build an XML for random entities, like zombies and animals'''
    xml = ""

    for i in range(num_entity):
        x = str(random.randint(-c.ARENA_WIDTH // 2, c.ARENA_WIDTH // 2))
        z = str(random.randint(-c.ARENA_BREADTH // 2,c.ARENA_BREADTH // 2))
        y = 207
        xml += f'<DrawEntity x="{x}" y="{y}" z="{z}" type="{mob_type}" yaw="0"/>'
    return xml

def getEntitySpawnXML():
    xml = ""
    for mob_type in c.ENTITIES_SPAWN:
        xml += getRandomMultiEntityXML(c.ENTITIES_SPAWN[mob_type], mob_type)

    return xml


def getItemSpawnXML():
    xml = ""
    for item_type in c.ITEMS_SPAWN:
        xml += getRandomMultiItemXML(c.ITEMS_SPAWN[item_type], item_type)
    return xml

def getMissionXML():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                <About>
                <Summary>Hello world!</Summary>
                </About>
                <ModSettings>
                    <MsPerTick>''' + str(c.MS_PER_TICK) + '''</MsPerTick>
                </ModSettings>

                <ServerSection>
                    <ServerInitialConditions>
                    <Time>
                        <StartTime>17000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <AllowSpawning>true</AllowSpawning>
                    <AllowedMobs>''' + c.MOB_TYPE + '''</AllowedMobs>
                </ServerInitialConditions>
                <ServerHandlers>
                    <FlatWorldGenerator generatorString="''' + c.GENERATOR_STRING + '''"/>
                    
                    <DrawingDecorator>
                        <DrawCuboid ''' + c.getCorner("1",True,True,y=205,expand=2) + " " + c.getCorner("2",False,False,y=226,expand=2) + ''' type="sea_lantern"/>
                        <DrawCuboid ''' + c.getCorner("1",True,True,y=206, expand=1) + " " + c.getCorner("2",False,False,y=226,expand=1) + ''' type="barrier"/>
                        <DrawCuboid ''' + c.getCorner("1",True,True,y=207) + " " + c.getCorner("2",False,False,y=226) + ''' type="air"/>
                        ''' + getItemSpawnXML() + getEntitySpawnXML() + '''
                    </DrawingDecorator>
                    <ServerQuitWhenAnyAgentFinishes />
                    
                </ServerHandlers>
                </ServerSection>
                
                <AgentSection mode="Survival">
                <Name>''' + c.PLAYER_NAME + '''</Name>
                <AgentStart>
                    ''' + c.PLAYER_SPAWN + '''
                    <Inventory>
                        <InventoryItem slot="0" type="diamond_sword"/>
                    </Inventory>
                </AgentStart>
                <AgentHandlers>
                    <ObservationFromFullStats/>
                    <ObservationFromNearbyEntities>
                        <Range name="entities" xrange="'''+str(c.ARENA_WIDTH)+'''" yrange="2" zrange="'''+str(c.ARENA_BREADTH)+'''" />
                    </ObservationFromNearbyEntities>
                    <HumanLevelCommands/>
                    <ChatCommands/>
                </AgentHandlers>
                </AgentSection>
            </Mission>'''

