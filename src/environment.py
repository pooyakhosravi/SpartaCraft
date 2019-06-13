import random
import src.constants as c



class MalmoEnvironment:
    def __init__(self, tickrate= c.TRAINING_DEFAULT_MS_PER_TICK, breadth = c.ARENA_BREADTH, width = c.ARENA_WIDTH, ents = c.ENTITIES_SPAWN, ents_spawn = c.ENTITIES_SPAWN_WITH_POSITION, movement_commands="automatic"):
        self.ARENA_BREADTH = breadth
        self.ARENA_WIDTH = width
        self.ENTITIES_SPAWN = ents
        self.ENTITIES_SPAWN_WITH_POSITION = ents_spawn
        self.ITEMS_SPAWN = c.ITEMS_SPAWN
        self.MS_PER_TICK = tickrate

        self.AGENT_TICK_RATE = int(2.5 * self.MS_PER_TICK)

        self.TIME_LIMIT = c.TIME_LIMIT
        self.movement_commands = c.AUTOMATIC_MOVEMENT_COMMANDS if movement_commands == 'automatic' else c.MANUAL_MOVEMENT_COMMANDS

        
    def getRandomMultiItemXML(self, num_item, item_type):
        ''' Build an XML string that contains some randomly positioned goal items'''
        xml=""
        for item in range(num_item):
            x = random.randint(-self.ARENA_WIDTH // 2, self.ARENA_WIDTH // 2)
            z = random.randint(-self.ARENA_BREADTH // 2,self.ARENA_BREADTH // 2)
            while (x)**2 + (z)**2 > (self.ARENA_BREADTH*self.ARENA_WIDTH/4):
                x = random.randint(-self.ARENA_WIDTH // 2, self.ARENA_WIDTH // 2)
                z = random.randint(-self.ARENA_BREADTH // 2,self.ARENA_BREADTH // 2)
            
            xml += '''<DrawItem x="''' + x + '''" y="210" z="''' + z + '''" type="''' + item_type + '''"/>'''
        return xml


    def getRandomMultiEntityXML(self, num_entity, mob_type):
        ''' Build an XML for random entities, like zombies and animals'''
        xml = ""
        for i in range(num_entity):
            x = random.randint(-self.ARENA_WIDTH // 2, self.ARENA_WIDTH // 2)
            z = random.randint(-self.ARENA_BREADTH // 2,self.ARENA_BREADTH // 2)
            while (x)**2 + (z)**2 > (self.ARENA_BREADTH*self.ARENA_WIDTH/4 - 5) or (x - c.PLAYER_X)**2 + (z-c.PLAYER_Z)**2 < 200:
                x = random.randint(-self.ARENA_WIDTH // 2, self.ARENA_WIDTH // 2)
                z = random.randint(-self.ARENA_BREADTH // 2,self.ARENA_BREADTH // 2)
            
            y = c.PLAYER_Y
            xml += f'<DrawEntity x="{x}" y="{y}" z="{z}" type="{mob_type}" yaw="0"/>'
        return xml

    def getEntityAtPosition(self, mob_type, positions):
        ''' Build an XML for entities, like zombies and animals, at given positions'''
        xml = ""
        for pos in positions:
            xml += f'<DrawEntity x="{pos[0]}" y="{pos[1]}" z="{pos[2]}" type="{mob_type}" yaw="-180"/>'
        return xml

    def getEntitySpawnWithPositionXML(self):
        xml = ""
        for mob_type, positions in self.ENTITIES_SPAWN_WITH_POSITION.items():
            xml += self.getEntityAtPosition(mob_type, positions)

        return xml

    def getEntityRandomSpawnXML(self):
        xml = ""
        # random.seed(2)
        for mob_type in self.ENTITIES_SPAWN:
            xml += self.getRandomMultiEntityXML(self.ENTITIES_SPAWN[mob_type], mob_type)

        return xml


    def getItemSpawnXML(self):
        xml = ""
        for item_type in self.ITEMS_SPAWN:
            xml += self.getRandomMultiItemXML(self.ITEMS_SPAWN[item_type], item_type)
        return xml


    def getRewards(self):
        xml = ""
        for entity in c.DAMAGE_ENTITY_REWARDS:
            xml += f'''<Mob type="{entity}" reward="{c.DAMAGE_ENTITY_REWARDS[entity]}"/>'''
        return xml

    def getCorner(self, index,top,left,expand=0,y=c.PLAYER_Y):
        ''' Return part of the XML string that defines the requested corner'''
        x = str(-(expand+self.ARENA_WIDTH//2)) if left else str(expand+self.ARENA_WIDTH//2)
        z = str(-(expand+self.ARENA_BREADTH//2)) if top else str(expand+self.ARENA_BREADTH//2)
        return 'x'+index+'="'+x+'" y'+index+'="' +str(y)+'" z'+index+'="'+z+'"'


    def getCornerXYZ(self, top,left,expand=0,y=c.PLAYER_Y):
        ''' Return part of the XML string that defines the requested corner'''
        x = -(expand+self.ARENA_WIDTH//2) if left else expand+self.ARENA_WIDTH//2
        z = -(expand+self.ARENA_BREADTH//2) if top else expand+self.ARENA_BREADTH//2
        return (x, y, z)

    def getCubiodMap(self):
        return f'''
        <DrawCuboid {self.getCorner("1",True,True,y=int(c.PLAYER_Y - 1), expand=1)} {self.getCorner("2",False,False,y=226, expand=1)} type="barrier"/>
        <DrawCuboid {self.getCorner("1",True,True,y=int(c.PLAYER_Y))} {self.getCorner("2",False,False,y=226)} type="air"/>'''


    def getCylindricalMap(self):
        result = ''
        expand = 3
        for x in range(-(expand+self.ARENA_WIDTH//2), expand+self.ARENA_WIDTH//2):
            for z in range(-(expand+self.ARENA_BREADTH//2), expand+self.ARENA_BREADTH//2):
                d = (x)**2 + (z)**2
                if d <= (self.ARENA_BREADTH*self.ARENA_WIDTH/4 + 2*expand):
                    result += f'''<DrawLine x1="{x}" y1="{int(c.PLAYER_Y)}" z1="{z}" x2="{x}" y2="{226}" z2="{z}" type="air"/>'''    
                elif int(d) < (self.ARENA_BREADTH*self.ARENA_WIDTH/4 + 10*expand):
                    result += f'''<DrawLine x1="{x}" y1="{int(c.PLAYER_Y)}" z1="{z}" x2="{x}" y2="{226}" z2="{z}" type="barrier"/>'''
                    
        return result
        # <DrawCuboid {self.getCorner("1",True,True,y=int(c.PLAYER_Y - 1), expand=1)} {self.getCorner("2",False,False,y=226, expand=1)} type="barrier"/>
        # <DrawCuboid {self.getCorner("1",True,True,y=int(c.PLAYER_Y))} {self.getCorner("2",False,False,y=226)} type="air"/>


    def getMissionXML(self):
        return f'''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                    <Summary>Colosseum World!</Summary>
                    </About>
                    <ModSettings>
                        <MsPerTick>{self.MS_PER_TICK}</MsPerTick>
                    </ModSettings>

                    <ServerSection>
                        <ServerInitialConditions>
                        <Time>
                            <StartTime>17000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <AllowSpawning>true</AllowSpawning>
                        <AllowedMobs>{c.MOB_TYPE}</AllowedMobs>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="{c.GENERATOR_STRING}"/>
                        
                        <DrawingDecorator>
                            <DrawCuboid {self.getCorner("1",True,True,y=int(c.PLAYER_Y - 2), expand=2)} {self.getCorner("2",False,False,y=226, expand=2)} type="sea_lantern"/>
                            {self.getCylindricalMap()}
                            {self.getItemSpawnXML()}
                            {self.getEntitySpawnWithPositionXML()}
                            {self.getEntityRandomSpawnXML()}
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes />
                        <ServerQuitFromTimeUp timeLimitMs="{self.TIME_LIMIT}" description="times_up"/>

                    </ServerHandlers>
                    </ServerSection>
                    
                    <AgentSection mode="Spectator">
                    <Name>{c.PLAYER_NAME}</Name>
                    <AgentStart>
                    <Inventory />
                        {c.PLAYER_SPAWN}
                    </AgentStart>
                    <AgentHandlers>
                        
                        <ObservationFromFullStats/>
                        <ObservationFromNearbyEntities>
                            <Range name="entities" xrange="{self.ARENA_WIDTH + 2}" yrange="3" zrange="{self.ARENA_BREADTH + 2}" />
                        </ObservationFromNearbyEntities>
                        {self.movement_commands}
                        <ChatCommands/>

                        <MissionQuitCommands quitDescription="killed_all"/>

                        <RewardForMissionEnd rewardForDeath="-1000.0">
                            <Reward description="killed_all" reward="0.0"/>
                            <Reward description="times_up" reward="-100.0"/>
                        </RewardForMissionEnd>
                    </AgentHandlers>
                    </AgentSection>
                </Mission>'''

            