import tkinter as tk
import src.constants as c

class Canvas:
    def __init__(self, width = c.ARENA_WIDTH, breadth = c.ARENA_BREADTH, mobsize = 4):
        self.breadth = breadth
        self.width = width
        self.mobsize = mobsize

    def init_canvas(self):
        self.root = tk.Tk()
        self.root.wm_title("Testing World!")
        self.canvas = tk.Canvas(self.root, width=c.CANVAS_WIDTH, height=c.CANVAS_HEIGHT, borderwidth=0, highlightthickness=0, bg="black")
        self.canvas.pack()
        self.root.update()

        return self
        

    def canvasX(self, x):
        return (c.CANVAS_BORDER//2) + (0.5 + x / float(self.width)) * (c.CANVAS_WIDTH-c.CANVAS_BORDER)

    def canvasY(self, y):
        return (c.CANVAS_BORDER// 2) + (0.5 + y / float(self.breadth)) * (c.CANVAS_HEIGHT-c.CANVAS_BORDER)


    def drawMobs(self, entities, flash):
        self.canvas.delete("all")
        if flash:
            self.canvas.create_rectangle(0,0,c.CANVAS_WIDTH,c.CANVAS_HEIGHT,fill="#ff0000") # Pain.
        self.canvas.create_rectangle(self.canvasX(-self.width // 2), self.canvasY(-self.breadth // 2), self.canvasX(self.width //2), self.canvasY(self.breadth // 2), fill="#888888")
        for ent in entities:
            fill = c.COLORS[ent["name"]] if ent["name"] in c.COLORS else "#00BFFF"
            self.canvas.create_oval(self.canvasX(ent["x"])-self.mobsize, self.canvasY(ent["z"])-self.mobsize, self.canvasX(ent["x"])+self.mobsize, self.canvasY(ent["z"])+self.mobsize, fill=fill)
        self.root.update()
