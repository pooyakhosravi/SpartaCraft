import tkinter as tk
import src.constants as c

def init_canvas():
    root = tk.Tk()
    root.wm_title("Testing World!")

    canvas = tk.Canvas(root, width=c.CANVAS_WIDTH, height=c.CANVAS_HEIGHT, borderwidth=0, highlightthickness=0, bg="black")
    canvas.pack()
    root.update()

    return root, canvas


def canvasX(x):
    return (c.CANVAS_BORDER//2) + (0.5 + x / float(c.ARENA_WIDTH)) * (c.CANVAS_WIDTH-c.CANVAS_BORDER)

def canvasY(y):
    return (c.CANVAS_BORDER// 2) + (0.5 + y / float(c.ARENA_BREADTH)) * (c.CANVAS_HEIGHT-c.CANVAS_BORDER)


def drawMobs(root, canvas, entities, flash):
    canvas.delete("all")
    if flash:
        canvas.create_rectangle(0,0,c.CANVAS_WIDTH,c.CANVAS_HEIGHT,fill="#ff0000") # Pain.
    canvas.create_rectangle(canvasX(-c.ARENA_WIDTH // 2), canvasY(-c.ARENA_BREADTH // 2), canvasX(c.ARENA_WIDTH //2), canvasY(c.ARENA_BREADTH // 2), fill="#888888")
    for ent in entities:
        fill = c.COLORS[ent["name"]] if ent["name"] in c.COLORS else "#00BFFF"
        canvas.create_oval(canvasX(ent["x"])-4, canvasY(ent["z"])-4, canvasX(ent["x"])+4, canvasY(ent["z"])+4, fill=fill)
    root.update()
