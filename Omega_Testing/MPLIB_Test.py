
import matplotlib, sys
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pylab as plt
from scipy import ndimage

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

root = Tk.Tk()
root.wm_title("minimal example")

image = plt.imread('Scanned.jpg')
fig = plt.figure(figsize=(5,4))
im = plt.imshow(image) # later use a.set_data(new_data)
ax = plt.gca()
ax.set_xticklabels([]) 
ax.set_yticklabels([]) 

# a tk.DrawingArea
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

def rotate(*args):
    print('rotate button press...')
    theta = 90
    rotated = ndimage.rotate(image, theta)
    im.set_data(rotated)
    canvas.draw()

def quit(*args):
    print('quit button press...')
    root.quit()     
    root.destroy() 

button_rotate = Tk.Button(master = root, text = 'Rotate', command = rotate)
button_quit = Tk.Button(master = root, text = 'Quit', command = quit)

button_quit.pack(side=Tk.LEFT)
button_rotate.pack()

Tk.mainloop()
