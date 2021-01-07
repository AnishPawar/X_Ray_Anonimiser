from tkinter import *
import tkinter as tk
from tkinter import filedialog, Text
from matplotlib.backends.backend_agg import FigureCanvasAgg

from numpy.core.records import fromarrays

from Backend_Beta import mouse,blur_function,ocr_function,auto_crop,image_parser

import cv2
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 

# Global Variables
og_img = np.zeros((), np.uint8)
temp_img = np.zeros((), np.uint8)

text = ''
fname = []

img_memory = []

fig = []

# GUI Main loop
root = tk.Tk()

canvas = tk.Canvas(root,height=1300,width = 1300,bg="#D0E69E")
canvas.pack()

frame = tk.Frame(root,bg="white")
frame.place(relwidth=1,relheight=0.1,relx=0,rely=0)

plot_frame = tk.Frame(root,bg="#D0E69E")
plot_frame.place(relwidth=0.65,relheight=0.8,relx=0.3,rely=0.15)



# Image Display
image = plt.imread('Ano.png')
fig = plt.figure(dpi=160)

im = plt.imshow(image,interpolation = 'nearest',aspect='equal') 
ax = plt.gca()
# plt.axis("equal")
ax.set_xticklabels([]) 
ax.set_yticklabels([]) 

canvas1 = FigureCanvasTkAgg(fig, master=plot_frame)
canvas1.draw()

toolbar = NavigationToolbar2Tk(canvas1, plot_frame)
toolbar.update()

canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1) 


def plot_new(img):
    im.set_data(img)
    canvas1.draw()

# Button Click Definitions
def open_btn_clicked(): 
    global fig
    # plt.close(fig)
    global fname,img_memory
    filename= filedialog.askopenfilename(initialdir="/Users/anishpawar/Robocon/Robocon_Anish/Matlab_IP/IP/Learning",title = "Select Image",filetypes=(("JPEG","*.jpg"),("PNG","*.png"),("all files","*.*")))
    # Name Pre-Processing 
    names = filename.split('/')
    fname = names[-1].split('.')

    global og_img,temp_img
    og_img = cv2.imread(filename)
    temp_img = og_img.copy()
    img_memory.append(temp_img)

    histogram = temp_img.ravel()
    if np.mean(histogram) <=120:
        temp_img = cv2.bitwise_not(temp_img)
    img_memory.append(temp_img)
    plot_new(og_img)
    
def fliph_btn_clicked():
    plt.close()
    global temp_img,img_memory,og_img
    temp_img = cv2.flip(temp_img,1)
    og_img = cv2.flip(og_img,1)
    img_memory.append(temp_img)
    plot_new(og_img)

def flipv_btn_clicked():
    plt.close()
    global temp_img,img_memory,og_img
    temp_img = cv2.flip(temp_img,0)
    og_img = cv2.flip(og_img,0)
    img_memory.append(temp_img)
    plot_new(og_img)

def ocr_btn_clicked():
    plt.close()
    global text,og_img
    print("Working.... \n")
    text,og_img = ocr_function(temp_img,og_img)
    plot_new(og_img)

def mcrop_btn_clicked():
    global img_memory
    image_parser(temp_img)
    cv2.imshow('Processed',temp_img)
    cv2.setMouseCallback('Processed',mouse)

def destroy_btn_clicked():
    cv2.destroyAllWindows()

def save_img_btn_clicked():
    cv2.imwrite("{}_Processed.{}".format(fname[0],fname[1]),og_img)

def auto_crop_btn_clicked():
    global temp_img,img_memory
    temp_img = auto_crop(temp_img)
    plot_new(temp_img)

def undo_btn_clicked():
    temp_img = img_memory[-2]
    img_memory.pop()
    plot_new(temp_img)

#Labels

label = tk.Label(frame,text='X-Ray Anomiser',fg= 'blue',bg='white',font =('Arial',25)) 
label.place(relx=0.4,rely=0.2)

label_1 = tk.Label(canvas,text='Image Loading:',fg= 'blue',bg='#D0E69E',font =('Arial',18)) 
label_1.place(relx=0.025,rely=0.115)

label_img = tk.Label(canvas,text='Processed Image',fg= 'blue',bg='#D0E69E',font =('Arial',18)) 
label_img.place(relx=0.55,rely=0.115)

label_2 = tk.Label(canvas,text='Cropping:',fg= 'blue',bg='#D0E69E',font =('Arial',18)) 
label_2.place(relx=0.025,rely=0.235)

label_3 = tk.Label(canvas,text='Transformations:',fg= 'blue',bg='#D0E69E',font =('Arial',18)) 
label_3.place(relx=0.025,rely=0.35)

# Button Definitions

open_btn = tk.Button(canvas,text="Open Image",padx=2,pady=10,fg="black",command=open_btn_clicked)
open_btn.place(relx=0.025,rely=0.16)

auto_crop_btn = tk.Button(canvas,text="Auto Crop",padx=7,pady=10,fg="black",command = auto_crop_btn_clicked)
auto_crop_btn.place(relx=0.025,rely=0.275)

mcrop_btn = tk.Button(canvas,text="Manual Crop",padx=2,pady=10,bg="#263D42",fg="black",command=mcrop_btn_clicked)
mcrop_btn.place(relx=0.15,rely=0.275)


FlipH_btn = tk.Button(canvas,text="Flip_H",padx=20,pady=10,fg="black",command= fliph_btn_clicked)
FlipH_btn.place(relx=0.025,rely=0.4)

FlipV_btn = tk.Button(canvas,text="Flip_V",padx=22,pady=10,fg="black",command= flipv_btn_clicked)
FlipV_btn.place(relx=0.15,rely=0.4)


undo_btn = tk.Button(canvas,text="Undo",padx=22,pady=10,fg="black",command= undo_btn_clicked)
undo_btn.place(relx=0.025,rely=0.475)

ocr_btn = tk.Button(canvas,text="Anomise",padx=13,pady=10,bg="#263D42",fg="black",command=ocr_btn_clicked)
ocr_btn.place(relx=0.15,rely=0.475)

save_img_btn = tk.Button(canvas,text=" Save \n Image",padx=18,pady=10,bg="#263D42",fg="green",command=save_img_btn_clicked)
save_img_btn.place(relx=0.025,rely=0.575)

destroy_btn = tk.Button(canvas,text="Close \n Windows",padx=9,pady=10,bg="#263D42",fg="red",command=destroy_btn_clicked)
destroy_btn.place(relx=0.15,rely=0.575)

plot_button = Button(master = canvas,height = 2,width = 8,padx = 4,text = "Display") 
plot_button.place(relx=0.15,rely=0.16) 


root.mainloop()