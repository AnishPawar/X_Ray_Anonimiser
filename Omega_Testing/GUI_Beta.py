from tkinter import *
import tkinter as tk
from tkinter import filedialog, Text
from matplotlib.backends.backend_agg import FigureCanvasAgg

from numpy.core.records import fromarrays

from Backend_Beta import ocr_function,auto_crop,image_parser,blur_correction

import cv2
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  NavigationToolbar2Tk) 

# Global Variables
og_img = np.zeros((), np.uint8)
temp_img = np.zeros((), np.uint8)

text = ''
fname = []

img_memory = []

fig = []

counter = 0
final_coor = []

file_list = []

# GUI Main loop
root = tk.Tk()

canvas = tk.Canvas(root,height=1300,width = 1300,bg="#D0E69E")
canvas.pack()

frame = tk.Frame(root,bg="white")
frame.place(relwidth=1,relheight=0.1,relx=0,rely=0)

plot_frame = tk.Frame(root,bg="#D0E69E")
plot_frame.place(relwidth=0.65,relheight=0.8,relx=0.3,rely=0.15,bordermode='inside')

# Image Display
image = plt.imread('Ano.png')
fig = plt.figure(dpi=160)

plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.2, hspace=0.2)

im = plt.imshow(image,interpolation = 'nearest',aspect='equal') 
ax = plt.gca()
ax.set_xticklabels([]) 
ax.set_yticklabels([]) 

canvas1 = FigureCanvasTkAgg(fig, master=plot_frame)
canvas1.draw()

# toolbar = NavigationToolbar2Tk(canvas1, plot_frame)
# toolbar.update()

canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1) 


def plot_new(img):
    im.set_data(img)
    canvas1.draw()

# Button Click Definitions
def open_btn_clicked(): 
    global fig,file_list
    # plt.close(fig)

    file_list = list(filedialog.askopenfilenames(initialdir="/home/oxidane/Desktop/XRAY/X_Ray_Data_Detector-main/Omega_Testing",title = "Select Image",filetypes=(("JPEG","*.jpg"),("PNG","*.png"),("all files","*.*"))))
    # Name Pre-Processing 
    load_img(file_list)

def load_img(file_list):
    global img_memory,og_img,temp_img,fname
    img_memory = []

    print(file_list)


    filename = file_list[0]

    names = filename.split('/')
    fname = names[-1].split('.')
    
    
    file_list.pop(0)


    og_img = cv2.imread(filename)

    temp_img = cv2.cvtColor(og_img,cv2.COLOR_BGR2GRAY)

    # temp_img = blur_correction(temp_img)

    histogram = temp_img.ravel()
    if np.mean(histogram) <=120:
        temp_img = cv2.bitwise_not(temp_img)
    img_memory.append(og_img)
    #plot_new(og_img)
    plot_new(temp_img)
    
def fliph_btn_clicked():
    plt.close()
    global temp_img,img_memory,og_img
    temp_img = cv2.flip(temp_img,1)
    og_img = cv2.flip(og_img,1)
    img_memory.append(og_img)
    plot_new(og_img)

def flipv_btn_clicked():
    plt.close()
    global temp_img,img_memory,og_img
    temp_img = cv2.flip(temp_img,0)
    og_img = cv2.flip(og_img,0)
    img_memory.append(og_img)
    plot_new(og_img)

def ocr_btn_clicked():
    plt.close()
    global text,og_img,img_memory,temp_img
    print("Working.... \n")
    text,og_img = ocr_function(temp_img,og_img)

    temp_img = og_img.copy()
    img_memory.append(og_img)
    plot_new(og_img)

def mcrop_btn_clicked():
    global temp_img,og_img
    # image_parser(temp_img)
    cv2.imshow('Processed',og_img)
    cv2.setMouseCallback('Processed',mouse)

def mouse(event,x,y,z,w):
    
    global counter,final_coor,og_img 
    marked = og_img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        counter = counter+1

        if counter == 1:
            pos1=(y,x)
            pos1_flip = (x,y)

            # final_coor.append(pos1_flip)
            final_coor.insert(0,pos1_flip)
            
            
            cv2.circle(marked,pos1_flip,60,(0,255,0),-1)
            cv2.imshow('Processed',marked)

        if counter == 2:
            pos2=(y,x)
            pos2_flip = (x,y)
            
            # final_coor.append(pos2_flip)
            final_coor.insert(1,pos2_flip)            
            
            cv2.circle(marked,pos2_flip,60,(0,255,0),-1)
            cv2.imshow('Processed',marked)

        if counter == 3:
            pos3=(y,x)
            pos3_flip = (x,y)
            
            # final_coor.append(pos3_flip)
            
            final_coor.insert(3,pos3_flip)

            cv2.circle(marked,pos3_flip,60,(0,255,0),-1)
            cv2.imshow('Processed',marked)

        if counter == 4:
            pos4=(y,x)      
            pos4_flip = (x,y)
            
            # final_coor.append(pos4_flip)

            final_coor.insert(2,pos4_flip)
            
            cv2.circle(marked,pos4_flip,60,(0,255,0),-1)
            print(final_coor)
            
            warp_function(final_coor,og_img)
            

def warp_function(coor,img):
    global og_img,temp_img,img_memory
    pts1 = np.float32(coor)
    pts2 = np.float32([(0, 0), (500, 0), (0, 600), (500, 600)])

    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    # global warped
    warped = cv2.warpPerspective(img, matrix, (500,600))            
    cv2.destroyAllWindows()
    og_img = warped
    temp_img = og_img.copy()
    img_memory.append(og_img)
    plot_new(og_img)

def rotate_r_btn_clicked(): 
    global temp_img,img_memory,og_img
    temp_img = cv2.rotate(temp_img,cv2.ROTATE_90_CLOCKWISE)
    og_img = cv2.rotate(og_img,cv2.ROTATE_90_CLOCKWISE)
    img_memory.append(og_img)
    plot_new(og_img)

def rotate_l_btn_clicked():
    global temp_img,img_memory,og_img
    temp_img = cv2.rotate(temp_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    og_img = cv2.rotate(og_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_memory.append(og_img)
    plot_new(og_img)


def destroy_btn_clicked():
    cv2.destroyAllWindows()

def save_img_btn_clicked():
    global file_list,fname,og_img
    print(fname)
    cv2.imwrite("{}_Processed.{}".format(fname[0],fname[1]),og_img)

    if file_list:
        load_img(file_list)


def auto_crop_btn_clicked():
    global og_img,temp_img,img_memory
    og_img = auto_crop(og_img)
    temp_img = og_img.copy()
    img_memory.append(og_img)

    plot_new(og_img)

def undo_btn_clicked():
    temp_img = img_memory[-2]
    img_memory.pop()
    plot_new(temp_img)

def show_credits():
    cred_frame = tk.Frame(canvas,bg="white")
    cred_frame.place(relwidth=0.22,relheight=0.1,relx=0.025,rely=0.85)

    label = tk.Label(cred_frame,text='Anish Pawar SY EXTC\nChaitanya Bandiwdekar SY IT',fg= 'black',bg='white',font =('Arial',16),justify='left') 
    label.place(relx=0.0,rely=0.2)

#Labels

label = tk.Label(frame,text='X-Ray Anonimiser',fg= 'blue',bg='white',font =('Arial',25)) 
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

open_btn = tk.Button(canvas,text="Open Image",height=2,width=10,fg="black",command=open_btn_clicked)
open_btn.place(relx=0.025,rely=0.16)

auto_crop_btn = tk.Button(canvas,text="Auto Crop",height=2,width=10,fg="black",command = auto_crop_btn_clicked)
auto_crop_btn.place(relx=0.025,rely=0.275)

mcrop_btn = tk.Button(canvas,text="Manual Crop",height=2,width=10,bg="#263D42",fg="black",command=mcrop_btn_clicked)
mcrop_btn.place(relx=0.15,rely=0.275)


FlipH_btn = tk.Button(canvas,text="Flip_H",height=2,width=10,fg="black",command= fliph_btn_clicked)
FlipH_btn.place(relx=0.025,rely=0.4)

FlipV_btn = tk.Button(canvas,text="Flip_V",height=2,width=10,fg="black",command= flipv_btn_clicked)
FlipV_btn.place(relx=0.15,rely=0.4)

rotate_r = tk.Button(canvas,text="Rotate_R",height=2,width=10,fg="black",command= rotate_r_btn_clicked)
rotate_r.place(relx=0.025,rely=0.475)

rotate_l = tk.Button(canvas,text="Rotate_L",height=2,width=10,bg="#263D42",fg="black",command=rotate_l_btn_clicked)
rotate_l.place(relx=0.15,rely=0.475)


undo_btn = tk.Button(canvas,text="Undo",height=2,width=10,fg="black",command= undo_btn_clicked)
undo_btn.place(relx=0.025,rely=0.55)

ocr_btn = tk.Button(canvas,text="Anonimise",height=2,width=10,bg="#263D42",fg="black",command=ocr_btn_clicked)
ocr_btn.place(relx=0.15,rely=0.55)

save_img_btn = tk.Button(canvas,text=" Save \n Image",height=2,width=10,bg="#263D42",fg="green",command=save_img_btn_clicked)
save_img_btn.place(relx=0.025,rely=0.625)

destroy_btn = tk.Button(canvas,text="Close \n Windows",height=2,width=10,bg="#263D42",fg="red",command=destroy_btn_clicked)
destroy_btn.place(relx=0.15,rely=0.625)

credit_button = Button(master = canvas,height=2,width=10,text = "Credits",command=show_credits) 
credit_button.place(relx=0.15,rely=0.16) 


root.mainloop()