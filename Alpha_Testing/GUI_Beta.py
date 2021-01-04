from tkinter import *
import tkinter as tk
from tkinter import filedialog, Text

from Backend_Beta import mouse,blur_function,ocr_function,auto_crop

import cv2
import numpy as np

import matplotlib.pyplot as plt

# Global Variables
og_img = np.zeros((), np.uint8)
temp_img = np.zeros((), np.uint8)

text = ''
fname = []

img_memory = []

# GUI Main loop
root = tk.Tk()

canvas = tk.Canvas(root,height=700,width = 700,bg="#6BF2BF")
canvas.pack()

frame = tk.Frame(root,bg="white")
frame.place(relwidth=0.6,relheight=0.9,relx=0.2,rely=0.05)

text_box = tk.Frame(frame,bg="#FDFFD6")
text_box.place(relwidth=0.6,relheight=0.6,relx=0.2,rely=0.2)

# Button Click Definitions
def open_btn_clicked():
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

    cv2.imshow('Original_Image',og_img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    cv2.imshow('Processed',temp_img)
    
def fliph_btn_clicked():
    global temp_img,img_memory,og_img
    temp_img = cv2.flip(temp_img,1)
    og_img = cv2.flip(og_img,1)
    img_memory.append(temp_img)
    cv2.imshow('Processed',temp_img)

def flipv_btn_clicked():
    global temp_img,img_memory,og_img
    temp_img = cv2.flip(temp_img,0)
    og_img = cv2.flip(og_img,0)
    img_memory.append(temp_img)
    cv2.imshow('Processed',temp_img)

def blur_btn_clicked():
    global img_memory
    blur_function(temp_img)

def show_btn_clicked():
    cv2.imshow('Processed',temp_img)
    
def ocr_btn_clicked():
    global text,og_img
    print("Working.... \n")
    text,og_img = ocr_function(temp_img,og_img)

def mcrop_btn_clicked():
    global img_memory
    cv2.imshow('Processed',temp_img)
    cv2.setMouseCallback('Processed',mouse)

def destroy_btn_clicked():
    cv2.destroyAllWindows()

def show_txt_btn_clicked():
    
    scrollbar = Scrollbar(text_box)
    scrollbar.pack(side=RIGHT, fill=Y)
    ocr = Text(text_box, wrap=WORD, yscrollcommand=scrollbar.set,bg = '#FDFFD6')
    ocr.insert('1.0',text)

    ocr.place(relx=0,rely=0,relheight = 1,relwidth =1)

    scrollbar.config(command=ocr.yview)

def save_img_btn_clicked():
    cv2.imwrite("{}_Processed.{}".format(fname[0],fname[1]),og_img)

def auto_crop_btn_clicked():
    global temp_img,img_memory
    temp_img = auto_crop(temp_img)
    cv2.imshow('Processed',temp_img)

def dummy_btn_clicked():
    global temp_img,img_memory
    
    # Invert or not decide! 
    histogram = temp_img.ravel()
    if np.mean(histogram) <=120:
        temp_img = cv2.bitwise_not(temp_img)
    
    # Add Any Number of Filters Below This For Testing


    img_memory.append(temp_img)
    cv2.imshow("Processed",temp_img)

def undo_btn_clicked():
    temp_img = img_memory[-2]
    img_memory.pop()
    cv2.imshow("Processed",temp_img)
    
#Labels

label = tk.Label(frame,text='Detected Text',fg= 'blue',bg='white',font =('Arial',22)) 
label.place(relx=0.3,rely=0.1)

# Buttons
open_btn = tk.Button(canvas,text="Open Image",padx=5,pady=10,fg="black",command=open_btn_clicked)
open_btn.place(relx=0.025,rely=0.05)

blur_btn = tk.Button(canvas,text="Blur Image",padx=5,pady=10,fg="black",command=blur_btn_clicked)
blur_btn.place(relx=0.025,rely=0.2)

auto_crop_btn = tk.Button(canvas,text="Auto Crop",padx=5,pady=10,fg="black",command = auto_crop_btn_clicked)
auto_crop_btn.place(relx=0.025,rely=0.375)

FlipH_btn = tk.Button(canvas,text="Flip_H",padx=14,pady=10,fg="black",command= fliph_btn_clicked)
FlipH_btn.place(relx=0.025,rely=0.775)

FlipV_btn = tk.Button(canvas,text="Flip_V",padx=14,pady=10,fg="black",command= flipv_btn_clicked)
FlipV_btn.place(relx=0.85,rely=0.775)

mcrop_btn = tk.Button(canvas,text="Manual Crop",padx=2,pady=10,bg="#263D42",fg="black",command=mcrop_btn_clicked)
mcrop_btn.place(relx=0.025,rely=0.5)

ocr_btn = tk.Button(canvas,text="OCR",padx=15,pady=10,bg="#263D42",fg="black",command=ocr_btn_clicked)
ocr_btn.place(relx=0.85,rely=0.05)

show_txt_btn = tk.Button(canvas,text="Show Text",padx=1,pady=10,bg="#263D42",fg="black",command=show_txt_btn_clicked)
show_txt_btn.place(relx=0.85,rely=0.2)

save_img_btn = tk.Button(canvas,text="Save Image",padx=1,pady=10,bg="#263D42",fg="green",command=save_img_btn_clicked)
save_img_btn.place(relx=0.85,rely=0.375)

show_btn = tk.Button(canvas,text="Show Original",padx=1,pady=10,bg="#263D42",fg="red",command=show_btn_clicked)
show_btn.place(relx=0.025,rely=0.895)

destroy_btn = tk.Button(canvas,text="Close \n Windows",padx=5,pady=10,bg="#263D42",fg="red",command=destroy_btn_clicked)
destroy_btn.place(relx=0.85,rely=0.87)

dummy_btn = tk.Button(canvas,text="Testing",padx=5,pady=10,bg="#263D42",fg="red",command=dummy_btn_clicked)
dummy_btn.place(relx=0.85,rely=0.575)

undo_btn = tk.Button(canvas,text="Undo",padx=14,pady=10,fg="black",command= undo_btn_clicked)
undo_btn.place(relx=0.025,rely=0.575)

root.mainloop()