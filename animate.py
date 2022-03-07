# Creating cartpole animation

import numpy as np
from math import pi, cos, sin
from tkinter import *
import time

def animate(sim):

    #############################################
    # Rotate function (rotates pole)
    # Inputs:
    #   vertices: array of points defining polygon shape
    #   theta: rotation angle in radians
    def rotate(theta,x):

        vertices = [-pole_width/2,0,pole_width/2,0,pole_width/2,pole_height,-pole_width/2,pole_height]
        # First translate object so that pivot point is at (0,0): (create matrix)
        m_vertices = np.array([[vertices[0],vertices[1]], [vertices[2],vertices[3]],
                    [vertices[4],vertices[5]], [vertices[6],vertices[7]]])

        r_vertices = np.zeros((4,2))
        for i in range(4):
            vector = m_vertices[i]
            rot_mtrx = np.array(( (cos(theta), -sin(theta)),
                                    (sin(theta), cos(theta)) ))
            r_vector = rot_mtrx.dot(vector)
            r_vertices[i] = [r_vector[0]+x,floor_y-r_vector[1]-cart_height]
        
        new_vertices = []
        for i in range(4):
            for j in range(2):
                new_vertices.append(r_vertices[i][j])
        
        return new_vertices

    #############################################
    # Initialize canvas
    gui = Tk()
    gui.geometry("800x600")
    gui.title("Cartpole Simulation")
    width = 800
    height = 600
    x_center = width/2
    y_center = height/2
    canvas = Canvas(gui, width=800,height=600,bg='white')
    canvas.pack()

    #############################################
    # Import and scale data
    x_unit = width/4.8
    xp_list = np.array(sim.x_list)*x_unit + width/2 # x history in tkinter scale

    dxp_list = np.diff(xp_list) # Divide by time step later
    dxp_list = np.insert(dxp_list,0,0)
    theta_list = sim.theta_list

    #############################################
    # Create animation objects
    floor_y = 500
    floor = canvas.create_line(0, floor_y, width, floor_y, width='2')

    cart_xi = xp_list[0] # center of cart
    cart_width = 100
    cart_height = 50
    cart = canvas.create_rectangle(cart_xi-cart_width/2,floor_y-cart_height,cart_xi+cart_width/2,floor_y, fill='black')

    pivot_radius = 6
    pivot = canvas.create_oval(cart_xi-pivot_radius,floor_y-cart_height-pivot_radius,cart_xi+pivot_radius,floor_y-cart_height+pivot_radius, fill='white')

    pole_width = 10
    pole_height = 120
    new_vertices = rotate(theta_list[0],cart_xi)
    pole = canvas.create_polygon(new_vertices, fill='red')

    #############################################
    # Animate frames
    for i in range(0,len(xp_list)):
        
        canvas.move(cart,dxp_list[i],0)
        canvas.move(pivot,dxp_list[i],0)
        
        new_vertices = rotate(-theta_list[i],xp_list[i])
        canvas.coords(pole,new_vertices)

        gui.update()
        time.sleep(.01)

    gui.mainloop()
