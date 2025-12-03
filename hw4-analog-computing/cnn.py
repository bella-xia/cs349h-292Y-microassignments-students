import networkx as nx
import numpy as np
import scipy.sparse  as sp
import scipy
import itertools 
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import random
import os 
from PIL import Image

import seaborn as sns




class CNNDiffeqSystem:

    def __init__(self,n,m):
        self.n = n
        self.m = m
        self.U = np.zeros(n*m)
        self.initVals = np.zeros(n*m)
        self.A = {}
        self.B = {}
        self.N = {}
        self.graph = nx.grid_2d_graph(self.n, self.m)

    def valid_coord(self,i,j):
        return i >= 0 and j >= 0 and i < self.n and j < self.m

    def initialize(self,z,A,B):
        for i in range(self.n):
            for j in range(self.m):
                k = i*self.m + j
                self.A[k] = np.zeros(self.n*self.m)
                self.B[k] = np.zeros(self.n*self.m)
                self.N[k] = np.zeros(self.n*self.m)
                for ti in range(3):
                    for tj in range(3):
                        ni = i+ti-1
                        nj = j+tj-1
                        if not self.valid_coord(ni,nj):
                            continue
                        nk = ni*self.m + nj
                        self.A[k][nk] = A[ti][tj]
                        self.B[k][nk] = B[ti][tj]
                        self.N[k][nk] = 1 



        self.Z = np.full(self.n*self.m, z)


                    

    def set_image(self,U,scale=1.0):
        for i in range(self.n):
            for j in range(self.m):
                k = i*self.m + j
                self.U[k] = U[i,j]*scale*2-1
                assert(self.U[k] <= 1.0)
                assert(self.U[k] >= -1.0)
    
    

    def diffeqs(self,x,t):
        vecFn = np.vectorize(lambda i: 
                             sum(np.multiply(self.A[i], np.clip(x,-1,1))) + sum(np.multiply(self.B[i],self.U)) + self.Z[i] - x[i])
                             # sum(np.multiply(self.A[i], 2 / (1 + np.exp(-x)) - 1)) + sum(np.multiply(self.B[i], self.U)) + self.Z[i] - x[i])
        
        dx = vecFn(np.arange(self.n*self.m)) # * 10
        return dx

    def run(self, time, npts=100):
        def ddt(time, x):
            return self.diffeqs(x,time) 
        print("<running solver>")
        times = np.linspace(0,time,npts)
        sol = scipy.integrate.solve_ivp(ddt, [0,time], self.initVals, t_eval=times)
        return sol.t, sol.y


    def render(self,filename,times,values):
        # draw the topology of the graph, what changes during animation
        # is just the color
        print("<rendering video>")

        plt.clf()
        fig = plt.figure()


        def init():
            sns.heatmap(np.zeros((28, 28)), vmax=1, square=True, cbar=False)

        def animate(i):
            variables=0.5*(np.clip(values[:,i],-1,1) + 1) # np.random.randint(2, size=200)
            image = variables.reshape((28,28))
            #data = np.random.rand(10, 10)
            sns.heatmap(image, vmax=1.0, square=True, cbar=False)

        anim = FuncAnimation(fig, animate, init_func=init, frames=range(len(times)), repeat = False)
        print("# of frames: %d" % (len(times)))
        anim.save(filename, writer='imagemagick')
 




def run_cnn():
    #choose a random MNIST image
    dataset = scipy.io.loadmat("dataset/caltech101_silhouettes_28.mat")

    index = 63
    pixelData= 1 - np.array(dataset["X"][index]).reshape((28,28))

    im = Image.fromarray(pixelData*255)
    im.save("cnn_input.png")

    #instantiate CNN
    cnn = CNNDiffeqSystem(28,28)
    # cnn.initVals = np.random.uniform(0, 0.1, 28*28)
    cnn.initVals = np.zeros(28*28)
    #set image as forcing function provided to U matrix.
    cnn.set_image(pixelData,scale=1.0)

    EXERCISE = 1
    
    # instantiate parameters
    A1 = [[0,0,0],[0,2,0],[0,0,0]]
    B1 = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    z1 = -0.5
    
    A2=[[-0.37, 0.68, -0.37], [0.68, 0.60, 0.68], [-0.37,0.68,-0.37]]
    B2=[[-1.21,-0.68,-1.21],[-0.68,7.76,-0.68],[-1.21,-0.68,-1.21]]
    z2=-1.77
 
    if EXERCISE == 1:
        cnn.initialize(z=z1, A=A1, B=B1)

    elif EXERCISE == 2:
        # PART B: introduce noise to A1, B1
        noise = 0.20
        AZ = np.random.uniform(-noise,noise,size=9).reshape((3,3))
        BZ = np.random.uniform(-noise,noise,size=9).reshape((3,3))
        ZZ = np.random.uniform(-noise,noise)

        ANZ1 = A1 + A1*AZ
        BNZ1 = B1 + B1*BZ
        ZNZ1 = z1 + z1*ZZ
        cnn.initialize(z=ZNZ1, A=ANZ1, B=BNZ1)

    elif EXERCISE == 3: 
        cnn.initialize(z=z2, A=A2, B=B2)

    elif EXERCISE == 4:
       
        noise = 0.20
        AZ = np.random.uniform(-noise,noise,size=9).reshape((3,3))
        BZ = np.random.uniform(-noise,noise,size=9).reshape((3,3))
        ZZ = np.random.uniform(-noise,noise)

        ANZ2 = A2 + A2*AZ
        BNZ2 = B2 + B2*BZ
        ZNZ2 = z2 + z2*ZZ
        cnn.initialize(z=ZNZ2, A=ANZ2, B=BNZ2)

    # run CNN and save time-evolution to image.
    times, values = cnn.run(2)
    cnn.render("E%d-cnn.gif" % EXERCISE, times,values)



run_cnn()
    

