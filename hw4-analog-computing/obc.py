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
import math
import os

from basic_units import cos, degrees, radians


khz = 1000

IMAGE_DIR="images-obc"
if not os.path.exists(IMAGE_DIR):
    os.mkdir(IMAGE_DIR)


def random_phase(variance):
    assert(variance <= 1.0)
    return random.uniform(-math.pi*variance,variance*math.pi) + math.pi

def random_freq(minfreq,maxfreq):
    assert(minfreq < maxfreq)
    return random.uniform(minfreq,maxfreq)

def random_uniform():
    return random.uniform(-1,1.0)

def rel_phase(phase_abs):
    phase_offset = 2*math.pi*math.floor(phase_abs/(2*math.pi))
    return phase_abs-phase_offset


class OBCDiffeqSystem:

    def __init__(self,n,naturalFreqNominal=None,SHILRatio=None,couplingStrength=100,injection_locked=False):
        self.injection_locked = injection_locked
    
        self.naturalFreqNominal = naturalFreqNominal
        self.SHILRatio = SHILRatio 
        self.couplingStrength = couplingStrength
        self.n = n
        self.couplings = {}
        self.graph = nx.Graph()
        for i in range(n):
            self.couplings[i] = np.zeros(self.n)
            self.graph.add_node(i)
            
        self.naturalFreq = np.full(self.n, 1)
        self.initialPhase = np.full(self.n, 0.0)
        self.locking = np.full(self.n, 0.2)

    def random_initial_phase(self):
        self.initialPhase = np.random.uniform(-np.pi, np.pi, size=self.n) 

    def set_coupling(self, w, i, j):
        self.couplings[i][j] = w
        self.couplings[j][i] = w
        self.graph.add_edge(i,j)

    def set_locking(self,locking):
        self.locking = np.full(self.n, locking)

    def set_natural_frequency(self,i,freqZero):
        print("osc %d: freq=%f" % (i,freqZero))
        self.naturalFreq[i] = freqZero

    def set_initial_phase(self, i, phiZero):
        print("osc %d: phase=%f" % (i,phiZero))
        self.initialPhase[i] = phiZero
    
    def diffeqs_injection_locked(self, t, x):
        phases = x[:self.n]
        freqs = x[self.n:2*self.n]
        deltaFreqs = self.naturalFreq-self.naturalFreqNominal
        dPhaseFn = np.vectorize(lambda i: 
                             1/self.n*self.couplingStrength*np.sum(np.multiply(self.couplings[i], 
                                                         np.sin(phases-np.full(self.n, phases[i])))) 
                             +self.naturalFreq[i]-self.locking[i]*np.sin(self.SHILRatio*x[i]))
        dPhase = dPhaseFn(np.arange(self.n))
        return dPhase 

    def diffeqs_natural(self, t, x):
        phases = x[:self.n]
        freqs = x[self.n:2*self.n]
        dPhaseFn = np.vectorize(lambda i: 
                             self.couplingStrength*np.sum(np.multiply(self.couplings[i], 
                                                         np.sin(phases-np.full(self.n, phases[i])))) 
                             +self.naturalFreq[i])
        
        dPhase = dPhaseFn(np.arange(self.n))
        return dPhase

    def phase_offset(self,times,freqs,phases):
        offsets = {}
        for j in range(self.n):
            total_offset = 0
            phase_offset = [phases[j][0]]
            for i in range(1,len(times)):
                dt = times[i] - times[i-1]
                total_offset += dt*freqs[j][i]
                phase_offset.append(float(phases[j][i] - total_offset))
            offsets[j] = phase_offset
        return offsets


    def render_timeseries(self,phasname,freqname,times,phases,freqs):
        plt.xlabel("time")
        #plt.ylabel("phase")
        npts = len(times)

        for i in range(self.n):
            plt.plot(times, list(map(lambda phase_abs: rel_phase(phase_abs), phases[i])), \
                yunits=radians, label="osc%d" % i)

        plt.legend()
        plt.savefig(IMAGE_DIR+"/"+phasname)
        plt.clf()


        for i in range(self.n):
            plt.plot(times, freqs[i],label="osc%d" % i)


        xmin, xmax, ymin, ymax = plt.axis()
        plt.gca().set_ylim(bottom=0,top=ymax*1.5)
        plt.xlabel("time")
        plt.ylabel("frequency")
        plt.plot()
        plt.legend()
        plt.savefig(IMAGE_DIR+"/"+freqname)
        plt.clf()



    def render_graph(self,figure, times,values):
        # draw the topology of the graph, what changes during animation
        # is just the color
        print("<rendering video>")
        pos = nx.spring_layout(self.graph)
        nodes = nx.draw_networkx_nodes(self.graph,pos)
        edges = nx.draw_networkx_edges(self.graph,pos)
        plt.axis('off')
        print(values)
        def update(i):
            nc = times[i] # np.random.randint(2, size=200)
            stvars = values[:,i] # np.random.randint(2, size=200)
            nodes.set_array(stvars)
            return nodes, 

        fig = plt.gcf()
        print("# of frames: %d" % (len(times)))
        ani = FuncAnimation(fig, update, interval=50, frames=range(len(times)), blit=True)
        ani.save(IMAGE_DIR+"/"+figure, writer='imagemagick',  savefig_kwargs={'facecolor':'white'}, fps=1)

    def render_oscillators(self,figure,times,phases,freqs,fps=30):
        maxfreq = max(map(lambda i: max(freqs[i]), range(self.n)))
        ntimepts = len(times)
        maxtime = 1.0/maxfreq*10
        times = np.linspace(0,maxtime,1000)
        def makesin(freq,phase):
            return np.array(list(map(lambda t: np.sin(freq*t+phase), times)))

        fig,axs = plt.subplots(self.n,1)
        lines = {}
        for i in range(self.n):
            axs[i].set_xlabel("time")
            axs[i].set_ylabel("phase")
            axs[i].set_title("oscillator %d" % i)
            axs[i].set_ylim(bottom=-1,top=1)
            lines[i], = axs[i].plot(times,makesin(self.naturalFreq[i],self.initialPhase[i]), color = "purple", lw=1)

        def animate(i):
            for j in range(self.n):
                freq = freqs[j][i]
                phase = phases[j][i]
                lines[j].set_data(times,makesin(freq,phase))
            
            return tuple(lines.values())

        interval_ms = 1.0/(fps/1000.0) 
        ani = FuncAnimation(fig, animate, interval=interval_ms, blit=True, repeat=False, frames=ntimepts, save_count=1000)    
        ani.save(IMAGE_DIR+"/"+figure, writer='imagemagick',fps=fps)


    def run(self, time, npts=100):
        if self.injection_locked:
            def ddt(x, time):
                return self.diffeqs_injection_locked(x,time)
        else:
            def ddt(x, time):
                return self.diffeqs_natural(x,time)

        print("<running solver>")
        times = np.linspace(0,time,npts)
        x0 = self.initialPhase
        sol = scipy.integrate.solve_ivp(ddt, [0,time], x0, t_eval=times)
        print("num pts: %d" % (len(sol.t)))
        phases = {}
        freqs = {}
        for i in range(self.n):
            phases[i] = sol.y[i,:]
            freqs[i]=np.abs(np.gradient(sol.y[i,:], sol.t[1]-sol.t[0]))
        return sol.t, phases, freqs



def simple_obc_phaseonly():
    EXERCISE =  4
    random.seed(5552321)
    nomFreq = math.pi
    # the natural frequency may only change by 30% of nominal
    coupleStrength = 4
    # the oscillators' locking strength should not dominate coupling strength
    lockStrength = 2

    # simple OBC without any coupling
    obc = OBCDiffeqSystem(4,naturalFreqNominal=nomFreq,couplingStrength=1, injection_locked=False)

    obc.set_natural_frequency(0,nomFreq)
    obc.set_natural_frequency(1,nomFreq)
    obc.set_natural_frequency(2,nomFreq)
    obc.set_natural_frequency(3,nomFreq)

    # let the simulation proceed for 20 oscillator iterations
    ncycles = 40
    runtime = 1/(nomFreq)*ncycles
    ptsPerCycle = 30

    flux = 1.0
    obc.set_initial_phase(0,random_phase(flux))
    obc.set_initial_phase(1,random_phase(flux))
    obc.set_initial_phase(2,random_phase(flux))
    obc.set_initial_phase(3,random_phase(flux))

    part2_couple_oscillators = (EXERCISE == 2)
    couple_weight = 1.0
    if part2_couple_oscillators:
        obc.set_coupling(couple_weight,0,1)
        obc.set_coupling(couple_weight,1,2)
        obc.set_coupling(couple_weight,2,3)

    part3_couple_oscillators = (EXERCISE == 3)
    couple_weight = 1.0
    if part3_couple_oscillators:
        obc.set_coupling(couple_weight,0,1)
        obc.set_coupling(0.5*couple_weight,1,2)
        obc.set_coupling(couple_weight,2,3)


    part4_couple_oscillators = (EXERCISE == 4)
    couple_weight = 1.0
    if part4_couple_oscillators:
        obc.set_coupling(couple_weight,0,1)
        obc.set_coupling(-couple_weight,1,2)
        obc.set_coupling(couple_weight,2,3)


    times,phases,freqs = obc.run(runtime,npts=ptsPerCycle*ncycles)
    obc.render_timeseries(f"A{EXERCISE}-phase.png",f"A{EXERCISE}-freq.png",times,phases,freqs)
    obc.render_oscillators(f"A{EXERCISE}-obc.gif",times,phases,freqs,fps=ptsPerCycle)


def simple_obc_freqonly():
    EXERCISE = 5
    minFreq = 5.6
    maxFreq = 6.2
    random.seed(5552321)
    couplingStrength = 1
    # simple OBC without any coupling
    obc = OBCDiffeqSystem(3,couplingStrength=couplingStrength, injection_locked=False)

    C1,C2 = 0,1
    A = 2


    Afreq = 3.0
    if EXERCISE == 2: 
        Afreq = 5.8
    elif EXERCISE == 3:
        Afreq = 1.0
    elif EXERCISE == 4:
        Afreq = 11.0
    elif EXERCISE == 5:
        Afreq = 3.0


    obc.set_natural_frequency(C1,5.8)
    obc.set_natural_frequency(C2,5.6)
    obc.set_natural_frequency(A,Afreq)

    # let the simulation proceed for 20 oscillator iterations
    ncycles = 40
    runtime = 1/(minFreq)*ncycles
    ptsPerCycle = 30

    flux = 1
    obc.set_initial_phase(0,random_phase(flux))
    obc.set_initial_phase(1,random_phase(flux))
    obc.set_initial_phase(2,random_phase(flux))

    couple_weight = 0.4
    obc.set_coupling(couple_weight,C1,C2)

    if EXERCISE != 1:
        couple_weight = 1.2
        obc.set_coupling(couple_weight,A,0)
        obc.set_coupling(couple_weight,A,1)


    
    times,phases,freqs= obc.run(runtime,npts=ptsPerCycle*ncycles)
    obc.render_timeseries(f"B{EXERCISE}-phase-{Afreq}.png",f"B{EXERCISE}-freq-{Afreq}.png",times,phases,freqs)
    obc.render_oscillators(f"B{EXERCISE}-obc-{Afreq}.gif",times,phases,freqs,fps=ptsPerCycle)




# simple_obc_phaseonly()
simple_obc_freqonly()
    

