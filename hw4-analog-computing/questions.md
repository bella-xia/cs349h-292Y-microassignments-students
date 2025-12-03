### Part A: Image Processing with Cellular Non-linear Networks [1 pt each, Q6 is 2 pts]

The `cnn.py` builds a cellular non-linear network for image processing problems, where an NxM cell cellular non-linear network
proesses an NxM image. The greyscale image is set with the `U` matrix, where 

You will be editing the `run_cnn()` function, which sets up the cellular non-linear network to process a random image from the 
MNIST dataset. The image is saved to `cnn_input.png`, and an animation of the CNN cell evolution over time is saved to `cnn.gif`.

Q1. Try the CNN with the A1, B1, and z1 parameters on the input image. What does the CNN do with the current parameter configuration? Does the result appear instantaneously?

The CNN provides a edge detection of the input image with the current parameter configuration. The result does not appear instantaneously, instead the edges gradually become brighter whereas the other pixels fade over the span of 10 seconds in the gif visualization.
 
----

Q2. Currently the cells initial values are all set to zero. How does the behavior change if they are randomly instantiated to a value between 0-0.1?

The background pixel intensities become less uniform. As the edges become brighter, it takes a much longer time for the background pixels to converge to black due to the non-uniformity.

----

Q3. Next, enable `EXERCISE=2` this introduces 20% relative mismatch into the parameters. What happens when all three are perturbed -- do you still get the same result? Which of these parameters has the most significant impact when perturbed?

When all three are perturbed, regions of the background becomes not dark enough to fully converge to black. Instead, they gradually lighten across iterations into white, disrupting the edge detection result. Among all the paramters, B most consistently have more significant impact when perturbed. Most of the times, with perturbation of B matrix alone the final result can be different.

---

Q4. Now, enable `EXERCISE=3` which uses a different set of parameters A2, B2, and z2. What does this parameterization of the CNN do?

This parameterization of CNN performs diagonal detection.

---

Q5. Now enable `EXERCISE=4` which introduces 20% relative mismatch to the parameters. What happens when all three are perturbed? How does this compare to 
when the template parameters A1, B1, z1 were mismatched.

It takes longer for the pixels to converge, but compared to the highly perturbed result when adding mismatch to A1, B1, z1, this does not significantly affect the final diagonal detection result.

---

Q6. Currently the CNN uses the clip function for saturation. What happens if your nonlinearity is imperfect, and actually implements a sigmoid? What happens? Do this by changing the `clip` function invocations to sigmoid function invocations and then rerunning some experiments. Make sure the sigmoid is properly centered so it performs the same operation as the clip function, but with smoother edges. 

It makes pixel values slower to converge to 1 and 0. This is because the saturation sigmoid exhibits on both edges.

---
Q7. Multiply the right-hand side of the differential equation by 10. This can be done by modifying the diffeqs function. What happens to the system? 

The speed of convergence becomes 10 times faster.

### Part B: Phase-Domain Oscillator-Based Computing [1pt/question]


We will next play with oscillator-based computing. I have written the code necessary to simulate the oscillator-based computing paradigm. The simulator
will generate the time-series plots of the phase over time, the frequency over time, and an animation of the oscillators in the `images-obc` directory.

We will first experiment with oscillators that have the same natural frequency, but start off with random phases. The experimental setup is in the 
`simple_osc_phaseonly` function. The natural frequency of the oscillators is 1 khz or 1000 hz, and the phase is randomly instantiated to be between 0 and 2\pi. You can change the exercise you're doing by setting the `EXERCISE` variable.

Q1. Run the function unmodified -- this will simulate a set of free-running oscillators. How do the phase and the frequency change over time? Do the oscillators synchronize? Does the phase ever stop changing? How do frequency and phase relate?

The frequency of the oscillators remains constant, whereas the phase increases from 0 to 2pi repeatedly with the same speed. The oscillators do not snychronize and the phase never stop changing. The frequency and the phase change is proportional as frequency denotes the speed of phase movement. 

---

Q1b. How do frequency and phase relate? How does frequency relate to the slope of the phase trajectories?

Frequency denotes the rate of phase change, and consequently is the slope of the phase trajectories. In the graph `A1-freq.png` and `A1-phase.png`, we can observe that all frequencies remain constant at 3, and consequently it takes approximately 2 unit times for phase to pass from 0 to 2pi (6.28 approx 3 * 2.09).

-----

Q2. Enable `EXERCISE=2`, which couples all the oscillators together. What happens to the frequency as the system evolves? What happens to the phase as the system evolves? How can you tell when the frequency is synchronized (look at `freq.png`), how can you tell when phase is synchronized (look at `phase.png`).

The system starts off with the frequency of the oscillator mostly dispersed. As time progresses the frequency starts to converge to around 3 then periodically oscillate around the value. Meanwhile, the phase shifts starts off to be distinct across all oscillators, but gradually converge to the same rate by the 4th unit time and continue to stay approximately the same for all later iterations. According to the graphs, frequency is synchronized when all oscillators share the same frequency at any time t with only minor drifts or oscillations, whereas phase is synhronized when all phase shifts collapse to map to the same movement in any time range [t, t + delta].  

----

Q3. Enable `EXERCISE=3`, which couples all the oscillators together, but makes the coupling strength between oscillators 1/2 very weak. How does the phase evolution of this system compare to the phase evolution from the previous configuration? Do the oscillators all synchronize by the end of the simulation? 

Since the coupling strength is weak, the phase convergence becomes much slower, taking around 2 additional unit times. Even after converging, the phase change rate is not as stable as the previous exercise, as the oscillators exhibit a notable out-of-sync segment around the 9th unit time. By the end of simulation, the oscillators are not fully synchronized, as they are still subject to periodic instability of large, unsynchronized phase changes.

----

Q4. Enable `EXERCISE=4`, which couples oscillators 0-1, and 2-3 together and negatively couples oscillators 1-2 together. How does the phase evolve over time in this problem setting? Which oscillators synchronize? Which oscillators do not?

Oscillator 0 and 1 synchronize in phase, so do 2 and 3. There is a constant relative phase difference of PI between 0, 1 oscillator pair and 2, 3 oscillator pair, suggesting that the two pairs are phase locked with a constant difference and thus are not synchronized.

### Part C: Frequency-Domain Oscillator-Based Computing with Non-idealities [1pt/question]

Next, we implement a modified version of the frequency-domain oscillator-based computing network from the "A Nanotechnology-Ready Computing Scheme based on a Weakly Coupled Oscillator Network" paper. This network has two "core" oscillators that operate at 5.6 and 5.8 Hz and one "input" oscillator A that can be set to anywhere between 5-6.8 Hz. The coupling strength between core oscillators is 0.4 and the coupling strength between the core and input oscillator is 1.2. The core oscillators will tend to synchronize when the input oscillator is in a certain frequency range. 

Q1. Enable `EXERCISE=1` -- this disconnects the input oscillator "A" from the network. What happens to the frequency of the core oscillators when this happens?

Since the two "core" oscillators starts off with relatively close frequencies, their frequencies gradually synchronize to the same value with the 0.4 coupling strength.

---

Q2. In Exercise 1, do the oscillators that synchronize in frequency also take on the same phase?

The oscillators that synchronize in frequency (i.e. the two core oscillators) also take on the same phase.

---

Q3. Enable `EXERCISE=2` -- this sets the frequency of oscillator "A" to a medium frequency (5.8) and reconnects input oscillator "A" to the system. What happens to the oscillator network? Which oscillators synchronize in frequency? Which oscillators synchronize in phase? 

All three oscillators appear to be relatively synchronized in phase and frequency. Nevertheless, there are still small drifts in oscillator frquency. Specifically, the two core oscillators tend to drift in the opposite direction as the input oscillator, which may suggest that the input oscillator is not fully synchronized in frequency.

----

Q4. Enable `EXERCISE=3` -- this sets the frequency of oscillator "A" to a low frequency (1.0) and reconnects oscillator "A". What happens to the oscillator network? Which oscillators synchronize in frequency? Which oscillators synchronize in phase?

The two core oscillators synchronize in both frequency and phase. The input oscillator starts off to seem that it is progressively synchronize in frequency, but then exhibits patterns of 

    1. periodic large perturbations, causing the core oscillators to have peak frequency and the input oscillator to have trough frequency

    2. intervals of stable constant frequency lock with a non-zero difference

This is also visible in the phase graph. the large perturbations tend to manifest as curvatures in input oscillator's phase pattern as it experience frquency changes. Since even when the input and core oscillators are frequency locked, they do not synchronize in frequency, they are never phase locked.  

----

Q5. Enable `EXERCISE=4` -- this sets the frequency of oscillator "A" to a high frequency (11.0). Again what happens?

The two core oscillators still synchronize in both frequency and phase. However, compared to previous trials, the frequency of the input oscillator is more unstable, as it repeatedly fluctuate between 7.5 and 13, with the core oscillators fluctuating in the opposite direction between 3 and 7.5. The core oscillator and the input oscillator do not become phase locked nor frequency locked. 

----

Q6. Enable `EXERCISE=5` - -this sets the frequency of oscillator "A" to a low frequency that is closer to 5.6-5.8. What is the synchronization behavior you observe?

The three oscillators are mostly synchronized in frequency, except for small drifts that the input oscillator exhibits periodically. The two core oscillators are synchronized in phase, whereas the input oscillator is phase locked with the core oscillators with a constant phase difference. 

### Part D: Grade Optimization with Integer Linear Programming [6 pts]

Next we will use integer linear programming to optimize your grace days to maximize your grade. Look at `grade_optimizer.py`, which has the scaffold and an example grade report. The scaffold code has comments indicating where you should add constraints and modify expressions.

Q1. What are the variables in this constraint problem? Are they integer or real variables? [1 pt]

variable per assignment:

    1. asgGraceDays (integer): the number of grade days assigned for the assignment

    2. assignmentGrade (real): the grade value of the assignment (produced by the overall grade deducting late penalty)

global variable:

    1. totalGraceDays (integer): the total number of grade days taken 

    2. totalGrade (real): the overall grade value scaled by each assignment's respective grade and proportion 

---

Q2. Fill in the constraints / update the necessary expressions and run the solver. Don't worry about handling the fact that we can't apply grace days to the final proposal deadline. What's the optimal assignment of grace days for the example grade report? What grade is achieved? [5 pts]

The optimal assignment is to assign 1 grade day to final and 2 grade days to survey. The grade achieved is 89.3 percent.
