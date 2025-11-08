# Assignment: Delay-Based Computing

In this assignment, we will implement numerous delay-based computing gates in `delay_gates.py`. To implement each gate, you need to fill in the `reset` and the `execute` functions:

 - The `reset` function should reset the state of the gate before execution. Recall delay-based computing gates are stateful and "remember" pulse information, and pulses are stored during a loading stage then released during execution. The reset function should define and reset whatever flags are necessary to correctly implement the operation of the gate.
 - The `execute` function should determine if the gate produces a pulse at a given timestep. The execute function takes as input the current time step (in nanoseconds), and a dictionary of input-value pairs. In this implementation, each port may have two values: PULSE (1) or NO_PULSE (0).

A `delay_circuit.py` Python file is provided to simulate a delay-based computing circuit and render the results, and a `delay_tester.py` Python file is provided to construct some simple single-gate circuits and simulate them for you. Each test function in the `delay_tester.py` produces a visualization of each simulation and a visualization of the circuit architecture. The measured time window is shaded.

## Part X: Understanding Delay-Based Encodings [8 points]

Execute the `delay_tester.py` file as is to investigate the behavior of delay-based input over time. The `test_input` function simulates the propagation of three constant delay-based values with labels ('X', 'Y', 'Z') through an empty circuit and outputs a pulse trace (`test_input.png`) and a summary of all the pulse arrivals/departures to console.

1. What does each numbered segment in the input image represent? What does it mean when a pulse (vertical line) is contained within a numbered segment?

The numbered segment represents discrete bins for each delay segment. When a pulse is contained within a numbered segment under number k, this means that the pulse exhibits k unit of delay in relation to the reference time. The segment number of each timestamp is calculated as

#seg = floor[(ts - settlement) / segment_window]

Where settlement stands for the offset of timestamp before the current operation (i.e. the time adjustment factor between absolute time and operation reference time). Segment window is the size of each segment.

2. What kind of value does an earlier pulse encode, what kind of value does a later pulse encode?

An earlier pulse encodes a smaller value, whereas a later pulse encodes a larger value. All delays encode non-negative integer values.

3. Run the simulation multiple times. Do you observe any variations in the plots from execution to execution?

Yes, the pulse positions fluctuate from trial to trial, and sometimes almost at the border between one delay value and another.

4. Look at the pulse summary printed to console. How is a gate's numerical values computed from the relative delay and segment time? How is the relative delay computed from the absolute delay and the gate's settling time? The settling time for a given port is the time required for the pulse to reach a particular port.

The gate's numerical value is computed via `val(g) = floor[relative delay / segment time]`

Meanwhile, the relative delay is computed as `relative-delay(g) = absolute delay - settling time`

# Part Y: Implementing the Delay Gates [8 pts]

Next, we will implement the simulations of first arrival, last arrival, delay, and inhibition gates associated with delay-based computing. Each delay gate is reset before execution, and then "executed" at each time step. The reset and execute functions may instantiate and change the gate's internal state, feel free to instantiate any internal variables needed.

1. Implement the reset and execute functions for the first arrival (`FirstArrival`) gate. Describe any internal state maintained by the gate, and what the reset operation and execution functions do for this gate. Uncomment the `test_first_arrival_gate` function in `delay_tester.py` to test the operation of the first arrival gate.

I added one internal state to the first arrival gate, which is `self.arrived` variable which tracks whether a signal has already arrived. This value is initialized to False. When the first signal arrives, the variable will be flipped and the gate will return PULSE. Any subsequent signals will not trigger the gate because now  `self.arrived` is True. During reset the variable is set back to False.

2. Implement the reset and execute functions for the last arrival (`LastArrival`) gate. Describe any internal state maintained by the gate, and what the reset operation and execution functions do for this gate. Uncomment the `test_last_arrival_gate` function in `delay_tester.py` to test the operation of the last arrival gate.

I added one internal state to track whether there is a previously arrived signal, called `self.first_arrived`. This variable is initially set to False, when the first signal passes through the gate it is flipped to True. Then when the second signal passes through the gate, since `self.first_arrived` is True, the gate will output PULSE. There is a minor edge case of when two signals arrive at the same timestamp. During this situation, the program will both flip `self.first_arrived` to True and return PULSE. In any other situations the gate will output NO_PULSE. During reset the variable is set back to False.

3. Implement the reset and execute functions for the delay (`Delay`) gate. Describe any internal state maintained by the gate, and what the reset operation and execution functions do for this gate. Uncomment the `test_delay_gate` function in `delay_tester.py` to test the operation of the delay gate. What does the circuit described in the `test_delay_gate` function do to the input value? 

I used one internal states to track the delayed time that the gate waits for, `self.await_ns`. The first tracks the time at which the signal arrives, the second tracks whether the delayed signal has already been delivered so the gate does not repeatedly send out signals. When a signal arrives, the gate sets `self.await_ns` to be the sum of current time tick and the expected delay. Then, at every other pass of execute function, it checks whether the time exceeds `self.await_ns`. At the first time tick exceeding the specified delay, the signal will be sent out and the `self.await_ns` will be immediately reset to None to indicate that the signal is already sent. During reset, `self.await_ns` is set to None to indicate there is no signal in yet.

4. Implement the reset and execute functions for the inhibition (`Inhibition`) gate. If two pulses arrive at the same time (which is unlikely in practice), you can assume that the gate emits a pulse with 50% probability. Describe any internal state maintained by the gate, and what the reset operation and execution functions do for this gate. Uncomment the `test_inh_gate` function in `delay_tester.py` to test the operation of the inhibition gate.

I used one internal state `self.a_arrived` to track whether signal at Input Gate A has arrived or not. At every input ts, I first checked whether `self.a_arrived` is still False. If A has arrived then nothing needs to be done and the system can simple output NO_PULSE. Otherwise, if Input Gate A signal has not arrived, in case Input Gate B arrives, the system will output a PULSE. In case Input Gate A arrives, `self.a_arrived` will be flipped to True. In the rare case where Input Gate A and Gate B arrives at the same timestamp, the system uses `random.random()` to generate a random float in range [0, 1) and output PULSE based on whether the value is above 0.5, thus creating a 1/2 probability proxy. During reset `self.a_arrived` is set back to False.

# Part Z: Using Delay-Based Computing [6 pts + 6 extra credit]

Implement a circuit that computes the numerical function.

    if max(X,Y) < 5:
        return max(X,Y)
    else:
        <no pulse>

You can use the `add_wire(src_gate,dst_gate,port)` function to add wires between gates, and the `add_gate(gate)` function to add a gate to the circuit. Feel free to use the `render_circuit("filename")` function to visualize the circuit that you built. As a reminder, in the simulation, the wire does not incur any delay but the gate introduces delay, and you need to make sure the inputs to the same gate have aligned delays. Reference the functions in `delay_tester.py` for examples of how to build and render a circuit. You do not need to worry about the `max(X,Y)=5` case. Test your circuit on a few values to verify that it computes the function correctly.

1. What is the structure of the circuit you implemented? Include a diagram of the circuit in your writeup. [4 pts]

The structure of the circuit I implemented is
```
-- Z [constant 5] --> Delay[0] --> 
                                   Inhibition --> 
-- X -->    
            LastArrival        -->
-- Y -->
```
where X and Y subsequently encode the two input values. I used a LastArrival gate to derive the expression max(X, Y). Then, for any max(X, Y) value that is larger than 5 (i.e. comes after 5), it will be inhibited by the inhibition gate. To guarantee that the constant 5 delay is aligned with the other inputs, I used a delay gate with delay time of 0. 

2. What if the `<` operation is replaced with `>`? What is the structure of the circuit you implemented? Include a diagram of the circuit in your writeup. [4 extra credit]

The structure of the circuit I implemented is
```            
-- X --> 
            LastArrival       --> 
-- Y -->
                                    Inhibition -->
-- Z [constant 5] -- Delay[0] -->
                                                    LastArrival -->
-- X2 [=X] --> 
                LastArrival -->     Delay[0]   -->
-- Y2 [=Y] -->
```
This essentially allows for the output of max(X, Y) conditioned on (5 if max(X, Y) > 5 else none), which gives max(X, Y) if the value is larger than 5.

During implementation I note that the simulation system seems to not allow one gate output to be connected to more than one gate inputs. This results in the (X2, Y2) input copy gates. Otherwise we can also directly use the [X, Y] -> LastArrival output for that part of the operation.

Researchers have recently explored using delay-based computing in the log domain. With this method, instead of computing directly over numeric values, a value `x` is encoded into `x'=-ln(x)` as a delay (you need not worry about encoding of negative values in this question), delay-based computing is used to implement the computation, and then you raise the delay-based value `x=e^{-x'}` to recover the original value. This method can encode a broader value range using the same delay range, and obtain results faster compared to naive encoding, especially when dealing with large values.

2. The log domain encoding also enables operations that are difficult to implement in the original value domain. Describe how you can implement multiplication and constant value scaling in log domain. [2 pts]

For constant value scaling, we have
```
cX = exp[-(-ln(c))]exp[-(-ln(X))]
   = exp[-dc]exp[-dX]
   = exp[-(dc + dX)]
```
Therefore we can encode constant value scaling by delaying the delay encoding of X (dX = -ln(X)) by the delay encoding of scaling constant c (dc = -ln(c))

This can similarly be applied to any multiplication, e.g.
```
XY = exp[-(dX + dY)]
```
Therefore, we would need to delay the delay encoding of X by the delay encoding of Y (or vice versa). Nevertheless, this will require programmable, input-dependent delay gates.

3. One downside of computing in log domain though is that addition and substraction become more difficult. However, there are effective approaches to approximate them. For example, substraction of two values `z=x-y` is `z'=-ln(e^{-x'}-e^{-y'})` in the log domain, but it can be approximated using delay logic as `z'=min(inhibit(x'+E0,y'+F0), ...,inhibit(x'+En,y'+Fn))`, where `Ei`'s and `Fi`'s are specially picked constants. Refer to `nLDE_approximation.png` in the folder for a visualization where `x'+y'=0`. Could you devise a similar approximation approach for addition? You may also focus on the case where `x'+y'=0`. Write the expression you use for the approximation, using only `min`, `max`, `inhibit` operations. Include a figure to show how good the approximation is in your writeup. [2 extra credit]

For the case of x' + y' = 0, we can further derive the equation as
```
z = x' + y'
  = -ln(e^(-x') + e^(-y'))
  = nLDE(x', y')
```
We know that the value has an upper bound where
```
nLDE(x', y') < min(x', y')
```
we can then create approximations such that
```
nLDE(x', y') approx min(x', y', max(x'-C, y'-D))
```
by iteratively making this approximation we ends at
```
nLDE(x', y') approx min(x', y', max(x'-C0, y'-D0), max(x'-C1, y'-D1), ... max(x'-Cn-1, y'-Dn-1))
```
where the tuples (C0, D0), (C1, D1), ... (Cn-1, Dn-1) are approximation terms that we derive through an optimization process. To reduce the introduction of negative values to non-negative delays, we can take 
c' = max(|C0|, |D0|, |C1|, |D1|, ...., |Cn-1|, |Dn-1|)
and instead perform
```
nLDE(x', y') + c' \approx min(x' + c', y' + c', max(x'+(c'-C0), y'+(c'-D0)), max(x'+(c'-C1), y'+(c'-D1)), ... max(x'+(c'-Cn-1), y'+(c'-Dn-1)))
```
I generated a figure using script `delay_log_addition.py`. The figure is attached as `nLDE_approx.png`.

# Part W: Digital Logic with Delay-Based Computing [14 pts]

Move to the `delay_digital.py` file. The file uses race-based computing to execute `(X | not Y) and (Z | X) and (Z | Y)`. I have already implemented the scaffold for this part of the project which builds the logic circuit, and convenience functions for performing the dual-rail encoding. The delay-signal visualization in this part may be too dense to read due to many number of gates, but you may read the text output for debugging.

1. Implement the `and` gate using delay logic. How did you implement this gate?

I implemented the `and` gate by having

positive rail: a LastArrival Gate that inputs both A and B, outputting the result R. This means that R only has a PULSE if both A and B have PULSE (i.e. A = B = True)

nagative rail: a FirstArrival Gate that inputs both not A and not B, outputting result not R. This means that not R only has no PULSE if neither not A nor not B has a PULSE (i.e. not A = not B = False)

I then return the representation (R, not R).

2. Implement the `or` gate using delay logic. How did you implement this gate?

I implemented the `or` gate by having

positive rail: FirstArrival Gate that inputs both A and B, outputting the result R. This means that R has a PULSE if either A or B has a PULSE. (i.e. A = True or B = True)

negative rail: a LastArrival Gate that inputs both not A and not B, outputting result not R. This means that not R has no PULSE if either not A or not B does not have a PULSE (i.e. not A = False or not B = False)

I then return the representation (R, not R).

3. Implement the `not` gate using delay logic. How did you implement this gate?

I implemented the `not` gate by simply shuffling the input rails. Specifically I take A as not R, and not A as R, then return the representation (R, not R).

4. Implement the digital readout gate in `delay_gates.py` (see class `DigitalReadOutGate`). This gate should remember if it saw a pulse or not.

The internal state I need to keep track of in the digital readout gate is the `self.has_pulse` arribute. As soon as the input has a pulse at any time, I flip the `self.has_pulse` to True. During reset and initialization, I set it to False.

5. Execute the completed implementation on a few digital values. What delay-based encoding does a digital "1" output correspond to? What delay-based encoding does the digital "0" output correspond to? What states are invalid in the current digital value encoding?

A signal "1" output corresponds to a delay-based encoding with PULSE on the positive rail and NO PULSE on the negative rail. A signal "0" output corresponds to a delay-based encoding with NO PULSE on the positive rail and PULSE on the negative rail. 

If both positive rail and negative rail have the same state (i.e. (PULSE, PULSE) or (NO PULSE, NO PULSE)), this is invalid in the current digital value encoding.
 
6. Currently, the implementation creates a fresh input signal for every usage of every variable instead of re-using the input signal. Why is this necessary? What would happen if you build the same circuit, but then reuse the same input for multiple gates?

Depending on whether this situation describes the simulation or practical level, I can think of two reasons.

On the simulation level, the current architecture does not support fan-out circuit design. Therefore, if any input signal gets re-used, instead of transmitting signals to both gates it will instead overwrite the second connection with the first one, causing the circuit to be incomplete.

On the practical level, if we build the same circuit, but resuse the same input for multiple gates, the signal strength will fade with the fan-out, and may become undetectable by the output gate. Creating a fresh input signal for every usage of every variable effectively eliminates the complications on quantitative aspect of signals.

7. What would happen if you don't reset the circuit between executions? Why is a circuit reset operation necessary for delay-based digital logic, but not necessary for conventional digital logic?

If the circuit is not reset, then some of the internal states retained in the gate may be stale, leading to operation errors. For example:

    i. if FirstArrival gate has previously fires a signal, in the next round it will have `self.arrived` still set to True and no signal will be outputted

    ii. if Inhibition gate receives signal from Input Gate A anytime in the previous round, it will keep the flag `self.a_arrived` as True, so that no signal from Input Gate B this round can pass the gate

    iii. if LastArrival gate receives signals previously, then `self.first_arrived` will become True. This will be inherited to the next round so the gate will fire whichever Input Gate gives the first signal, instead of outputting the second signal as expected

The reason why delay-based digital logic requires explicit reset is that different from conventional digital logic, which is primiarily stateless and only needs to consider the current input, delay-based logic is stateful and requires keeping internal states on the history of signals (e.g. whether a signal has arrived for LastArrival, FirstArrival and Inhibition, the time of signal for DelayGate). This means that between different operations, the circuits need to re-initialize all the internal states so that those from a previous operation does not corrupt the new operation.