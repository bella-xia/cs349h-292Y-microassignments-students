#### Part A: Stochastic Computing [2 pt/question, 8 points]

The provided scaffold code runs 10,000 trials of the `1/2*(0.8*0.4 + 0.6)` stochastic computation and plots the distribution of results. The `PART_A_example_computation` function executes the above numeric expression, and the `run_stochastic_computation` runs a stochastic computation for <ntrials> trials and plots the distribution + reports the mean and standard deviation of the stochastic executions. The vertical red line shows the expected result. Currently, the stochastic computing operations and value-to-bitstream / bitstream-to-value conversion routines are not implemented. 

Implement the stochastic computing paradigm by filling in the function stubs in the PosStochComputing class. Don't worry about supporting negative numbers just yet.

- `to_stoch` : convert a value in [0,1] to a stochastic bitstream.
- `from_stoch`: convert a stochastic bitstream to a numerical value
- `stoch_add`: perform scaled addition over two stochastic bitstreams.
- `stoch_mul`: perform multiplication over two stochastic bitstreams.

Execute the stochastic computation for bitstream lengths 10, 100, and 1000 with `stochastic.py`, and answer the following questions.

Q1. How does the mean change with increasing bitstream length? How does the variance change?
```bash
--- terminal output ---
bitstream length: 10, total trial: 1000
ref=0.460000
mean=0.459890
std=0.157925
bitstream length: 100, total trial: 1000
ref=0.460000
mean=0.459904
std=0.049670
bitstream length: 1000, total trial: 1000
ref=0.460000
mean=0.460002
std=0.015585
```
With increasing bitstream length, the mean becomes closer to the precise value (-1.1e-4 -> -9.6e-5 -> +2e-6).
Meanwhile, the variance becomes smaller.

Q2. What is the smallest representable numeric value in a single 1000-bit stochastic bitstream? What happens when you try to generate a bitstream for this value -- do the bitstream values converge to the desired value?

***I am a little confused about what is the "coverge" refering to specifically? something needs to increse for convergence?***
```bash
--- terminal output --- 
bitstream length: 10, total trial: 10000
ref=0.100000
mean=0.100440
std=0.094931
bitstream length: 100, total trial: 10000
ref=0.010000
mean=0.009941
std=0.009867
bitstream length: 200, total trial: 10000
ref=0.005000
mean=0.005067
std=0.005043
bitstream length: 500, total trial: 10000
ref=0.002000
mean=0.001967
std=0.001973
bitstream length: 1000, total trial: 10000
ref=0.001000
mean=0.001003
std=0.001002
```
The smallest non-zero representable numeric value in a single 1000-bit stochastic bitstream is 1 / 1000. I am supposing this is what is suggested by the question cause the value 0 does not have a convergence problem, since a Bernoulli distribution with p(x=1) = 0 would always be accurately representing 0. 

When attempting to represent the value 1 / 1000 over increasing bitstream length from 10 to 1000, the mean value across 10000 generations does converge to the desired value. For a 1000-bit stream, the mean was 0.001003, deviating by only +3e-6. Additionally, the variance decreases approximately inversely to the bitstream length, thus becoming smaller as the bitstream length increases.

Q3. What stochastic bitstream length L do we need to represent a rational number V accurately with a single bitstream, assuming V is in (0,1]? Write the equation. How does this L scale with V? Denote L' as the number bits needed to represent V accurately as an integer (removing the leading "0."). How does L' scale with V? 

As the convergence in Q2 suggests, to guarantee accurately representing a rational number V, V needs to be equal or greater than 1 / bitstream length (i.e. it needs to take on at least the weight of a single bit in the bitstream). This means that for any rational number V, we would need

L = ceil(1 / V)

This equation can also be rewritten as finding the smallest positive integer L such that it satisfies

L * V >= 1

This means that L will scale inversely with V, or scale proportionally with 1 / V.

Meanwhile, suppose we have V' = f(V) where f denotes the operation of removing the leading "0." and representing V as an integer. For integer-level presentation in traditional computing model, we use binary stream with increasing weights moving up the digits, where for any 1 in bit i (counting from right to left):

[0, 0, 0, 1, 0, 1]
         i=2

The encoded value is 2^i. Therefore, to represent V' acccurately, we need to find the bit with the largest value encoded. We can derive this using

L' = floor(log_2 V') + 1 (the 0th bit)

or, this equation can also be rewritten as finding the smallest positive integer L' such that it satisfies

V' < 2^L'

In this computing model, L' will scale logarithmically with the value of V', which means it will also scale logarithmically with V provided that all V has the same floating-point precision.

Q4. Design a stochastic computation for bitstreams with a length n=1000, such that the end result is not accurately representable by a single bitstream. You must accomplish this with stochastic operations, all initial values must be >= 0.1 and every constant must be a uniquely generated bitstream.

To do so, we need to expected value at the end or at any intermediate point of operation to be smaller than the minimum representable value with 1000-bit bitstream. We can do that using the operation sequence:

0.1 * 0.1 * 0.1 * 0.1 = 0.0001 < 0.001

This means that no single bitstream is able to accurately represent the product. If we then trying to run this operation, we get statistics below:
```bash
--- terminal output --- 
bitstream length: 1000, total trial: 10000
ref=0.000100
mean=0.000100
std=0.000319
```
Even though we see that mean itself is still converging to 0.0001, the standard deviation is 3roughly 3 time the mean, signaling that the the operation results are highly inprecise. From the distribution bar chart, we would observe that a significant number of trials has 0 as the product whereas many remaining produces 0.001. This situation results in an approximate mean but extremely high variance.

#### Part X: Non-Idealities Stochastic Computing [2 pt/question, 6 points]

Next, we'll experiment with introducing non-idealities into the stochastic computation `1/2*(0.8*0.4 + 0.6)`. We will introduce two kinds of non-idealities:

- bit-shift errors: this class of errors result from timing issues in circuits. This non-ideality causes one bitstream to lag behind the other during computation. For example, a bit shift error at index 1 of a bit stream would transform some stream 00101 to 00010 (the 0 at index 1 is replicated). We do not consider the case where more than one bit shift occurs at the same index, as it is unlikely to happen in practice.

- bit-flip errors: this class of errors result from bit flips in storage elements, or in the computational circuit. This non-ideality introduces bit flips in bitstreams. For example, a bit flip error at index 1 of a bit stream would transform some stream 00101 to 01101.

Fill in the `apply_bitshift` and `apply_bitflip` functions in the stochastic computing class and apply these non-idealities at the appropriate points in the stochastic computing model. Make sure these non-idealities are only enabled for this section of the homework.

Q1. What happens to the computational results when you introduce a per-bit bit-flip error probability of 0.0001? What happens when the per-bit bit flip error probability is 0.1?
```bash
--- terminal output --- 
----- disable per-bit bit-flip error -----
ref=0.460000
mean=0.460065
std=0.015921
----- per-bit bit-flip error probability = 0.0001 -----
ref=0.460000
mean=0.459944
std=0.015625
----- per-bit bit-flip error probability = 0.1 -----
ref=0.460000
mean=0.486118
std=0.015812
```
When introducing an error probability of 0.0001, the result does not exhibit significant deviation from if no bit-flip error exists. However, when the error probability is 0.1, the empirical mean begins to deviate significantly from the expected value whereas the variance stays constant, suggesting higher inaccuracy.

Q2. What happens to the computational results when you introduce a per-bit bit-shift error probability of 0.0001? What happens when the per-bit bit shift error probability is 0.1?
```bash
--- terminal output --- 
----- disable per-bit bit-shift error -----
ref=0.460000
mean=0.459792
std=0.015904
----- per-bit bit-shift error probability = 0.0001 -----
ref=0.460000
mean=0.459920
std=0.015514
----- per-bit bit-shift error probability = 0.1 -----
ref=0.460000
mean=0.459899
std=0.017028
```
Both when introducing an error probability of 0.0001 and 0.1, the epirical mean remains relatively identical to when no bit-shift error occurs. However, when bit bit-shift error probability is 0.1, the variance of the computation result increases, suggesting higher imprecision.

Q3. In summary, is the computation affected by these non-idealities? Do you see any changes in behavior as the bitstream length grows?
```bash
--- terminal output --- 
----- no error contrast trial -----
bitstream length: 10, total trial: 10000
ref=0.460000
mean=0.457700
std=0.157685
bitstream length: 100, total trial: 10000
ref=0.460000
mean=0.459766
std=0.049684
bitstream length: 1000, total trial: 10000
ref=0.460000
mean=0.459852
std=0.015695
----- per-bit bit-flip error probability = 0.1 -----
bitstream length: 10, total trial: 10000
ref=0.460000
mean=0.485550
std=0.159750
bitstream length: 100, total trial: 10000
ref=0.460000
mean=0.485682
std=0.050775
bitstream length: 1000, total trial: 10000
ref=0.460000
mean=0.485800
std=0.015780
----- per-bit bit-shift error probability = 0.1 -----
bitstream length: 10, total trial: 10000
ref=0.460000
mean=0.457520
std=0.169698
bitstream length: 100, total trial: 10000
ref=0.460000
mean=0.459418
std=0.053126
bitstream length: 1000, total trial: 10000
ref=0.460000
mean=0.459657
std=0.017125
```
Yes, computation is generally affected by the unidealities, provided that the occurance probability is high (e.g. a 0.1 erro probability). bitflip errors tends to make computation results more inaccurate, whereas bitshift errors have less impact on accuracy but tends to make result more inprecise. As the bitstream length increases (10 -> 100 -> 1000), the standard gradually decreases for all cases. Meanwhile, the pattern of accuracy drift in bitflip and precision drift in bitshift remain consistent, suggesting that the growth of length does not significantly affect unideality-induced deviations.

#### Part Y: Statically Analyzing Stochastic Computations [2 pt/question, 6 points]

Next, we'll build a simple static analysis for stochastic computations. A _static analysis_ is a type of analysis that is able to infer information about a program without ever running the computation. The analysis we will be building determines the minimum bitstream size necessary for a computation, given a set of precisions for each of the arguments. We define a bitstream size V to be sufficient if for any possible values t generated during the computation, V*t>=1. For example, to compute the bitstream length for the following expression:

    (x + y) + z

We will set up the static analyzer as follows:

    `analysis = StochasticComputingStaticAnalysis()`
    `analysis.stoch_add(analysis.stoch_add(prec_x, prec_y), prec_z)`
    `N = analysis.get_size()`

where `prec_x`, `prec_y`, and `prec_z` are the precisions of x, y, and z respectively. Precision of a variable is defined as the smallest value of the variable. If the precision of x is 0.01, the precision of y is 0.02 and the precision of z is 0.03, then the minimum bitstream length is 100. In this exercise, you will be populating the `StochasticComputingStaticAnalysis` class, which offers the following functions:

    - `stoch_var`, given a variable with a desired precision `prec`, update the static analyzer to incorporate this information.
    - `stoch_add`, given two stochastic bitstreams that can represent values with precision `prec1` and `prec2` respectively, figure out the precision required for the result stochastic bitstream given an addition operation is performed. Update the static analyzer to incorporate any new information.
    - `stoch_mul`, given two stochastic bitstreams that can represent values with precision `prec1` and `prec2` respectivesly, figure out the precision required for the result stochastic bitstream given a multiplication operation is performed. Update the static analyzer to incorporate any new information.
    - `get_size`, given all of the operations and variables analyzed so far, return the smallest possible bitstream size that accurately executes all operations, and can accurately represent all values.

We will use this static analysis to figure out what stochastic bistream length to use for the computation 1/2*(w*x + b), where the precision of w is 0.01, the precision of x is 0.1, and the precision of b is 0.1. For convenience, the scaffold file provides helper functions `PART_Y_analyze_wxb_function` for analyzing the `1/2*(w*x+b)` function, given a dictionary of precisions for variables `w`, `x`, and `b`, a `PART_Y_execute_wxb_function` which executes the `1/2*(w*x+b)` function using stochastic computing given a dictionary of variable values for `w`, `x`, and `b`, and a `PART_Y_test_analysis` function which uses the static analysis to find the best bitstream size for the `1/2*(w*x+b)` expresison, and then uses the size returned by the static analyzer to execute the `1/2*(w*x+b)` for ten random variable values that have the promised precisions.

Q1. Describe how your precision analysis works. Specifically, how do you propagate the precisions through the entire computation? How do you determine the final size?

The final size is determined by a variable called`min_prec`, which stores the minimum value ever stored or computed by the system. The default value for `min_prec` is 1, which is the upper bound of the any value within the unipolar stochastic range [0, 1] and only require one bit to represent. Everytime `stoch_var` is evoked, it compares the argument variable to the current `min_prec` and update `min_prec` if needed. Then it returns the argument variable to proceed with further calculations.

For `stoch_add` and `stoch_mul`, the function assumes that the values passed-in are already recorded by `stoch_var` (since they are passed-in as prec values). So the operation will only update `stoch_var` on the computation result, namely 1/2(prec1 + prec2) for `stoch_add` and prec1 * prec2 for `stoch_mul`.

Finally, when `get_size` is evoked, the system passes its `min_prec` value into `req_length` function to obtain the minimum bitstream length required.

Q2. What bitstream length did your analysis return?

My analysis returns 1000 as the bitstream length.

Q3. How did the random executions perform when parametrized with the analyzer-selected bitstream length?
```bash
--- terminal output ---
best size: 1000
{'x': 0.3, 'w': 0.09, 'b': 0.7}
ref=0.363500
mean=0.363586
std=0.015194

{'x': 0.7, 'w': 0.3, 'b': 0.9}
ref=0.555000
mean=0.554971
std=0.015845

{'x': 1.0, 'w': 0.96, 'b': 0.8}
ref=0.880000
mean=0.880052
std=0.010258

{'x': 0.7, 'w': 0.52, 'b': 0.2}
ref=0.282000
mean=0.281973
std=0.014400

{'x': 0.8, 'w': 0.39, 'b': 0.5}
ref=0.406000
mean=0.405799
std=0.015538

{'x': 0.4, 'w': 0.24, 'b': 0.2}
ref=0.148000
mean=0.147999
std=0.011166

{'x': 0.2, 'w': 0.67, 'b': 0.1}
ref=0.117000
mean=0.117044
std=0.010198

{'x': 0.1, 'w': 0.72, 'b': 0.1}
ref=0.086000
mean=0.086014
std=0.008838

{'x': 0.0, 'w': 0.58, 'b': 0.9}
ref=0.450000
mean=0.450117
std=0.015585

{'x': 0.5, 'w': 0.81, 'b': 0.5}
ref=0.452500
mean=0.452559
std=0.015737
```
The random executions, when parametrized with the analyzer-selected bitstream length, appear to be both stable and accuracy. Across the trials, the standard deviation is consistantly below 0.016, with some trials having as low as 1/2 or 2/3 of standard deviation upper bound. Manwhile, the empirical mean deviate in units of 1e-4 to 1e-6. Considering that the suggested bitstream length is 1000, providing an intrinsic precision approximately 1e-3, this means that the mean is sufficiently accurate.
 
#### Part Z: Sources of Error in Stochastic Computing [2 pt/question, 4 points]

Next, we will investigate the `PART_Z_execute_rng_efficient_computation` stochastic computaton. This computation implements `1/2*(x*x+x)`, and implements an optimization (`save_rngs=True`) that reuses the bitstream for x to reduce the number of random number generators.

Q1. Does the accuracy of the computation change when the `save_rngs` optimization is enabled? Why or why not?
```bash
---- part z: one-rng optimization ---
x = 1.0
running with save_rngs disabled
ref=1.000000
mean=1.000000
std=0.000000
running with save_rngs enabled
ref=1.000000
mean=1.000000
std=0.000000
x = 0.8
running with save_rngs disabled
ref=0.720000
mean=0.719963
std=0.014169
running with save_rngs enabled
ref=0.720000
mean=0.799860
std=0.012730
x = 0.6
running with save_rngs disabled
ref=0.480000
mean=0.479904
std=0.015792
running with save_rngs enabled
ref=0.480000
mean=0.600118
std=0.015543
x = 0.0
running with save_rngs disabled
ref=0.000000
mean=0.000000
std=0.000000
running with save_rngs enabled
ref=0.000000
mean=0.000000
std=0.000000
x = 0.9
running with save_rngs disabled
ref=0.855000
mean=0.855051
std=0.011134
running with save_rngs enabled
ref=0.855000
mean=0.899976
std=0.009417
```
Yes, then Does `save_rngs` optimization is enabled, except for cases where x = 0 or 1 which results in deterministic rngs, all other trials exhibit significantly higher empirical mean. This is because by using the same bitstream across the three instances of x, the system introduces unexpected correlations and the probability model is disrupted.

We start by navigating the common scenario, that is, if the three rng are all indepdent bitstreams. we will have 

x1, x2, x3 ~ Binomial(n, x) / n

This means

for the multiplication operation, we have

E(x1 AND x2) = E(x1 * x2) = E(x1)E(x2) = x*x

for the add operation, with another y = Binomial(n, 0.5), we have:

E(MUX y, x1x2, x3)= E[y(x1 * x2) +  (1-y)x3] = 1/2[E(x1)E(x2) + E(x3)] = 1/2(x*x+x)

However, in the case where

x' ~ Binomial(n, p) / n is reused three times, we instead have

for the multiplication operation

E(x' AND x') = E(x') = x

and for the add operation

E(MUX y, x', x') = E(x') = x

Thus we have instead computed result with an expected value of x. This is consistent with the empirical mean as we have 0.600118 approx. 0.6, 0.799860 approx. 0.8, and 0.899976 approx. 0.9.

Q2. Devise an alternate method for implementing $x*x+x$ from a single stochastic bitstream. There is a way to do this with a single (N+k)-bit bitstream, where k is a small constant value.
```bash
--- terminal output --- 
x = 0.3
running with save_rngs disabled
ref=0.195000
mean=0.194941
std=0.012634
running with save_rngs enabled
ref=0.195000
mean=0.195004
std=0.015164
x = 0.4
running with save_rngs disabled
ref=0.280000
mean=0.280151
std=0.014171
running with save_rngs enabled
ref=0.280000
mean=0.279900
std=0.017785
x = 0.3
running with save_rngs disabled
ref=0.195000
mean=0.195035
std=0.012580
running with save_rngs enabled
ref=0.195000
mean=0.194955
std=0.015243
x = 0.3
running with save_rngs disabled
ref=0.195000
mean=0.194867
std=0.012367
running with save_rngs enabled
ref=0.195000
mean=0.194912
std=0.015147
x = 0.8
running with save_rngs disabled
ref=0.720000
mean=0.720069
std=0.014278
running with save_rngs enabled
ref=0.720000
mean=0.720335
std=0.019506
```
Since all operations between two bitstreams theoretically happen on single-bit level, this means as long as the current two bits involved in the operation (either AND, MUX, or any other) are not strongly correlated, this can significantly reduce the deviation from the probability model that informs the computing model. By allocating a k-bit longer bitstream, and representing x, x2, and x3 as N-bit overlapping sliding windows on it, we can effectively reduce the correlation between any bit-wise operation. To minimize the space allocated, we can always set k = number of occurences of the shared variable - 1. So that for the ith occurance of the variable x (assume i starts with 0), we will have x_i = ARR[i:N+i].

According to the result, this effectively makes the empirical mean very close to both the expected value and the uncorrelated computation results. Nevertheless, the shifting method does seem to lead to higher variance, suggesting higher imprecision in the computation. The increase in variance appears negligible for the current calculation, but may introduce further biases or imprecisions in more complex logic.
 
#### Part W: Extend the Stochastic Computing Paradigm [15 points]

Come up with your own extension, application, or analysis tool for the stochastic computing paradigm. This is your chance to be creative. Describe what task you chose, how it was implemented, and describe any interesting results that were observed. There is no need to pursue great results, e.g., beating some state-of-the-art approaches. It is fine to obtain minimum results just to show that your idea works. Here are some ideas to get you started:

- Implement a variant of stochastic computing, such as deterministic stochastic computing or bipolar stochastic computing. You may also modify the existing stochastic computing paradigm to incorporate a new source of hardware error -- you will need to justify your hardware error model. 

- Build an stochastic computing analysis of your choosing. You may build up the existing bitstream size analysis to work with abstract syntax trees, or you may devise a new analysis that studies some other property of the computation, such as error propagation or correlation.

- Implement an application using the stochastic computing paradigm. Examples from literature include image processing, ML inference, and LDPC decoding.