import numpy as np
import matplotlib.pyplot as plt

'''
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
'''

if __name__ == '__main__':
    xs = np.arange(-5, 5)
    ys = xs * -1
    true_pos = -np.log(np.exp(-xs) + np.exp(-ys))
    approx_one = np.array([min(x, y) for x, y in zip(xs, ys)])
    approx_two = np.array([min(min(x, y), max(x-1, y-1)) for x, y in zip(xs, ys)])
    approx_three = np.array([min(min(x, y), max(x-0.5, y-0.5)) for x, y in zip(xs, ys)])
    
    fig = plt.figure()
    plt.plot(xs, true_pos, label='nLDE(x, y)')
    plt.plot(xs, approx_one, label='min(x, y)', alpha=0.5, linestyle='--')
    plt.plot(xs, approx_two, label='min(x, y, max(x-1, y-1))', alpha=0.5, linestyle='--')
    plt.plot(xs, approx_three, label='min(x, y, max(x-0.5, y-0.5))', alpha=0.5, linestyle='--')
    plt.legend()
    plt.savefig('nLDE_approx.png')


