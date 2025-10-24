import numpy as np
import matplotlib.pyplot as plt
import math

class PosStochasticComputing:
    APPLY_FLIPS = False
    APPLY_SHIFTS = False
    FLIPERR_PROB = 0.0001
    SHIFTERR_PROB = 0.0001

    @classmethod
    def apply_bitshift(cls, bitstream : np.ndarray) -> None:
        if not cls.APPLY_SHIFTS:
            return

        err_mask : np.ndarray = np.random.binomial(1, cls.SHIFTERR_PROB, (bitstream.shape[0]-1, )).astype(np.bool_) # last bit cannot constitute a shift error
        digits = np.where(err_mask)[0]
        for digit in digits:
            tmp: np.uint8 = bitstream[digit]
            bitstream[digit+1:] = bitstream[digit:-1]
            bitstream[digit] = tmp
        return

        # unreachable
        raise Exception("apply the bitshift error to the bitstream with probability 0.0001")

    @classmethod
    def apply_bitflip(cls, bitstream : np.ndarray) -> None:
        if not cls.APPLY_FLIPS:
            return
        
        err_mask : np.ndarray = np.random.binomial(1, cls.FLIPERR_PROB, (bitstream.shape[0], )).astype(np.bool_)
        bitstream[err_mask] ^= 1
        return

        # unreachable
        raise Exception("apply the to the bitstream with probability 0.0001")

    @classmethod
    def to_stoch(cls, prob : float, nbits : int) -> np.ndarray:
        assert 0.0 <= prob <= 1.0, "invalid bound for stochastic generation, expected [0, 1]"

        rng : np.ndarray = np.random.binomial(1, prob, (nbits,)).astype(np.uint8) 
        cls.apply_bitflip(rng) # computation circuit
        # cls.apply_bitflip(rng) # write storage
        return rng

        # unreachable
        raise Exception("convert a decimal value in [0,1] to an <nbit> length bitstream.") 

    @classmethod
    def stoch_add(cls, bitstream : np.ndarray, bitstream2 : np.ndarray) -> np.ndarray:
        assert bitstream.shape == bitstream2.shape, "invalid bitstream addition, expected same length" 

        rng: np.ndarray = np.random.binomial(1, 0.5, (bitstream.shape[0], )).astype(np.uint8)
        cls.apply_bitflip(rng) # computation circuit
        rng = rng.astype(np.bool_)
        # cls.apply_bitflip(bitstream); cls.apply_bitflip(bitstream2) # read storage
        cls.apply_bitshift(rng); cls.apply_bitshift(bitstream); cls.apply_bitshift(bitstream2) # computation
        weighted_sum : np.ndarray = np.where(rng, bitstream, bitstream2)
        cls.apply_bitflip(weighted_sum) # computation circuit
        # cls.apply_bitflip(weighted_sum) # write storage
        return weighted_sum

        # unreachable
        raise Exception("add two stochastic bitstreams together")

    @classmethod
    def stoch_mul(cls, bitstream : np.ndarray, bitstream2 : np.ndarray) -> np.ndarray:
        assert bitstream.shape == bitstream2.shape, "invalid bitstream multiplication, expected same length" 

        # cls.apply_bitflip(bitstream); cls.apply_bitflip(bitstream2) # read storage
        cls.apply_bitshift(bitstream); cls.apply_bitshift(bitstream2) # computation
        prod : np.ndarray = bitstream & bitstream2
        cls.apply_bitflip(prod) # computation circuit
        # cls.apply_bitflip(prod) # write storage
        return prod

        # unreachable
        raise Exception("multiply two stochastic bitstreams together")

    @classmethod
    def from_stoch(cls, result : np.ndarray) -> float:
        # cls.apply_bitflip(result) # read storage
        cnt1, cnt2 = np.sum(result, dtype=np.uint32), result.shape[0]
        return cnt1 / cnt2

        # unreachable
        raise Exception("convert a stochastic bitstream to a numerical value")

class StochasticComputingStaticAnalysis:

    def __init__(self):
        self.min_prec = 1.0
        # pass

    def req_length(self, smallest_value : float) -> int:
        return math.ceil(1 / smallest_value)

        # unreachable
        raise Exception("figure out the smallest bitstream length necessary represent the input decimal value. This is also called the precision.")

    def stoch_var(self, prec : int) -> int:
        if prec < self.min_prec:
            self.min_prec = prec
        return prec

        # unreachable
        raise Exception("update static analysis -- the expression contains a variable with precision <prec>.")
        result_prec = None
        return result_prec


    def stoch_add(self, prec1, prec2):
        return self.stoch_var((prec1 + prec2) / 2)

        # unreachable
        raise Exception("update static analysis -- the expression adds together two bitstreams with precisions <prec1> and <prec2> respectively.")
        result_prec = None
        return result_prec


    def stoch_mul(self, prec1, prec2):
        return self.stoch_var(prec1 * prec2)

        # unreachable
        raise Exception("update static analysis -- the expression multiplies together two bitstreams with precisions <prec1> and <prec2> respectively.")
        result_prec = None
        return res_prec

    def get_size(self):
        return self.req_length(self.min_prec)

        # unreachable
        raise Exception("get minimum bitstream length required by computation.")


# run a stochastic computation for ntrials trials
def run_stochastic_computation(lambd, ntrials, visualize=True, summary=True):
    results = []
    reference_value, _ = lambd()
    for i in range(ntrials):
        _,result = lambd()
        results.append(result)

    if visualize:
        nbins = math.floor(np.sqrt(ntrials))
        plt.hist(results,bins=nbins)
        plt.axvline(x=reference_value, color="red")
        plt.show()
    if summary:
        print("ref=%f" % (reference_value))
        print("mean=%f" % np.mean(results))
        print("std=%f" % np.std(results))

def PART_A_example_computation(bitstream_len):
    # expression: 1/2*(0.8 * 0.4 + 0.6)
    reference_value = 1/2*(0.8 * 0.4 + 0.6)
    w = PosStochasticComputing.to_stoch(0.8, bitstream_len)
    x = PosStochasticComputing.to_stoch(0.4, bitstream_len)
    y = PosStochasticComputing.to_stoch(0.6, bitstream_len)
    tmp = PosStochasticComputing.stoch_mul(x, w)
    result = PosStochasticComputing.stoch_add(tmp, y)
    return reference_value, PosStochasticComputing.from_stoch(result)


def PART_Y_analyze_wxb_function(precs):
    # 1/2*(w*x + b)
    analysis = StochasticComputingStaticAnalysis()
    w_prec = analysis.stoch_var(precs["w"])
    x_prec = analysis.stoch_var(precs["x"])
    b_prec = analysis.stoch_var(precs["b"])
    res_prec = analysis.stoch_mul(w_prec, x_prec)
    analysis.stoch_add(res_prec, b_prec)
    N = analysis.get_size()
    print("best size: %d" % N)
    return N

def PART_Y_execute_wxb_function(values, N):
    # expression: 1/2*(w*x + b)
    w = values["w"]
    x = values["x"]
    b = values["b"]
    reference_value = 1/2*(w*x + b)
    w = PosStochasticComputing.to_stoch(w, N)
    x = PosStochasticComputing.to_stoch(x, N)
    b = PosStochasticComputing.to_stoch(b, N)
    tmp = PosStochasticComputing.stoch_mul(x, w)
    result = PosStochasticComputing.stoch_add(tmp, b)
    return reference_value, PosStochasticComputing.from_stoch(result)


def PART_Y_test_analysis():
    precs = {"x": 0.1, "b":0.1, "w":0.01}
    # apply the static analysis to the w*x+b expression, where the precision of x and b is 0.1 and
    # the precision of w is 0.01
    N_optimal = PART_Y_analyze_wxb_function(precs)
    print("best size: %d" % N_optimal)

    variables = {}
    for _ in range(10):
        variables["x"] = round(np.random.uniform(),1)
        variables["w"] = round(np.random.uniform(),2)
        variables["b"] = round(np.random.uniform(),1)
        print(variables)
        run_stochastic_computation(lambda : PART_Y_execute_wxb_function(variables,N_optimal), ntrials=10000, visualize=True)
        print("")


def PART_Z_execute_rng_efficient_computation(value,N,save_rngs=True): # assume a small constant value of k
    # expression: 1/2*(x*x+x)

    xv = value
    reference_value = 1/2*(xv*xv + xv)
    if save_rngs:
        ref = PosStochasticComputing.to_stoch(xv, N+2)
        x = ref[:N]
        x2 = ref[1:N+1]
        x3 = ref[2:]
    else:
        x = PosStochasticComputing.to_stoch(xv, N)
        x2 = PosStochasticComputing.to_stoch(xv, N)
        x3 = PosStochasticComputing.to_stoch(xv, N)

    tmp = PosStochasticComputing.stoch_mul(x, x2)
    result = PosStochasticComputing.stoch_add(tmp,x3)
    return reference_value, PosStochasticComputing.from_stoch(result)


if __name__ == '__main__':
    ntrials = 10000

    # ===== A.Q1 ===== 
    def run_PART_A_Q1():
        print("---- part a Q1: effect of length on stochastic computation ---")
        print('bitstream length: 10, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=10), ntrials)
        print('bitstream length: 100, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=100), ntrials)
        print('bitstream length: 1000, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=1000), ntrials)

    # run_PART_A_Q1()

    # ===== A.Q2 ===== 
    def run_PART_A_Q2():
        print("---- part a Q2: effect of length on stochastic computation ---")
        def PART_A_Q2_computation(bitstream_len):
            reference_value = 1 / bitstream_len
            x = PosStochasticComputing.to_stoch(reference_value, bitstream_len)
            return reference_value, PosStochasticComputing.from_stoch(x)

        print('bitstream length: 10, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_Q2_computation(bitstream_len=10), ntrials)
        print('bitstream length: 100, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_Q2_computation(bitstream_len=100), ntrials)
        print('bitstream length: 200, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_Q2_computation(bitstream_len=200), ntrials)
        print('bitstream length: 500, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_Q2_computation(bitstream_len=500), ntrials)
        print('bitstream length: 1000, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_Q2_computation(bitstream_len=1000), ntrials)
        print('bitstream length: 2000, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_Q2_computation(bitstream_len=1000), ntrials)
    run_PART_A_Q2()

    # ==== A.Q4 ====
    def run_PART_A_Q4():
        print("---- part a Q4: effect of length on stochastic computation ---")
        def PART_A_Q4_computation(bitstream_len):
            # expression: 0.1 * 0.1 * 0.1 * 0.1
            reference_value = 0.1 * 0.1 * 0.1 * 0.1
            a = PosStochasticComputing.to_stoch(0.1, bitstream_len)
            b = PosStochasticComputing.to_stoch(0.1, bitstream_len)
            c = PosStochasticComputing.to_stoch(0.1, bitstream_len)
            d = PosStochasticComputing.to_stoch(0.1, bitstream_len)
            tmp = PosStochasticComputing.stoch_mul(a, b)
            tmp = PosStochasticComputing.stoch_mul(tmp, c)
            result = PosStochasticComputing.stoch_mul(tmp, d)
            return reference_value, PosStochasticComputing.from_stoch(result)
        print('bitstream length: 1000, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_Q4_computation(bitstream_len=1000), ntrials)

    # run_PART_A_Q4()

    # Part X, introduce non-idealities

    # test non-idealities
    def test_Nonidealities():
        PosStochasticComputing.APPLY_FLIPS = True
        PosStochasticComputing.APPLY_SHIFTS = True
        PosStochasticComputing.FLIPERR_PROB = 0.05
        PosStochasticComputing.SHIFTERR_PROB = 0.05

        print('checking flip behavior')
        for _ in range(3):
            x = PosStochasticComputing.to_stoch(0.5, 20)
            print('before:', x)
            PosStochasticComputing.apply_bitflip(x)
            print('after :', x)
            print()

        print('checking shift behavior')
        for _ in range(3):
            x = PosStochasticComputing.to_stoch(0.5, 20)
            print('before:', x)
            PosStochasticComputing.apply_bitshift(x)
            print('after :', x)
            print()

        PosStochasticComputing.APPLY_FLIPS = False
        PosStochasticComputing.APPLY_SHIFTS =False

    # test_Nonidealities()

    def run_PART_X():
        print("---- part x: effect of bit flips ---")
        PosStochasticComputing.APPLY_SHIFTS = False
        print("----- disable per-bit bit-flip error -----")
        PosStochasticComputing.APPLY_FLIPS = False
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=1000), ntrials)
        print("----- per-bit bit-flip error probability = 0.0001 -----")
        PosStochasticComputing.APPLY_FLIPS = True
        PosStochasticComputing.FLIPERR_PROB = 0.0001
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=1000), ntrials)
        print("----- per-bit bit-flip error probability = 0.1 -----")
        PosStochasticComputing.FLIPERR_PROB = 0.1
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=1000), ntrials)
        PosStochasticComputing.APPLY_FLIPS = False
        print("----- disable per-bit bit-shift error -----")
        PosStochasticComputing.APPLY_SHIFTS = False
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=1000), ntrials)
        print("----- per-bit bit-shift error probability = 0.0001 -----")
        PosStochasticComputing.APPLY_SHIFTS = True
        PosStochasticComputing.SHIFTERR_PROB = 0.0001
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=1000), ntrials)
        print("----- per-bit bit-shift error probability = 0.1 -----")
        PosStochasticComputing.SHIFTERR_PROB = 0.1
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=1000), ntrials)
        PosStochasticComputing.APPLY_FLIPS = False
        PosStochasticComputing.APPLY_SHIFTS =False

    # run_PART_X()

    def run_PART_X_Q3():
        print("---- part x: effect of bit flips ---")
        PosStochasticComputing.APPLY_SHIFTS = False
        PosStochasticComputing.APPLY_FLIPS = False
        print("----- no error contrast trial -----")
        print('bitstream length: 10, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=10), ntrials)
        print('bitstream length: 100, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=100), ntrials)
        print('bitstream length: 1000, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=1000), ntrials)
        PosStochasticComputing.APPLY_SHIFTS = False
        PosStochasticComputing.APPLY_FLIPS = True
        PosStochasticComputing.FLIPERR_PROB = 0.1
        print("----- per-bit bit-flip error probability = 0.1 -----")
        print('bitstream length: 10, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=10), ntrials)
        print('bitstream length: 100, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=100), ntrials)
        print('bitstream length: 1000, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=1000), ntrials)
        PosStochasticComputing.APPLY_FLIPS = False
        PosStochasticComputing.APPLY_SHIFTS = True
        PosStochasticComputing.SHIFTERR_PROB = 0.1
        print("----- per-bit bit-shift error probability = 0.1 -----")
        print('bitstream length: 10, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=10), ntrials)
        print('bitstream length: 100, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=100), ntrials)
        print('bitstream length: 1000, total trial: 10000')
        run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=1000), ntrials)
        PosStochasticComputing.APPLY_FLIPS = False
        PosStochasticComputing.APPLY_SHIFTS =False

    run_PART_X_Q3()


    # Part Y, apply static analysis
    def run_PART_Y():
        print("---- part y: apply static analysis ---")
        PART_Y_test_analysis()

    # run_PART_Y()

    # Part Z, resource efficent rng generation
    def run_PART_Z():
        print("---- part z: one-rng optimization ---")
        for _ in range(5):
            v = round(np.random.uniform(),1)
            print(f"x = {v}")
            print("running with save_rngs disabled")
            run_stochastic_computation(lambda : PART_Z_execute_rng_efficient_computation(value=v, N=1000, save_rngs=False), ntrials)
            print("running with save_rngs enabled")
            run_stochastic_computation(lambda : PART_Z_execute_rng_efficient_computation(value=v, N=1000, save_rngs=True), ntrials)

    # run_PART_Z()
