import string
import random
import operator
import numpy as np
import tqdm
import matplotlib.pyplot as plt


class HDC:
    SIZE = 10000

    @classmethod
    def rand_vec(cls) -> np.ndarray:
        hvec : np.ndarray = np.random.binomial(1, 0.5, (cls.SIZE,)).astype(np.uint8)
        return hvec

        # unreachable
        raise Exception("generate atomic hypervector with size HDC.SIZE")

    @classmethod
    def dist(cls, x1 : np.ndarray, x2 : np.ndarray) -> float:
        xs_xor : np.ndarray = x1 ^ x2
        d : float = np.sum(xs_xor) / cls.SIZE
        return d

        # unreachable
        raise Exception("Hamming distance between hypervectors")

    @classmethod
    def bind(cls, x1 : np.ndarray , x2 : np.ndarray) -> np.ndarray:
        xs_xor : np.ndarray = x1 ^ x2 
        return xs_xor

        # unreachable
        raise Exception("bind two hypervectors together")

    @classmethod
    def bind_all(cls, xs : list[np.ndarray]) -> np.ndarray:
        if len(xs) == 0:
            return np.zeros((cls.SIZE,)).astype(np.uint8)

        xs_xor = np.bitwise_xor.reduce(xs).astype(np.uint8)
        return xs_xor
        
        # unreachable 
        raise Exception("convenience function. bind together a list of hypervectors")

    @classmethod
    def bundle(cls, xs : list[np.ndarray]) -> np.ndarray:
        thres : float = len(xs) / 2

        # produce majority vote
        xs_sum : np.ndarray = np.sum(np.stack(xs), axis=0) 
        xs_maj : np.ndarray = (xs_sum > thres).astype(np.uint8)
        
        # break ties
        xs_tie : np.ndarray = xs_sum == thres
        num_ties : int = np.sum(xs_tie)
        xs_maj[xs_tie] = np.random.binomial(1, 0.5, size=num_ties).astype(np.uint8)
        return xs_maj

        # unreachable
        raise Exception("bundle together xs, a list of hypervectors")

    @classmethod
    def permute(cls, x : np.ndarray, i : int) -> np.ndarray:
        x_per : np.ndarray = np.roll(x, shift=i)
        return x_per

        # unreachable
        raise Exception("permute x by i, where i can be positive or negative")

    @classmethod
    def apply_bit_flips(cls, x: np.ndarray, p: float = 0.0) -> None:

        x_err : np.ndarray = np.random.binomial(1, p, (x.shape[0], )).astype(np.bool_)
        x[x_err] ^= 1
        return
        
        # unreachable
        raise Exception("return a corrupted hypervector, given a per-bit bit flip probability p")


class HDItemMem:

    def __init__(self, name=None, ber=0.0) -> None:
        self.name = name
        self.item_mem = {}
        # per-bit bit flip probabilities for the Hamming distance
        self.ber = ber

    def add(self, key, hv):
        assert (hv is not None)
        self.item_mem[key] = hv

    def get(self, key):
        return self.item_mem[key]

    def has(self, key):
        return key in self.item_mem

    def distance(self, query):
        wks = {}
        for k, v in self.item_mem.items():
            v_copy = v.copy()
            HDC.apply_bit_flips(v_copy, self.ber)
            d : float = HDC.dist(query, v_copy)
            wks[k] = d
        
        return wks

        # unreachable
        raise Exception("compute hamming distance between query vector and each row in item memory. Introduce bit flips if the bit flip probability is nonzero")

    def all_keys(self):
        return list(self.item_mem.keys())

    def all_hvs(self):
        return list(self.item_mem.values())

    def wta(self, query):
        wks = self.distance(query)
        wk = min(wks, key=wks.get)
        wd = wks[wk]
        
        return wk, wd

        # unreachable
        raise Exception("winner-take-all querying")

    def matches(self, query, threshold=0.49):
        
        wks = self.distance(query)
        wks = dict(filter(lambda w: w[1] <= threshold, wks.items()))
        
        return wks

        # unreachable
        raise Exception("threshold-based querying")

# a codebook is simply an item memory that always creates a random hypervector
# when a key is added.
class HDCodebook(HDItemMem):

    def __init__(self, name=None):
        HDItemMem.__init__(self, name)

    def add(self, key):
        self.item_mem[key] = HDC.rand_vec()


def make_letter_hvs():
    cb : HDCodebook = HDCodebook(name='letters')
    for c in range(ord('a'), ord('z') + 1):
        cb.add(chr(c))
    
    return cb
    
    # unreachable
    raise Exception("return a codebook of letter hypervectors")


def make_word(letter_codebook : HDCodebook, word : str) -> np.ndarray | None:
        
    xs : list[np.ndarray] = []

    for i, c in enumerate(word):
        # Task 1, Q2
        # xs.append(letter_codebook.get(c))
        xs.append(HDC.permute(letter_codebook.get(c), i))

    return HDC.bundle(xs)

    # unreachable
    raise Exception("make a word using the letter codebook")


def monte_carlo(fxn, trials):
    results = list(map(lambda i: fxn(), tqdm.tqdm(range(trials))))
    return results


def plot_dist_distributions(key1, dist1, key2, dist2):
    plt.hist(dist1,
            alpha=0.75,
            label=key1)

    plt.hist(dist2,
            alpha=0.75,
            label=key2)

    plt.legend(loc='upper right')
    plt.title('Distance distribution for Two Words')
    plt.show()
    plt.clf()


def study_distributions():
    def gen_codebook_and_words(w1 : np.ndarray, w2 : np.ndarray, prob_error : float =0.0) -> float:
        letter_cb : HDCodebook = make_letter_hvs()
        hv1 : np.ndarray = make_word(letter_cb, w1)
        HDC.apply_bit_flips(hv1, p=prob_error)
        hv2 : np.ndarray = make_word(letter_cb, w2)
        d : float = HDC.dist(hv1, hv2)
        return d
        
        # unreachable
        raise Exception("encode words and compute distance")

    trials = 1000
    d1 = monte_carlo(lambda: gen_codebook_and_words("fox", "box"), trials)
    d2 = monte_carlo(lambda: gen_codebook_and_words("fox", "car"), trials)
    plot_dist_distributions("box", d1, "car", d2)
    
    
    # Task 1, Q3
    # for err_rate in range(11):
    #     perr = err_rate * 0.1
    #     print(f'pdderr = {perr}')
    #     d1 = monte_carlo(lambda: gen_codebook_and_words("fox", "box", prob_error=perr), trials)
    #     d2 = monte_carlo(lambda: gen_codebook_and_words("fox", "car", prob_error=perr), trials)
    #     plot_dist_distributions("box", d1, "car", d2)

    # Task 1, Q4
    # perr = 0.1
    # for size in range(10, 1, -1):
    #     HDC.SIZE = size
    #     print(f'HDC size: {HDC.SIZE}')
    #     d1 = monte_carlo(lambda: gen_codebook_and_words("fox", "box", prob_error=perr), trials)
    #     d2 = monte_carlo(lambda: gen_codebook_and_words("fox", "car", prob_error=perr), trials)
    #     plot_dist_distributions("box", d1, "car", d2)

if __name__ == '__main__':
    HDC.SIZE = 10000

    letter_cb = make_letter_hvs()
    hv1 = make_word(letter_cb, "fox")
    hv2 = make_word(letter_cb, "box")
    hv3 = make_word(letter_cb, "xfo")
    hv4 = make_word(letter_cb, "car")

    print(HDC.dist(hv1, hv2))
    print(HDC.dist(hv1, hv3))
    print(HDC.dist(hv1, hv4))
    
    study_distributions()
