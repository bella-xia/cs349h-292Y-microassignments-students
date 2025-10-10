import numpy as np
from stochastic import PosStochasticComputing, StochasticComputingStaticAnalysis

class AST:
    def __init__(self) -> None:
        self.flag : str = '' # op / num
        self.val: float = 0.0
        self.op : str = ''
        self.lhs : AST = None
        self.rhs : AST = None
        self.alloc : int = 0

    def __str__(self) -> str:
        return self._get_str()

    def _get_str(self, prefix='') -> str:
        if self.flag == 'num':
            rep = prefix + f'NUM[{self.val}]\n'
            assert not self.lhs and not self.rhs, 'Error: numeric values should not have lhs and rhs values'
            return rep

        rep = prefix + f'OP[{self.op}\n'
        rep += self.lhs._get_str(prefix + ' ')
        rep += self.rhs._get_str(prefix + ' ')
        rep += prefix + ']\n'
        return rep

    def _update_bitlen(self, analysis : StochasticComputingStaticAnalysis) -> float:
        
        if self.flag == 'num':
            prec = analysis.stoch_var(self.val)
            return prec

        lhs_prec = self.lhs._update_bitlen(analysis)
        rhs_prec = self.rhs._update_bitlen(analysis)
        if self.op == '+':
            return analysis.stoch_add(lhs_prec, rhs_prec)
        
        return analysis.stoch_mul(lhs_prec, rhs_prec)

    def _update_ref(self, ref_dict : dict[float, int]) -> None:
        if self.flag == 'num':
            ref_dict[self.val] = ref_dict.get(self.val, 0) + 1
            return

        self.lhs._update_ref(ref_dict)
        self.rhs._update_ref(ref_dict)

    def _clean_alloc(self) -> None:
        self.alloc = 0
        
        if self.flag == 'op':
            self.lhs._clean_alloc()
            self.rhs._clean_alloc()
     
    def do_stoch(self, bitstream_len : int, rng_record : dict[float, tuple[np.ndarray, int]]) -> np.ndarray:

        if self.flag == 'num':
            if rng_record.get(self.val, None):
                idx = rng_record[self.val][1]
                rng_record[self.val] = (rng_record[self.val][0], idx+1)
                return rng_record[self.val][0][idx:bitstream_len+idx]
            rng = PosStochasticComputing.to_stoch(self.val, bitstream_len)
            self.alloc += rng.nbytes
            return rng

        lhs = self.lhs.do_stoch(bitstream_len, rng_record)
        rhs = self.rhs.do_stoch(bitstream_len, rng_record)
        self.alloc += self.lhs.alloc + self.rhs.alloc

        if self.op == '+':
            return  PosStochasticComputing.stoch_add(lhs, rhs)

        return PosStochasticComputing.stoch_mul(lhs, rhs)
    
    def eval_standard(self) -> float:
        if self.flag == 'num': 
            return self.val

        lhs = self.lhs.eval_standard()
        rhs = self.rhs.eval_standard()
        return (lhs + rhs) / 2 if self.op == '+' else lhs * rhs

    def eval_stoch(self, optimize : bool = False, 
                    bitstream_len : int | None = None, 
                    num_rounds : int = 1000) -> float:
        if not bitstream_len:
            analysis = StochasticComputingStaticAnalysis()
            self._update_bitlen(analysis)
            bitstream_len = analysis.get_size()
            print(f'using static analysis to derive bitstream length: {bitstream_len}')

        if optimize:
            ref_dict = {}
            self._update_ref(ref_dict)

        res, allocs = [], []
        
        for _ in range(num_rounds):
            self._clean_alloc()
            rng_record = {}
            if optimize:
                for k, v in ref_dict.items():
                    rng = PosStochasticComputing.to_stoch(k, bitstream_len + v - 1)
                    self.alloc += rng.nbytes
                    rng_record[k] = (rng, 0)

            res.append(PosStochasticComputing.from_stoch(self.do_stoch(bitstream_len, rng_record)))
            allocs.append(self.alloc)
        print(f'average allocation of numpy array memory: {np.mean(allocs, dtype=np.uint32)} bytes')
        return np.mean(res) 

class Parser:

    def __init__(self, eq: str):
        self.eq = eq.strip()
        self.i = 0
        self.ast = None
        pass

    def parse(self) -> AST | None:
        return self._parse_add()

    def _peek(self) -> str:
        if self.i >= len(self.eq): 
            return ''

        return self.eq[self.i]


    def _consume(self) -> str:
        assert self.i < len(self.eq), 'Error: attempted consumption at end of string'
        c : str = self.eq[self.i]
        self.i += 1
        return c

    def _parse_space(self) -> str:
        
        while self._peek() == ' ':
            self._consume()
 
    def _parse_digit(self) -> AST:
        ast = AST()
        ast.flag = 'num'
        ast.val = 0.0

        # pre floating point
        while '0' <= self._peek() <= '9':
            ast.val *= 10
            ast.val += int(self._consume())

        if self._peek() != '.':
            self._parse_space()
            return ast

        # post floating point
        self.i += 1
        fp : int = 0.1

        while '0' <= self._peek() <= '9':
            ast.val += fp * int(self._consume())
            fp /= 10
        
        self._parse_space()
        return ast
    
    def _parse_opcode(self) -> AST:
        opcode = self._consume()
        ast = AST()
        ast.flag = 'op'
        ast.op = opcode
        self._parse_space()

        return ast

    def _parse_expr(self) -> AST | None:
        if (self._peek()) == '(':
            self._consume()
            node = self._parse_add()
            assert self._peek() == ')', 'Error: incorrect bracket termination'
            self._consume()
            self._parse_space()
            return node

        elif self._peek().isdigit():
            return self._parse_digit()

        elif self._peek() == '+' or self._peek() == '*':
            return self._parse_opcode()
        
        elif self._peek() == '':
            return None

        raise Exception('Error: unexpected end of equation')

    def _parse_mult(self) -> AST | None:
        lhs = self._parse_expr()
        while self._peek() == '*':
            op = self._parse_opcode()
            op.lhs = lhs
            op.rhs = self._parse_expr()
            lhs = op
        return lhs

    def _parse_add(self) -> AST | None:
        lhs = self._parse_mult()
        while self._peek() == '+':
            op = self._parse_opcode()
            op.lhs = lhs
            op.rhs = self._parse_mult()
            lhs = op
        return lhs

if __name__ == '__main__':
    
    eq : str = ''
    spec : str = "Enter the expression to evaluate ('exit' or 'e' to quit): "

    while True:
        eq = input(spec)
        if eq == 'e' or eq == 'exit':
            print('exiting parser')
            break
        parser = Parser(eq)
        node = parser.parse()
        print(f'expression: {eq}')
        print('AST repr:')
        print(node)
        print('evaluated via standard computing')
        print('res:', node.eval_standard())
        print('\nevaluated via stochastic computing (non-optimized)')
        print('res:', node.eval_stoch())
        print('\nevaluated via stochastic computing (optimized)')
        print('res:', node.eval_stoch(optimize=True))
