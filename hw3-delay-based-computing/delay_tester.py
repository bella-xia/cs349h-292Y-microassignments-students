from delay_gates import *
from delay_circuit import *


def summarize_outputs(outputs):
    print("------")
    for (gate,port), value, settle, segment, abs_t, rel in outputs:
        print("%s port:%s = %s" % (gate,port,value))
        print("   settling time =\t%e" % (settle))
        print("   relative delay =\t%e" % (rel))
        print("   absolute delay =\t%e" % (abs_t))
        print("   segment time =\t%e" % (segment) )
        print("")


def test_input():
    circ = DelayBasedCircuit()
    inx = circ.add_gate(Input("X"))
    inx = circ.add_gate(Input("Y"))
    inx = circ.add_gate(Input("Z"))
    timing, traces = circ.simulate({"X":4, "Y":7, "Z":2})
    circ.render("test_input.png",timing,traces)
    circ.render_circuit("test_input_circ")
    summarize_outputs(circ.get_outputs(timing,traces))


def test_last_arrival_gate():
    circ = DelayBasedCircuit()
    inx = circ.add_gate(Input("X"))
    iny = circ.add_gate(Input("Y"))
    la = circ.add_gate(LastArrival())
    circ.add_wire(inx,la,"A")
    circ.add_wire(iny,la,"B")

    timing,traces = circ.simulate({"X":4,"Y":7})
    circ.render("test_la.png",timing,traces)
    circ.render_circuit("test_la_circ")
    summarize_outputs(circ.get_outputs(timing,traces))


def test_first_arrival_gate():
    circ = DelayBasedCircuit()
    inx = circ.add_gate(Input("X"))
    iny = circ.add_gate(Input("Y"))
    fa = circ.add_gate(FirstArrival())
    circ.add_wire(inx,fa,"A")
    circ.add_wire(iny,fa,"B")

    timing,traces = circ.simulate({"X":4,"Y":7})
    circ.render("test_fa.png",timing,traces)
    circ.render_circuit("test_fa_circ")
    summarize_outputs(circ.get_outputs(timing,traces))


def test_delay_gate():
    circ = DelayBasedCircuit()
    inx = circ.add_gate(Input("X"))
    delay = circ.add_gate(DelayGate(2*circ.segment_time))
    circ.add_wire(inx,delay,"A")

    circ.render_circuit("test_del_circ")
    timing,traces = circ.simulate({"X":4})
    circ.render("test_del1.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

    timing,traces = circ.simulate({"X":6})
    circ.render("test_del2.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))


def test_inh_gate():
    circ = DelayBasedCircuit()
    inx = circ.add_gate(Input("X"))
    iny = circ.add_gate(Input("Y"))
    inh = circ.add_gate(Inhibition())
    circ.add_wire(inx,inh,"A")
    circ.add_wire(iny,inh,"B")

    circ.render_circuit("test_inh_circ")
    timing,traces = circ.simulate({"X":4,"Y":7})
    circ.render("test_inh1.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

    timing,traces = circ.simulate({"X":7,"Y":4})
    circ.render("test_inh2.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

def test_custom_logic():
    circ = DelayBasedCircuit()
    inx = circ.add_gate(Input("X"))
    iny = circ.add_gate(Input("Y"))
    inc = circ.add_gate(Input("Z")) # this gate will have constant 5 delay
    la = circ.add_gate(LastArrival())  # max(X, Y) operation
    inh = circ.add_gate(Inhibition())
    delay = circ.add_gate(DelayGate(0))
    circ.add_wire(inx, la, "A")
    circ.add_wire(iny, la, "B")
    circ.add_wire(inc, delay, "A")
    circ.add_wire(delay, inh, "A")
    circ.add_wire(la, inh, "B")

    circ.render_circuit("test_custom_logic")
    
    # case 1.1: X > 5, Y < 5
    timing,traces = circ.simulate({"X":7,"Y":4, "Z":5})
    circ.render("test_custom_logic11.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

    # case 1.2: X < 5, Y > 5
    timing,traces = circ.simulate({"X":3,"Y":8, "Z":5})
    circ.render("test_custom_logic12.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

    # case 1.3: X > 5, Y > 5
    timing,traces = circ.simulate({"X":8,"Y":6, "Z":5})
    circ.render("test_custom_logic13.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

    # case 2.1: X < 5, Y < 5
    timing,traces = circ.simulate({"X":3,"Y":4, "Z":5})
    circ.render("test_custom_logic21.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

    # case 2.2: X = Y < 5
    timing,traces = circ.simulate({"X":1,"Y":1, "Z":5})
    circ.render("test_custom_logic22.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

def test_custom_logic2():
    circ = DelayBasedCircuit()
    inx = circ.add_gate(Input("X"))
    inx2 = circ.add_gate(Input("X2")) # x and x2 have the same delay
    iny = circ.add_gate(Input("Y"))
    iny2 = circ.add_gate(Input("Y2")) # y and y2 have the same delay
    inc = circ.add_gate(Input("Z")) # this gate will have constant 5 delay
    la = circ.add_gate(LastArrival())  # max(X, Y) operation
    la2 = circ.add_gate(LastArrival()) # max(X2, Y2) operation, essentially creating another copy
    la3 = circ.add_gate(LastArrival()) # max(max(X2, Y2), 5) operation
    delay = circ.add_gate(DelayGate(0))
    delay2 = circ.add_gate(DelayGate(0))
    inh = circ.add_gate(Inhibition())

    circ.add_wire(inx, la, "A")
    circ.add_wire(iny, la, "B")
    circ.add_wire(inc, delay, "A")
    circ.add_wire(inx2, la2, "A")
    circ.add_wire(iny2, la2, "B")
    circ.add_wire(la, inh, "A")
    circ.add_wire(delay, inh, "B")
    circ.add_wire(la2, delay2, "A")
    circ.add_wire(inh, la3, "A")
    circ.add_wire(delay2, la3, "B")

    circ.render_circuit("test_custom_logic2")
    
    # case 1.1: X > 5, Y < 5
    timing,traces = circ.simulate({"X":7,"Y":4, "X2":7,"Y2":4,"Z":5})
    circ.render("test_custom_logic211.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

    # case 1.2: X < 5, Y > 5
    timing,traces = circ.simulate({"X":3,"Y":8, "X2":3,"Y2":8,"Z":5})
    circ.render("test_custom_logic212.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

    # case 1.3: X > 5, Y > 5
    timing,traces = circ.simulate({"X":8,"Y":6, "X2":8,"Y2":6,"Z":5})
    circ.render("test_custom_logic213.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

    # case 2.1: X < 5, Y < 5
    timing,traces = circ.simulate({"X":3,"Y":4, "X2":3,"Y2":4,"Z":5})
    circ.render("test_custom_logic221.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

    # case 2.2: X = Y < 5
    timing,traces = circ.simulate({"X":1,"Y":1, "X2":1,"Y2":1,"Z":5})
    circ.render("test_custom_logic222.png",timing,traces)
    summarize_outputs(circ.get_outputs(timing,traces))

# test_input()
# test_first_arrival_gate()
# test_last_arrival_gate()
# test_inh_gate()
# test_delay_gate()
# test_custom_logic()
# test_custom_logic2()