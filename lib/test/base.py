import numpy as np
import random

from ..mytorch.functional import *
from ..util import RunMode, set_run_mode

atol = 1e-5

def np_randn(*args):
    res = np.random.randn(*args)
    return res.astype(np.float32)


def answer_gen(func, *args):
    set_run_mode(RunMode.TORCH)
    return func(*args)


def test_leaky_relu(runmode):
    test_i = np_randn(4, 4)
    test_grad = random.random()
    answer = answer_gen(leaky_relu, test_i, test_grad, "leaky_relu_answer")

    set_run_mode(runmode)
    output = leaky_relu(test_i, test_grad, "leaky_relu_test")

    if output is None:
        print(f"leaky_relu in {runmode} not implemented")
    elif answer.shape != output.shape:
        print(f"leaky_relu in {runmode} test failed.\n output shape does not match answer shape. {answer.shape} != {output.shape}")
    elif np.allclose(answer, output, atol=atol):
        print(f"leaky_relu in {runmode} test passed.\n l2 error:{np.linalg.norm(answer-output)}")
    else:
        print(f"leaky_relu in {runmode} test failed.\n l2 error:{np.linalg.norm(answer-output)}")


def test_batch_norm(runmode):
    n, c, h, w = 1, 3, 3, 3
    test_i = np_randn(n, c, h, w)
    test_rm = np_randn(c)
    test_rv = np_randn(c)
    test_rv = np.abs(test_rv)
    test_w = np_randn(c)
    test_b = np_randn(c)

    answer = answer_gen(batch_norm, test_i, test_rm, test_rv, test_w, test_b, "batch_norm_answer")

    set_run_mode(runmode)
    output = batch_norm(test_i, test_rm, test_rv, test_w, test_b, "batch_norm_test")

    if output is None:
        print(f"batch_norm in {runmode} not implemented")
    elif answer.shape != output.shape:
        print(f"batch_norm in {runmode} test failed.\n output shape does not match answer shape. {answer.shape} != {output.shape}")
    elif np.allclose(answer, output, atol=atol):
        print(
            f"batch_norm in {runmode} test passed.\n l2 error:{np.linalg.norm(answer-output)}")
    else:
        print(f"batch_norm in {runmode} test failed.\n l2 error:{np.linalg.norm(answer-output)}")

    
def test_conv2d(runmode):
    n = 1
    ic, h, w = 3, 16, 16
    oc, kh, kw = 10, 3, 3
    s = 1
    test_i = np_randn(n, ic, h, w)
    test_w = np_randn(oc, ic, kh, kw)
    test_b = np_randn(oc)

    answer = answer_gen(conv2d, test_i, test_w, test_b, s, "conv2d_answer")

    set_run_mode(runmode)
    output = conv2d(test_i, test_w, test_b, s, "conv2d_test")

    if output is None:
        print(f"conv2d in {runmode} not implemented")
    elif answer.shape != output.shape:
        print(f"conv2d in {runmode} test failed.\n output shape does not match answer shape. {answer.shape} != {output.shape}")
    elif np.allclose(answer, output, atol=atol):
        print(
            f"conv2d in {runmode} test passed.\n l2 error:{np.linalg.norm(answer-output)}")
    else:
        print(f"conv2d in {runmode} test failed.\n l2 error:{np.linalg.norm(answer-output)}")


def test_conv2d_stride2(runmode):
    n = 3
    ic, h, w = 3, 5, 5
    oc, kh, kw = 10, 3, 3
    s = 2
    test_i = np_randn(n, ic, h, w)
    test_w = np_randn(oc, ic, kh, kw)
    test_b = np_randn(oc)

    answer = answer_gen(conv2d, test_i, test_w, test_b, s, "conv2d_stride2_answer")

    set_run_mode(runmode)
    output = conv2d(test_i, test_w, test_b, s, "conv2d_stride2_test")

    if output is None:
        print(f"conv2d with stride 2 in {runmode} not implemented")
    elif answer.shape != output.shape:
        print(f"conv2d with stride 2 in {runmode} test failed.\n output shape does not match answer shape. {answer.shape} != {output.shape}")
    elif np.allclose(answer, output, atol=atol):
        print(f"conv2d with stride 2 in {runmode} test passed.\n l2 error:{np.linalg.norm(answer-output)}")
    else:
        print(f"conv2d with stride 2 in {runmode} test failed.\n l2 error:{np.linalg.norm(answer-output)}")


def test_conv2d_no_bias(runmode):
    n = 3
    ic, h, w = 3, 16, 16
    oc, kh, kw = 10, 3, 3
    s = 1
    test_i = np_randn(n, ic, h, w)
    test_w = np_randn(oc, ic, kh, kw)

    answer = answer_gen(conv2d, test_i, test_w, None, s, "conv2d_no_bias_answer")

    set_run_mode(runmode)
    output = conv2d(test_i, test_w, None, s, "conv2d_no_bias_test")

    if output is None:
        print(f"conv2d without bias in {runmode} not implemented")
    elif answer.shape != output.shape:
        print(f"conv2d without bias in {runmode} test failed.\n output shape does not match answer shape. {answer.shape} != {output.shape}")
    elif np.allclose(answer, output, atol=atol):
        print(f"conv2d without bias in {runmode} test passed.\n l2 error:{np.linalg.norm(answer-output)}")
    else:
        print(f"conv2d without bias in {runmode} test failed.\n l2 error:{np.linalg.norm(answer-output)}")


def test_max_pool2d(runmode):
    n, c, h, w = 1, 1, 6, 6
    k, s = 2, 2
    test_i = np_randn(n, c, h, w)

    answer = answer_gen(max_pool2d, test_i, k, s, "max_pool2d_answer")

    set_run_mode(runmode)
    output = max_pool2d(test_i, k, s, "max_pool2d_test")

    if output is None:
        print(f"max_pool2d in {runmode} not implemented")
    elif answer.shape != output.shape:
        print(f"max_pool2d in {runmode} test failed.\n output shape does not match answer shape. {answer.shape} != {output.shape}")
    elif np.allclose(answer, output, atol=atol):
        print(f"max_pool2d in {runmode} test passed.\n l2 error:{np.linalg.norm(answer-output)}")
    else:
        print(f"max_pool2d in {runmode} test failed.\n l2 error:{np.linalg.norm(answer-output)}")


def test_pad(runmode):
    n, c, h, w = 1, 1, 6, 6
    p = 1, 2, 3, 4 # 3 h 4 / 1 w 2
    v = 1.
    test_i = np_randn(n, c, h, w)

    answer = answer_gen(pad, test_i, p, v, "pad_answer")

    set_run_mode(runmode)
    output = pad(test_i, p, v, "pad_test")

    if output is None:
        print(f"pad in {runmode} not implemented")
    elif answer.shape != output.shape:
        print(f"pad in {runmode} test failed.\n output shape does not match answer shape. {answer.shape} != {output.shape}")
    elif np.allclose(answer, output, atol=atol):
        print(f"pad in {runmode} test passed.\n l2 error:{np.linalg.norm(answer-output)}")
    else:
        print(f"pad in {runmode} test failed.\n l2 error:{np.linalg.norm(answer-output)}")


ns = [2**0, 2**4]
cs = [2**4, 2**8]
ncs = [2**4, 2**8]
hws = [2**8, 2**10]
khws = [2**1, 2**3]
ss = [2**0, 2**3]
ps = [2**1, 2**5]

class IndexGen:
    def __init__(self, num_select):
        self.curr = 0
        self.last = num_select
        self.indexes = [0] * num_select

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr < self.last:
            self.indexes[self.curr-1] = 0
            self.indexes[self.curr] = 1
            self.curr += 1
            return self.indexes
        else:
            raise StopIteration

def stress_conv2d(runmode):
    set_run_mode(runmode)

    ig = IndexGen(6)
    for i in ig:
        n = ns[i[0]] * (2 if runmode != RunMode.C else 1)
        ic = cs[i[1]] * (2 if runmode != RunMode.C else 1)
        oc = cs[i[2]] * (2 if runmode != RunMode.C else 1)
        hw = hws[i[3]] * (2 if runmode != RunMode.C else 1)
        khw = khws[i[4]] * (2 if runmode != RunMode.C else 1)
        s = ss[i[5]] * (2 if runmode != RunMode.C else 1)

        test_i = np_randn(n, ic, hw, hw)
        test_w = np_randn(oc, ic, khw, khw)
        test_b = np_randn(oc)
        conv2d(test_i, test_w, test_b, s, f"stress_conv2d_n{n}ic{ic}oc{oc}hw{hw}khw{khw}")

def stress_batch_norm(runmode):
    set_run_mode(runmode)

    ig = IndexGen(3)
    for i in ig:
        n = ns[i[0]] * (2 if runmode != RunMode.C else 1)
        c = cs[i[1]] * (2 if runmode != RunMode.C else 1)
        hw = hws[i[2]] * (2 if runmode != RunMode.C else 1)
        
        test_i = np_randn(n, c, hw, hw)
        test_rm = np_randn(c)
        test_rv = np_randn(c)
        test_rv = np.abs(test_rv)
        test_w = np_randn(c)
        test_b = np_randn(c)

        batch_norm(test_i, test_rm, test_rv, test_w, test_b, f"stress_batch_norm_n{n}c{c}hw{hw}")

def stress_leaky_relu(runmode):
    set_run_mode(runmode)

    ig = IndexGen(2)
    for i in ig:
        nc = ncs[i[0]] * (2 if runmode != RunMode.C else 1)
        hw = hws[i[1]] * (2 if runmode != RunMode.C else 1)
        test_grad = random.random()
        test_i = np_randn(nc, hw)
        leaky_relu(test_i, test_grad, f"stress_leaky_relu_nc{nc}hw{hw}")


def stress_maxpool_2d(runmode):
    set_run_mode(runmode)

    ig = IndexGen(5)
    for i in ig:
        n = ns[i[0]] * (2 if runmode != RunMode.C else 1)
        c = cs[i[1]] * (2 if runmode != RunMode.C else 1)
        hw = hws[i[2]] * (2 if runmode != RunMode.C else 1)
        k = khws[i[3]] * (2 if runmode != RunMode.C else 1)
        s = ss[i[4]] * (2 if runmode != RunMode.C else 1)

        test_i = np_randn(n, c, hw, hw)
        max_pool2d(test_i, k, s, f"stress_max_pool2d_n{n}c{c}hw{hw}k{k}s{s}")

def stress_pad(runmode):
    set_run_mode(runmode)

    ig = IndexGen(4)
    for i in ig:
        n = ns[i[0]] * (2 if runmode != RunMode.C else 1)
        c = cs[i[1]] * (2 if runmode != RunMode.C else 1)
        hw = hws[i[2]] * (2 if runmode != RunMode.C else 1)
        p = ps[i[3]] * (2 if runmode != RunMode.C else 1)

        v = 1.
        test_i = np_randn(n, c, hw, hw)

        pad(test_i, (p, p, p, p), v, f"stress_pad_n{n}c{c}hw{hw}p{p}")
