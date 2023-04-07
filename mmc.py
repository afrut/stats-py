# M/M/c System Calculator
# Check: c = 2  l = 15   m = 20    ET = 0.05818181818181818
import math

# Computes (1/n!)(l/m)**n
def f1(n, l, m):
    prod = 1
    while True:
        if n <= 0:
            break
        prod = prod * (1/n) * (l/m)
        n -= 1
    return prod

# Computes (1/(n - 1)!)(l/m)**n
def f2(n, l, m):
    prod = l/m
    n -= 1
    while True:
        if n <= 0:
            break
        prod = prod * (l / m) / n
        n -= 1
    return prod

# Probability of having 0 jobs in system
def pi0(c, l, m):
    p = l/(c*m)

    # Loop for n
    total = 1
    for i in range(0,max(c-1,1)):
        n = i + 1
        total = total + f1(n, l, m)
    total += f1(c, l, m) * (1 / (1 - p))
    return total**(-1)

def etq(c, l, m, _pi0 = None):
    if _pi0 is None:
        _pi0 = pi0(c, l, m)
    return f2(c, l, m) * m / (((c * m) - l) ** 2) * _pi0

def enq(c, l, m, _pi0 = None):
    if _pi0 is None:
        _pi0 = pi0(c, l, m)
    return f2(c, l, m) * l * m / (((c * m) - l) ** 2) * _pi0

def et(c, l, m, _etq  = None):
    if _etq is None:
        _etq = etq(c, l, m)
    return (1 / m) + _etq

def en(c, l, m, _et = None):
    if _et is None:
        _et = et(c, l, m)
    return l * _et

def erlang(c, l, m, _pi0 = None):
    if _pi0 is None:
        _pi0 = pi0(c, l, m)
    p = l / (c * m)
    return f1(c, l, m) * _pi0 / (1 - p)

def fc(l, m, Pmax):
    c = int(l/m) + 1
    while True:
        if erlang(c, l, m) < Pmax:
            break
        c+=1
    return c

def mmc(c, l, m, Pmax, display = False):
    _f1 = f1(c, l, m)
    _f2 = f2(c, l, m)
    _pi0 = pi0(c, l, m)
    _etq = etq(c, l, m, _pi0)
    _enq = enq(c, l, m, _pi0)
    _et = et(c, l, m, _etq)
    _en = en(c, l, m, _et)
    _erlang = erlang(c, l, m, _pi0)
    _fc = fc(l, m, Pmax)
    if display:
        print(f"f1 = {_f1:.6f}")
        print(f"f2 = {_f2:.6f}")
        print(f"pi0 = {_pi0:.6f}")
        print(f"etq = {_etq:.6f}")
        print(f"enq = {_enq:.6f}")
        print(f"et = {_et:.6f}")
        print(f"en = {_en:.6f}")
        print(f"erlang = {_erlang:.6f}")
        print(f"fc(0.01) = {_fc}")
    return (_f1,_f2,_pi0,_etq,_enq,_et,_en,_erlang,_fc)

def mmcc(c, l, m, display = False):
    sum = 1
    orig_c = c
    while True:
        if c <= 0:
            break
        sum += f1(c, l, m)
        c -= 1
    _pi0 = 1 / sum
    ls = [_pi0]

    c = orig_c
    for n in range(1, c + 1):
        ls.append(f1(n, l, m) * _pi0)
    _pin = f1(c, l, m) / sum
    if display:
        print(f"_pi0= {_pi0:.6f}")
        print(f"_pin= {_pin:.6f}")
    return (_pi0, _pin, ls)


if __name__ == "__main__":
    c, l, m = (2, 15, 20)
    assert et(c, l, m) - 0.058181 < 1e-6

    c, l, m, Pmax = (2, 3, 60 / 18, 0.01)
    assert f1(c, l, m) - 0.405000 < 1e-6
    assert f2(c, l, m) - 0.810000 < 1e-6
    assert pi0(c, l, m) - 0.379310 < 1e-6
    assert etq(c, l, m) - 0.076176 < 1e-6
    assert enq(c, l, m) - 0.228527 < 1e-6
    assert et(c, l, m) - 0.376176 < 1e-6
    assert en(c, l, m) - 1.128527 < 1e-6
    assert erlang(c, l, m) - 0.279310 < 1e-6
    assert fc(l, m, Pmax) - 5 == 0
    _f1,_f2,_pi0,_etq,_enq,_et,_en,_erlang,_fc = mmc(c, l, m, Pmax)

    c, l, m, Pmax = (2, 400, 1, 0.2)
    _f1,_f2,_pi0,_etq,_enq,_et,_en,_erlang,_fc = mmc(c, l, m, Pmax)
    print(f"fc(0.2) = {_fc}")
    assert fc(l, m, Pmax) == 422


    print("----------------------------------------")
    print("  Assignment 7 # 1")
    print("----------------------------------------")
    c, l, m, Pmax = (2, 10, 12, 0.5)
    _f1,_f2,_pi0,_etq,_enq,_et,_en,_erlang,_fc = mmc(c, l, m, Pmax, True)
    for c in range(2, 21):
        _f1,_f2,_pi0,_etq,_enq,_et,_en,_erlang,_fc = mmc(c, l, m, Pmax)
        print(f"revenue {c} = {(l - (0.2 * l * _erlang) - c):.6f}")

    print("----------------------------------------")
    print("  Assignment 7 # 2")
    print("----------------------------------------")
    c, l, m, Pmax = (2, 1, 1.5, 0.5)
    _pi0, _pin, ls = mmcc(c, l, m, True)
    print(f"Answer = {_pin * l * 60:.6f} jobs per hour")

    print("----------------------------------------")
    print("  Assignment 7 # 3a")
    print("----------------------------------------")
    p = 0.95
    m = 1
    Pmax = 0.5
    for k in [2**x for x in range(6)]:
        l = p * k * m
        _f1,_f2,_pi0,_etq,_enq,_et,_en,_erlang,_fc = mmc(k, l, m, Pmax)
        print(f"k = {k}, rho = {l / (k * m):.6f}, Pq = {_erlang:.6f}, E[T] = {_et:.6f}")

    print("----------------------------------------")
    print("  Assignment 7 # 3b")
    print("----------------------------------------")
    k = 32
    l = p * k * m
    R = l / m
    Pmax = 0.2
    sq = R + (math.sqrt(R) * 1.06)
    print(f"num_servers by square root rule: {sq:.6f}")
    print(f"num_servers exact: {fc(l, m, Pmax):.6f}")