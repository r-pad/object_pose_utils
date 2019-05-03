import numpy as np

def subrandom_iter(c = (np.sqrt(5)-1)/2.):
    r = np.random.rand()
    while 1:
        yield r
        r = np.mod(r + c, 1)

def subrandom(N, c = None, dim=1):
    if c is None:
        if(dim == 1):
            c = [(np.sqrt(5)-1)/2.]
        elif(dim == 2):
            c = [0.5545497, 0.308517]
        else:
            from sympy import prime
            c = [prime(j+1) for j in range(dim)]
    
    if(type(c) is float):
        c = [c]

    assert(dim == len(c))
    values = []
    for c_i in c:
        v_i = []
        for j, x in enumerate(subrandom_iter(c_i)):
            if(j >= N):
                break
            v_i.append(x)
        
        values.append(v_i)
    
    values = np.array(values).T
    if(dim == 1):
        values = values.flatten()
    
    return values

