from sisepuede.core.attribute_table import AttributeTable



# parameters a, b, c, d
_PARAMS_SEQUESTRATION_CURVE_EST = np.array([0.1323, 1.0642, 6.3342, 3.455], )






# NOTE: keys must be immutable! What can we use?
dictionary = {
    key_1: value_1,
    key_2: value_2,
    .
    .
    .``
}

# NOTE: keys must be immutable! What can we use?
#   0



def bubble_sort(
    vec: list,
) -> list:
    """
    Implement bubble sort
    """

    n = len(vec)
    
    for j in range(n - 1):
        
        # assume no swaps occur
        swap_occured = False

        for i in range(1, n - j):
            if vec[i] >= vec[i - 1]: continue
                
            # swap
            vec[i - 1], vec[i] = vec[i], vec[i -1 ]
            swap_occured = True
        
        # no sense continuing if we don't have to
        if not swap_occured: break

    return vec





def o_1(
   vec: list,
) -> list:
   """
   Demonstrate an O(1)--here, the output is independent of list size
   """
   n = len(vec)
   return n**2



def o_n(
    vec: np.ndarray,
) -> np.ndarray:
    
    """
    Demonstrate an O(n)--here, the number of operations scales linearly
    """
    n = vec.shape[0]
    out = np.zeros(vec.shape)

    for i in range(n):
        out[i] = vec[i]**2 if vec[i] < i**2 else vec[i]**-0.89

    return out

        

def o_n2(
    vec: np.ndarray,
) -> np.ndarray:
    
    """
    Demonstrate an O(n^2)--here, the number of operations scales in polynomial 
        time
    """
    n = vec.shape[0]
    out = np.zeros(n, n)

    for i in range(n):
        for j in range(n):
            out[i, j] = func(i, j, n)

    return out



def o_2n(
    vec: np.ndarray,
) -> np.ndarray:
    
    """
    Demonstrate an O(2^n)--here, the number of operations scales exponentially
    """
    n = vec.shape[0]
    ps = power_set(n)
    out = np.zeros(len(ps))

    for i, subset in enumerate(out):
        out[i] = func(subset)

    return out




v_best = None
k_best = None

for k, v in dict.items():
    v_best = v if v_best is None else (
        v if v < v_best else v_best
    )

    v_best = v if v_best is None else (
        v if v < v_best else v_best
    )