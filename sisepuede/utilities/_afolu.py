"""
Store functions to support AFOLU model
"""

import numpy as np
import scipy.optimize as sco


def _get_estimated_parameters(
    vec_factors:  np.ndarray, # y1
    vec_initial_condition: np.ndarray, # np.array([pb, pc, pd_val])
) -> 'scipy.optimize._optimize.OptimizeResult':
    
    result = sco.minimize(
        _objective_sequestration_parameters,
        vec_initial_condition,#np.array(pb, pc, pd),
        args = (vec_factors, ),
        bounds = [(0, None), (0, None), (0, None)],
        method = "SLSQP",#"Nelder-Mead"
    )

    return result



def _objective_sequestration_parameters(
    x_vec: np.ndarray,
    y_factors: np.ndarray,
    widths = (20, 80)
) -> float:
    """
    Find distance between mean value of integrals and seqestration factors
        using parameters b

    Function Arguments
    ------------------
    - x_vec: vector of x_a, x_b, x_c, and x_d to use in objective function
    - y_factors: ordered array of young secondary, old secondary, and primary 
        growth factors.  

        NOTE: `a` in sequestration_curve() is always the primary growth factor 
        since lim as t -> inf is a
    """

    out = _suquestration_int_est(
        y_factors[-1], 
        *x_vec,
        widths = widths,
    ) 
    
    out = np.linalg.norm(out - y_factors[0:-1])
    
    return out



def _sequestration_curve(
    t: float,
    a: float,
    b: float, 
    c: float,
    d: float,
) -> float:
    """
    Project sequestration in new forests as a function of time 

    See Zhou et al 2015 (https://dx.doi.org/10.1002/2015JG002943) for 
        information on the structure of the curve, as well as 
        Repo et al 2021 (https://doi.org/10.1016/j.foreco.2021.119507)
        for secondary (qualitative) source.
    """

    out = b*((t/c)**d) - 1
    out /= np.exp(t/c)
    out += 1
    out *= a

    return out



def _suquestration_int_est(
    *args,
    center: float = 0.5,
    widths: tuple = (20, 180), 
) -> float:
    """
    Get the crude integral estimate of the curve
    
    800 years to primary forest:
    https://www.kloranebotanical.foundation/en/projects/return-primary-forest-europe#:~:text=Primary%20forests%20can%20be%20recreated,provided%20they're%20left%20alone.
    """

    out = np.zeros(len(widths))

    for i, w in enumerate(widths):
        
        flr = widths[i - 1] if i >= 1 else 0
        flr += center
        
        totals = [_sequestration_curve(x, *args) for x in flr + np.arange(w)]
        avg = sum(totals)/w

        out[i] = avg

    return out



