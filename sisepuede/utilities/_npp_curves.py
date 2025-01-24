"""
Store functions to support AFOLU model
"""

import numpy as np
import scipy.optimize as sco
import sisepuede.utilities._toolbox as sf
from typing import *


# some global settings
_PARAMS_DEFAULT_GAMMA = np.array([235.5, 0.426, -0.00484], )  # see tang et al. table 1 for Boreal GPP
_PARAMS_DEFAULT_SEM = np.array([0.1323, 1.0642, 6.3342, 3.455], )
_WIDTHS_DEFAULT = (20, 180)  #, 1000)



################################
#    INITIALIZE SOME CURVES    #
################################

class NPPCurve:
    """
    Store information about the NPP Curve
    """
    def __init__(self,
        func: callable,
        bounds: callable,
        defaults: np.ndarray,
        jacobian: Union[callable, None] = None,
        name: Union[str, None] = None,
    ) -> None:
        
        self.bounds = bounds
        self.defaults = defaults
        self.function = func
        self.jacobian = jacobian
        self.name = name
        self.is_npp_curve = True

        return None



    def __call__(self,
        *args,
        **kwargs,
    ) -> float:
        out = self.function(
            *args,
            **kwargs,
        )

        return out
    


    def get_parameters(self,
        vec_params: Union[list, np.ndarray, None] = None,
        stop_on_error: bool = False,
    ) -> Union[np.ndarray, None]:
        """
        Get parameters--if an invalid vector is 
        """


        return_def = not sf.islistlike(vec_params)
        return_def |= (len(vec_params) != len(self.defaults)) if not return_def else return_def

        if return_def:
            if stop_on_error and vec_params is not None:
                tp = str(type(vec_params))
                raise RuntimeError(f"Invalid type")

        out = (
            self.defaults
            if return_def
            else vec_params
        )

        return out
    



##  GAMMA
#`
def _bounds_gamma(
    *args,
    **kwargs,
) -> float:
    """
    Get bounds for minimization of gamma
    """
    # final parameter k2 is negative
    out = [(0, None), (None, 1), (None, 0)]

    return out



def _curve_gamma(
    t: float,
    k0: float,
    k1: float,
    k2: float,
) -> float:
    """
    Project sequestration in new forests as a function of time using NPP in
        gamma curve. The curve takes the following form:

        k0*(t**k1)*(e**(k2*t))
    
    """

    out = k0*(t**k1)*np.exp(k2*t)

    return out



def _jacobian_gamma(
    t: float,
    k0: float,
    k1: float,
    k2: float,
) -> float:
    """
    Project sequestration in new forests as a function of time using NPP in
        gamma curve. The curve takes the following form:

        k0*(t**k1)*(e**(k2*t))
    
    """

    comp_k1k2 = (t**k1)*np.exp(k2*t)

    out = np.array(
        [
            comp_k1k2,
            k0*np.log(t)*comp_k1k2,
            k0*t*comp_k1k2,
        ]
    )

    return out






# SEM CURVE
#

def _curve_sem(
    t: float,
    a: float,
    b: float, 
    c: float,
    d: float,
) -> float:
    """
    Use the SEM curve (Chen et al.) for NPP 
    """

    out = b*((t/c)**d) - 1
    out /= np.exp(t/c)  
    out += 1
    out *= a

    return out



def _bounds_sem(
    vec_target: np.ndarray,
    *args,
    force_convergence: bool = True,
    **kwargs,
) -> float:
    """
    Set bounds for minimization of gamma
    """ 

    out = [(0, None) for x in range(4)]
    if force_convergence:
        out[0] = (vec_target[-1], vec_target[-1])

    return out







class NPPCurves:
    """
    Generate curves for net primary production (NPP), which estimates
        net "carbon gain by plants". As noted by Chapin and Eviner,


        NPP is the net carbon gain by plants. It is the balance between the 
        carbon gained by gross primary production (GPP – i.e., net 
        photosynthesis measured at the ecosystem scale) and carbon released by 
        plant mitochondrial respiration, both expressed per unit land area. Like 
        GPP, NPP is generally measured at the ecosystem scale over relatively 
        long time intervals, such as a year (g biomass or g C m −2 year− 1). NPP 
        includes the new biomass produced by plants, the soluble organic 
        compounds that diffuse or are secreted into the environment (root or 
        phytoplankton exudation), the carbon transfers to microbes that are 
        symbiotically associated with roots (e.g., mycorrhizae and 
        nitrogen-fixing bacteria), and the volatile emissions that are lost from 
        leaves to the atmosphere (Clark et al., 2001). 

        (see 10.6.2.1, https://www.sciencedirect.com/referencework/9780080983004/treatise-on-geochemistry)

        
    For more on NPP, see: https://www.sciencedirect.com/topics/earth-and-planetary-sciences/net-primary-production

    
    Includes the following curves:

        - SEM (chen et al. 2003)
        - Gamma (Tang et al. 2014)

        
    Sources
    -------
    See Zhou et al 2015 (https://dx.doi.org/10.1002/2015JG002943) for 
        information on the structure of the curve, as well as 
        Repo et al 2021 (https://doi.org/10.1016/j.foreco.2021.119507)
        for secondary (qualitative) source.

    See Li et al. (https://bg.copernicus.org/articles/21/625/2024/) for a 
        comparison of different sequestration curves. In general, the SEM and
        Gamma curves show the best performance. 

    
    Initialization Arguments
    ------------------------
    - sequestration_targets: ordered list of tuples of the form

        [(targ_0, width_0), ... , (targ_{k - 1}, width_{k - 1})]

        that give

            - targ_i: target average annual sequestration factor
            - width_i: window width
    
    Optional Arguments
    ------------------
    - dt: width for crude integral estimate
    - stop_on_bad_target_spec: raise error if a target is specified incorrectly? 
        If False, continues with well-specified values.

    
    """

    def __init__(self,
        sequestration_targets: List[Tuple],
        dt: float = 0.01,
        stop_on_bad_target_spec: bool = True,
    ) -> None:
        

        self._initialize_curves()
        self._initialize_sequestration_targets(
            sequestration_targets,
            dt, 
            stop_on_error = stop_on_bad_target_spec,
        )

        return None
    


    def _initialize_curves(self,
    ) -> None:
        """
        initialize valid curves. Sets the following properties:

            - self.curves
            - self.dict_curves
        """
        # keys for dictionary
        key_func = "function"
        key_params_default = "parameters_default"

        # curve names
        curve_name_gamma = "gamma"
        curve_name_sem = "sem"
        
        # set up objects
        curve_gamma = NPPCurve(
            _curve_gamma,
            _bounds_gamma,
            _PARAMS_DEFAULT_GAMMA,
            jacobian = _jacobian_gamma,
            name = curve_name_gamma,
        )

        curve_sem = NPPCurve(
            _curve_sem,
            _bounds_sem,
            _PARAMS_DEFAULT_SEM,
            name = curve_name_sem,
        )


        dict_curves = {
            curve_name_gamma: curve_gamma,
            curve_name_sem: curve_sem,
        }
        curves = sorted(list(dict_curves.keys()))


        ##  SET PROPERTIES

        self.curve_name_gamma = curve_name_gamma
        self.curve_name_sem = curve_name_sem
        self.curves = curves
        self.dict_curves = dict_curves
        self.key_func = key_func
        self.key_params_default = key_params_default

        return None
    


    def _initialize_sequestration_targets(self,
        sequestration_targets: List[Tuple],
        dt: float,
        stop_on_error: bool = True,
    ) -> None:
        """
        Check the sequestration targets and initialize the following properties:

            - self.dt
            - self.targets
            - self.widths
        """

        # some checks
        if not isinstance(sequestration_targets, list):
            raise RuntimeError(f"sequestration_targets must be a list if tuples")
        
        if not sf.isnumber(dt):
            raise RuntimeError(f"dt must be a number")
        

        # init
        targets = []
        widths = []

        for i, tup in enumerate(sequestration_targets):

            skip = not isinstance(tup, tuple)
            skip |= len(tup) < 2 if not skip else skip
            skip |= (not (sf.isnumber(tup[0]) & sf.isnumber(tup[1]))) if not skip else skip
            skip |= (min(tup) < 0) if not skip else skip

            if skip:
                if stop_on_error:
                    raise RuntimeError(f"Invalid target/width pair found at position {i} in sequestration_targets. Must be a two-ple.")
                continue

            targets.append(tup[0])
            widths.append(tup[1])


        ##  SET PROPERTIES

        self.dt = dt
        self.targets = targets
        self.widths = widths

        return None



    def estimate_integral(self,
        curve: Union[callable, str, NPPCurve],
        *args,
        dt: Union[float, None] = None,
        vec_widths: Union[np.ndarray, None] = None,
    ) -> float:
        """
        Get the crude integral estimate of the curve
        
        800 years to primary forest:
        https://www.kloranebotanical.foundation/en/projects/return-primary-forest-europe#:~:text=Primary%20forests%20can%20be%20recreated,provided%20they're%20left%20alone.
        """
        # init
        curve = self.get_curve(curve, self.curve_name_sem, )
        dt = self.get_dt(dt)
        vec_widths = self.get_widths(vec_widths)
        
        out = np.zeros(len(vec_widths))

        for i, w in enumerate(vec_widths):

            flr = vec_widths[i - 1] if i >= 1 else 0
            t = np.arange(flr, w, dt) + dt/2
            
            totals = curve(t, *args)
            avg = totals.sum()*dt
            out[i] = avg/(w - flr) if w != flr else 0.0

        return out
    


    def fit(self,
        curve: Union[str, NPPCurve], 
        force_convergence: bool = True, 
        method: str = "SLSQP",  # "Nelder-Mead"
        vec_params_0: Union[np.ndarray, None] = None,  # np.array([pb, pc, pd_val])
        vec_targets: Union[np.ndarray, None] = None,
        **kwargs,
    ) -> 'scipy.optimize._optimize.OptimizeResult':
        """Get the estimated parameters for the sequestration curve

        Arguments
        ---------
        curve : Union[str, NPPCurve]
            Curve to use; can be "gamma", "sem", or other NPPCurve object
        
        Keyword Arguments
        -----------------
        force_convergence : bool
            in SEM method, force convergence to the final average sequestration
            value provided in curves.targets? If True, sets param a to the last
            target.    
        method : str
            minimization method used; pased to sco.minimie
        vec_params_0 : Union[np.ndarray, None]
            Optional starting parameter vector. If None, defaults to 
            NPPCurve.default for the given curve
        vec_targets : Union[np.ndarray, None]
            Optional target vector to match. If None, defaults to 
            NPPCurves.target (initialization set)
        """

        # init
        vec_targets = self.get_targets(vec_targets, )
        curve = self.get_curve(curve, )
        bounds = curve.bounds(
            vec_targets, 
            force_convergence = force_convergence, 
        )

        vec_params_0 = (
            curve.defaults
            if not isinstance(vec_params_0, np.ndarray)
            else vec_params_0
        )

        result = sco.minimize(
            self.objective,
            vec_params_0,
            args = (curve, vec_targets, ),
            bounds = bounds,
            method = method,
            **kwargs,
        )

        return result



    def get_bounds(self,
        curve: Union[callable, str, NPPCurve, ],
        vec_targets: Union[np.ndarray, None] = None,
    ) -> List[Tuple]:
        """
        Retrieve bounds for parameter searches
        """

        curve = self.get_curve(curve)
        if curve is None:
            return None
        
        vec_targets = self.get_targets(vec_targets, )
        out = curve.bounds(vec_targets, )

        return out



    def get_curve(self,
        curve: Union[callable, str, NPPCurve, ],
        return_on_none: Any = None,
    ) -> Union[callable, None]:
        """
        Retrieve a curve for parameter estimation
        """

        if isinstance(curve, NPPCurve) | callable(curve):
            return curve
        
        out = self.dict_curves.get(curve, return_on_none)

        return out



    def get_dt(self,
        dt: Union[float, None] = None,
    ) -> float:
        """
        Retrieve dt
        """
        dt = self.dt if not sf.isnumber(dt) else float(dt)

        return dt
        


    def get_targets(self,
        vec_targets: Union[np.ndarray, None] = None,
    ) -> Union[np.ndarray, None]:
        """
        Retrieve the targets
        """
        vec_targets = self.targets if vec_targets is None else vec_targets

        return vec_targets
    


    def get_widths(self,
        vec_widths: Union[np.ndarray, None] = None,
    ) -> Union[callable, None]:
        """
        Retrieve the widths
        """
        vec_widths = self.widths if vec_widths is None else vec_widths

        return vec_widths
    


    def objective(self,
        vec_params: np.ndarray,
        curve: Union[callable, str, NPPCurve, ],
        vec_targets: Union[np.ndarray, None] = None,
        vec_widths: Union[np.ndarray, None] = None,
    ) -> float:
        """
        Find distance between mean value of integrals and seqestration factors
            using parameters b

        Function Arguments
        ------------------
        - vec_params: vector of parameters to pass to fit function
            - for Gamma: k0, k1, k2
            - for SEM: b, c, d (a is specified as limiting value for primary
                since since lim f_SEM(t) as t -> inf is a)
        - curve: curve to use:
            - "gamma"
            - "sem"
    
            
        Keyword Arguments
        -----------------
        - vec_targets: optional targets to pass. Defaults to self-defined
            - ordered array of young secondary, old secondary, and 
                primary growth factors
        - vec_widths: optional widths to pass
        """

        curve = self.get_curve(curve, )
        vec_targets = self.get_targets(vec_targets, )
        vec_widths = self.get_widths(vec_widths, )
        
        out = self.estimate_integral(
            curve,
            *vec_params, 
            vec_widths = vec_widths,
        ) 
        
        out = np.linalg.norm(out - vec_targets)
        
        return out

