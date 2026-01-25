"""
Store functions to support AFOLU model
"""

import numpy as np
import pandas as pd
import scipy.optimize as sco
import sisepuede.utilities._toolbox as sf
from typing import *



##########################
#    GLOBAL VARIABLES    #
##########################

# module uuid
_MODULE_UUID = "C437BEFC-FFA4-4412-AF98-E704A556DD18"

# some global settings
_PARAMS_DEFAULT_GAMMA = np.array([235.5, 0.426, -0.00484], )  # see tang et al. table 1 for Boreal GPP
_PARAMS_DEFAULT_SEM = np.array([0.1323, 1.0642, 6.3342, 3.455], )
_WIDTHS_DEFAULT = (20, 180)  #, 1000)

# some errors
class InvalidNorm(Exception):
    pass



class MissingNPPParameters(Exception):
    pass




################################
#    INITIALIZE SOME CURVES    #
################################

class NPPCurve:
    """Store information about the NPP Curve
    """
    def __init__(self,
        func: callable,
        bounds: callable,
        defaults: np.ndarray,
        derivative: Union[callable, None] = None,
        jacobian: Union[callable, None] = None,
        name: Union[str, None] = None,
        norm: Union[float, None] = None,
    ) -> None:
        
        self._initialize_attributes(
            func,
            bounds,
            defaults,
            derivative,
            jacobian,
            name,
        )

        self._initialize_norm(
            norm,
        )

        self._initialize_uuid()
        

        return None



    def __call__(self,
        *args,
        **kwargs,
    ) -> float:
        
        out = self.project(*args, **kwargs, )

        return out
    


    def _initialize_attributes(self,
        func: callable,
        bounds: callable,
        defaults: np.ndarray,
        derivative: Union[callable, None],
        jacobian: Union[callable, None],
        name: Union[str, None],
    ) -> None:
        """Initialiize the norm for the curve.
        """

        #function = np.vectorize(func, )#otypes = [float], )

        self.bounds = bounds
        self.defaults = defaults
        self.derivative = derivative
        self.function = func
        self.jacobian = jacobian
        self.name = name
        self.is_npp_curve = True

        return None
    


    def _initialize_norm(self,
        norm: Union[float, None],
    ) -> None:
        """Initialiize the norm for the curve.
        """

        norm = None if not sf.isnumber(norm) else norm
        if norm == 0:
            raise InvalidNorm(f"Keyword argument 'norm' cannot have value equal to zero.")

        # case where number
        self.norm = norm

        return None
    


    def _initialize_uuid(self,
    ) -> None:
        """Initialize the UUID
        """

        self.is_npp_curve = True
        self._uuid = _MODULE_UUID

        return None




    def get_parameters(self,
        vec_params: Union[list, np.ndarray, None] = None,
        stop_on_error: bool = False,
    ) -> Union[np.ndarray, None]:
        """Get parameters--if an invalid vector is 
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
    


    def project(self,
        *args,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """project the npp curve in time. Note that, in self.function(), args[0]
            is always the time. 

        Tries to retrieve norm from kwargs; if not present, defaults to 
            initialization value.
        """

        # make sure the norm isn't passed to the raw function
        norm = kwargs.pop("norm", self.norm, )
        t = np.array(args[0]) if sf.islistlike(args[0]) else args[0]
        out = self.function(t, *args[1:], **kwargs)

        if norm is not None:
            out /= norm
            
        return out

    



##  GAMMA
#`
def _BOUNDS_GAMMA(
    *args,
    **kwargs,
) -> float:
    """Get bounds for minimization of gamma
    """
    # final parameter k2 is negative
    out = [(0, None), (None, 1), (None, 0)]

    return out



def _CURVE_GAMMA(
    t: float,
    k0: float,
    k1: float,
    k2: float,
) -> float:
    """Project sequestration in new forests as a function of time using NPP in
        gamma curve. The curve takes the following form:

        k0*(t**k1)*(e**(k2*t))
    
    """

    out = k0*(t**k1)*np.exp(k2*t)

    return out



def _JACOBIAN_GAMMA(
    t: float,
    k0: float,
    k1: float,
    k2: float,
) -> float:
    """Project sequestration in new forests as a function of time using NPP in
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

def _BOUNDS_SEM(
    vec_target: np.ndarray,
    *args,
    force_convergence: bool = True,
    **kwargs,
) -> float:
    """Set bounds for minimization of gamma
    """ 

    out = [(0, None) for x in range(4)]
    if force_convergence:
        out[0] = (vec_target[-1], vec_target[-1])

    return out



def _CURVE_SEM(
    t: float,
    a: float,
    b: float, 
    c: float,
    d: float,
) -> float:
    """Use the SEM curve (Chen et al.) for NPP 
    """

    out = b*((t/c)**d) - 1
    out /= np.exp(t/c)  
    out += 1
    out *= a

    return out



def _DERIV_SEM(
    t: float,
    a: float,
    b: float, 
    c: float,
    d: float,
) -> float:
    """First-order derivative of the SEM curve (Chen et al.) for NPP (wrt t)
    """

    # intermediate variables
    u = (t**d)*b/(c**d)
    exp = np.exp(t/c)

    out = u*d/t - u/c + 1/c
    out *= a/exp

    return out






class NPPCurves:
    """Generate curves for net primary production (NPP), which estimates net 
        "carbon gain by plants". As noted by Chapin and Eviner,


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
    sequestration_targets : List[Tuple]
        Ordered list of tuples of the form

        [(targ_0, width_0), ... , (targ_{k - 1}, width_{k - 1})]

        that give

            - targ_i: target average annual sequestration factor
            - width_i: window width
    
    Optional Arguments
    ------------------
    dt : float
        Width for crude integral estimate
    norm : Union[float, None]
        Optional target area for integration; will normalize the NPP curve so
        that the Reimann sum (or cumulative mass) will be equivalent to this 
        area over the span from 0 - N, where N is the when a forest reaches
        primary (e.g., 1000 years)
    stop_on_bad_target_spec : bool
        Raise error if a target is specified incorrectly? If False, continues 
        with well-specified values.

    
    """

    def __init__(self,
        sequestration_targets: List[Tuple],
        dt: float = 0.01,
        norm: Union[float, None] = None,
        stop_on_bad_target_spec: bool = True,
    ) -> None:
        

        self._initialize_curves(
            norm = norm,
        )
        self._initialize_sequestration_targets(
            sequestration_targets,
            dt, 
            stop_on_error = stop_on_bad_target_spec,
        )

        self._initialize_uuid()

        return None
    


    def _initialize_curves(self,
        norm: Union[float, None],
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
            _CURVE_GAMMA,
            _BOUNDS_GAMMA,
            _PARAMS_DEFAULT_GAMMA,
            jacobian = _JACOBIAN_GAMMA,
            name = curve_name_gamma,
            norm = norm,
        )

        curve_sem = NPPCurve(
            _CURVE_SEM,
            _BOUNDS_SEM,
            _PARAMS_DEFAULT_SEM,
            derivative = _DERIV_SEM,
            name = curve_name_sem,
            norm = norm,
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
        self.norm = norm

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
    


    def _initialize_uuid(self,
    ) -> None:
        """Initialize the UUID
        """

        self.is_npp_curves = True
        self._uuid = _MODULE_UUID

        return None



    def estimate_integral(self,
        curve: Union[callable, str, NPPCurve],
        *args,
        dt: Union[float, None] = None,
        vec_widths: Union[np.ndarray, None] = None,
    ) -> float:
        """Get the crude integral estimate of the curve
        
        800 years to primary forest:
        https://www.kloranebotanical.foundation/en/projects/return-primary-forest-europe#:~:text=Primary%20forests%20can%20be%20recreated,provided%20they're%20left%20alone.
        """
        # init
        curve = self.get_curve(curve, self.curve_name_sem, )
        dt = self.get_dt(dt)
        vec_widths = self.get_widths(vec_widths)
        
        out = np.zeros(len(vec_widths))

        for i, w in enumerate(vec_widths):

            flr = sum(vec_widths[0:i]) if i >= 1 else 0
            t = np.arange(flr, flr + w, dt) + dt/2
            
            totals = curve(t, *args)
            avg = totals.sum()*dt
            out[i] = avg/w if w != 0 else 0.0

        return out
    


    def fit(self,
        curve: Union[str, NPPCurve], 
        assign_as_default: bool = True,
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
        assign_as_default : bool
            Assign resulting parameters to the "params" property? If True, sets
            parameters to self.defaults and vec_targets to self.targets.
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

        if assign_as_default:
            curve.defaults = result.x

        return result
    


    def get_assumed_steady_state_sem(self,
        *args,
        dt: Union[float, None] = None,
        tol: float = 0.0000001,
        vec_widths: Union[np.ndarray, None] = None,
    ) -> int:
        """Find the starting point for assumed steady state
        """
        curve_npp = self.get_curve("sem")
        vec_widths = self.get_widths(vec_widths, )
        dt = self.get_dt(dt, )

        window_max = sum(vec_widths)

        # get domain and derivative
        t = np.arange(0, window_max, dt)
        out = curve_npp.derivative(t, *args, )
        i_0 = self.get_assumed_steady_state_sem_p0(out, )
        if i_0 is None:
            return None

        # iterate forward
        i_1 = self.get_assumed_steady_state_sem_p1(out, i_0, tol = tol, )

        return i_1
    


    def get_assumed_steady_state_sem_cumulative_mass(self,
        *args,
        dt: Union[float, None] = None,
        force_imax_ceiling: bool = True,
        return_type: str = "vector",
        tol: float = 0.0000001,
        vec_widths: Union[np.ndarray, None] = None,
    ) -> np.ndarray:
        """For a parameterization, get the cumulative fraction of biomass used. 
        

        Arguments
        ---------
        *args : floats
            ordered parameters passed to curve

        Keyword Arguments
        -----------------
        dt : Union[float, None]
            Optional dt step used to estimate integral. If None, defaults to 
            self.dt
        force_imax_ceiling : bool
            Set to True to force the assumed steady state arrive to be an 
            integer, i.e., the ceiling of the determined i_max. This is 
            convenient for setting tp-based output (e.g., if 
            return_type == "array_collapsed_to_tp")
        return_array : str
            "array" : return an N x 2 array, where the first column are time 
                periods (in units of dt) and the second period is the cumulative 
                mass
            "array_collapsed_to_tp"  return an N_TP x 2 array, where the first 
                column are time periods (in units of time period) and the second 
                period is the cumulative mass
            "vector" : return a vector of cmf values by dt to imax
        tol : float
            convergence tolerance for steady state id
        vec_widths : array_like
            vector of integration windows used to determine domain of 
            integration and search
        """
        # get 
        curve_npp = self.get_curve("sem")
        vec_widths = self.get_widths(vec_widths, )
        dt = self.get_dt(dt, )

        # y
        i_max = self.get_assumed_steady_state_sem(
            *args,
            tol = tol,
            vec_widths = vec_widths,
        )
        if i_max is None:
            return None
        
        if force_imax_ceiling:
            i_max = np.ceil(i_max*dt)/dt


        # init 
        t = np.arange(i_max)*dt + dt/2
        vals = curve_npp(t, *args, )
        cmf = np.cumsum(vals)
        cmf /= cmf[-1]

        # return the vector alone?
        if return_type == "vector":
            return cmf
        
        
        ##  CONTINUE WITH ARRAY BUILDING IF NECESSARY

        arr_out = np.zeros((len(t), 2))
        arr_out[:, 0] = t
        arr_out[:, 1] = cmf

        if return_type == "array_collapsed_to_tp":
            arr_out = self.get_assumed_steady_state_sem_mass_collapse(
                arr_out,
                return_type = "array",
            )

        return arr_out



    def get_assumed_steady_state_sem_mass_collapse(self,
        arr_out: np.ndarray,
        field_t: str = "t",
        field_t_new: str = "t_new",
        field_val: str = "value",
        return_type: str = "array",
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Do the collapse to return the collapsed mass array. Valid return_type
            values are "array" or "dataframe"
        """

        df_out = pd.DataFrame(
            arr_out, 
            columns = [field_t, field_val],
        )
        df_out[field_t_new] = np.floor(df_out[field_t])

        df_out = (
            df_out[[field_t_new, field_val]]
            .groupby([field_t_new])
            .agg(
                {
                    field_t_new: "first",
                    field_val: "max",
                }
            )
            .reset_index(drop = True, )
        )

        df_out = (
            df_out
            if return_type == "dataframe"
            else df_out.to_numpy()
        )

        return df_out
    


    def get_assumed_steady_state_sem_p0(self,
        out: np.ndarray,
    ) -> Union[int, None]:
        """Find the starting point to iterate until steady state is reached.
        """

        # iterate over window
        i_max = None
        i_min = None
        
        for i, val in enumerate(out):
            if i == 0:
                df = 0
                continue
        
            df_last = df
            df = val - out[i - 1]
        
            # (np.sign(df) != np.sign(df_last))
            if ((df < 0) & (df_last > 0)) & (i > 1):
                i_max = i
                
            elif ((df > 0) & (df_last < 0)) & (i > 1):
                i_min = i
                

        # need to cross the max first too
        if (i_max is None) or (i_min is None):
            return None

        return i_min



    def get_assumed_steady_state_sem_p1(self,
        out: np.ndarray,
        i_0: int,
        tol: float = 0.0000001,
    ) -> Union[int, None]:
        """Iterate from i_0 until steady state
        """

        i = i_0
        while (np.abs(out[i]) > tol) & (i < len(out) - 1):
            i += 1

        return i



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
        """Retrieve a curve for parameter estimation
        """

        if isinstance(curve, NPPCurve) | callable(curve):
            return curve
        
        out = self.dict_curves.get(curve, return_on_none)

        return out



    def get_dt(self,
        dt: Union[float, None] = None,
    ) -> float:
        """Retrieve dt
        """
        dt = self.dt if not sf.isnumber(dt) else float(dt)

        return dt
        


    def get_targets(self,
        vec_targets: Union[np.ndarray, None] = None,
    ) -> Union[np.ndarray, None]:
        """Retrieve the targets
        """
        vec_targets = self.targets if vec_targets is None else vec_targets

        return vec_targets
    


    def get_widths(self,
        vec_widths: Union[np.ndarray, None] = None,
    ) -> Union[callable, None]:
        """Retrieve the widths
        """
        vec_widths = self.widths if vec_widths is None else vec_widths

        return vec_widths
    


    def objective(self,
        vec_params: np.ndarray,
        curve: Union[callable, str, NPPCurve, ],
        vec_targets: Union[np.ndarray, None] = None,
        vec_widths: Union[np.ndarray, None] = None,
    ) -> float:
        """Find distance between mean value of integrals and seqestration 
            factors using parameters b

        Function Arguments
        ------------------
        vec_params : np.ndarray
            Vector of parameters to pass to fit function
            - for Gamma: k0, k1, k2
            - for SEM: b, c, d (a is specified as limiting value for primary
                since since lim f_SEM(t) as t -> inf is a)
        curve : Union[callable, str, NPPCurve, ]
            Curve to use:
                * "gamma"
                * "sem"
    
            
        Keyword Arguments
        -----------------
        vec_targets : Union[np.ndarray, None]
            Optional targets to pass. Defaults to self-defined
            - ordered array of young secondary, old secondary, and 
                primary growth factors
        vec_widths : Union[np.ndarray, None]
            Optional integration widths to pass
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




###################################
###                             ###
###    SOME SIMPLE FUNCTIONS    ###
###                             ###
###################################


def is_npp_curve(
    obj: Any,
) -> bool:
    """Check if obj is a SISEPUEDE NPPCurve
    """
    out = hasattr(obj, "is_npp_curve")
    uuid = getattr(obj, "_uuid", None)
    
    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out



def is_npp_curves(
    obj: Any,
) -> bool:
    """Check if obj is a SISEPUEDE NPPCurves
    """
    out = hasattr(obj, "is_npp_curves")
    uuid = getattr(obj, "_uuid", None)
    
    out &= (
        uuid == _MODULE_UUID
        if uuid is not None
        else False
    )

    return out
