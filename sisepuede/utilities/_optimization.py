#import cyipopt
import numpy as np
import qpsolvers
import scipy.optimize as sco
import warnings
from typing import *

import sisepuede.utilities._toolbox as sf



######################
#    SOME GLOBALS    #
#####################3#`

class ConstraintError(Exception):
    pass

class ShapeError(Exception):
    pass


# 
#
#

class QAdjuster:
    """Adjust a transition matrix Q to match requirements from land use 
        reallocation factor.

    Uses a Minimum Cost approach to minimize the distance between:

        - prevalence targets (highest cost, applies only to targets that aren't
            ignored)
        - transitions on diagonal (mid-cost)
        - transitions off diagonal (lowest-cost)

        
    Optional Arguments
    ------------------
    flag_ignore : float
        Flag to signify target classes that can be ignored in optimization
    min_solveable_diagonal : float
        Optional specification of minimum allowable diagonal in solved
        transition matrices.
        NOTE: This is called "solveable" because existing, unadjustd on-diagonal 
            transitions that are lower than min_solveable_diagonal are 
            preserved.
    """
    
    def __init__(self,
        flag_ignore: float = -999.,
        min_solveable_diagonal: float = 0.98,
    ) -> None:
        
        self._initialize_properties(
            flag_ignore = flag_ignore,
            min_solveable_diagonal = min_solveable_diagonal,
        )

        return None

    

    def __call__(self,
        *args,
        **kwargs,
    ) -> Union[np.ndarray, None]:
        
        out = self.correct_transitions(
            *args,
            **kwargs,
        )

        return out



    def _initialize_properties(self,
        flag_ignore: float = -999.,
        min_solveable_diagonal: float = 0.98,
    ) -> None:
        """Initialize key properties, including:

            * self.flag_ignore
            * self.min_solveable_diagonal
        """

        # checks
        flag_ignore = (
            -999. 
            if not sf.isnumber(flag_ignore) 
            else float(flag_ignore)
        )
        min_solveable_diagonal = (
            0.98 
            if not sf.isnumber(min_solveable_diagonal) 
            else max(min(min_solveable_diagonal, 1.0), 0.0)
        )


        ##  SET PROPERTIES

        self.flag_ignore = flag_ignore
        self.min_solveable_diagonal = min_solveable_diagonal

        return None
    


    #######################
    #    KEY FUNCTIONS    #
    #######################

    def clean_and_reshape_solver_output(self,
        Q: np.ndarray,
        Q_solve: np.ndarray,
        thresh_to_zero: float = 0.00000001,
        **kwargs,
    ) -> np.ndarray:
        """Reshape the output from QAdjuster.solve to match the original 
            transition matrix shape. Additionally, chop near-zero elements to 
            zero and ensure proper normalization. 
        """
        Q_solve_out = Q_solve.copy()
        Q_solve_out[np.abs(Q_solve_out) <= thresh_to_zero] = 0.0
        Q_solve_out = sf.check_row_sums(Q_solve_out.reshape(Q.shape))

        return Q_solve_out



    def f_obj_mce(self,
        x: np.ndarray,
        p_0: np.ndarray, # prevalence vector at time 0
        p_1: np.ndarray, # prevalence vector at time 1
        flag_ignore: Union[float, None] = None,
    ) -> float:
        """Minimize the distance between the new matrix and the original 
            transition matrix for the Minimize Calibration Error (MCE) approach

        Function Arguments
        ------------------
        x : np.ndarray
            Variable vector
        p_0 : np.ndarray
            Initial prevalence
        p_1 : np.ndarray
            Next-step prevalence

        Keyword Arguments
        -----------------
        flag_ignore : Union[float, None]
            Optional flag to use to ignore classes in the target
        """

        # build objective function
        n = len(p_0)
        obj = np.dot(p_0, x.reshape((n, n)))
        obj = ((obj - p_1)**2)

        # some weights--no change in objective for ignored targets
        vec = np.ones(obj.shape)
        if sf.isnumber(flag_ignore):
            vec[p_1 == flag_ignore] = 0

        obj = np.dot(obj, vec)


        return obj

    

    def grad_mce(self,
        x: np.ndarray,
        p_0: np.ndarray,
        p_1: np.ndarray,
        flag_ignore: Union[float, None] = None,
    ) -> np.ndarray:
        """Generate the gradient vector for f_obj_mce()

        Function Arguments
        ------------------
        x : np.ndarray
            Variable vector
        p_0 : np.ndarray
            Initial prevalence
        p_1 : np.ndarray
            Next-step prevalence
        
        Keyword Arguments
        -----------------
        flag_ignore : Union[float, None]
            Optional flag to use to ignore classes in the target
        """

        n = p_0.shape[0]

        # initialize a matrix and gradient vector
        Q_cur = x.reshape((n, n))
        vec_grad = np.zeros(n**2).astype(float)

        area = p_0.sum()
        ignore = sf.isnumber(flag_ignore)
        
        # iterate 
        for k in range(n**2):
            # column and row in Q
            j = k%n
            i = int((k - j)/n)

            val = 2*p_0[i]*(p_0.dot(Q_cur[:, j]) - p_1[j])
            if ignore:
                val *= int(p_1[j] != flag_ignore)

            val /= area**2

            vec_grad[k] = val

        return vec_grad
    


    def f_obj_hess(self, 
        x: np.ndarray,
        x_try: np.ndarray,
    ) -> np.ndarray:
        """Set the Hessian for the objective function
        """
        out = np.diag(2*np.ones(len(x)))

        return out
    


    def flat_index(self,
        i:int, 
        j:int, 
        n:int,
    ) -> int:
        """For matrix indices i, j in an n x n matrix, get the indices of 
            elements in the flat vector of length n^2.
        """
        out = i*n + j
        
        return out



    def flat_index_inverse(self,
        k:int,
        n:int,
    ) -> int:
        """For indices of elements in a flat vector of length n^2, get the 
            matrix indices of original elements.
        """
        #n_root = Int64(n^0.5)
        col = k%n
        row = int((k - col)/n)
            
        out = (row, col)
        
        return out




    def get_constraint_coeffs_error(self,
        matrix_0: np.ndarray,
        *,
        error_type: str = "additive",
        inds_absorp: Union[list, None] = None,
        infimum: float = 0.0,
        infimum_diag: float = 0.99,
        max_error: float = 0.01,
        preserve_zeros: bool = True,
        supremum: float = 0.99999,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a matrix of coefficients used to ensure that values do not
            exceed some error. Returns

            (
                mat_coeffs, # n x n^2 matrix
                vec_inf,
                vec_sup,
            )


        Function Arguments
        ------------------
        - matrix_0: initial transition matrix (n x n)

        Keyword Arguments
        -----------------
        - error_type: one of the following:
            * "additive": error as a +/-
            * "scalar": error as a fraction of base value
        - inds_absorp: optional indices to pass as valid absorption states
        - infimum: minimum value allowed
        - infimum_diag: minimum value allowed along diagonal
        - max_error: maximum error fraction
        - preserve_zeros: preserve zeros in the matrix? if False, additive 
            errors can lead to the introduction of transitions where 0s 
            previously were found
        - supremum: max value allowed
        """

        error_type = (
            "additive" 
            if error_type not in ["additive", "scalar"] 
            else error_type
        )
        
        inds_absorp = [] if not sf.islistlike(inds_absorp) else list(inds_absorp)
        n = matrix_0.shape[0]
        

        # initialize output matrix
        mat_coeffs = np.zeros((n**2, n**2))
        vec_inf = np.zeros(n**2)
        vec_sup = np.zeros(n**2)

        for k in range(n**2):
            # get indices from the iterator
            j = int(k%n)
            i = int((k - j)/n)

            mat_coeffs[k, k] = 1.0
            inf_cur = infimum_diag if (i == j) else infimum

            # ensure the inf/sup don't interfere with current estimates 
            inf_cur = min(inf_cur, matrix_0[i, j])
            sup_cur = supremum if not (i in inds_absorp) else 1.0#max(supremum, matrix_0[i, j])

            # set inf/sup based on current value
            if error_type == "additive":
                # get infimum
                inf_k = (
                    matrix_0[i, j]
                    if (matrix_0[i, j] == 0) & preserve_zeros
                    else max(matrix_0[i, j] - max_error, inf_cur)
                )

                # get supremum
                sup_k = (
                    matrix_0[i, j]
                    if (matrix_0[i, j] == 0) & preserve_zeros
                    else min(matrix_0[i, j] + max_error, sup_cur)
                )

                vec_inf[k] = inf_k
                vec_sup[k] = sup_k
                    
            elif error_type == "scalar":
                vec_inf[k] = max(matrix_0[i, j]*(1 - max_error), inf_cur)
                vec_sup[k] = min(matrix_0[i, j]*(1 + max_error), supremum)



        out = (
            mat_coeffs,
            vec_inf,
            vec_sup,
        )

        return out



    def get_constraint_coeffs_max_area(self,
        x_0: np.ndarray,
        vector_bounds: np.ndarray,
        flag_ignore: float,
    ) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        """Generate maximum area constraint coefficients (B_1) for land use 
            adjustment optimization

        Returns a tuple with A and b for inequality (Ax <= b)
            (
                A,  # matrix with dims (k, n^2), where k is number of bounds
                b,  # bounds
            )

        If no valid bounds are specified, returns None.
        

        Function Arguments
        ------------------
        - x_0: prevalence vector
        - vector_bounds: vector includng bounds to apply
        - flag_ignore: flag in vector_bounds used 
        """

        # if there are no constraints, don't bother including
        w = np.where(vector_bounds != flag_ignore)[0]
        if len(w) == 0:
            return None

        n = x_0.shape[0]
        n_w = len(w)
        
        # initialize output matrices - start with inequality
        A_coeffs_ineq = np.zeros((n_w, n**2))
        b_ineq = vector_bounds[w]

        # add constraint on upper bound
        for i, ind in enumerate(w):
            inds = np.arange(n)*n + ind
            A_coeffs_ineq[i, inds] = x_0


        ##  BUILD OUTPUTS

        out = (
            A_coeffs_ineq,
            b_ineq,
        )

        return out



    def get_constraint_coeffs_min_area(self,
        matrix_0: np.ndarray,
        x_0: np.ndarray,
        vector_bounds: np.ndarray,
        flag_ignore: float,
    ) -> Union[Dict[str, Union[np.ndarray, None]], None]:
        """Generate minimum area constraint coefficients (B_0) for land use 
            adjustment optimization

        Returns a tuple with A and b for inequality (Ax <= b
            (
                A,  # matrix with dims (k, n^2), where k is number of bounds
                b,  # bounds
            )

        If no valid bounds are specified, returns None.
        

        Function Arguments
        ------------------
        - matrix_0: initial transition matrix (n x n)
        - x_0: prevalence vector
        - vector_bounds: vector includng bounds to apply
        - flag_ignore: flag in vector_bounds used 
        """

        # if there are no constraints, don't bother including
        w = np.where(vector_bounds != flag_ignore)[0]
        if len(w) == 0:
            return None

        n = matrix_0.shape[0]
        n_w = len(w)
        
        # initialize output matrices - start with inequality
        A_coeffs_ineq = np.zeros((n_w, n**2))
        b_ineq = np.zeros(n_w)

        
        for i, ind in enumerate(w):
            # check if it's necessary to stop the loss of land
            constraint_low = vector_bounds[ind]
            stop_loss = x_0[ind] <= constraint_low

            # force the class remaining probability = 1 by setting it to >= 1 if the current area is below the minimum
            A_coeffs_ineq[i, ind*(n + 1)] = -1 if stop_loss else -x_0[ind]
            b_ineq[i] = -1 if stop_loss else -constraint_low


        ##  BUILD OUTPUTS

        out = (
            A_coeffs_ineq,
            b_ineq,
        )

        return out
    


    def get_constraint_coeffs_min_diag(self,
        matrix_0: np.ndarray,
        lb_diag_artificial: float,
    ) -> Tuple[np.ndarray]:
        """Generate coefficients to promote a minimum value on the diagonal. 
            Prevents the solver from artificially reallocating entire land use
            classes.

            NOTE: Any existing constraints on the diagonal below 
            lb_diag_artificial are considered lower bounds for that class.

        Returns a tuple of the form
        
        (
            A,  # matrix with dims (n, n^2)
            b,  # vector with dim (n, )
        )

        Function Arguments
        ------------------
        matrix_0 : np.ndarray
            Initial transition matrix (n x n)
        lb_diag_artificial : float
            Lower bound on values on the diagoal; it is called "artificial"
            because it only applies to those whose unadjusted transitions are
            greater than this. 

            For each diagonal element x_{ii}, the lower bound is set as

            min(x_{ii}, lb_diag_artifical, ) 
        """

        n = matrix_0.shape[0]

        # initialize output matrices
        A = np.zeros((n, n**2))
        b = np.zeros(n)

        # update constraints
        for i in range(n):
            A[i, i*(n + 1)] = -1
            b[i] = -min(matrix_0[i, i], lb_diag_artificial, )


        out = (
            A, 
            b,
        )

        return out



    def get_constraint_coeffs_preserve_zeros(self,
        matrix_0: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """Generate a matrix of coefficients used to ensure that values in
            matrix_0 that are 0 also zero in the solution. Returns a tuple in 
            the form of

            (
                A,  # matrix with dims (n, n^2)
                b,  # vector with dim (n, )
            )

        Function Arguments
        ------------------
        matrix_0 : np.ndarray
            Initial transition matrix (n x n)
        """

        n = matrix_0.shape[0]

        # initialize output matrix
        A = np.zeros((n, n))
        A[matrix_0 == 0] = 1.0
        A = np.array([A.flatten()])

        b = np.zeros(1)

        out = (
            A,
            b,
        )

        return out
    


    def get_constraint_coeffs_prohibited_transitions(self,
        matrix_0: np.ndarray,
        prohibited_transitions: Union[List[Tuple], None],
    ) -> Tuple[np.ndarray]:
        """Generate a matrix of coefficients used to ensure that any transitions
            that are not allowed are not passed. Returns a tuple of the form

            (
                A,  # matrix with dims (n, n^2)
                b,  # vector with dim (n, )
            )

        Function Arguments
        ------------------
        matrix_0 : np.ndarray
            Initial transition matrix (n x n)
        prohibited_transitions : Union[List[Tuple], None]
            If specifying prohibited transitions from i -> j, specify here using
            a list of tuples, e.g., [(i0, j0), (i1, j1), ...]. 
        """

        if not isinstance(prohibited_transitions, list):
            return None
        
        # init
        n = matrix_0.shape[0]
        n_pt = len(prohibited_transitions)
        G = np.zeros((n_pt, n**2))
        h = np.zeros(n_pt, )
        
        k = 0
        for tup in prohibited_transitions:

            # some checks
            if not isinstance(tup, tuple): continue
            if len(tup) != 2: continue

            ind = self.flat_index(*tup, n,)
            G[k, ind] = 1
            h[k] = 0

            k += 1 

        # if no specifications are successful, no constraint it set
        if k == 0:
            return None

        # otherwise, return matrices for valid prohibitions
        out = (
            G[0:k],
            h[0:k],
        )

        return out
    
    
    
    def get_constraint_coeffs_row_stochastic(self,
        matrix_0: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """Generate a matrix of coefficients used to ensure that the resulting
            matrix is row-stochastic

        Returns a tuple of the form
        
        (
            A,  # matrix with dims (n, n^2)
            b,  # vector with dim (n, )
        )

        Function Arguments
        ------------------
        matrix_0 : np.ndarray
            Initial transition matrix (n x n)
        """

        n = matrix_0.shape[0]

        # initialize output matrix
        A = np.zeros((n, n**2))

        for i in range(n):
            b0 = i*n
            b1 = (i + 1)*n
            A[i, b0:b1] = 1

        
        # equal one
        b = np.ones(n)

        out = (
            A, 
            b,
        )

        return out



    def get_costs(self,
        matrix_0: np.ndarray,
        costs_qij: Union[dict, np.ndarray, None] = None,
        costs_x: Union[dict, np.ndarray, None] = None,
        cost_basic_deault: Union[float, int] = 1.0,
        cost_factor_qii_default: Union[float, int, dict] = 10.0,
        cost_factor_prev_default: Union[float, int, None] = None,
        **kwargs,
    ) -> np.ndarray:
        """Get the costs specified for each qij and the prevalence vector 

        Function Arguments
        ------------------
        matrix_0 : np.ndarray
            Unadjusted transition matrix

        Keyword Arguments
        -----------------
        costs_qij : Union[dict, np.ndarray, None]
            Specification of transition costs directly, either as a 
            dictionary or as a numpy array (n x n). Dictionary is used to 
            overwrite defaults with (row, column) index tuples as keys mapping 
            to costs as values
        costs_x : Union[dict, np.ndarray, None]
            Specification of prevalence costs directly, either as a 
            dictionary or as a numpy array (n x 1). Dictionary is used to
            overwrite defaults with the index as a key mapping to the costs as 
            a value
        cost_basic_default : Union[float, int]
            Basic cost to use for individual land use transitions. If specified 
            as a dictionary, must 
        cost_factor_qii_default : Union[float, int, dict]
            Scalar applied to cost_basic to create costs on diagonals as default
        cost_factor_prev_default : Union[float, int, None]
            Default factor to use for prevalence costs. If None, is calculated 
            as 10 times the sum of all transition costs.         
        """
        ##  GET COSTS FOR TRANSITION ERRORS

        if isinstance(costs_qij, np.ndarray):
            if costs_qij.shape == matrix_0.shape:
                costs_transition = costs_qij
            else:
                costs_qij = None
        
        if not isinstance(costs_qij, np.ndarray):
            costs_transition = cost_basic_deault*np.ones(matrix_0.shape)
            np.fill_diagonal(costs_transition, cost_basic_deault*cost_factor_qii_default)

            # fill values?
            if isinstance(costs_qij, dict):
                for k, v in costs_qij.items():

                    cont = not isinstance(k, tuple)
                    cont |= (len(k) != 2) if not cont else cont
                    cont |= not sf.isnumber(v)

                    if cont:
                        continue
                    
                    # try to assign; if indices are invalid, skip
                    i, j = k
                    try:
                        costs_transition[i, j] = v
                    except:
                        continue

        costs_transition = costs_transition.flatten()
        
        
        ##  GET COSTS FOR PREVALENCE ERROR

        cost_factor_prev_default = (
            cost_factor_prev_default
            if sf.isnumber(cost_factor_prev_default)
            else 10000*costs_transition.sum()
        )

        if isinstance(costs_x, np.ndarray):
            if costs_x.shape == (matrix_0.shape[0], ):
                costs_prevalence = costs_x
            else:
                costs_x = None
        
        if not isinstance(costs_x, np.ndarray):
            costs_prevalence = cost_factor_prev_default*np.ones(matrix_0.shape[0])

            # fill values?
            if isinstance(costs_x, dict):
                for k, v in costs_x.items():

                    cont = not sf.isnumber(k, integer = True, )
                    cont |= not sf.isnumber(v)
                    if cont:
                        continue
                    
                    # try to assign; if indices are invalid, skip
                    try:
                        costs_prevalence[k] = v
                    except:
                        continue
        
        out = (
            costs_transition,
            costs_prevalence,
        )

        return out
    

    def get_problem_components_objective(self,
        matrix_0: np.ndarray,
        x_0: np.ndarray,
        x_target: np.ndarray,
        return_component_matrices: bool = False,
        **kwargs,
    ) -> Tuple:
        """Generate objective value components for QP in land use optimization

        Returns a tuple of the form

        (
            M,
            c,
        )
        

        Function Arguments
        ------------------
        matrix_0 : np.ndarray
            Initial transition matrix (n x n)
        x_0 : np.ndarray
            Prevalence vector
        x_target : np.ndarray
            Target prevalence vector. Classes without a target can be ignored 
            using flag_ignore

        Keyword Arguments
        -----------------
        return_component_matrices : bool
            If True, returns a tuple in the form:
            (
                M_prevalence,   # prevalence distance quadratic component
                c_prevalence,   # prevalence distance linear component
                M_transitions,  # transition matrix distance quadratic component
                c_transitions,  # transition matrix distance linear component
            )

        - **kwargs : 
            Passed to get_costs
        """

        ##  GET OBJECTIVE VALUES

        # get costs to impose
        costs_transition, costs_prevalence = self.get_costs(
            matrix_0, 
            **kwargs,
        )

        # M and c in the objective function, used to calculate euclidean distance
        M_prev, c_prev = self.get_qp_component_vectors_euclidean_prevalence(
            x_0, 
            x_target, 
            weights = costs_prevalence,
        )

        # set the transition costs
        M_tran = np.diag(costs_transition)
        c_tran = -2*costs_transition*matrix_0.flatten()

        M = 2*(M_prev + M_tran)# + np.diag(0.0000001 * np.ones(n**2))
        c = c_tran + c_prev

        out = (
            (M, c, )
            if not return_component_matrices
            else (M_prev, c_prev, M_tran, c_tran, )
        )

        return out


    
    def get_problem_components_constraints(self,
        matrix_0: np.ndarray,
        x_0: np.ndarray,
        vec_infima: np.ndarray,
        vec_suprema: np.ndarray,
        flag_ignore: float,
        prohibited_transitions: Union[List[Tuple[int]], None] = None,
        **kwargs,
    ) -> Tuple:
        """Generate constraints for land use optimization QP. Returns a tuple in 
            the following form:

            (
                A,
                b,
                G,
                h,      # 
                v_0,    # vector of lower bounds
                v_1,    # vector of upper bounds
            )

        Function Arguments
        ------------------
        matrix_0 : np.ndarray
            Initial transition matrix (n x n)
        x_0 : np.ndarray
            Prevalence vector
        vec_infima : np.ndarray
            Vector specifying class infima; use flag_ignore to set no infimum 
            for a class
        vec_suprema : np.ndarray
            Vector specifying class suprema; use flag_ignore to set no supremum 
            for a class
        flag_ignore : float
            Float specifying

        Keyword Arguments
        -----------------
        prohibited_transitions : bool
            Optional list of transitions from i -> j that are prohibited. If
            None, no restriction is included.
        **kwargs : 
            Ignored
        """

        ##  GET CONSTRAINTS

        constraint_min_area = self.get_constraint_coeffs_min_area(
            matrix_0,
            x_0,
            vec_infima,
            flag_ignore,
        );

        constraint_max_area = self.get_constraint_coeffs_max_area(
            x_0,
            vec_suprema,
            flag_ignore,
        );


        constraint_row_stochastic = self.get_constraint_coeffs_row_stochastic(matrix_0, )
        constraint_preserve_zeros = self.get_constraint_coeffs_preserve_zeros(matrix_0, )
        constraint_prohibited_transitions = self.get_constraint_coeffs_prohibited_transitions(
            matrix_0,
            prohibited_transitions,
        )

        global constraint_prohibited_transitions0
        constraint_prohibited_transitions0 = constraint_prohibited_transitions

        ##  SET EQ CONSTRAINTS

        A = constraint_row_stochastic[0]
        b = constraint_row_stochastic[1]


        ##  BUILD INEQ (G and h in QPsolve for some reason)

        G = []
        h = []

        constraint_list = [
            constraint_min_area,
            constraint_max_area,
            constraint_preserve_zeros,
            constraint_prohibited_transitions
        ]


        for i, constraint in enumerate(constraint_list):
            if constraint is None: continue
                
            G.append(constraint[0])
            h.append(constraint[1])


        G = np.concatenate(G)
        h = np.concatenate(h)

        # return outputs as a tuple
        out = (
            A,
            b,
            G,
            h
        )

        return  out
    


    def get_qp_component_vectors_euclidean_prevalence(self,
        x_0: np.ndarray,
        x_1: np.ndarray,
        weights: Union[np.ndarray, None] = None,
    ) -> Tuple[np.ndarray]:
        """Get the objective function component matrix and vectors necessary for
            euclidean distance between a transition matrix estimate and a 
            target. Includes ability to weight individual classes.

        NOTE: does not include the 1/2 coefficient in a standard QP
            
        Function Arguments
        ------------------
        - x_0: initial state vector (1 x n)
        - x_1: target state vector (1 x n)

        Keyword Arguments
        -----------------
        - weights: optional vector of weights to place on prevalence classes
        """
        
        n = x_0.shape[0]
        n2 = n**2
        
        # build vector c (first degree terms) and matrix M (second degree)
        vec_c = np.zeros(n2)
        M = np.zeros((n2, n2))
        weights = (
            np.ones(n) 
            if not isinstance(weights, np.ndarray)
            else weights
        )

        
        ##  ITEATE OVER EACH ROW/COL TO ADD VALUES WHERE NEEDED
        
        # build vec_c
        for k in range(n2):
            i, j = self.flat_index_inverse(k, n)
            vec_c[k] = -2*x_0[i]*x_1[j]*weights[j]


        """
        build matrix - this algebra sets up the Euclidean distance

            (obj) sum_{i <= n} (u.dot(Q[:,j]) - v[j])^2

        between Q^Tu and v as the objective function, where

            x[k] = Q[i, j], k = i*n + j

        """

        for k1 in range(n2):
            for k2 in range(n2):

                # get rows/columns
                row_1, col_1 = self.flat_index_inverse(k1, n)
                row_2, col_2 = self.flat_index_inverse(k2, n)

                # skip if columns are not the same
                if (col_1 != col_2):
                    continue

                # set values along diagonal
                if k1 == k2:
                    M[k1, k2] = (x_0[row_1]**2)*weights[col_1]
                    continue

                # otherwise, set to product of initial states associated with row
                M[k1, k2] = x_0[row_1]*x_0[row_2]*weights[col_1]

        
        out = (M, vec_c)
        
        return out





    ##########################
    #    SOLVER FUNCTIONS    #
    ###########################

    def solve(self,
        Q: np.ndarray,
        x_0: np.ndarray, 
        x_target: np.ndarray, 
        vec_infima: np.ndarray,
        vec_suprema: np.ndarray,
        flag_ignore: float,
        *,
        #perturbation_diag: float = 0.000001,
        prohibited_transitions: Union[List[Tuple[int]], None] = None,
        return_unadj_on_infeas: bool = True,
        solver: str = "quadprog",
        stop_on_error: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Solve the Minimize Calibration Error (MCE) problem; attempts in two
            stages:

            (1) direct solution of the quadratic program;
            (2) numerical attempt using scipy.optimize.minimize

            Support function for correct_transitions()

        NOTE: If return_unadj_on_infeas is False, then returns None.

        Function Arguments
        ------------------
        Q : np.ndarray
            Unadjusted transition matrix (2-d, n x n)
        x_0 : np.ndarray
            Initial prevalence (1-d, n x 1)
        x_target : np.ndarray
            Next-step prevalence vector (can ignore classes; 1-d, n x 1)
        vec_infima : np.ndarray
            vector specifying class infima; use flag_ignore to set no infimum 
            for a class (1-d, n x 1)
        vec_suprema : np.ndarray
            Vector specifying class suprema; use flag_ignore to set no supremum 
            for a class (1-d, n x 1)
        flag_ignore : float
            Flag in vector_bounds used 

        Keyword Arguments
        -----------------
        perturbation_diag : float
            Perturbation to apply to the diagonal (positive) to ensure P is 
            positive definite, a requirement for some solvers.
        prohibited_transitions : bool
            Optional list of transitions from i -> j that are prohibited. If
            None, no restriction is included.
        return_unadj_on_infeas : bool
            If both 
        solver : str
            Valid solver passed to qpsolvers.solve_qp()
        stop_on_error : bool
            Stop if errors occur in solutions?
        verbose : bool
            Print all solver output?
        **kwargs :
            Passed to get_constraint_coeffs_error() and sco.minimize()
        """

        # try to retrieve objective components
        try:
            M, c = self.get_problem_components_objective(
                Q,
                x_0,
                x_target,
                **kwargs,
            )

        except Exception as e:
            msg = f"Error retrieving objective values in QAdjuster: {e}"
            raise RuntimeError(msg)
        

        # try to retrieve constraint components
        try:
            A, b, G, h = self.get_problem_components_constraints(
                Q,
                x_0,
                vec_infima,
                vec_suprema,
                flag_ignore,
                prohibited_transitions = prohibited_transitions,
                **kwargs,
            )

        except Exception as e:
            msg = f"Error retrieving constraints in QAdjuster: {e}"
            raise RuntimeError(msg)

        """
        global dict_out
        dict_out = {
            "M": M,
            "c": c,
            "A": A,
            "b": b,
            "G": G,
            "h": h,
            "solver": solver,
            "Q": Q,
            "x_0": x_0,
            "x_target": x_target,
        }
        """;

        try:

            sol = qpsolvers.solve_qp(
                M, 
                c, 
                A = A,
                b = b,
                G = G,
                h = h,
                lb = np.zeros(M.shape[0]),
                ub = np.ones(M.shape[0]),
                solver = solver,
                verbose = verbose,
            )

            # print("QP SUCCEEDED!")

        except Exception as e:
            msg = f"Error trying to solve 'Minimize Calibration Error' problem in QCorrector using qpsolvers.solve_qp: {e}."
            if stop_on_error:
                raise RuntimeError(msg)
            
            warnings.warn(f"{msg} Trying scipy.optimize.minimize...")
        

        if sol is not None:
            sol = self.clean_and_reshape_solver_output(
                Q,
                sol,
                **kwargs, # optional passing of thresh_to_zero
            )

            return sol



        ##  TRY SCO.MINIMIZE

        try:

            _, kwargs_pass = sf.get_args(
                sco.minimize, 
                include_defaults = True,
            )
            kwargs_pass = dict((k, v) for k, v in kwargs.items() if k in kwargs_pass.keys())

            sol = self.solve_sco_min(
                Q,
                x_0,
                x_target,
                A,
                b,
                G,
                h,
                flag_ignore = flag_ignore,
                min_diag = kwargs.get("min_diag"),
                verbose = verbose,
                **kwargs_pass,
            )

            # print("SCO SUCCEEDED!")

        except Exception as e:
            if stop_on_error:
                msg = f"Error trying to solve 'Minimize Calibration Error' problem in QCorrector using scipy.optimize.minimize: {e}"
                raise RuntimeError(msg)

            # print(f"EVERYTHING FAILED {e}! RETURNING ORIGINAL")
            sol = None
        

        # reshape output and check sums
        if sol is not None:
            sol = self.clean_and_reshape_solver_output(
                Q,
                sol,
                **kwargs,
            )

            return sol
        

        ##  RETURN OUTPUT

        out = Q if ((sol is None) & return_unadj_on_infeas) else sol

        return out



    def solve_sco_min(self,
        Q: np.ndarray,
        x_0: np.ndarray, 
        x_target: np.ndarray, 
        A: np.ndarray,
        b: np.ndarray,
        G: np.ndarray,
        h: np.ndarray,
        false_area: float = 10000.,
        flag_ignore: Union[float, None] = None,
        min_diag: Union[float, None] = None,
        verbose: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Call scipy.optimize.minimize to attempt to solve the problem. 

        Function Arguments
        ------------------
        Q : np.ndarray
            Unadjusted transition matrix (2-d, n x n)
        x_0 : np.ndarray
            Initial prevalence (1-d, n x 1)
        x_target : np.ndarray
            Next-step prevalence vector (can ignore classes; 1-d, n x 1)
        A : np.ndarray
            EQUALITY constraint matrix (Ax == b)
        b : np.ndarray
            EQUALITY constraints (Ax == b)
        G : np.ndarray
            INEQUALITY constraint matrix (Gx <= h)
        h : np.ndarray
            INEQUALITY constraints (Gx <= h)
        flag_ignore : float
            Flag in vector_bounds used 

        Keyword Arguments
        -----------------
        false_area : float
            Dummy area to use to inflate target vectors for numerical precision
        flag_ignore : Union[float, None]
            Flag in vector_bounds used 
        min_diag : Union[float, None]
            Minimum diagonal attributable to solver change; if an existing 
            (unadjusted) diagonlal transition is below this level, then that is 
            used as the minimum. If None, defaults to 
            self.min_solveable_diagonal
        verbose : bool
            Print all output? (CURRENTLY NOT PASSED)
        **kwargs : 
            Passed to sco.minimimze()
        """

        ##  INITIALIZATION

        min_diag = (
            self.min_solveable_diagonal 
            if not sf.isnumber(min_diag) 
            else min(max(min_diag, 0.0), 1.0)
        )

        # prepare prevalence vectors
        x_0_cur = false_area*x_0
        x_target_cur = x_target.copy()
        x_target_cur[x_target_cur != flag_ignore] *= false_area

        # define functions to pass
        def _obj_func_cur(
            x: np.ndarray,
        ) -> float:
            """Passable function to sco.minimize for current iteration.
            """
            
            out = self.f_obj_mce(
                x,
                x_0_cur,
                x_target_cur,
                flag_ignore = flag_ignore,
            )
            
            return out
        


        def _obj_grad_cur(
            x: np.ndarray
        ) -> float:
            """Passable function to sco.minimize for current iteration. 
            """
            out = self.grad_mce(
                x,
                x_0_cur,
                x_target_cur,
                flag_ignore = flag_ignore,
            )
            
            return out
        


        ##  GET SOME CONSTRAINTS
        
        # add the min diag
        G_md, h_md = self.get_constraint_coeffs_min_diag(Q, min_diag, )
        G_new = np.concatenate([G, G_md])
        h_new = np.concatenate([h, h_md])

        n = x_0.shape[0]**2
        constraints = [
            sco.LinearConstraint(G_new, ub = h_new, ),
            sco.LinearConstraint(A, lb = b, ub = b, ),
            sco.LinearConstraint(np.identity(n), lb = np.zeros(n), ub = np.ones(n), )
        ]


        ##  FINALLY, CALL MINIMIZE
        
        sol = sco.minimize(
            _obj_func_cur,
            Q.flatten(),
            constraints = constraints,
            jac = _obj_grad_cur,
            **kwargs,
        )

        out = sol.x if sol.success else None

        return out






    



class LivestockDietEstimator:
    """Estimate how livestock split their diet based on crop, residue, and 
        pasture availability. Solves for 10 potential pathways (by index):

        [0] crop residues               (existing)
        [1] crops, cereals              (existing)
        [2] crops, non-cereals          (existing)
        [3] pastures                    (existing)
        [4] crops, cereals              (new)
        [5] crops, non-cereals          (new)
        [6] pastures                    (new)
        [7] crop imports, cereals       (used to satisfy slack)
        [8] crop imports, non-cereals   (used to satisfy slack)
        [9] pasture imports             (e.g., alfalfa, hay--used to satisfy 
                                            slack if present. Generally, this is
                                            higher cost than crop imports)

    Uses a Minimum Cost approach to prioritize dietary pathways over others 
        while constraining dietary fractions for livestock.
    
    NOTE: Conceptual constraint arrays A are oriented as 

        n_lvst x n_pathways = m x n

              |  p_0  |  p_1  |  ...  |  p_n-1  |
        ----
        l_0   |  a_00 |  a_01 |  ...  |  a_0n-1 |

        l_1   |  a_10 |  a_11 |  ...  |  a10n-1 |

        ...

        The vector of unknowns is x \in \mathbb{R}^{mn}.


    Initialization Arguments
    ------------------------
    n_livstock_categories : int
        Number of livestock categories to use
    
    Optional Arguments
    ------------------
    flag_ignore : float
        Flag to signify target classes that can be ignored in optimization

        

    KEY ARRAYS
    ----------
    vec_max_crops_cereals_fraction : np.ndarray
        Vector (n_livstock_categories x 1) of maximum dietary fractions from 
        cereal crops
    vec_max_crops_non_cereals_fraction : np.ndarray
        Vector (n_livstock_categories x 1) of maximum dietary fractions from 
        non-cereal crops
    vec_max_crop_residues_fraction : np.ndarray
        Vector (n_livstock_categories x 1) of maximum dietary fractions from 
        crop residues
    vec_max_pastures_fraction : np.ndarray
        Vector (n_livstock_categories x 1) of maximum dietary fractions from 
        pastures
    vec_min_crops_cereals_fraction : np.ndarray
        Vector (n_livstock_categories x 1) of minimum dietary fractions from 
        cereal crops
    vec_min_crops_non_cereals_fraction : np.ndarray
        Vector (n_livstock_categories x 1) of minimum dietary fractions from 
        non-cereal crops
    vec_min_crop_residues_fraction : np.ndarray
        Vector (n_livstock_categories x 1) of minimum dietary fractions from 
        crop residues
    vec_min_pastures_fraction : np.ndarray
        Vector (n_livstock_categories x 1) of minimum dietary fractions from 
        pastures
    """
    
    def __init__(self,
        n_livstock_categories: int,
        flag_ignore: float = -999.,
    ) -> None:
        
        self._initialize_properties(n_livstock_categories, )
        self._initialize_derivative_properties()

        return None

    

    def __call__(self,
        *args,
        **kwargs,
    ) -> Union[np.ndarray, None]:
        
        out = self.correct_transitions(
            *args,
            **kwargs,
        )

        return out
    


    def _initialize_arrays(self,
    ) -> None:
        """Initialize arrays that are filled and cleared. Initializes the
            following properties:

        """
        return None


    
    def _initialize_derivative_properties(self,
    ) -> None:
        """Initialize some properties that rely on previous instantiation. Sets
            the following properties:

            * self.inds_flat_new_classes

        """

        # get some special indices
        inds_flat_new_classes = self.get_flat_inds_new_classes()
        

        ##  SET PROPERTIES

        self.inds_flat_new_classes = inds_flat_new_classes

        return None



    def _initialize_properties(self,
        n_livstock_categories: int,
    ) -> None:
        """Initialize key vector proporties and the information, such as number
            of livestock categories. Verifies that 
        """

        # set some names
        nm_crop_imports_cereals = "crop_imports_cereals"
        nm_crop_imports_non_cereals = "crop_imports_non_cereals"
        nm_crop_residues = "crop_residues"
        nm_crop_residues_new = "crop_residues_new"
        nm_crops_cereals = "crops_cereals"
        nm_crops_cereals_new = "crops_cereals_new"
        nm_crops_non_cereals = "crops_non_cereals"
        nm_crops_non_cereals_new = "crops_non_cereals_new"
        nm_pasture_imports = "pasture_imports"
        nm_pastures = "pastures"
        nm_pastures_new = "pastures_new"
        
        labels_col = [
            nm_crop_residues,
            nm_crops_cereals,
            nm_crops_non_cereals,
            nm_pastures,
            nm_crop_residues_new,
            nm_crops_cereals_new,
            nm_crops_non_cereals_new,
            nm_pastures_new,
            nm_crop_imports_cereals,
            nm_crop_imports_non_cereals,
            nm_pasture_imports
        ]

        # map to index
        dict_types_to_inds = dict((v, i) for i, v in enumerate(labels_col))

        
        ##  BUILD SOME DICTIONARIES FOR TYPE MAPPING 

        # map indices to associated import indices
        ind_crop_imports_cereals = dict_types_to_inds.get(nm_crop_imports_cereals)
        ind_crop_imports_non_cereals = dict_types_to_inds.get(nm_crop_imports_non_cereals)
        ind_crop_residues = dict_types_to_inds.get(nm_crop_residues)
        ind_crop_residues_new = dict_types_to_inds.get(nm_crop_residues_new)
        ind_crops_cereals = dict_types_to_inds.get(nm_crops_cereals)
        ind_crops_cereals_new = dict_types_to_inds.get(nm_crops_cereals_new)
        ind_crops_non_cereals = dict_types_to_inds.get(nm_crops_non_cereals)
        ind_crops_non_cereals_new = dict_types_to_inds.get(nm_crops_non_cereals_new)
        ind_pasture_imports = dict_types_to_inds.get(nm_pasture_imports)
        ind_pastures = dict_types_to_inds.get(nm_pastures)
        ind_pastures_new = dict_types_to_inds.get(nm_pastures_new)

        # map the standard index to import index
        dict_ind_to_imp_ind = {
            ind_crops_cereals: ind_crop_imports_cereals,
            ind_crops_non_cereals: ind_crop_imports_non_cereals,
            ind_pastures: ind_pasture_imports,
        }

        # map the standard index to "new" index
        dict_ind_to_new_ind = {
            ind_crop_residues: ind_crop_residues_new,
            ind_crops_cereals: ind_crops_cereals_new,
            ind_crops_non_cereals: ind_crops_non_cereals_new,
            ind_pastures: ind_pastures_new,
        }
        dict_new_ind_to_ind_existing = sf.reverse_dict(dict_ind_to_new_ind, )


        # some shortcuts for matrices
        m = n_livstock_categories
        n = len(dict_types_to_inds)


        ##  SET PROPERTIES

        # set some index properties
        for k, v in dict_types_to_inds.items():
            property_name = f"ind_{k}"
            setattr(self, property_name, v)

        # explicit
        self.dict_ind_to_imp_ind = dict_ind_to_imp_ind
        self.dict_ind_to_new_ind = dict_ind_to_new_ind
        self.dict_new_ind_to_ind_existing = dict_new_ind_to_ind_existing
        self.dict_types_to_inds = dict_types_to_inds
        self.labels_col = labels_col
        self.m = m
        self.mn = m*n
        self.n = n
        self.nm_crop_imports_cereals = nm_crop_imports_cereals
        self.nm_crop_imports_non_cereals = nm_crop_imports_non_cereals
        self.nm_crop_residues = nm_crop_residues
        self.nm_crop_residues_new = nm_crop_residues_new
        self.nm_crops_cereals = nm_crops_cereals
        self.nm_crops_cereals_new = nm_crops_cereals_new
        self.nm_crops_non_cereals = nm_crops_non_cereals
        self.nm_crops_non_cereals_new = nm_crops_non_cereals_new
        self.nm_pasture_imports = nm_pasture_imports
        self.nm_pastures = nm_pastures
        self.nm_pastures_new = nm_pastures_new

        return None
    


    def _verify_constraint_arrays(self,
        vec_max: np.ndarray,
        vec_min: np.ndarray,
    ) -> None:
        """Check that vectors are properly specified for use.
        """

        ##  CHECK TYPES
        
        if not isinstance(vec_max, np.ndarray):
            tp = str(type(vec_max))
            raise TypeError(f"Invalid type {tp} for max constraint: must be a NumPy array.")
        
        if not isinstance(vec_min, np.ndarray):
            tp = str(type(vec_min))
            raise TypeError(f"Invalid type {tp} for min constraint: must be a NumPy array.")
        

        ##  CHECK SHAPES

        if vec_max.shape != (self.m, ):
            raise ShapeError(f"Invalid shape for max constraint: must have be a vector of length {self.m}")
        
        if vec_min.shape != (self.m, ):
            raise ShapeError(f"Invalid shape for min constraint: must have be a vector of length {self.m}")


        ##  CHECK CONSTRAINT RELATIONSHIP

        vec_diff = vec_max - vec_min
        if vec_diff.min() < 0:
            raise ConstraintError(f"Error in constraint: max must always be greater than equal than min.")

        return None        
    



    #######################
    #    KEY FUNCTIONS    #
    #######################

    def check_import_frac(self,
        vec_import_frac: Any,          
    ) -> bool:
        """Chcek if the import fraction specification is useful for implementing
            the forced import fraction constraint.
        """
        # add the import fraction constraint?
        imp_constraint = isinstance(vec_import_frac, np.ndarray)
        imp_constraint &= (
            vec_import_frac.shape[0] == len(self.dict_ind_to_imp_ind)
            if imp_constraint
            else False
        )

        return imp_constraint
    


    def collapse_into_categories(self,
        mat: np.ndarray,
    ) -> np.ndarray:
        """Collapse imports and new into base categories of crop residues,
            crop cereals, crop non-cereals, and pastures.
        """

        ##  CHECKS

        if not isinstance(mat, np.ndarray):
            return None
        
        # check if only passing a vector
        return_first_row = False
        if len(mat.shape) == 1:
            return_first_row = True
            mat = np.array([mat])


        ##  INITIALIZE SHORTCUTS

        ind_cc = self.ind_crops_cereals
        ind_cnc = self.ind_crops_non_cereals
        ind_cr = self.ind_crop_residues
        ind_p = self.ind_pastures

        ordered_inds = [
            ind_cr,
            ind_cc,
            ind_cnc,
            ind_p
        ]

        A = np.zeros((mat.shape[0], len(ordered_inds)))

        
        ##  ITERATE

        for j, ind in enumerate(ordered_inds):
            
            vec = mat[:, ind].copy()

            # try to get imports
            ind_imp = self.dict_ind_to_imp_ind.get(ind, )
            if ind_imp is not None:
                vec += mat[:, ind_imp]

            # try to get new
            ind_new = self.dict_ind_to_new_ind.get(ind, )
            if ind_new is not None:
                vec += mat[:, ind_new]

            # add to output
            A[:, j] = vec

        A = A[0] if return_first_row else A

        return A



    def display_as_matrix(self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Take a vector of free variables (or coefficients) and reshape to
            properly-sized matrix.
        """
        
        if x.shape == (self.mn, ):
            out = x.reshape((self.m, self.n))
        elif x.shape == (self.mn + 1, ):
            out = x[:-1].reshape((self.m, self.n))
            

        return out



    def flat_index(self,
        i:int, 
        j:int, 
    ) -> int:
        """For matrix indices i, j in an n x n matrix, get the indices of 
            elements in the flat vector of length n^2.
        """
        out = i*self.n + j
        
        return out



    def flat_index_inverse(self,
        k:int,
    ) -> int:
        """For indices of elements in a flat vector of length n^2, get the 
            matrix indices of original elements.
        """
        #n_root = Int64(n^0.5)
        col = k%self.n
        row = int((k - col)/self.n)
            
        out = (row, col)
        
        return out



    def flat_indices_col(self,
        j: int,
    ) -> np.ndarray:
        """Get the flat indices for a given column
        """
        out = np.arange(self.m)*self.n + j
        return out


    def get_bounds(self,
        allow_new: bool = False, 
        for_carrying_capacity: bool = False, 
    ) -> np.ndarray:
        """Reformat costs for use in program

        Function Arguments
        ------------------
        vec_costs : np.ndarray
            Base vector of costs for 1:n

        Keyword Arguments
        -----------------
        for_carrying_capacity : bool
            Set to True to build the constraint for the carrying capacity 
            problem, which scales a population up or down to meet land use 
            availability.
        """
        N = self.mn if not for_carrying_capacity else self.mn + 1
        vec_bounds_lower = np.zeros(N, )
        vec_bounds_upper = np.inf*np.ones(N, )

        # update upper bounds if not allowed
        if not allow_new:
            vec_bounds_upper[self.inds_flat_new_classes] = 0.0
        #else:
        #    # NOTE: additional constraint needed to prevent exceeding existing area
        #    vec_bounds_lower[self.inds_flat_new_classes] = -np.inf
        
        # turn into a list of tuples
        out = list(zip(vec_bounds_lower, vec_bounds_upper, ))

        return out



    def get_constraint_coeffs_eq_agrc_imports(self,
        vec_import_frac: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Imports are a fraction of total consumption; this constraint sets
            crop imports to be a function of production. Returns a tuple of the
            form 

            (
                A,
                b,
            )
        
        Function Arguments
        ------------------
        vec_import_frac : np.ndarray
            Vector of import fractions of cereals and non-cereals. Used to 
            adjust import supply availability if new crops are planted.

        Keyword Arguments
        -----------------
        """

        ##  ADD IN IMPORT AVAILABILITY BASED ON IMPORT FRACTION

        n_rows = len(self.dict_ind_to_imp_ind)
        A = np.zeros((n_rows, self.mn, ))
        b = np.zeros(n_rows)

        # get import fractions by type
        dict_ind_imp_fracs = {
            self.ind_crops_cereals: vec_import_frac[0],
            self.ind_crops_non_cereals: vec_import_frac[1],
            self.ind_pastures: vec_import_frac[2],
        }

        i = 0

        # update with reduction to import suppply constraint commensurate with growth
        for k, v in self.dict_ind_to_imp_ind.items():
            
            # imports and new crops
            ind_new = self.dict_ind_to_new_ind.get(k, )         # index associated with new crop type
            
            inds_imports = self.flat_indices_col(v, )           # column indices associated with imported crop type
            inds_new_crop = self.flat_indices_col(ind_new, )    # column indices associated with new crop type being planted
            inds_orig_crop = self.flat_indices_col(k, )         # column indices associated with original crops
            

            f = dict_ind_imp_fracs.get(k, )                     # import fraction for crop type
            
            # check value
            set_inf = not sf.isnumber(f)
            set_inf |= ((f > 1) | (f < 0)) if not set_inf else set_inf
            
            # by setting to inf, the constraint will be removed by the routine 
            # self.reduce_matrices_with_inf()
            cons = (-f/(1 - f)) if not set_inf else np.inf

            # imports are equal to import fraction applied to new and orig crops
            A[i, inds_imports] = 1
            A[i, inds_new_crop] = cons
            A[i, inds_orig_crop] = cons

            i += 1
        

        out = (A, b, )

        return out
    


    def get_constraint_coeffs_eq_lvst_feed_balance(self,
        vec_demands: np.ndarray,
        for_carrying_capacity: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a matrix of coefficients used to ensure that, for each 
            livestock class, feed demands are met. Returns

            (
                mat_coeffs,     # m x mn matrix
                vec_demands,    # m x 1
            )


        Function Arguments
        ------------------
        vec_demands : np.ndarray
            Vector of feed demands, in mass, by livestock category

        Keyword Arguments
        -----------------
        for_carrying_capacity : bool
            Set to True to build the constraint for the carrying capacity 
            problem, which scales a population up or down to meet land use 
            availability.
        """

        A = np.zeros((self.m, self.mn))
        for i in range(self.m):
            inds = range(i*self.n, (i+1)*self.n)
            A[i, inds] = 1
        
        if not for_carrying_capacity:
            out = (A, vec_demands, )
            return out
        
        
        ##  ADD AN EXTRA COLUMN IF RUNNING FOR CARRYING CAPACITY

        A_cc = np.zeros((self.m, self.mn + 1))
        A_cc[:,0:-1] = np.nan_to_num(
            (A.transpose()/vec_demands).transpose(),
            nan = 0.0,
            posinf = 1.0,
        )

        # add in the scalar
        A_cc[:, -1] = -1*np.sign(vec_demands, )
        b_cc = np.zeros(vec_demands.shape[0], )

        out = (A_cc, b_cc, )

        return out
    


    def get_constraint_coeffs_eq_lvst_new_residues(self,
        vec_residue_generation_factor: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a matrix of coefficients used to ensure that, for each 
            livestock class, feed demands are met. Returns

            (
                mat_coeffs,     # m x mn matrix
                vec_demands,    # m x 1
            )


        Function Arguments
        ------------------
        vec_residue_generation_factor : np.ndarray
            For new crop classes, vector (2, ), ordered as 
            Crops - Cereals, Crops - Non-Cereals, storing rates at which 
            residues are available for livestock feed as a proportion of yield

        Keyword Arguments
        -----------------
        """

        ##  ADD IN AVAILABILITY OF RESIDUES FROM NEW CROP CLASSES

        # initialize some indices
        ind_cr = self.ind_crop_residues
        ind_cr_new = self.dict_ind_to_new_ind.get(ind_cr, )
        inds_cc_new = self.flat_indices_col(self.ind_crops_cereals_new, )
        inds_cnc_new = self.flat_indices_col(self.ind_crops_non_cereals_new, )

        # get factors for residue generation
        rho_cc = vec_residue_generation_factor[0]
        rho_cnc = vec_residue_generation_factor[1]

        # update with reduction
        A = np.zeros((1, self.mn))
        b = np.zeros(1, )
        
        # force new crop residuals to be equal to generation by new crops
        # using generation factor
        A[0, ind_cr_new] = 1
        A[0, inds_cc_new] = -rho_cc
        A[0, inds_cnc_new] = -rho_cnc

        out = (A, b, )
        
        return out
    


    def get_constraint_coeffs_leq_lvst_dietary_fractions(self,
        vec_demands: np.ndarray,
        vec_max_crop_residues_fraction: np.ndarray,
        vec_max_crops_cereals_fraction: np.ndarray,
        vec_max_crops_non_cereals_fraction: np.ndarray,
        vec_max_pastures_fraction: np.ndarray,
        vec_min_crop_residues_fraction: np.ndarray,
        vec_min_crops_cereals_fraction: np.ndarray,
        vec_min_crops_non_cereals_fraction: np.ndarray,
        vec_min_pastures_fraction: np.ndarray,
        for_carrying_capacity: bool = False,
    ) -> Tuple[np.ndarray]:
        """Get the inquality constraints for dietary fractions.
        """

        ##  INITIALIZATION
        
        # get bound arrays
        vecs_frac_max, vecs_frac_min = self.get_vecs_diet_frac_bounds(
            vec_max_crop_residues_fraction,
            vec_max_crops_cereals_fraction,
            vec_max_crops_non_cereals_fraction,
            vec_max_pastures_fraction,
            vec_min_crop_residues_fraction,
            vec_min_crops_cereals_fraction,
            vec_min_crops_non_cereals_fraction,
            vec_min_pastures_fraction, 
        )

        # shortcut for number of vec_frac conceptual constraints to deal with
        n_vf = len(vecs_frac_max)
        
        # index shortcuts
        ind_cc = self.ind_crops_cereals
        ind_cnc = self.ind_crops_non_cereals
        ind_cr = self.ind_crop_residues
        ind_p = self.ind_pastures
        
        
        # initialize bounds (which have to take into account LFI) and an inner array
        arr_base = np.zeros((self.m, self.n), )
        b = np.zeros(2*n_vf*self.m)

        # A will be dependent on whether the carrying capacity estimation is being conducted
        ncol_a = self.mn if not for_carrying_capacity else self.mn + 1
        A = np.zeros((2*n_vf*self.m, ncol_a), )
        
        
        # ordered indices in LDE--use index names
        ordered_for_vecs_inds_lde = [
            ind_cr,
            ind_cc,
            ind_cnc,
            ind_p
        ]
        
        
        ##  BUILD MIN AND MAX CONSTRAINTS INSIDE THE SAME LOOP

        for i in range(self.m):
            for ind_j_lde in range(n_vf):
                
                j = ordered_for_vecs_inds_lde[ind_j_lde]
                ind_import = self.dict_ind_to_imp_ind.get(j, )
                ind_new = self.dict_ind_to_new_ind.get(j, )

                # get some output indices
                base = i*n_vf + ind_j_lde
                row_min_cr = 2*base
                row_max_cr = 2*base + 1
                
                
                ##  MIN CONSTRAINT
            
                # clear array and get some shortcuts
                arr_base[:] = 0
                frac_min = vecs_frac_min[j][i]
                
                # min constraint
                arr_base[i, j] = -1
                # arr_base[i, ind_l] = -1*frac_min    # this would be in place if using generic Livestock Feed Imports


                # any imports or new area
                if ind_import is not None:
                    arr_base[i, ind_import] = -1

                if ind_new is not None:
                    arr_base[i, ind_new] = -1
                
                factor_dem_min = vec_demands[i]*frac_min

                # assign
                if not for_carrying_capacity:
                    A[row_min_cr] = arr_base.flatten().copy()
                    b[row_min_cr] = -1*factor_dem_min

                else:
                    # adjust A; b remains 0 in this case
                    arr_base /= factor_dem_min if (factor_dem_min > 0) else 1
                    A[row_min_cr, 0:-1] = arr_base.flatten().copy()
                    A[row_min_cr, -1] = float(factor_dem_min > 0)

                    
            
                
                ##  MAX CONSTRAINT
            
                # clear array and get some shortcuts
                arr_base[:] = 0
                frac_max = vecs_frac_max[j][i]
                
                # min constraint
                arr_base[i, j] = 1
                # arr_base[i, ind_l] = frac_max   # this would be in place if using generic Livestock Feed Imports

                # any imports or new area
                if ind_import is not None:
                    arr_base[i, ind_import] = 1

                if ind_new is not None:
                    arr_base[i, ind_new] = 1


                factor_dem_max = vec_demands[i]*frac_max

                # assign
                if not for_carrying_capacity:
                    A[row_min_cr] = arr_base.flatten().copy()
                    b[row_min_cr] = factor_dem_max

                else:
                    # adjust A; b remains 0 in this case
                    arr_base /= factor_dem_max if (factor_dem_max > 0) else 1
                    A[row_max_cr, 0:-1] = arr_base.flatten().copy()
                    A[row_max_cr, -1] = -1*float(factor_dem_max > 0)

        # return
        out = (A, b, )

        return out



    def get_constraint_coeffs_leq_lvst_feed_supply(self,
        vec_supplies: np.ndarray,
        vec_residue_generation_factor: np.ndarray,
        for_carrying_capacity: bool = False,
        vec_import_frac: Union[np.ndarray, None] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a matrix of coefficients used to ensure that, for each 
            feed pathway, feed availability is not exceeded. Returns

            (
                mat_coeffs, # n x n^2 matrix
                vec_supplies,
            )


        Function Arguments
        ------------------
        vec_residue_generation_factor : np.ndarray
            For new crop classes, vector (2, ), ordered as 
            Crops - Cereals, Crops - Non-Cereals, storing rates at which 
            residues are available for livestock feed as a proportion of yield
        vec_supplies : np.ndarray
            Vector of feed availability, in mass, by feed pathway. Only applies
            to categories 0 -- (n - 1) (the slack category is not bounded)

        Keyword Arguments
        -----------------
        for_carrying_capacity : bool
            Set to True to build the constraint for the carrying capacity 
            problem, which scales a population up or down to meet land use 
            availability.

        """

        ##  SETUP CONSTRAINED INDICES

        # initialize some indices
        ind_cr = self.ind_crop_residues
        ind_cr_new = self.dict_ind_to_new_ind.get(ind_cr, )
        inds_cc_new = self.flat_indices_col(self.ind_crops_cereals_new, )
        inds_cnc_new = self.flat_indices_col(self.ind_crops_non_cereals_new, )

        # imports are not constrained because they are a function of how much
        # is planted in original/new croplands
        inds_skip = [
            self.ind_crop_residues_new
        ]

        if self.check_import_frac(vec_import_frac, ):
            inds_skip += [
                self.ind_crop_imports_cereals,
                self.ind_crop_imports_non_cereals,
                self.ind_pasture_imports
            ]

        inds_constrained = [x for x in range(self.n) if x not in inds_skip]
        
        # initialize the matrix with basic constraints for each supply and output 
        A = np.zeros((self.n, self.mn))

        # assign summation
        for j in inds_constrained:
            inds = self.flat_indices_col(j, )
            A[j, inds] = 1


        ##  ADD IN CROP RESIDUE AVAILABILITY

        # get factors for residue generation
        rho_cc = vec_residue_generation_factor[0]
        rho_cnc = vec_residue_generation_factor[1]
        
        # crop residuals supplies are increased by these factors*new yields
        A[ind_cr, inds_cc_new] = -rho_cc
        A[ind_cr, inds_cnc_new] = -rho_cnc

        # add extra column for Y if running carrying capacity model
        if for_carrying_capacity:
            A = self.format_matrix_for_carrying_capacity(A, )

        # setup output
        out = (
            A[inds_constrained, :],
            vec_supplies[inds_constrained], 
        )

        return out



    def get_constraint_coeffs_leq_new_lands_lb(self,                   
        vec_supplies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a matrix of coefficients used to ensure that new land
            classes (use total yield mass of crops and pastures) cannot shift 
            away more than exists.

            (
                mat_coeffs, # n x n^2 matrix
                vec_supplies,
            )


        let:
            S  = supply
            x  = use of crop x
            x* = new crops

            Note that 
                x* >= x_c - Sc 

            i.e., that new crop reductions cannot go below the slack between
            available supply and how much was actually used. This gives

            x_c - x* <= Sc



        Function Arguments
        ------------------
        vec_supplies : np.ndarray
            Vector of feed availability, in mass, by feed pathway. Only applies
            to categories 0 -- (n - 1) (the slack category is not bounded)

        Keyword Arguments
        -----------------
        """
        # initialize some indices

        dict_iter = self.dict_ind_to_new_ind
        n_cons = len(dict_iter)

        # build the matrix with basic constraints
        A = np.zeros((n_cons, self.mn))
        b = np.zeros(n_cons, )

        i = 0
        for j, j_new in dict_iter.items():
            inds_orig = self.flat_indices_col(j, ) 
            inds_new = self.flat_indices_col(j_new, ) 

            A[i, inds_orig] = 1        # + total mass produced from original area 
            A[i, inds_new] = -1        # - total mass produced from new area
            b[i] = vec_supplies[j]     # supply associated with current area

            i += 1                     # next row of constraint
    
        out = (A, b, )

        return out
    


    def get_costs(self,
        vec_costs: np.ndarray,
        allow_new: bool = False, 
        for_carrying_capacity: bool = False,
        override_import_costs: bool = False,
        vec_import_frac: Union[np.ndarray, None] = None, # HERE123--deal with this
        **kwargs,
    ) -> np.ndarray:
        """Reformat costs for use in program. Ensures that crop imports have 0 
            cost (they are a function of planting for livestock domestic demand 
            and import fraction)

        Function Arguments
        ------------------
        vec_costs : np.ndarray
            Base vector of costs for 1:n. 
            NOTE: If running with `for_carrying_capacity = True`, this is 
                ignored, as a cost of -1 is specified for increasing the 
                population scalar Y.
        
        Keyword Arguments
        -----------------
        allow_new : bool
            Allow new crop/pastures? 
        for_carrying_capacity : bool
            Set to True to build the constraint for the carrying capacity 
            problem, which scales a population up or down to meet land use 
            availability.
        override_import_costs : bool
            If True, will not force import costs to be same as crop costs.
        vec_import_frac : Union[np.ndarray, None]
            Optional vector of import fractions of cereals and non-cereals. Used 
            to adjust import supply availability if new crops are planted. 
                * If NumPy vector:  Uses the ratio of imports as a hard 
                                    constraint. Can only be used if allowing new
                * Otherwise:        Allows for imports to be a free variable.
                                    Generally the case if allow_new = False
        kwargs : 
            Ignored
        """

        # return costs for carrying capacity approach
        if for_carrying_capacity:
            c = np.zeros(self.mn + 1, )
            c[-1] = -1
            return c


        vec_out = vec_costs.copy()

        # ensure costs are correct for imports by setting to 0 (depend only on 
        # planting decisions)
        set_zero_imp_costs = not override_import_costs
        set_zero_imp_costs &= self.check_import_frac(vec_import_frac, )
        if set_zero_imp_costs:
            for v in self.dict_ind_to_imp_ind.values():
                vec_out[v] = 0

        # apply to each row in the matrix
        c = np.ones((self.m, self.n))*vec_out
        c = c.flatten()

        return c
    


    def get_flat_inds_new_classes(self,
    ) -> np.ndarray:
        """
        """
        inds_new = sorted(
            list(self.dict_ind_to_new_ind.values())
        )

        # get indices associated with "new"
        inds_flat = []
        for ind in inds_new:
            inds_flat.extend(
                list(self.flat_index(np.arange(self.m, ), ind ))
            )

        inds_flat = np.array(inds_flat, )
        
        return inds_flat

    
    
    def get_problem_components_constraints(self,
        allow_new: bool,
        vec_demand: np.ndarray,
        vec_max_crop_residues_fraction: np.ndarray,
        vec_max_crops_cereals_fraction: np.ndarray,
        vec_max_crops_non_cereals_fraction: np.ndarray,
        vec_max_pastures_fraction: np.ndarray,
        vec_min_crop_residues_fraction: np.ndarray,
        vec_min_crops_cereals_fraction: np.ndarray,
        vec_min_crops_non_cereals_fraction: np.ndarray,
        vec_min_pastures_fraction: np.ndarray,
        vec_residue_generation_factor: np.ndarray,
        vec_supply: np.ndarray,
        for_carrying_capacity: bool = False,
        vec_import_frac: Union[np.ndarray, None] = None,
    ) -> Tuple:
        """Generate constraints for livestock dietary estimation optimization 
            LP. Returns a tuple in the following form:

            (
                A,
                b,
                G,
                h,
            )

        Function Arguments
        ------------------
        allow_new : bool
            Allow new land use classes? If False and forcing vec_import_frac,
            this can cause infeasibilities.
        vec_demand : np.ndarray
            Vector of demand for livestock
        vec_residue_generation_factor : np.ndarray
            For new crop classes, vector (2, ), ordered as 
            Crops - Cereals, Crops - Non-Cereals, storing rates at which 
            residues are available for livestock feed as a proportion of yield
        vec_supply : np.ndarray
            Vector of supplies by category (ordered--the final category, for
            livestock feed imports, is ignored)
        
        Keyword Arguments
        -----------------
        for_carrying_capacity : bool
            Set to True to build the constraint for the carrying capacity 
            problem, which scales a population up or down to meet land use 
            availability.
        vec_import_frac : Union[np.ndarray, None]
            Optional vector of import fractions of cereals and non-cereals. Used to 
            adjust import supply availability if new crops are planted. 
                * If NumPy vector:  Uses the ratio of imports as a hard 
                                    constraint. Can only be used if allowing new
                * Otherwise:        Allows for imports to be a free variable.
                                    Generally the case if allow_new = False
        **kwargs : 
            Ignored
        """

        ##  INITIALIZATION

        # initialize output matrices--use A, b for leq and G, h for eq
        A = []
        G = []
        b = []
        h = []

        # dietary fraction bounds, ordered
        args = (
            vec_max_crop_residues_fraction,
            vec_max_crops_cereals_fraction,
            vec_max_crops_non_cereals_fraction,
            vec_max_pastures_fraction,
            vec_min_crop_residues_fraction,
            vec_min_crops_cereals_fraction,
            vec_min_crops_non_cereals_fraction,
            vec_min_pastures_fraction,
        )
    

        ##  GET EQUALITY CONSTRAINTS

        # totals must equal demanded
        A_fb, b_fb = self.get_constraint_coeffs_eq_lvst_feed_balance(
            vec_demand,
            for_carrying_capacity = for_carrying_capacity, 
        )
        G.append(A_fb, )
        h.append(b_fb, )

        """
        # add the new crop residue forcing constraint
        A_cr, b_cr = self.get_constraint_coeffs_eq_lvst_new_residues(
            vec_residue_generation_factor,
        )
        G.append(A_cr, )
        h.append(b_cr, )
        """;

        # add the import fraction forcing constraint?
        if self.check_import_frac(vec_import_frac, ):

            # import values as a function of existing + new crop plants
            A_imp, b_imp = self.get_constraint_coeffs_eq_agrc_imports(
                vec_import_frac.astype(float),
            )
            if for_carrying_capacity:
                A_imp = self.format_matrix_for_carrying_capacity(A_imp, )

            G.append(A_imp, )
            h.append(b_imp, )

            if not allow_new:
                msg = f"""Note: allow_new if False and import fractions are 
                    fixed. This can cause infeasibilities if enough feed is not
                    provided.
                """
                warnings.warn(msg)



        ##  GET INEQUALITY CONSTRATINTS

        # dietary fractions
        A_df, b_df = self.get_constraint_coeffs_leq_lvst_dietary_fractions(
            vec_demand,
            *args,
            for_carrying_capacity = for_carrying_capacity, 
        )
        A.append(A_df, )
        b.append(b_df, )

        # supply balance
        A_sb, b_sb = self.get_constraint_coeffs_leq_lvst_feed_supply(
            vec_supply.astype(float), 
            vec_residue_generation_factor.astype(float),
            for_carrying_capacity = for_carrying_capacity, 
            vec_import_frac = vec_import_frac,
        )
        A.append(A_sb, )
        b.append(b_sb, )

        # new area removals can't exceed existing area (area balance)
        A_ab, b_ab = self.get_constraint_coeffs_leq_new_lands_lb(
            vec_supply.astype(float), 
        )
        if for_carrying_capacity:
            A_ab = self.format_matrix_for_carrying_capacity(A_ab, )

        A.append(A_ab, )
        b.append(b_ab, )


        # concatenate
        A = np.concat(A)
        G = np.concat(G, )
        b = np.concat(b, )
        h = np.concat(h, )

        # clean out any rows that might have infinite constraint values
        A, b = self.reduce_matrices_with_inf(A, b, )
        G, h = self.reduce_matrices_with_inf(G, h, )


        # return outputs as a tuple
        out = (
            A,
            b,
            G,
            h
        )

        return  out



    def get_vecs_diet_frac_bounds(self,
        vec_max_crop_residues_fraction: np.ndarray,
        vec_max_crops_cereals_fraction: np.ndarray,
        vec_max_crops_non_cereals_fraction: np.ndarray,
        vec_max_pastures_fraction: np.ndarray,
        vec_min_crop_residues_fraction: np.ndarray,
        vec_min_crops_cereals_fraction: np.ndarray,
        vec_min_crops_non_cereals_fraction: np.ndarray,
        vec_min_pastures_fraction: np.ndarray,
        return_ordered_groups: bool = True,
    ) -> Tuple[np.ndarray]:
        """Enter dietary fraction bound vectors by type, then check them. Once
            verified, return. If any issues arise, an error will be raised.

        Keyword Arguments
        -----------------

        return_ordered_groups : bool
            * True:     Return two tuples in order of index in LDE matrices,
                        i.e., as 
                        (
                            (
                                vec_max_crop_residues_fraction,
                                vec_max_crops_cereals_fraction,
                                vec_max_crops_non_cereals_fraction,
                                vec_max_pastures_fraction,
                            ),
                            (
                                vec_min_crop_residues_fraction,
                                vec_min_crops_cereals_fraction,
                                vec_min_crops_non_cereals_fraction,
                                vec_min_pastures_fraction,
                            )
                        )
            * False:    Return a tuple (length 8) of vectors in same order as 
                        input
        """

        # check contraints
        self._verify_constraint_arrays(
            vec_max_crops_cereals_fraction,
            vec_min_crops_cereals_fraction,
        )

        self._verify_constraint_arrays(
            vec_max_crops_non_cereals_fraction,
            vec_min_crops_non_cereals_fraction,
        )

        self._verify_constraint_arrays(
            vec_max_crop_residues_fraction,
            vec_min_crop_residues_fraction,
        )

        self._verify_constraint_arrays(
            vec_max_pastures_fraction,
            vec_min_pastures_fraction,
        )

        out = (
            vec_max_crop_residues_fraction,
            vec_max_crops_cereals_fraction,
            vec_max_crops_non_cereals_fraction,
            vec_max_pastures_fraction,
            vec_min_crop_residues_fraction,
            vec_min_crops_cereals_fraction,
            vec_min_crops_non_cereals_fraction,
            vec_min_pastures_fraction,
        )

        if return_ordered_groups:
            out = (out[0:4], out[4:])

        return out
    

    
    def format_matrix_for_carrying_capacity(self,
        mat: np.ndarray,
    ) -> np.ndarray:
        """Format a coefficient matrix that is otherwise unchanged to fit with 
            the carrying capacity problem. Adds a column to the right. 
        """
        m, n = mat.shape
        mat_out = np.zeros((m, n + 1), )
        mat_out[:, 0:-1] = mat

        return mat_out



    def reduce_matrices_with_inf(self,
        A: np.ndarray,
        b: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """Remove any constraint rows associated with np.inf in b. Returns a 
            tuple of revised matrices A, b
        """

        inds = np.where(np.abs(b) != np.inf)[0]
        out = (
            A[inds, :],
            b[inds],
        )

        return out





    ##########################
    #    SOLVER FUNCTIONS    #
    ###########################

    def solve(self,
        vec_costs: np.ndarray,
        vec_demand: np.ndarray,
        vec_max_crop_residues_fraction: np.ndarray,
        vec_max_crops_cereals_fraction: np.ndarray,
        vec_max_crops_non_cereals_fraction: np.ndarray,
        vec_max_pastures_fraction: np.ndarray,
        vec_min_crop_residues_fraction: np.ndarray,
        vec_min_crops_cereals_fraction: np.ndarray,
        vec_min_crops_non_cereals_fraction: np.ndarray,
        vec_min_pastures_fraction: np.ndarray,
        vec_residue_generation_factor: np.ndarray,
        vec_supply: np.ndarray,
        *,
        allow_new: bool = False,
        for_carrying_capacity: bool = False,
        stop_on_error: bool = False,
        vec_import_frac: Union[np.ndarray, None] = None,
        **kwargs,
    ) -> np.ndarray:
        """Solve for estimated dietary consumption based on available land use
            (dry matter intake from pastures); crops produced for livestock
            constumption; and crop residuals.

        

        NOTE: If return_unadj_on_infeas is False, then returns None.

        Function Arguments
        ------------------
        vec_costs : np.ndarray
            Vector (n_sources x 1) of costs per each crop
        vec_demand : np.ndarray
            Vector (n_livstock_categories x 1) of total feed demands from each
            livetock type
        vec_max_crop_residues_fraction : np.ndarray
            Vector (n_livstock_categories x 1) of maximum dietary fractions from 
            crop residues
        vec_max_crops_cereals_fraction : np.ndarray
            Vector (n_livstock_categories x 1) of maximum dietary fractions from 
            cereal crops
        vec_max_crops_non_cereals_fraction : np.ndarray
            Vector (n_livstock_categories x 1) of maximum dietary fractions from 
            non-cereal crops
        vec_max_pastures_fraction : np.ndarray
            Vector (n_livstock_categories x 1) of maximum dietary fractions from 
            pastures
        vec_min_pastures_fraction : np.ndarray
            Vector (n_livstock_categories x 1) of minimum dietary fractions from 
            pastures
        vec_min_crops_cereals_fraction : np.ndarray
            Vector (n_livstock_categories x 1) of minimum dietary fractions from 
            cereal crops
        vec_min_crops_non_cereals_fraction : np.ndarray
            Vector (n_livstock_categories x 1) of minimum dietary fractions from 
            non-cereal crops
        vec_min_crop_residues_fraction : np.ndarray
            Vector (n_livstock_categories x 1) of minimum dietary fractions from 
            crop residues
        vec_residue_generation_factor : np.ndarray
            For new crop classes, vector (2, ), ordered as 
            Crops - Cereals, Crops - Non-Cereals, storing rates at which 
            residues are available for livestock feed as a proportion of yield
        vec_supply : np.ndarray
            Vector (n_sources x 1) of supplies available per type (final one
            ignored)
        
        Keyword Arguments
        -----------------
        allow_new : bool
            Allow sourcing from new classes? 
        for_carrying_capacity : bool
            Set to True to build the constraint for the carrying capacity 
            problem, which scales a population up or down to meet land use 
            availability.
        stop_on_error : bool
            Stop if errors occur in solutions?
        vec_import_frac : Union[np.ndarray, None]
            Optional vector of import fractions of cereals and non-cereals. Used to 
            adjust import supply availability if new crops are planted. 
                * If NumPy vector:  Uses the ratio of imports as a hard 
                                    constraint. Can only be used if allowing new
                * Otherwise:        Allows for imports to be a free variable.
                                    Generally the case if allow_new = False
        **kwargs :
            Passed to sco.linprog()
        """
        # some init--not that the carrying capacity run cannot allow new
        allow_new &= not for_carrying_capacity
        class_name = str(self.__class__).split(".")[-1].split("'")[0]
        sol = None
        
        # try to get bounds
        try:
            list_bounds = self.get_bounds(
                allow_new = allow_new, 
                for_carrying_capacity = for_carrying_capacity,
            )

        except Exception as e:
            msg = f"Error retrieving bounds in {class_name}: {e}"
            raise RuntimeError(msg)
        
      
        # try to retrieve objective components
        try:
            c = self.get_costs(
                vec_costs,
                allow_new = allow_new,
                for_carrying_capacity = for_carrying_capacity,
                vec_import_frac = vec_import_frac,
                **kwargs,
            )

        except Exception as e:
            msg = f"Error retrieving costs values in {class_name}: {e}"
            raise RuntimeError(msg)


        # try to retrieve constraint components
        try:
            A, b, G, h = self.get_problem_components_constraints(
                allow_new,
                vec_demand,
                vec_max_crop_residues_fraction,
                vec_max_crops_cereals_fraction,
                vec_max_crops_non_cereals_fraction,
                vec_max_pastures_fraction,
                vec_min_crop_residues_fraction,
                vec_min_crops_cereals_fraction,
                vec_min_crops_non_cereals_fraction,
                vec_min_pastures_fraction,
                vec_residue_generation_factor,
                vec_supply,
                for_carrying_capacity = for_carrying_capacity,
                vec_import_frac = vec_import_frac,
            )

        except Exception as e:
            msg = f"Error retrieving constraints in {class_name}: {e}"
            raise RuntimeError(msg)

        try:
            sol = sco.linprog(
                c, 
                A_ub = A,
                b_ub = b,
                A_eq = G,
                b_eq = h,
                bounds = list_bounds,
                **kwargs,
            )


        except Exception as e:
            msg = f"Error trying to solve {class_name}: {e}."
            if stop_on_error:
                raise RuntimeError(msg)
            
            warnings.warn(f"{msg} Trying scipy.optimize.minimize...")
        

        if sol is not None:
            """
            sol = self.clean_and_reshape_solver_output(
                Q,
                sol,
                **kwargs, # optional passing of thresh_to_zero
            )
            """
        return sol







    


    





    


    





    