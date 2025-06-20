#import cyipopt
import numpy as np
import qpsolvers
import scipy.optimize as sco
import warnings
from typing import *

import sisepuede.utilities._toolbox as sf




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
    ) -> np.ndarray:
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
    ) -> np.ndarray:
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
    
    
    
    def get_constraint_coeffs_row_stochastic(self,
        matrix_0: np.ndarray,
    ) -> np.ndarray:
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
        **kwargs,
    ) -> Tuple:
        """Generate constraints for land use optimization QP. Returns a tuple in 
            the following form:

            (
                A,
                b,
                G,
                h,
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
        **kwargs : 
            Ignored
        """

        ##  GET CONSTRAINTS

        constraint_min_area = self.get_constraint_coeffs_min_area(
            matrix_0,
            x_0,
            vec_infima,
            flag_ignore
        );

        constraint_max_area = self.get_constraint_coeffs_max_area(
            x_0,
            vec_suprema,
            flag_ignore,
        );


        constraint_row_stochastic = self.get_constraint_coeffs_row_stochastic(matrix_0, )
        constraint_preserve_zeros = self.get_constraint_coeffs_preserve_zeros(matrix_0, )


        ##  SET EQ CONSTRAINTS

        A = constraint_row_stochastic[0]
        b = constraint_row_stochastic[1]


        ##  BUILD INEQ (G and h in QPsolve for some reason)

        G = []
        h = []

        constraint_list = [
            constraint_min_area,
            constraint_max_area,
            constraint_preserve_zeros
        ]


        for i, constraint in enumerate(constraint_list):

            if constraint is None:
                continue
                
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






    


    





    