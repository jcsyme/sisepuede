#import cyipopt
import numpy as np
import os, os.path
import pandas as pd
import qpsolvers
import scipy.optimize as sco
import sys
from typing import *


import sisepuede.utilities._toolbox as sf



# 
#
#

class QAdjuster:
    """
    Adjust a transition matrix Q to match requirements from land use 
        reallocation factor.
    """
    
    def __init__(self,
    ) -> None:

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




    #######################
    #    KEY FUNCTIONS    #
    #######################


        
    def get_constraints(self,
        Q: np.ndarray,
        x_0: np.ndarray,
        x_1: np.ndarray,
        epsilon_constraint: float = 0.000001,
        error_type = "additive",
        max_error: float = 0.01,
        preserve_zeros: bool = True,
        sup: float = 1.0,
    ) -> List[sco.LinearConstraint]:
        """
        Retrieve constraints for the Minimize Calibration Error (MCE) problem.
            Returns two constraints and:

            * ConstraintRowStochastic
            * ConstraintErrorBounds

        Function Arguments
        ------------------
        - Q: initial transition matrix
        - x_0: initial (time 0) prevalence vector
        - x_1: second step (time 1) prevalence vector

        Keyword Arguments
        -----------------
        - epsilon_constraint: acceptable numerical error in constraints; note
            that, due to floating point errors, some sums can exceed the 
            constraints, meaning that the initial state is infeasible. 
        - error_type: 
            * "additive": add the max_error as bounds for initial estimates of
                free variables
            * "scalar": apply the max_error as a scalar to initial esimates of
                free variables
        - max_error: maximium error for transition probabilities (e.g., 
            0.01 = 1% error from baseline estimates)
        - preserve_zeros: preserve zeros in the bounds?
        - sup: supremum for transitions that are output; recommended to avoid
            setting to 1
        """

        n = Q.shape[0]

        # get matrices for constraints
        A_row_stochastic = self.get_constraint_coeffs_row_stochastic(Q, )
        A_error, vec_inf_err, vec_sup_err = self.get_constraint_coeffs_error(
            Q, 
            error_type = error_type,
            max_error = max_error,
            preserve_zeros = preserve_zeros,
            supremum = sup,
        )

        # constraints to pass to solvers
        constraints = [
            # force rows to sum to 1 (ConstraintRowStochastic)
            sco.LinearConstraint(
                A_row_stochastic, 
                lb = np.ones(n) - epsilon_constraint, 
                ub = np.ones(n) + epsilon_constraint,
                keep_feasible = True,
            ), 

            # constraint on acceptable error (ConstraintErrorBounds)
            sco.LinearConstraint(
                A_error, 
                lb = vec_inf_err, 
                ub = vec_sup_err,
                keep_feasible = True,
            )
        ]

        return constraints



    def correct_transitions(self,
        Q: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        *,
        approach: str = "minimize_transition_error",
        epsilon_constraint: float = 0.000001,
        error_type: str = "additive",
        inds_absorp: Union[list, None] = None,
        infimum_diag: float = 0.99,
        max_error: float = 0.01,
        perturbation_diag: float = 0.000001,
        preserve_zeros: bool = False,
        solver_qp: str = "cvxopt",
        supremum: float = 0.99999,
    ) -> Union[np.ndarray, None]:
        """
        Correct the initial transition matrix, which is derived from one data
            source (e.g., Copernicus), to ensure that it reproduces observed 
            prevalences from other sources (e.g., FAO)

        Function Arguments
        ------------------
        - Q: initial transition matrix
        - u: initial (time 0) prevalence vector
        - v: second step (time 1) prevalence vector

        Keyword Arguments
        -----------------
        - approach: problem approach. One of the following (can be entered as
            any combination of upper or lower case)

            * "minimize_estimated_angle" or "mea" (LP):
                
                Minimize the estimated angle between prevalence vectors. This 
                approach scales each dimension in the prevalence vector 
                uniformly, attempting to close the gap between the two vectors. 
                The program is formulated as a Linear Program. 

                This minimization problem seeks to minimize 
                arcos(dot(Q^Tu, v)/||Q^Tu||||v||). Noting that arcos is 
                monotonic on the interval [0, Pi/2],

                lim arcos as dot(Q^Tu, v) -> ||Q^Tu||||v|| -> 1 is 0. To avoid
                second degree terms in the formulation, we estimate the
                denominator as ||v||^2, and the problem is


                min -dot(Q^Tu, v)

                s.t.

                    dot(Q^Tu, v) <= ||v||^2  # angle approaches from one side
                    \forall i: \sum_{j <= n} Q[i, j] = 1  # row stochastic
                    \forall i, j: Q[i, j] - e <= Q[i, j] <= Q[i, j] + e  # acceptable error

                    where e is an allowable error term

                    (if preserve_zeros, Q[i, j] = 0 => e = 0)


            * "minimize_calibration_error" or "mce" (QP): 
            
                Minimize the distance between the observed prevalence and the 
                resulting prevalence while subjecting transitions to error 
                constraints. 

                This minimization problem seeks to minimize the Euclidean 
                distance between the estimated prevalence vector, Q^Tu, and the 
                observed vector, v; i.e.,

                    (obj) sum_{i <= n} (u.dot(Q[:,j]) - v[j])^2

                Q \in R^{n x n} is flattened as x[k], k in [1,..., n^2]

                    - x[k] = Q[i, j], k = i*n + j, 1 <= k <= n^2
                    - u is the initial prevalence vector
                    - v is the target prevalence vector


            * "minimize_transiton_error" or "mte": minimize the error in the 
                transition matrix while constraing the solution to reflect 
                target prevalence. May require that estimated transition 
                probabilities with value 0 are relaxed.

        - epsilon_constraint: acceptable numerical error in constraints; note
            that, due to floating point errors, some sums can exceed the 
            constraints, meaning that the initial state is infeasible. 
        - error_type: 
            * "additive": add the max_error as bounds for initial estimates of
                free variables
            * "scalar": apply the max_error as a scalar to initial esimates of
                free variables
        - inds_absorp: optional indices for allowable absorption states
        - infimum_diag: minimum value allowed along diagonal
        - max_error: maximium error for transition probabilities (e.g., 
            0.01 = 1% error from baseline estimates)
        - perturbation_diag: perturbation to apply to the diagonal (positive) to
            ensure P is positive definite, a requirement for some solvers.
        - preserve_zeros: if True, ensures that a solution matrix x' cannot 
            introduce non-zero transitions to edges where they wre not present 
            in x
        - solver_qp: default solver to use for Quadtratic Programs
        - supremum: supremum for transitions that are output; recommended to 
            avoid setting to 1
        """

        # set up components
        matrix = Q.copy()
        matrix = (matrix.transpose()/matrix.sum(axis = 1)).transpose()
        n = matrix.shape[0]


        ##  GET PROBLEM SETUP
        
        approach = approach.lower()
        valid_approaches = [
            "minimize_estimated_angle",
            "mea",
            "minimize_calibration_error", 
            "mce",
            "minimize_transition_error", 
            "mte"
        ]
        approach = "mce" if (approach not in valid_approaches) else approach


        # Minimize Calibration Error
        if approach in ["minimize_calibration_error", "mce"]:

            sol = self.solve_mce(
                matrix,
                u,
                v,
                error_type = error_type,
                inds_absorp = inds_absorp,
                infimum_diag = infimum_diag,
                max_error = max_error,
                perturbation_diag = perturbation_diag,
                preserve_zeros = preserve_zeros,
                solver = solver_qp,
                supremum = supremum,
            )


        elif approach in ["minimize_transition_error", "mte"]:
            """
            constraints = self.get_constraints_mte(
                matrix,
                x_0,
                x_1,
                epsilon_constraint = epsilon_constraint,
                max_error = max_error,
            )
            """
            raise RuntimeError(f"APPROACH {approach} UNDEFINED IN optimization_utilities.py")


       
     
        return sol
        
        

    def f_obj_mce(self,
        x: np.ndarray,
        p_0: np.ndarray, # prevalence vector at time 0
        p_1: np.ndarray, # prevalence vector at time 1
    ) -> float:
        """
        Minimize the distance between the new matrix and the original 
            transition matrix for the Minimize Calibration Error (MCE) approach

        Function Arguments
        ------------------
        - x: vector to solve for
        - p_0: initial prevalence
        - p_1: next-step prevalence
        """
        #obj = ((x - vec_prev)**2).sum()
        #grad = 2*(x - vec_prev)
        n = len(p_0)
        obj = np.dot(p_0, x.reshape((n, n)))
        obj = ((obj - p_1)**2).sum()

        # get gradient
        grad = self.grad_mce(x, p_0, p_1, )

        return obj



    def f_obj_mte(self,
        x: np.ndarray,
        p_0: np.ndarray, # prevalence vector at time 0
        p_1: np.ndarray, # prevalence vector at time 1
    ) -> float:
        """
        Minimize the distance between the new matrix and the original 
            transition matrix for the Minimize Transition Error (MTE) approach

        Function Arguments
        ------------------
        - x: vector to solve for
        - p_0: initial prevalence
        - p_1: next-step prevalence
        """
        obj = ((x - vec_prev)**2).sum()
        grad = 2*(x - vec_prev)
       
        return (obj, grad)

    

    def grad_mce(self,
        x: np.ndarray,
        p_0: np.ndarray,
        p_1: np.ndarray,
    ) -> np.ndarray:
        """
        Generate the gradient vector for f_obj_mce()

        Function Arguments
        ------------------
        - x: variable vector
        - p_0: initial prevalence
        - p_1: next-step prevalence
        """

        n = p_0.shape[0]

        # initialize a matrix and gradient vector
        Q_cur = x.reshape((n, n))
        vec_grad = np.zeros(n**2).astype(float)

        area = p_0.sum()

        
        # iterate 
        for k in range(n**2):
            # column and row in Q
            j = k%n
            i = int((k - j)/n)

            val = 2*p_0[i]*(p_0.dot(Q_cur[:, j]) - p_1[j])
            val /= area**2

            vec_grad[k] = val

        return vec_grad
    


    def f_obj_hess(self, 
        x: np.ndarray,
        x_try: np.ndarray,
    ) -> np.ndarray:
        """
        Set the Hessian for the objective function
        """
        out = np.diag(2*np.ones(len(x)))

        return out
    


    def flat_index(self,
        i:int, 
        j:int, 
        n:int,
    ) -> int:
        """
        For matrix indices i, j in an n x n matrix, get the indices of elements
            in the flat vector of length n^2.
        """
        out = i*n + j
        
        return out



    def flat_index_inverse(self,
        k:int,
        n:int,
    ) -> int:
        """
        For indices of elements in a flat vector of length n^2, get the matrix indices
            of original elements.
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
        """
        Generate a matrix of coefficients used to ensure that values do not
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
        """
        Generate maximum area constraint coefficients (B_1) for land use 
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
        print(f"n_w = {n_w}")
        
        # initialize output matrices - start with inequality
        A_coeffs_ineq = np.zeros((n_w, n**2))
        b_ineq = vector_bounds[w]

        # add constraint on upper bound
        for i, ind in enumerate(w):
            inds = np.arange(n)*n + ind
            print(x_0) if i == 0 else None
            #np.put(A_coeffs_ineq[i], inds, x_0)
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
        """
        Generate minimum area constraint coefficients (B_0) for land use 
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



    def get_constraint_coeffs_preserve_zeros(self,
        matrix_0: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a matrix of coefficients used to ensure that values in matrix_0 
            that are 0 also zero in the solution. Returns a tuple in the form of

        (
            A,
            b,
        )

        Function Arguments
        ------------------
        - matrix_0: initial transition matrix (n x n)
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
        """
        Generate a matrix of coefficients used to ensure that the resulting
            matrix is row-stochastic

        Returns a tuple of the form
        
        (
            A,  # matrix with dims (n, n^2)
            b,  # vector with dim (n, )
        )

        Function Arguments
        ------------------
        - matrix_0: initial transition matrix (n x n)
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
    ) -> np.ndarray:
        """
        Get the costs specified for each qij and the prevalence vector 

        Function Arguments
        ------------------
        - matrix_0: unadjusted transition matrix

        Keyword Arguments
        -----------------
        - costs_qij: specification of transition costs directly, either as a 
            dictionary or as a numpy array (n x n). Dictionary is used to 
            overwrite defaults with (row, column) index tuples as keys mapping 
            to costs as values
        - costs_x: specification of prevalence costs directly, either as a 
            dictionary or as a numpy array (n x 1). Dictionary is used to
            overwrite defaults with the index as a key mapping to the costs as 
            a value
        - cost_basic: basic cost to use for individual land use transitions. If
            specified as a dictionary, must 
        - cost_factor_qii: scalar applied to cost_basic to create costs on
            diagonals as default
        - cost_factor_prev_default: default factor to use for prevalence costs.
            If None, is calculated as 10 times the sum of all transition costs.
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
            else 100*costs_transition.sum()
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
    


    def get_qp_component_vectors_euclidean_prevalence(self,
        x_0: np.ndarray,
        x_1: np.ndarray,
        weights: Union[np.ndarray, None] = None,
    ) -> Tuple[np.ndarray]:
        """
        Get the objective function component matrix and vectors necessary for
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

    def solve_mce(self,
        Q: np.ndarray,
        u: np.ndarray, 
        v: np.ndarray, 
        *,
        perturbation_diag: float = 0.000001,
        solver: str = "cvxopt",
        **kwargs,
    ) -> sco._optimize.OptimizeResult:
        """
        Solve the Minimize Calibration Error (MCE) problem (QP). Support 
            function for correct_transitions()

        Function Arguments
        ------------------
        - x: variable vector
        - u: initial prevalence
        - v: next-step prevalence

        Keyword Arguments
        -----------------
        - perturbation_diag: perturbation to apply to the diagonal (positive) to
            ensure P is positive definite, a requirement for some solvers.
        - solver: valid solver passed to qpsolvers.solve_qp()
        - **kwargs: passed to get_constraint_coeffs_error()
        """

        # get constraints 
        A_err, vec_lb, vec_ub = sf.call_with_varkwargs(
            self.get_constraint_coeffs_error,
            Q,
            dict_kwargs = kwargs,
            include_defaults = True,
        )
        
        A_rs = self.get_constraint_coeffs_row_stochastic(Q,)
        b_rs = np.ones(len(A_rs))

        # get objective components
        M, vec_c = self.get_qp_component_vectors_euclidean_prevalence(
            Q, u, v,
        )

        # double to set up for standard form [(1/2)x^tMx + Cx] and add perturbation
        M = 2*M + perturbation_diag
        sol = None

        try:
            sol = qpsolvers.solve_qp(
                M, 
                vec_c, 
                A = A_rs,
                b = b_rs,
                lb = vec_lb,
                ub = vec_ub,
                solver = solver,
            )

        except Exception as e:
            msg = "Error trying to solve 'Minimize Calibration Error' problem in QCorrector as Quadratic Program: {e}. Trying IPOPT..."
            #log
            warnings.warn(msg)
        
        if sol is not None:
            return sol


        ##  TRY IPOPT

        try:
            const = sf.call_with_varkwargs(
                self.get_constraints_mce,
                Q,
                u,
                v,
                dict_kwargs = kwargs,
                include_defaults = True,
            
            )

            sol = cyipopt.scipy_interface.minimize_ipopt(
                self.f_obj_mce,
                Q.flatten(),
                args = (u, v),
                constraints = const,
            )

            sol = None if not sol.success else sol.x
            

        except Exception as e:
            msg = "Error trying to solve 'Minimize Calibration Error' problem in QCorrector using IPOPT: {e}"
            sol = None
            raise RuntimeError(msg)
            

        return sol
    

    


    





    