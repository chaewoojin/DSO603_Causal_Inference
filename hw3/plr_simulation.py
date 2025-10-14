"""
PLR Simulation Study

This module implements a version of the PLR simulation that
handles high-dimensional problems.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning
from scipy import stats
from joblib import Parallel, delayed
import warnings


class PLRSimulation:
    """
    PLR simulation with options for high-dimensional settings.
    
    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs for simulation.
    """
    
    def __init__(self, random_state=42, n_jobs=-1):
        self.random_state = random_state
        np.random.seed(random_state)
        self.n_jobs = n_jobs
        
    def generate_data_setting1(self, n=500, p=10, rho=0.0, random_state=None):
        """
        Generate data for Setting 1: Low-dimensional, nonlinear nuisances.
        
        Parameters
        ----------
        n : int
            Number of samples.
        p : int
            Number of features.
        rho : float
            Correlation between U and V.
        random_state : int or None
            Random seed.
        
        Returns
        -------
        X : ndarray, shape (n, p)
            Covariate matrix.
        W : ndarray, shape (n,)
            Treatment variable.
        Y : ndarray, shape (n,)
            Outcome variable.
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        X = np.random.multivariate_normal(np.zeros(p), np.eye(p), n)
        
        cov_matrix = np.array([[1, rho], [rho, 1]])
        U, V = np.random.multivariate_normal(np.zeros(2), cov_matrix, n).T
        
        g0 = lambda x: np.sin(x[:, 0]) + x[:, 1]**2
        e0 = lambda x: 0.5 * x[:, 2] - 0.3 * x[:, 3]
        
        W = e0(X) + V
        Y = 2 * W + g0(X) + U
        
        return X, W, Y
    
    def generate_data_setting2(self, n, p, s, rho=0.0, random_state=None):
        """
        Generate data for Setting 2: High-dimensional, linear sparse nuisances.
        
        Parameters
        ----------
        n : int
            Number of samples.
        p : int
            Number of features.
        s : int
            Sparsity (number of nonzero coefficients).
        rho : float
            Correlation between U and V.
        random_state : int or None
            Random seed.
        
        Returns
        -------
        X : ndarray, shape (n, p)
            Covariate matrix.
        W : ndarray, shape (n,)
            Treatment variable.
        Y : ndarray, shape (n,)
            Outcome variable.
        beta0 : ndarray, shape (p,)
            True coefficients for Y.
        gamma0 : ndarray, shape (p,)
            True coefficients for W.
        S : ndarray, shape (s,)
            Indices of nonzero coefficients.
        """
        if random_state is not None:
            np.random.seed(random_state)
        # Generate covariance matrix for X with regularization
        Sigma_X = np.zeros((p, p))
        for j in range(p):
            for k in range(p):
                Sigma_X[j, k] = 0.5 ** abs(j - k)
        # Add small regularization for numerical stability
        Sigma_X += 1e-6 * np.eye(p)
        try:
            X = np.random.multivariate_normal(np.zeros(p), Sigma_X, n)
        except np.linalg.LinAlgError:
            X = np.random.multivariate_normal(np.zeros(p), np.eye(p), n)

        S = np.random.choice(p, size=s, replace=False)
        beta0 = np.zeros(p)
        gamma0 = np.zeros(p)
        beta_signs = np.random.choice([-1, 1], size=s)
        gamma_signs = np.random.choice([-1, 1], size=s)
        beta_magnitudes = np.random.uniform(0.3, 0.6, s)
        gamma_magnitudes = np.random.uniform(0.3, 0.6, s)
        beta0[S] = beta_signs * beta_magnitudes
        gamma0[S] = gamma_signs * gamma_magnitudes
        cov_matrix = np.array([[1, rho], [rho, 1]])
        U, V = np.random.multivariate_normal(np.zeros(2), cov_matrix, n).T

        # print(f"{X[:, S]}", flush=True)
        # print(f"{gamma0[S]}", flush=True)
        # print(f"{beta0}: {beta0}", flush=True)
        # print("X:", np.any(np.isnan(X[:, S])), np.any(np.isinf(X[:, S])), X[:, S].shape)
        # print("gamma0:", np.any(np.isnan(gamma0[S])), np.any(np.isinf(gamma0[S])), gamma0[S].shape)
        # print("beta0:", np.any(np.isnan(beta0)), np.any(np.isinf(beta0)), beta0[S].shape)

        # Only use active set S for multiplication
        W = X[:, S] @ gamma0[S] + V
        Y = 2 * W + X[:, S] @ beta0[S] + U
        
        return X, W, Y, beta0, gamma0, S
    
    def naive_plugin_ols(self, X, W, Y, setting=1):
        """
        Naïve plug-in OLS estimator.
        
        Parameters
        ----------
        X : ndarray, shape (n, p)
            Covariate matrix.
        W : ndarray, shape (n,)
            Treatment variable.
        Y : ndarray, shape (n,)
            Outcome variable.
        setting : int
            1 for low-dimensional, 2 for high-dimensional.
        
        Returns
        -------
        theta_hat : float
            Estimated treatment effect.
        """
        if setting == 1:
            rf = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=1)
            rf.fit(X, Y)
            g_hat = rf.predict(X)
        else:
            # Use Ridge as fallback if Lasso fails
            try:
                lasso = LassoCV(cv=10, random_state=self.random_state, n_jobs=1, 
                               alphas=np.logspace(-4, 1, 20), max_iter=2000)
                lasso.fit(X, Y)
                g_hat = lasso.predict(X)
            except:
                # Fallback to Ridge
                ridge = RidgeCV(cv=10, alphas=np.logspace(-4, 1, 20))
                ridge.fit(X, Y)
                g_hat = ridge.predict(X)
        
        # Check for numerical issues
        if np.any(np.isnan(g_hat)) or np.any(np.isinf(g_hat)):
            g_hat = np.zeros_like(Y)
        
        # Compute theta with numerical stability
        numerator = np.sum(W * (Y - g_hat))
        denominator = np.sum(W**2)
        
        if abs(denominator) < 1e-10:
            theta_hat = 0.0
        else:
            theta_hat = numerator / denominator
        
        return theta_hat
    
    def dml_estimator(self, X, W, Y, setting=1, K=5, return_models=False):
        """
        Double Machine Learning estimator with cross-fitting.
        
        Parameters
        ----------
        X : ndarray, shape (n, p)
            Covariate matrix.
        W : ndarray, shape (n,)
            Treatment variable.
        Y : ndarray, shape (n,)
            Outcome variable.
        setting : int
            1 for low-dimensional, 2 for high-dimensional.
        K : int
            Number of folds for cross-fitting.
        return_models : bool
            If True, return fitted nuisance models (for sparsity analysis).
        
        Returns
        -------
        theta_hat : float
            Estimated treatment effect.
        g_hat : ndarray, shape (n,)
            Estimated E[Y|X].
        e_hat : ndarray, shape (n,)
            Estimated E[W|X].
        g_models, e_models : list (optional)
            Fitted models for each fold (if return_models is True).
        """
        n = len(Y)
        kf = KFold(n_splits=K, shuffle=True, random_state=self.random_state)
        g_hat = np.zeros(n)
        e_hat = np.zeros(n)
        g_models = []
        e_models = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            W_train, W_val = W[train_idx], W[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            if setting == 1:
                g_model = GradientBoostingRegressor(n_estimators=50, random_state=self.random_state)
                e_model = GradientBoostingRegressor(n_estimators=50, random_state=self.random_state)
            else:
                try:
                    g_model = LassoCV(cv=10, random_state=self.random_state, n_jobs=1,
                                     alphas=np.logspace(-4, 0, 15), max_iter=2000)
                    e_model = LassoCV(cv=10, random_state=self.random_state, n_jobs=1,
                                     alphas=np.logspace(-4, 0, 15), max_iter=2000)
                except:
                    g_model = RidgeCV(cv=10, alphas=np.logspace(-4, 1, 20))
                    e_model = RidgeCV(cv=10, alphas=np.logspace(-4, 1, 20))
            try:
                g_model.fit(X_train, Y_train)
                g_hat[val_idx] = g_model.predict(X_val)
            except:
                g_hat[val_idx] = np.mean(Y_train)
            try:
                e_model.fit(X_train, W_train)
                e_hat[val_idx] = e_model.predict(X_val)
            except:
                e_hat[val_idx] = np.mean(W_train)
            if setting == 2 and return_models:
                g_models.append(g_model)
                e_models.append(e_model)
        g_hat = np.nan_to_num(g_hat, nan=0.0, posinf=0.0, neginf=0.0)
        e_hat = np.nan_to_num(e_hat, nan=0.0, posinf=0.0, neginf=0.0)
        Y_tilde = Y - g_hat
        W_tilde = W - e_hat
        numerator = np.sum(W_tilde * Y_tilde)
        denominator = np.sum(W_tilde**2)
        if abs(denominator) < 1e-10:
            theta_hat = 0.0
        else:
            theta_hat = numerator / denominator
        if setting == 2 and return_models:
            return theta_hat, g_hat, e_hat, g_models, e_models
        return theta_hat, g_hat, e_hat
    
    def compute_confidence_interval(self, X, W, Y, theta_hat, g_hat, e_hat, alpha=0.05):
        """
        Compute confidence interval for theta_hat.
        
        Parameters
        ----------
        X : ndarray, shape (n, p)
            Covariate matrix.
        W : ndarray, shape (n,)
            Treatment variable.
        Y : ndarray, shape (n,)
            Outcome variable.
        theta_hat : float
            Estimated treatment effect.
        g_hat : ndarray, shape (n,)
            Estimated E[Y|X].
        e_hat : ndarray, shape (n,)
            Estimated E[W|X].
        alpha : float
            Significance level for confidence interval.
        
        Returns
        -------
        ci_lower : float
            Lower bound of confidence interval.
        ci_upper : float
            Upper bound of confidence interval.
        se : float
            Standard error estimate.
        """
        n = len(Y)
        
        Y_tilde = Y - g_hat
        W_tilde = W - e_hat
        
        # Check for numerical issues
        Y_tilde = np.nan_to_num(Y_tilde, nan=0.0, posinf=0.0, neginf=0.0)
        W_tilde = np.nan_to_num(W_tilde, nan=0.0, posinf=0.0, neginf=0.0)
        
        J = np.mean(W_tilde**2)
        Omega = np.mean(W_tilde**2 * (Y_tilde - theta_hat * W_tilde)**2)
        
        # Check for numerical issues
        if J < 1e-10 or Omega < 1e-10:
            se = 1.0  # Large standard error as fallback
        else:
            var_theta = Omega / (J**2)
            se = np.sqrt(max(var_theta / n, 1e-10))  # Ensure positive variance
        
        z_alpha = stats.norm.ppf(1 - alpha/2)
        ci_lower = theta_hat - z_alpha * se
        ci_upper = theta_hat + z_alpha * se
        
        return ci_lower, ci_upper, se
    
    def single_replication_setting1(self, n, p, rho, r):
        """
        Single replication for Setting 1.
        
        Parameters
        ----------
        n : int
            Number of samples.
        p : int
            Number of features.
        rho : float
            Correlation between U and V.
        r : int
            Replication index (for random seed).
        
        Returns
        -------
        dict
            Dictionary with results for this replication.
        """
        try:
            # Generate data
            X, W, Y = self.generate_data_setting1(n, p, rho, self.random_state + r)
            
            # Naïve plug-in OLS
            theta_naive = self.naive_plugin_ols(X, W, Y, setting=1)
            
            # DML
            theta_dml, g_hat, e_hat = self.dml_estimator(X, W, Y, setting=1)
            
            # Confidence intervals
            ci_naive = self.compute_confidence_interval(X, W, Y, theta_naive, 
                                                       np.zeros(n), np.zeros(n))
            ci_dml = self.compute_confidence_interval(X, W, Y, theta_dml, g_hat, e_hat)
            
            return {
                'theta_naive': theta_naive,
                'theta_dml': theta_dml,
                'ci_naive': ci_naive,
                'ci_dml': ci_dml,
                'success': True
            }
        except Exception as e:
            return {
                'theta_naive': 0.0,
                'theta_dml': 0.0,
                'ci_naive': (0.0, 0.0, 1.0),
                'ci_dml': (0.0, 0.0, 1.0),
                'success': False,
                'error': str(e)
            }
    
    def single_replication_setting2(self, n, p, s, rho, r):
        """
        Single replication for Setting 2, with sparsity reporting.
        
        Parameters
        ----------
        n : int
            Number of samples.
        p : int
            Number of features.
        s : int
            Sparsity (number of nonzero coefficients).
        rho : float
            Correlation between U and V.
        r : int
            Replication index (for random seed).
        
        Returns
        -------
        dict
            Dictionary with results for this replication.
        """
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        np.seterr(all='ignore')
        try:
            # Generate data
            X, W, Y, beta0, gamma0, S = self.generate_data_setting2(n, p, s, rho, self.random_state + r)
            
            # Naïve plug-in OLS
            theta_naive = self.naive_plugin_ols(X, W, Y, setting=2)
            
            # DML with model return
            theta_dml, g_hat, e_hat, g_models, e_models = self.dml_estimator(X, W, Y, setting=2, return_models=True)
            
            # Confidence intervals
            ci_naive = self.compute_confidence_interval(X, W, Y, theta_naive, np.zeros(n), np.zeros(n))
            ci_dml = self.compute_confidence_interval(X, W, Y, theta_dml, g_hat, e_hat)
            
            # Compute selected sparsity for each fold, then average
            s_g_list = []
            s_e_list = []
            for gm, em in zip(g_models, e_models):
                # Only count for LassoCV models
                if hasattr(gm, 'coef_'):
                    s_g_list.append(np.sum(np.abs(gm.coef_) > 1e-8))
                if hasattr(em, 'coef_'):
                    s_e_list.append(np.sum(np.abs(em.coef_) > 1e-8))
            hat_s_g = float(np.mean(s_g_list)) if s_g_list else np.nan
            hat_s_e = float(np.mean(s_e_list)) if s_e_list else np.nan
            
            return {
                'theta_naive': theta_naive,
                'theta_dml': theta_dml,
                'ci_naive': ci_naive,
                'ci_dml': ci_dml,
                'S': S,
                'hat_s_g': hat_s_g,
                'hat_s_e': hat_s_e,
                'success': True
            }
        except Exception as e:
            return {
                'theta_naive': 0.0,
                'theta_dml': 0.0,
                'ci_naive': (0.0, 0.0, 1.0),
                'ci_dml': (0.0, 0.0, 1.0),
                'S': np.array([]),
                'hat_s_g': np.nan,
                'hat_s_e': np.nan,
                'success': False,
                'error': str(e)
            }
    
    def run_simulation_setting1(self, n=500, p=10, rho_values=[0.0, 0.3], R=100):
        """
        Run simulation for Setting 1.
        
        Parameters
        ----------
        n : int
            Number of samples.
        p : int
            Number of features.
        rho_values : list
            List of correlation values to simulate.
        R : int
            Number of replications.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with simulation results.
        """
        results = {
            'rho': [],
            'method': [],
            'bias': [],
            'variance': [],
            'rmse': [],
            'coverage': [],
            'ci_length': []
        }
        
        for rho in rho_values:
            print(f"Running Setting 1 with rho={rho} (R={R})")
            
            # Run parallel computation with error handling
            replication_results = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(self.single_replication_setting1)(n, p, rho, r) 
                for r in range(R)
            )
            
            # Filter successful replications
            successful_results = [r for r in replication_results if r['success']]
            failed_count = len(replication_results) - len(successful_results)
            
            if failed_count > 0:
                print(f"Warning: {failed_count} replications failed")
            
            if len(successful_results) < R // 2:
                print(f"Error: Too many failures ({failed_count}/{R}). Skipping rho={rho}")
                continue
            
            # Extract results
            naive_estimates = [r['theta_naive'] for r in successful_results]
            dml_estimates = [r['theta_dml'] for r in successful_results]
            naive_cis = [r['ci_naive'] for r in successful_results]
            dml_cis = [r['ci_dml'] for r in successful_results]
            
            # Compute metrics for both methods
            for method, estimates, cis in [('Naive', naive_estimates, naive_cis),
                                         ('DML', dml_estimates, dml_cis)]:
                estimates = np.array(estimates)
                bias = np.mean(estimates) - 2
                var = np.var(estimates)
                rmse = np.sqrt(np.mean((estimates - 2)**2))
                coverage = np.mean([1 if 2 >= ci[0] and 2 <= ci[1] else 0 for ci in cis])
                ci_length = np.mean([ci[1] - ci[0] for ci in cis])
                
                results['rho'].append(rho)
                results['method'].append(method)
                results['bias'].append(bias)
                results['variance'].append(var)
                results['rmse'].append(rmse)
                results['coverage'].append(coverage)
                results['ci_length'].append(ci_length)
        
        return pd.DataFrame(results)
    
    def run_simulation_setting2(self, n_values=[500, 1000], p_values=[800, 1200], 
                                     s_values=[10, 30], rho_values=[0.0, 0.3], R=100):
        """
        Run simulation for Setting 2, with sparsity reporting.
        
        Parameters
        ----------
        n_values : list
            List of sample sizes.
        p_values : list
            List of feature dimensions.
        s_values : list
            List of sparsity levels.
        rho_values : list
            List of correlation values.
        R : int
            Number of replications.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with simulation results.
        """
        results = {
            'n': [], 'p': [], 's': [], 'rho': [], 'method': [],
            'bias': [], 'variance': [], 'rmse': [], 'coverage': [], 'ci_length': [],
            'avg_sparsity_g': [], 'avg_sparsity_e': []
        }
        
        # Use more conservative parameters to avoid numerical issues
        for n in n_values:
            for p in p_values:
                if p <= n:
                    continue
                for s in s_values:
                    for rho in rho_values:
                        print(f"Running Setting 2 with n={n}, p={p}, s={s}, rho={rho} (R={R})")
                        # Run parallel computation with error handling
                        replication_results = Parallel(n_jobs=self.n_jobs, verbose=1)(
                            delayed(self.single_replication_setting2)(n, p, s, rho, r) 
                            for r in range(R)
                        )
                        
                        # Filter successful replications
                        successful_results = [r for r in replication_results if r['success']]
                        failed_count = len(replication_results) - len(successful_results)
                        
                        if failed_count > 0:
                            print(f"Warning: {failed_count} replications failed")
                        
                        if len(successful_results) < R // 2:
                            print(f"Error: Too many failures ({failed_count}/{R}). Skipping this configuration")
                            continue
                        
                        # Extract results
                        naive_estimates = [r['theta_naive'] for r in successful_results]
                        dml_estimates = [r['theta_dml'] for r in successful_results]
                        naive_cis = [r['ci_naive'] for r in successful_results]
                        dml_cis = [r['ci_dml'] for r in successful_results]
                        
                        # Collect sparsity for DML
                        sparsity_g = [r['hat_s_g'] for r in successful_results]
                        sparsity_e = [r['hat_s_e'] for r in successful_results]
                        
                        # Compute metrics for both methods
                        for method, estimates, cis in [('Naive', naive_estimates, naive_cis),
                                                     ('DML', dml_estimates, dml_cis)]:
                            estimates = np.array(estimates)
                            bias = np.mean(estimates) - 2
                            var = np.var(estimates)
                            rmse = np.sqrt(np.mean((estimates - 2)**2))
                            coverage = np.mean([1 if 2 >= ci[0] and 2 <= ci[1] else 0 for ci in cis])
                            ci_length = np.mean([ci[1] - ci[0] for ci in cis])
                            results['n'].append(n)
                            results['p'].append(p)
                            results['s'].append(s)
                            results['rho'].append(rho)
                            results['method'].append(method)
                            results['bias'].append(bias)
                            results['variance'].append(var)
                            results['rmse'].append(rmse)
                            results['coverage'].append(coverage)
                            results['ci_length'].append(ci_length)
                            if method == 'DML':
                                results['avg_sparsity_g'].append(np.nanmean(sparsity_g))
                                results['avg_sparsity_e'].append(np.nanmean(sparsity_e))
                            else:
                                results['avg_sparsity_g'].append(np.nan)
                                results['avg_sparsity_e'].append(np.nan)
        
        return pd.DataFrame(results)
    
    def run_quick_simulation(self, R=50):
        """
        Run a quick simulation for testing.
        
        Parameters
        ----------
        R : int
            Number of replications.
        
        Returns
        -------
        results1 : pd.DataFrame
            Results for Setting 1.
        results2 : pd.DataFrame
            Results for Setting 2.
        """
        print("Running Quick Simulation...")
        
        # Setting 1 with smaller parameters
        results1 = self.run_simulation_setting1(n=200, p=5, rho_values=[0.0, 0.3], R=R)
        
        # Setting 2 with much smaller parameters for stability
        results2 = self.run_simulation_setting2(
            n_values=[200], 
            p_values=[300],  
            s_values=[5], 
            rho_values=[0.0, 0.3], 
            R=R
        )
        
        return results1, results2


def main():
    """
    Main function to run the PLR simulation study.
    """
    print("Starting PLR Simulation Study...")
    print("This version includes numerical stability improvements")
    
    # Initialize simulation
    sim = PLRSimulation(random_state=42, n_jobs=-1)
    
    # Run quick simulation first
    print("\n" + "="*60)
    print("Running Quick Simulation (R=50)")
    print("="*60)
    results1_quick, results2_quick = sim.run_quick_simulation(R=50)
    
    print("Quick Results - Setting 1:")
    print(results1_quick.round(4))
    print("\nQuick Results - Setting 2:")
    print(results2_quick.round(4))
    
    # Ask user for full simulation
    response = input("\nRun full simulation? (y/n): ")
    
    if response.lower() == 'y':
        # Run Setting 1
        print("\n" + "="*60)
        print("Running Full Setting 1")
        print("="*60)
        results1 = sim.run_simulation_setting1()
        
        # Run Setting 2 with conservative parameters
        print("\n" + "="*60)
        print("Running Full Setting 2 (with conservative parameters)")
        print("="*60)
        results2 = sim.run_simulation_setting2(
            n_values=[500, 1000], 
            p_values=[800, 1200],  
            s_values=[10, 30],     
            rho_values=[0.0, 0.3], 
            R=100
        )
        
        # Save results
        results1.to_csv('results_setting1.csv', index=False)
        results2.to_csv('results_setting2.csv', index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("RESULTS - SETTING 1")
        print("="*60)
        print(results1.round(4))
        
        print("\n" + "="*60)
        print("RESULTS - SETTING 2")
        print("="*60)
        print(results2.round(4))
        
        print("\nSimulation completed! Results saved.")
    else:
        print("Skipping full simulation. Quick results completed.")

if __name__ == "__main__":
    main()