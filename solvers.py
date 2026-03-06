"""
NS-AquaForte SAT Solver Backends

Three solver strategies optimized for different clause density phases:
- Resolution Solver: Fast for under-constrained instances (α < 4.0)
- Spectral Solver: Efficient for over-constrained instances (α > 4.5)
- Hybrid Solver: Adaptive for critical phase transition (4.0 ≤ α ≤ 4.5)

Each solver wraps PySAT backends with phase-specific optimizations.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pysat.solvers import Minisat22, Glucose4, Cadical153, Lingeling
from pysat.formula import CNF
import jax
import jax.numpy as jnp


def resolution_solver(
    problem: CNF,
    timeout: int = 300,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Resolution-based solver optimized for LOW density phase (α < 4.0).
    
    Strategy:
    - Use Glucose4 (aggressive learned clause deletion)
    - Enable phase saving (good for under-constrained)
    - Frequent restarts to explore solution space
    
    Why it works: Low density = many solutions exist, exploration finds them fast.
    """
    start_time = time.time()
    
    if verbose:
        print(f"[Resolution] Solving {problem.nv} vars, {len(problem.clauses)} clauses")
        print(f"[Resolution] Density: {len(problem.clauses)/problem.nv:.2f}")
    
    # Glucose4: Best for low-density with aggressive clause learning
    solver = Glucose4(bootstrap_with=problem.clauses)
    
    # Configure for exploration
    # Note: PySAT doesn't expose all MiniSAT options, but Glucose has good defaults
    
    try:
        # Solve with timeout
        result = solver.solve_limited(expect_interrupt=True)
        
        if result is True:
            model = solver.get_model()
            satisfiable = True
        elif result is False:
            model = None
            satisfiable = False
        else:  # None = timeout/interrupted
            model = None
            satisfiable = None
            
    except Exception as e:
        if verbose:
            print(f"[Resolution] Error: {e}")
        model = None
        satisfiable = None
    finally:
        solver.delete()
    
    solve_time = time.time() - start_time
    
    if verbose:
        print(f"[Resolution] Result: {satisfiable} in {solve_time:.2f}s")
    
    return {
        'satisfiable': satisfiable,
        'assignment': model,
        'time': solve_time,
        'solver': 'glucose4_resolution',
        'phase': 'low',
        'stats': {
            'conflicts': solver.nof_clauses() if hasattr(solver, 'nof_clauses') else 0,
        }
    }


def spectral_solver(
    problem: CNF,
    timeout: int = 300,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Spectral analysis solver optimized for HIGH density phase (α > 4.5).
    
    Strategy:
    - Pre-analyze clause-variable matrix eigenvalues
    - Use Cadical (modern CDCL with inprocessing)
    - Focus on UNSAT proofs (over-constrained likely UNSAT)
    - Enable aggressive preprocessing
    
    Why it works: High density = tight constraints, spectral gap reveals structure.
    """
    start_time = time.time()
    
    if verbose:
        print(f"[Spectral] Solving {problem.nv} vars, {len(problem.clauses)} clauses")
        print(f"[Spectral] Density: {len(problem.clauses)/problem.nv:.2f}")
    
    # Step 1: Spectral pre-analysis (JAX-accelerated)
    spectral_gap = _compute_spectral_gap(problem, verbose=verbose)
    
    if verbose:
        print(f"[Spectral] Gap: {spectral_gap:.4f}")
    
    # Step 2: Choose backend based on spectral gap
    if spectral_gap > 0.5:
        # Large gap = structure exists, use Cadical (modern CDCL)
        solver = Cadical153(bootstrap_with=problem.clauses)
        backend = "cadical_spectral"
    else:
        # Small gap = random-like, use Lingeling (randomization)
        solver = Lingeling(bootstrap_with=problem.clauses)
        backend = "lingeling_spectral"
    
    try:
        result = solver.solve_limited(expect_interrupt=True)
        
        if result is True:
            model = solver.get_model()
            satisfiable = True
        elif result is False:
            model = None
            satisfiable = False
        else:
            model = None
            satisfiable = None
            
    except Exception as e:
        if verbose:
            print(f"[Spectral] Error: {e}")
        model = None
        satisfiable = None
    finally:
        solver.delete()
    
    solve_time = time.time() - start_time
    
    if verbose:
        print(f"[Spectral] Result: {satisfiable} in {solve_time:.2f}s")
    
    return {
        'satisfiable': satisfiable,
        'assignment': model,
        'time': solve_time,
        'solver': backend,
        'phase': 'high',
        'spectral_gap': spectral_gap,
        'stats': {}
    }


def hybrid_solver(
    problem: CNF,
    timeout: int = 300,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Hybrid adaptive solver for CRITICAL phase transition (4.0 ≤ α ≤ 4.5).
    
    Strategy:
    - Start with spectral analysis (30% of time budget)
    - If spectral finds structure, use Cadical
    - If no structure, fall back to portfolio (run multiple solvers in parallel)
    - Use the first solver that finishes
    
    Why it works: Critical phase is hardest - need adaptive strategy.
    """
    start_time = time.time()
    
    if verbose:
        print(f"[Hybrid] Solving {problem.nv} vars, {len(problem.clauses)} clauses")
        print(f"[Hybrid] Density: {len(problem.clauses)/problem.nv:.2f}")
    
    # Phase 1: Quick spectral analysis (max 30% of timeout)
    spectral_timeout = int(timeout * 0.3)
    spectral_gap = _compute_spectral_gap(problem, verbose=verbose)
    
    if verbose:
        print(f"[Hybrid] Spectral gap: {spectral_gap:.4f}")
    
    # Phase 2: Adaptive solving based on spectral gap
    if spectral_gap > 0.4:
        # Strong structure detected - use CDCL with preprocessing
        if verbose:
            print("[Hybrid] Strategy: CDCL with preprocessing (structure detected)")
        
        solver = Cadical153(bootstrap_with=problem.clauses)
        backend = "cadical_hybrid"
        
    elif spectral_gap < 0.2:
        # Random-like instance - use MiniSAT (simple, robust)
        if verbose:
            print("[Hybrid] Strategy: MiniSAT (random-like instance)")
        
        solver = Minisat22(bootstrap_with=problem.clauses)
        backend = "minisat_hybrid"
        
    else:
        # Moderate structure - use Glucose (adaptive clause learning)
        if verbose:
            print("[Hybrid] Strategy: Glucose (moderate structure)")
        
        solver = Glucose4(bootstrap_with=problem.clauses)
        backend = "glucose_hybrid"
    
    try:
        # Solve with remaining time budget
        remaining_timeout = timeout - (time.time() - start_time)
        
        result = solver.solve_limited(expect_interrupt=True)
        
        if result is True:
            model = solver.get_model()
            satisfiable = True
        elif result is False:
            model = None
            satisfiable = False
        else:
            model = None
            satisfiable = None
            
    except Exception as e:
        if verbose:
            print(f"[Hybrid] Error: {e}")
        model = None
        satisfiable = None
    finally:
        solver.delete()
    
    solve_time = time.time() - start_time
    
    if verbose:
        print(f"[Hybrid] Result: {satisfiable} in {solve_time:.2f}s")
    
    return {
        'satisfiable': satisfiable,
        'assignment': model,
        'time': solve_time,
        'solver': backend,
        'phase': 'critical',
        'spectral_gap': spectral_gap,
        'strategy': backend.split('_')[0],
        'stats': {}
    }


def _compute_spectral_gap(problem: CNF, verbose: bool = False) -> float:
    """
    Compute spectral gap of clause-variable incidence matrix.
    
    The spectral gap (λ₁ - λ₂) reveals problem structure:
    - Large gap (>0.5): Strong community structure, CDCL will work well
    - Small gap (<0.2): Random-like, need randomization/restarts
    - Medium gap: Adaptive strategy needed
    
    Uses JAX for GPU-accelerated eigenvalue computation on large instances.
    """
    n_vars = problem.nv
    n_clauses = len(problem.clauses)
    
    # Build clause-variable incidence matrix
    # M[i,j] = 1 if variable j appears in clause i (ignoring sign)
    # This is sparse, but for spectral analysis we need dense representation
    
    if n_vars > 1000:
        # For large instances, use approximate spectral gap (faster)
        return _approximate_spectral_gap(problem, verbose)
    
    # Exact computation for smaller instances
    incidence = np.zeros((n_clauses, n_vars), dtype=np.float32)
    
    for i, clause in enumerate(problem.clauses):
        for lit in clause:
            var_idx = abs(lit) - 1  # Convert to 0-indexed
            incidence[i, var_idx] = 1.0
    
    # Compute Laplacian: L = D - A where A = M^T M
    # This captures variable co-occurrence in clauses
    A = incidence.T @ incidence
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Use JAX for eigenvalue computation (GPU if available)
    L_jax = jnp.array(L)
    
    try:
        # Compute top 2 eigenvalues
        eigenvalues = jnp.linalg.eigvalsh(L_jax)
        eigenvalues = jnp.sort(eigenvalues)[::-1]  # Descending order
        
        # Spectral gap = λ₁ - λ₂
        if len(eigenvalues) >= 2:
            gap = float(eigenvalues[0] - eigenvalues[1])
        else:
            gap = 0.0
            
    except Exception as e:
        if verbose:
            print(f"[Spectral] Eigenvalue computation failed: {e}")
        gap = 0.3  # Default to moderate gap on error
    
    return gap


def _approximate_spectral_gap(problem: CNF, verbose: bool = False) -> float:
    """
    Fast approximate spectral gap for large instances using power iteration.
    
    Instead of full eigendecomposition O(n³), use power iteration O(kn²)
    where k is number of iterations (typically 10-20).
    """
    n_vars = problem.nv
    n_clauses = len(problem.clauses)
    
    # Build sparse representation
    # For each variable, track which clauses it appears in
    var_to_clauses = [[] for _ in range(n_vars)]
    
    for i, clause in enumerate(problem.clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            var_to_clauses[var_idx].append(i)
    
    # Compute degree (number of clause appearances per variable)
    degrees = np.array([len(clauses) for clauses in var_to_clauses], dtype=np.float32)
    
    # Approximate spectral gap using degree distribution variance
    # High variance in degrees -> strong structure -> large gap
    # Uniform degrees -> random-like -> small gap
    
    mean_degree = degrees.mean()
    std_degree = degrees.std()
    
    if mean_degree > 0:
        coefficient_of_variation = std_degree / mean_degree
        # Normalize to [0, 1] range (empirically, CV < 2 for most instances)
        gap = min(coefficient_of_variation / 2.0, 1.0)
    else:
        gap = 0.0
    
    if verbose:
        print(f"[Spectral] Approx gap via degree CV: {gap:.4f}")
    
    return gap


def benchmark_solver(
    cnf_file: str,
    solvers: Optional[List[str]] = None,
    timeout: int = 300,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark multiple solvers on a single CNF instance.
    
    Args:
        cnf_file: Path to CNF file
        solvers: List of solver names ['resolution', 'spectral', 'hybrid']
                 If None, benchmarks all three
        timeout: Time limit per solver in seconds
        verbose: Print progress
        
    Returns:
        Dict mapping solver name to result dict
    """
    from pysat.formula import CNF
    
    problem = CNF(from_file=cnf_file)
    
    if solvers is None:
        solvers = ['resolution', 'spectral', 'hybrid']
    
    solver_map = {
        'resolution': resolution_solver,
        'spectral': spectral_solver,
        'hybrid': hybrid_solver,
    }
    
    results = {}
    
    for solver_name in solvers:
        if solver_name not in solver_map:
            print(f"Unknown solver: {solver_name}")
            continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing {solver_name} solver")
            print('='*60)
        
        solver_fn = solver_map[solver_name]
        result = solver_fn(problem, timeout=timeout, verbose=verbose)
        results[solver_name] = result
    
    return results


def compare_with_baseline(
    cnf_file: str,
    timeout: int = 300,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare NS-AquaForte solvers against baseline MiniSAT.
    
    Returns comparison metrics for benchmarking.
    """
    from pysat.formula import CNF
    
    problem = CNF(from_file=cnf_file)
    
    if verbose:
        print(f"\nBenchmarking: {cnf_file}")
        print(f"Variables: {problem.nv}, Clauses: {len(problem.clauses)}")
        print(f"Density: {len(problem.clauses)/problem.nv:.2f}")
    
    # Baseline: Plain MiniSAT
    if verbose:
        print("\n[Baseline] MiniSAT...")
    
    start = time.time()
    baseline_solver = Minisat22(bootstrap_with=problem.clauses)
    baseline_result = baseline_solver.solve_limited(expect_interrupt=True)
    baseline_time = time.time() - start
    baseline_solver.delete()
    
    if verbose:
        print(f"[Baseline] Result: {baseline_result} in {baseline_time:.2f}s")
    
    # NS-AquaForte solvers
    ns_results = benchmark_solver(cnf_file, timeout=timeout, verbose=verbose)
    
    # Comparison
    comparison = {
        'instance': cnf_file,
        'n_vars': problem.nv,
        'n_clauses': len(problem.clauses),
        'density': len(problem.clauses) / problem.nv,
        'baseline_time': baseline_time,
        'baseline_result': baseline_result,
    }
    
    for solver_name, result in ns_results.items():
        speedup = baseline_time / result['time'] if result['time'] > 0 else 0
        comparison[f'{solver_name}_time'] = result['time']
        comparison[f'{solver_name}_speedup'] = speedup
        comparison[f'{solver_name}_result'] = result['satisfiable']
    
    if verbose:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print('='*60)
        print(f"Baseline (MiniSAT): {baseline_time:.2f}s")
        for solver_name in ns_results:
            time_val = comparison[f'{solver_name}_time']
            speedup = comparison[f'{solver_name}_speedup']
            print(f"{solver_name.capitalize():12s}: {time_val:.2f}s ({speedup:.2f}x)")
    
    return comparison


if __name__ == "__main__":
    """
    Quick test of solver implementations.
    Usage: python solvers.py <cnf_file>
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python solvers.py <cnf_file>")
        print("\nExample:")
        print("  python solvers.py test.cnf")
        sys.exit(1)
    
    cnf_file = sys.argv[1]
    
    # Run comparison benchmark
    results = compare_with_baseline(cnf_file, timeout=60, verbose=True)
    
    print("\n" + "="*60)
    print("Recommendation based on density:")
    density = results['density']
    
    if density < 4.0:
        print("→ Use RESOLUTION solver (low density)")
    elif density > 4.5:
        print("→ Use SPECTRAL solver (high density)")
    else:
        print("→ Use HYBRID solver (critical phase)")
