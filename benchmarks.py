"""
NS-AquaForte Benchmarking Utilities

Tools for systematic evaluation of NS-AquaForte against baseline solvers.
Includes data generation, batch processing, and result visualization.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    instance: str
    n_vars: int
    n_clauses: int
    density: float
    solver: str
    time: float
    satisfiable: Optional[bool]
    phase: str
    spectral_gap: Optional[float] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


def generate_random_3sat(
    n_vars: int,
    n_clauses: int,
    seed: int = 42
) -> List[List[int]]:
    """
    Generate random 3-SAT instance.
    
    Args:
        n_vars: Number of variables
        n_clauses: Number of clauses
        seed: Random seed for reproducibility
        
    Returns:
        List of clauses (each clause is list of 3 literals)
    """
    np.random.seed(seed)
    
    clauses = []
    for _ in range(n_clauses):
        # Pick 3 distinct variables
        vars_in_clause = np.random.choice(n_vars, size=3, replace=False) + 1
        
        # Randomly negate (50% chance per literal)
        signs = np.random.choice([-1, 1], size=3)
        clause = [int(sign * var) for sign, var in zip(signs, vars_in_clause)]
        
        clauses.append(clause)
    
    return clauses


def save_cnf(clauses: List[List[int]], filepath: str, n_vars: Optional[int] = None):
    """
    Save clauses to CNF file in DIMACS format.
    
    Format:
        p cnf <n_vars> <n_clauses>
        <lit1> <lit2> <lit3> 0
        ...
    """
    if n_vars is None:
        # Infer from max variable index
        n_vars = max(abs(lit) for clause in clauses for lit in clause)
    
    n_clauses = len(clauses)
    
    with open(filepath, 'w') as f:
        f.write(f"p cnf {n_vars} {n_clauses}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")


def generate_phase_transition_suite(
    output_dir: str,
    n_vars: int = 250,
    densities: Optional[List[float]] = None,
    instances_per_density: int = 5,
    seed: int = 42
) -> List[str]:
    """
    Generate suite of instances across phase transition.
    
    Args:
        output_dir: Directory to save CNF files
        n_vars: Number of variables
        densities: List of clause/var ratios (default: 3.0 to 6.0)
        instances_per_density: How many instances per density point
        seed: Base random seed
        
    Returns:
        List of generated CNF file paths
    """
    if densities is None:
        # Sample across phase transition
        densities = np.linspace(3.0, 6.0, 13)  # 3.0, 3.25, ..., 6.0
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    for alpha in densities:
        n_clauses = int(alpha * n_vars)
        
        for trial in range(instances_per_density):
            # Generate instance
            instance_seed = seed + int(alpha * 1000) + trial
            clauses = generate_random_3sat(n_vars, n_clauses, seed=instance_seed)
            
            # Save to file
            filename = f"phase_alpha_{alpha:.2f}_trial_{trial}.cnf"
            filepath = os.path.join(output_dir, filename)
            save_cnf(clauses, filepath, n_vars=n_vars)
            
            generated_files.append(filepath)
    
    print(f"Generated {len(generated_files)} instances in {output_dir}")
    return generated_files


def run_benchmark_suite(
    cnf_files: List[str],
    solvers: List[str] = ['resolution', 'spectral', 'hybrid'],
    baseline: bool = True,
    timeout: int = 300,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Run systematic benchmarks on a suite of CNF instances.
    
    Args:
        cnf_files: List of CNF file paths
        solvers: Which NS-AquaForte solvers to test
        baseline: Whether to include MiniSAT baseline
        timeout: Time limit per instance per solver
        output_file: CSV file to save results (optional)
        
    Returns:
        DataFrame with all benchmark results
    """
    from .solvers import resolution_solver, spectral_solver, hybrid_solver
    from pysat.formula import CNF
    from pysat.solvers import Minisat22
    
    solver_map = {
        'resolution': resolution_solver,
        'spectral': spectral_solver,
        'hybrid': hybrid_solver,
    }
    
    results = []
    
    for cnf_file in tqdm(cnf_files, desc="Benchmarking"):
        try:
            problem = CNF(from_file=cnf_file)
            
            instance_info = {
                'instance': os.path.basename(cnf_file),
                'n_vars': problem.nv,
                'n_clauses': len(problem.clauses),
                'density': len(problem.clauses) / problem.nv,
            }
            
            # Baseline MiniSAT
            if baseline:
                start = time.time()
                baseline_solver = Minisat22(bootstrap_with=problem.clauses)
                baseline_result = baseline_solver.solve_limited(expect_interrupt=True)
                baseline_time = time.time() - start
                baseline_solver.delete()
                
                results.append(BenchmarkResult(
                    solver='minisat_baseline',
                    time=baseline_time,
                    satisfiable=baseline_result if baseline_result is not None else None,
                    phase='baseline',
                    **instance_info
                ))
            
            # NS-AquaForte solvers
            for solver_name in solvers:
                solver_fn = solver_map[solver_name]
                result = solver_fn(problem, timeout=timeout, verbose=False)
                
                results.append(BenchmarkResult(
                    solver=solver_name,
                    time=result['time'],
                    satisfiable=result['satisfiable'],
                    phase=result['phase'],
                    spectral_gap=result.get('spectral_gap'),
                    **instance_info
                ))
        
        except Exception as e:
            print(f"Error processing {cnf_file}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame([r.to_dict() for r in results])
    
    # Save if requested
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return df


def analyze_results(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze benchmark results and compute summary statistics.
    
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Overall statistics
    solvers = df['solver'].unique()
    
    for solver in solvers:
        solver_df = df[df['solver'] == solver]
        
        analysis[solver] = {
            'mean_time': solver_df['time'].mean(),
            'median_time': solver_df['time'].median(),
            'std_time': solver_df['time'].std(),
            'instances_solved': solver_df['satisfiable'].notna().sum(),
            'sat_instances': (solver_df['satisfiable'] == True).sum(),
            'unsat_instances': (solver_df['satisfiable'] == False).sum(),
        }
    
    # Compute speedups vs baseline
    if 'minisat_baseline' in solvers:
        baseline_df = df[df['solver'] == 'minisat_baseline'].set_index('instance')
        
        for solver in solvers:
            if solver == 'minisat_baseline':
                continue
            
            solver_df = df[df['solver'] == solver].set_index('instance')
            
            # Merge on instance to compare same problems
            merged = baseline_df[['time']].join(
                solver_df[['time']], 
                rsuffix='_solver',
                how='inner'
            )
            
            speedups = merged['time'] / merged['time_solver']
            
            analysis[solver]['speedup_vs_baseline'] = {
                'mean': speedups.mean(),
                'median': speedups.median(),
                'std': speedups.std(),
                'min': speedups.min(),
                'max': speedups.max(),
            }
    
    # Phase-specific analysis
    phases = df['phase'].unique()
    analysis['by_phase'] = {}
    
    for phase in phases:
        if phase == 'baseline':
            continue
        
        phase_df = df[df['phase'] == phase]
        analysis['by_phase'][phase] = {
            'mean_time': phase_df['time'].mean(),
            'instances': len(phase_df),
        }
    
    return analysis


def plot_phase_transition_performance(
    df: pd.DataFrame,
    output_file: Optional[str] = None
):
    """
    Plot performance across phase transition.
    
    Creates two subplots:
    1. Solve time vs clause density
    2. Speedup vs clause density
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get unique solvers
    solvers = df['solver'].unique()
    colors = sns.color_palette("husl", len(solvers))
    
    # Plot 1: Solve time vs density
    for solver, color in zip(solvers, colors):
        solver_df = df[df['solver'] == solver]
        
        # Group by density and compute mean
        grouped = solver_df.groupby('density')['time'].agg(['mean', 'std'])
        
        ax1.plot(grouped.index, grouped['mean'], 'o-', 
                label=solver, color=color, linewidth=2, markersize=6)
        ax1.fill_between(grouped.index, 
                         grouped['mean'] - grouped['std'],
                         grouped['mean'] + grouped['std'],
                         alpha=0.2, color=color)
    
    ax1.axvline(x=4.26, color='red', linestyle='--', alpha=0.5, label='Critical Phase')
    ax1.set_xlabel('Clause Density (α = clauses/vars)', fontsize=12)
    ax1.set_ylabel('Solve Time (seconds)', fontsize=12)
    ax1.set_title('Performance Across Phase Transition', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Speedup vs density (NS-AquaForte vs baseline)
    if 'minisat_baseline' in solvers:
        baseline_df = df[df['solver'] == 'minisat_baseline'].set_index(['instance', 'density'])
        
        for solver, color in zip(solvers, colors):
            if solver == 'minisat_baseline':
                continue
            
            solver_df = df[df['solver'] == solver].set_index(['instance', 'density'])
            
            # Compute speedup
            merged = baseline_df[['time']].join(
                solver_df[['time']], 
                rsuffix='_solver',
                how='inner'
            ).reset_index()
            
            merged['speedup'] = merged['time'] / merged['time_solver']
            
            # Group by density
            grouped = merged.groupby('density')['speedup'].agg(['mean', 'std'])
            
            ax2.plot(grouped.index, grouped['mean'], 'o-',
                    label=solver, color=color, linewidth=2, markersize=6)
            ax2.fill_between(grouped.index,
                             grouped['mean'] - grouped['std'],
                             grouped['mean'] + grouped['std'],
                             alpha=0.2, color=color)
        
        ax2.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5)
        ax2.axvline(x=4.26, color='red', linestyle='--', alpha=0.5, label='Critical Phase')
    
    ax2.set_xlabel('Clause Density (α = clauses/vars)', fontsize=12)
    ax2.set_ylabel('Speedup vs MiniSAT', fontsize=12)
    ax2.set_title('NS-AquaForte Speedup by Phase', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.show()


def print_summary_table(df: pd.DataFrame):
    """
    Print formatted summary table of results.
    """
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Overall stats by solver
    print("\nOverall Performance:")
    print("-" * 80)
    
    summary = df.groupby('solver').agg({
        'time': ['mean', 'median', 'std'],
        'satisfiable': 'count'
    }).round(2)
    
    print(summary)
    
    # Speedup vs baseline
    if 'minisat_baseline' in df['solver'].values:
        print("\nSpeedup vs MiniSAT Baseline:")
        print("-" * 80)
        
        baseline_df = df[df['solver'] == 'minisat_baseline'].set_index('instance')
        
        for solver in df['solver'].unique():
            if solver == 'minisat_baseline':
                continue
            
            solver_df = df[df['solver'] == solver].set_index('instance')
            
            merged = baseline_df[['time']].join(
                solver_df[['time']], 
                rsuffix='_solver',
                how='inner'
            )
            
            speedups = merged['time'] / merged['time_solver']
            
            print(f"\n{solver}:")
            print(f"  Mean speedup:   {speedups.mean():.2f}x")
            print(f"  Median speedup: {speedups.median():.2f}x")
            print(f"  Best speedup:   {speedups.max():.2f}x")
            print(f"  Worst speedup:  {speedups.min():.2f}x")
    
    # Phase-specific performance
    print("\nPerformance by Phase:")
    print("-" * 80)
    
    phase_summary = df[df['solver'] != 'minisat_baseline'].groupby(['solver', 'phase']).agg({
        'time': 'mean',
        'instance': 'count'
    }).round(2)
    
    print(phase_summary)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    """
    Example benchmark workflow.
    """
    print("NS-AquaForte Benchmark Suite")
    print("="*80)
    
    # Generate test instances
    print("\n1. Generating phase transition instances...")
    cnf_files = generate_phase_transition_suite(
        output_dir='./benchmark_instances',
        n_vars=100,  # Start small for testing
        densities=[3.5, 4.0, 4.26, 4.5, 5.0],
        instances_per_density=3
    )
    
    # Run benchmarks
    print("\n2. Running benchmarks...")
    results_df = run_benchmark_suite(
        cnf_files,
        solvers=['resolution', 'spectral', 'hybrid'],
        baseline=True,
        timeout=60,
        output_file='benchmark_results.csv'
    )
    
    # Analyze results
    print("\n3. Analyzing results...")
    analysis = analyze_results(results_df)
    print(json.dumps(analysis, indent=2))
    
    # Print summary
    print_summary_table(results_df)
    
    # Plot results
    print("\n4. Generating plots...")
    plot_phase_transition_performance(
        results_df,
        output_file='phase_transition_performance.png'
    )
    
    print("\nBenchmark complete!")
