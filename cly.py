"""Command-line interface for NS-AquaForte."""

import argparse
import sys
from pathlib import Path
from .core import make
from .benchmarks import compare_with_baseline, generate_phase_transition_suite, run_benchmark_suite


def main():
    """NS-AquaForte CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NS-AquaForte: LLM-Guided SAT Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve single instance
  ns-aquaforte solve problem.cnf
  
  # Compare with MiniSAT
  ns-aquaforte benchmark problem.cnf
  
  # Generate test suite
  ns-aquaforte generate --vars 100 --densities 3.5,4.0,4.5 --output ./tests
  
  # Run full benchmark suite
  ns-aquaforte suite ./instances/*.cnf --timeout 60
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Solve command
    solve_parser = subparsers.add_parser('solve', help='Solve SAT instance')
    solve_parser.add_argument('cnf_file', help='Path to CNF file')
    solve_parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds')
    solve_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark vs baseline')
    bench_parser.add_argument('cnf_file', help='Path to CNF file')
    bench_parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate test instances')
    gen_parser.add_argument('--vars', type=int, default=100, help='Number of variables')
    gen_parser.add_argument('--densities', type=str, default='3.5,4.0,4.26,4.5,5.0',
                           help='Comma-separated density values')
    gen_parser.add_argument('--instances', type=int, default=5, 
                           help='Instances per density')
    gen_parser.add_argument('--output', type=str, default='./instances',
                           help='Output directory')
    
    # Suite command
    suite_parser = subparsers.add_parser('suite', help='Run benchmark suite')
    suite_parser.add_argument('files', nargs='+', help='CNF files to benchmark')
    suite_parser.add_argument('--timeout', type=int, default=300, help='Timeout per instance')
    suite_parser.add_argument('--output', type=str, default='results.csv',
                             help='Output CSV file')
    
    args = parser.parse_args()
    
    if args.command == 'solve':
        solve_command(args)
    elif args.command == 'benchmark':
        benchmark_command(args)
    elif args.command == 'generate':
        generate_command(args)
    elif args.command == 'suite':
        suite_command(args)
    else:
        parser.print_help()
        sys.exit(1)


def solve_command(args):
    """Solve a single SAT instance."""
    from .core import load_cnf
    
    print(f"Loading {args.cnf_file}...")
    problem = load_cnf(args.cnf_file)
    
    print(f"Creating solver...")
    solver = make(timeout=args.timeout, verbose=args.verbose)
    
    print(f"Solving...")
    solution, stats = solver.run(problem)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if solution.satisfiable is True:
        print(f"✓ SAT - Satisfiable")
        if args.verbose and solution.assignment:
            print(f"  Assignment: {solution.assignment[:10]}...")
    elif solution.satisfiable is False:
        print(f"✗ UNSAT - Unsatisfiable")
    else:
        print(f"? TIMEOUT - Could not determine")
    
    print(f"\nTime: {stats['time']:.2f}s")
    print(f"Phase: {stats['detected_phase']}")
    print(f"Algorithm: {stats['selected_algorithm']}")
    if 'confidence' in stats:
        print(f"Confidence: {stats['confidence']:.2f}")


def benchmark_command(args):
    """Compare with baseline solver."""
    print(f"Running benchmark: {args.cnf_file}")
    print("="*60)
    
    results = compare_with_baseline(
        args.cnf_file,
        timeout=args.timeout,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    baseline_time = results['baseline_time']
    
    for solver in ['resolution', 'spectral', 'hybrid']:
        if f'{solver}_time' in results:
            time_val = results[f'{solver}_time']
            speedup = results[f'{solver}_speedup']
            
            marker = "🏆" if speedup == max([
                results.get(f'{s}_speedup', 0) 
                for s in ['resolution', 'spectral', 'hybrid']
            ]) else "  "
            
            print(f"{marker} {solver.capitalize():12s}: {time_val:.2f}s ({speedup:.2f}x)")
    
    print(f"\nBaseline (MiniSAT): {baseline_time:.2f}s")


def generate_command(args):
    """Generate test suite."""
    densities = [float(d) for d in args.densities.split(',')]
    
    print(f"Generating test suite:")
    print(f"  Variables: {args.vars}")
    print(f"  Densities: {densities}")
    print(f"  Instances per density: {args.instances}")
    print(f"  Output: {args.output}")
    
    files = generate_phase_transition_suite(
        output_dir=args.output,
        n_vars=args.vars,
        densities=densities,
        instances_per_density=args.instances
    )
    
    print(f"\n✓ Generated {len(files)} instances")


def suite_command(args):
    """Run benchmark suite."""
    print(f"Running benchmark suite on {len(args.files)} instances...")
    
    results = run_benchmark_suite(
        args.files,
        solvers=['resolution', 'spectral', 'hybrid'],
        baseline=True,
        timeout=args.timeout,
        output_file=args.output
    )
    
    print(f"\n✓ Results saved to {args.output}")
    
    # Print summary
    from .benchmarks import print_summary_table
    print_summary_table(results)


if __name__ == '__main__':
    main()
