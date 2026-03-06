"""Core NS-AquaForte solver implementation.

LLM-guided phase detection + adaptive SAT solver selection using JAX-based algorithms.
"""

import os
import time
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from pysat.formula import CNF

from .phase_detection import detect_phase_llm
from .solvers import resolution_solver, spectral_solver, hybrid_solver


def load_cnf(filepath: str) -> CNF:
    """Load a CNF file into pysat's internal representation."""
    return CNF(from_file=filepath)


def make(
    llm_provider: str = "anthropic",
    llm_model: str = "claude-sonnet-4",
    api_key: Optional[str] = None,
    timeout: int = 300,
    verbose: bool = True,
) -> "NSAquaForteSolver":
    """
    Factory function to create an NS-AquaForte solver instance.

    Args:
        llm_provider: LLM backend ("anthropic", "openai", "google", "xai", ...)
        llm_model: Specific model name for the chosen provider
        api_key: Optional explicit API key (falls back to env var)
        timeout: Maximum time (seconds) for the solver phase
        verbose: Whether to print progress information

    Returns:
        Configured NSAquaForteSolver instance
    """
    class NSAquaForteSolver:
        def __init__(self):
            self.llm_provider = llm_provider.lower()
            self.llm_model = llm_model
            self.api_key = api_key
            self.timeout = timeout
            self.verbose = verbose

            # Early validation
            if self.api_key is None:
                env_var = _get_api_key_env_var(self.llm_provider)
                self.api_key = os.environ.get(env_var)
                if self.api_key is None:
                    raise ValueError(
                        f"No API key provided for provider '{self.llm_provider}' "
                        f"and environment variable '{env_var}' is not set."
                    )

        def run(self, problem: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """
            Solve a SAT instance using LLM-guided phase detection and adaptive solver.

            Args:
                problem: A pysat CNF object or compatible SAT instance

            Returns:
                (solution, stats): solver output + metadata
            """
            start_time = time.perf_counter()

            if self.verbose:
                print(f"[NS-AquaForte] Starting solve (timeout={self.timeout}s)")

            # Phase 1: LLM phase detection
            try:
                phase, confidence = detect_phase_llm(
                    problem=problem,
                    provider=self.llm_provider,
                    model=self.llm_model,
                    api_key=self.api_key,
                )
            except Exception as e:
                if self.verbose:
                    print(f"[NS-AquaForte] LLM phase detection failed: {e}")
                phase, confidence = "critical", 0.5  # safe fallback

            if self.verbose:
                print(f"[NS-AquaForte] Phase detected: {phase} (confidence: {confidence:.2f}) "
                      f"using {self.llm_provider}/{self.llm_model}")

            # Phase 2: Algorithm selection
            solver_map = {
                "low": resolution_solver,
                "high": spectral_solver,
                "critical": hybrid_solver,
            }
            solver_fn = solver_map.get(phase, hybrid_solver)  # default to hybrid

            if self.verbose:
                print(f"[NS-AquaForte] Selected solver: {solver_fn.__name__}")

            # Phase 3: Execute solver (can be jitted in the solver implementations)
            solution = solver_fn(problem, timeout=self.timeout)

            total_time = time.perf_counter() - start_time

            # Gather statistics
            stats = {
                "detected_phase": phase,
                "phase_confidence": confidence,
                "selected_algorithm": solver_fn.__name__,
                "solve_time_seconds": solution.get("time", 0.0),
                "total_time_seconds": total_time,
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "timeout_used": self.timeout,
            }

            if self.verbose:
                print(f"[NS-AquaForte] Solve completed in {total_time:.2f}s")

            return solution, stats

        def __repr__(self) -> str:
            return (f"NSAquaForteSolver(provider={self.llm_provider!r}, "
                    f"model={self.llm_model!r}, timeout={self.timeout}s)")

    return NSAquaForteSolver()


def _get_api_key_env_var(provider: str) -> str:
    """Map provider name to conventional environment variable name."""
    mapping = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GEMINI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "xai": "XAI_API_KEY",
        "grok": "XAI_API_KEY",
        # Extend here for Mistral, Cohere, Ollama, etc.
    }
    return mapping.get(provider, "LLM_API_KEY")


# Optional: convenience aliases
def create_solver(**kwargs):
    """Alias for make()"""
    return make(**kwargs)
