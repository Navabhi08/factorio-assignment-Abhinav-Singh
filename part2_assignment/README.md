# Factory Steady State & Bounded Belts

This repository contains an implementation of the ERP.AI Part 2 assessment.  The
solution is organised as a single Python package under `part2_assignment/` and
provides two deterministic command line tools:

* `factory/main.py` computes steady-state crafting rates that satisfy machine
  and supply limits.
* `belts/main.py` computes bounded belt flows with lower and upper capacity
  constraints.

The code base is dependency-free (Python 3.11 standard library only) so that it
is simple to run in constrained environments.

## Solution architecture

### Linear optimisation helpers

The linear-programming routines now live alongside the factory solver in
`factory/main.py`.  Keeping everything in a single module simplifies the folder
layout while still exposing the same utilities to:

* Construct coefficient matrices in a deterministic recipe order.
* Invoke a simplex-style feasibility search implemented in pure Python.
* Recover primal and dual certificates that are later formatted as JSON.

The belts solver retains its standalone implementation and no longer depends on
a separate helper package.

### Factory solver

`factory/main.py` models the production network using recipe and item sets that
are pre-sorted to guarantee repeatable output.  The solver assembles a linear
program with the following constraints:

1. Each recipe produces or consumes items according to its stoichiometry.
2. Optional productivity and speed modules adjust the effective output and
   craft time for every machine type.
3. Raw resource usage is clamped to the configured supply caps and machine
   counts are bounded.

The optimisation first attempts to satisfy the requested production rate.  If
no feasible solution exists, it performs a binary search on the target rate
while reusing the same constraint matrix to find the maximum achievable
throughput.  Bottleneck hints are derived from the dual solution and sorted to
provide deterministic guidance.

### Belts solver

`belts/core.py` transforms the bounded flow problem into a standard max-flow
instance by splitting nodes, shifting lower bounds, and introducing a
super-source/sink pair.  The transformation mirrors the process outlined in the
assessment PDF and results in a single call to a deterministic Edmonds–Karp
routine.  When infeasible, the solver reports the offending min-cut along with
per-edge deficit information to help the reader understand the blockage.

### Command line interfaces

`factory/main.py` and `belts/main.py` are thin wrappers that:

1. Read a JSON payload from `stdin`.
2. Call the corresponding solver.
3. Emit a minified JSON object to `stdout` with keys ordered for stability.

Because both modules live inside a package, they can be executed either as
scripts (`python part2_assignment/factory/main.py`) or via `python -m`.

## Optional utilities

The assessment allows several optional helpers and generators.  They are
provided under `part2_assignment/tests/`:

* `verify_factory.py` and `verify_belts.py` perform quick post-solution
  validations against the JSON contract.
* `gen_factory.py` and `gen_belts.py` generate deterministic pseudo-random test
  instances that stress the solver edge cases.

These utilities are intentionally lightweight so that they run quickly during a
live defence session while still demonstrating how the solvers can be probed.

## Testing strategy

Unit tests in `part2_assignment/tests/test_factory.py` and
`part2_assignment/tests/test_belts.py` capture representative scenarios for
both domains (feasible, infeasible, and tie-breaking cases).  The tests insert
the repository root onto `sys.path`, allowing direct imports without
installation.

The repository also ships a `run_samples.py` helper that mirrors the commands in
`RUN.md` so that reviewers can reproduce all checks with a single script.

## Determinism and logging

All modules avoid non-deterministic language features (such as unordered set
iteration) by relying on explicit sorting.  No debug logging is emitted; only
the final JSON payload is written to `stdout` to comply with the CLI contract.
