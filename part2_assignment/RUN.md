# Running the solution

All commands should be executed from the repository root so that the
`part2_assignment` package is on the module search path.

## Quick validation

```bash
python part2_assignment/run_samples.py "python part2_assignment/factory/main.py" "python part2_assignment/belts/main.py"
```

The helper executes a pair of smoke tests using inline JSON payloads and checks
that the returned JSON is valid.

## Unit tests

```bash
FACTORY_CMD="python part2_assignment/factory/main.py" \
BELTS_CMD="python part2_assignment/belts/main.py" \
pytest -q part2_assignment/tests
```

The environment variables are read by the optional verification helpers if you
choose to incorporate them into the test suite.

## Manual execution

Both tools conform to the stdin/stdout contract described in the assessment. A
sample command looks like this:

```bash
cat sample_factory.json | python part2_assignment/factory/main.py > factory_output.json
```

Replace the JSON file with any payload that follows the published schema.
