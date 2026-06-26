"""
Ground-truth task definitions for the ablation study.
Each task has:
  - goal        : str  — the prompt fed to AdaptEvolve-MAS
  - test_code   : str  — driver that imports the evolved module and runs assertions
  - check(code) : bool — quick in-process correctness check for SS-1 / analysis
"""
import ast, textwrap, traceback, time

# ── Task definitions ──────────────────────────────────────────────────────────

TASKS = {
    "bubble_sort": {
        "name": "BubbleSort Optimization",
        "goal": (
            "Implement an optimized bubble sort algorithm in Python. "
            "It must sort a list of integers in ascending order, handle empty lists, "
            "and be as fast as possible. Function signature: def bubble_sort(arr: list) -> list"
        ),
        "reference": textwrap.dedent("""\
            def bubble_sort(arr):
                a = list(arr)
                n = len(a)
                for i in range(n):
                    swapped = False
                    for j in range(n - i - 1):
                        if a[j] > a[j + 1]:
                            a[j], a[j + 1] = a[j + 1], a[j]
                            swapped = True
                    if not swapped:
                        break
                return a
        """),
        "test_cases": [
            ([], []),
            ([1], [1]),
            ([3, 1, 2], [1, 2, 3]),
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
            (list(range(50, 0, -1)), list(range(1, 51))),
        ],
        "entry_fn": "bubble_sort",
    },
    "prime_sieve": {
        "name": "Prime Sieve",
        "goal": (
            "Implement the Sieve of Eratosthenes in Python to find all prime numbers "
            "up to a given limit n. Optimize for speed and memory. "
            "Function signature: def sieve(n: int) -> list[int]"
        ),
        "reference": textwrap.dedent("""\
            def sieve(n):
                if n < 2:
                    return []
                is_prime = bytearray([1]) * (n + 1)
                is_prime[0] = is_prime[1] = 0
                for i in range(2, int(n**0.5) + 1):
                    if is_prime[i]:
                        is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
                return [i for i, v in enumerate(is_prime) if v]
        """),
        "test_cases": [
            (0, []),
            (1, []),
            (10, [2, 3, 5, 7]),
            (30, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]),
            (100, [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]),
        ],
        "entry_fn": "sieve",
    },
    "matrix_multiply": {
        "name": "Matrix Multiply",
        "goal": (
            "Implement an efficient matrix multiplication function in Python (no numpy). "
            "It must multiply two 2D lists of floats and return a 2D list. "
            "Function signature: def matmul(A: list, B: list) -> list"
        ),
        "reference": textwrap.dedent("""\
            def matmul(A, B):
                rows_A, cols_A = len(A), len(A[0])
                cols_B = len(B[0])
                C = [[0.0] * cols_B for _ in range(rows_A)]
                for i in range(rows_A):
                    for k in range(cols_A):
                        if A[i][k] == 0:
                            continue
                        for j in range(cols_B):
                            C[i][j] += A[i][k] * B[k][j]
                return C
        """),
        "test_cases": [
            ([[1,2],[3,4]], [[5,6],[7,8]], [[19,22],[43,50]]),
            ([[1,0],[0,1]], [[3,4],[5,6]], [[3,4],[5,6]]),  # identity
            ([[2]], [[3]], [[6]]),
        ],
        "entry_fn": "matmul",
    },
}


def check_correctness(code: str, task_key: str) -> float:
    """
    Execute generated code and run ground-truth test cases.
    Returns correctness ratio in [0, 1].
    """
    task = TASKS[task_key]
    fn_name = task["entry_fn"]
    test_cases = task["test_cases"]

    try:
        ast.parse(code)
    except SyntaxError:
        return 0.0

    ns = {}
    try:
        exec(compile(code, "<evolved>", "exec"), ns)
    except Exception:
        return 0.0

    fn = ns.get(fn_name)
    if not callable(fn):
        return 0.0

    passed = 0
    for case in test_cases:
        try:
            if task_key == "matrix_multiply":
                inp_a, inp_b, expected = case
                result = fn(inp_a, inp_b)
                # Allow float tolerance
                flat_r = [x for row in result for x in row]
                flat_e = [x for row in expected for x in row]
                ok = all(abs(a - b) < 1e-6 for a, b in zip(flat_r, flat_e))
            else:
                inp, expected = case
                result = fn(inp if isinstance(inp, list) else inp)
                ok = result == expected
            if ok:
                passed += 1
        except Exception:
            pass

    return passed / len(test_cases)


def time_solution(code: str, task_key: str, n_trials: int = 3) -> float:
    """Run the entry function with a medium-sized input and return avg ms. -1 if fails."""
    task = TASKS[task_key]
    fn_name = task["entry_fn"]

    ns = {}
    try:
        exec(compile(code, "<evolved>", "exec"), ns)
    except Exception:
        return -1.0

    fn = ns.get(fn_name)
    if not callable(fn):
        return -1.0

    # Benchmark inputs
    bench_inputs = {
        "bubble_sort": ([list(range(200, 0, -1))], {}),
        "prime_sieve": ([500], {}),
        "matrix_multiply": ([[list(range(i, i+10)) for i in range(10)],
                             [list(range(i, i+10)) for i in range(10)]], {}),
    }
    args, kwargs = bench_inputs.get(task_key, ([], {}))

    times = []
    for _ in range(n_trials):
        try:
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            times.append((time.perf_counter() - t0) * 1000)
        except Exception:
            return -1.0

    return sum(times) / len(times)
