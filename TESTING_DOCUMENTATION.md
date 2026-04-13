# Documentation Code Examples Testing

## Overview

All code examples in MPSTAB's documentation are **automatically tested** to ensure they are executable and correct.

## How It Works

A pytest suite (`tests/test_rst_code_blocks.py`) automatically:

1. **Scans all `.rst` files** in the `docs/` directory
2. **Extracts code blocks** marked with `::` in RST format
3. **Validates code** is proper Python syntax (not bash/shell commands)
4. **Executes each block** as an isolated test
5. **Reports failures** with exact location and error details

## Running Tests

Run all documentation code block tests:

```bash
poetry run pytest tests/test_rst_code_blocks.py -v
```

## For Contributors: Writing Code Examples

When adding code examples to documentation, **ensure the code is executable**:

### ✅ Best Practices

1. **Include all necessary imports** in the code block itself
   ```rst
   Here's an example::

       from mpstab import HSMPO
       from qibo import Circuit, gates

       circuit = Circuit(5)
       simulator = HSMPO(ansatz=circuit)
   ```

2. **Define all variables** used in the code
   ```rst
   Example::

       from mpstab import HSMPO
       from qibo import Circuit, gates

       # Create and setup
       circuit = Circuit(5)
       circuit.add(gates.H(0))

       # Use it
       simulator = HSMPO(ansatz=circuit)
   ```

3. **Keep examples focused and copy-pasteable**
   - Show the key functionality
   - Remove unnecessary complexity
   - Add helpful comments

4. **Mark code blocks properly** with `::` in RST
   ```rst
   Description of what the code does::

       # Your code here
   ```

### ⚠️ Common Issues to Avoid

- ❌ Missing imports (add them to the code block)
- ❌ Undefined variables (define them in the code block)
- ❌ Observable length mismatch (e.g., `"ZZ"` for 5-qubit circuit)

### ✔️ Validation

After updating documentation, verify your code examples work:

```bash
poetry run pytest tests/test_rst_code_blocks.py -v
```

Any failures will show:
- Which file and code block failed
- The exact error
- The code that failed

Fix the issue and re-run until all tests pass. That's it!
