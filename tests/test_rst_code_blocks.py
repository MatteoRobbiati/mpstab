"""
Test suite that executes all code examples from RST documentation.

Scans .rst files, extracts code blocks, and executes them to verify they work.
All code blocks are expected to include necessary imports.
"""

import ast
import os
from pathlib import Path

import pytest


def is_directive(line):
    """Check if a line is an RST directive (.. name::) rather than a code block (::)."""
    stripped = line.lstrip()
    # Directives start with .. followed by a directive name
    return stripped.startswith("..")


def is_python_code(code):
    """
    Check if code is valid Python by attempting to parse it as AST.

    This is the most reliable method - if it parses as valid Python, it's Python.
    Bash/shell scripts will fail to parse and return False.
    """
    stripped = code.strip()

    # Skip if it's empty
    if not stripped:
        return False

    # Try to parse as Python AST
    try:
        ast.parse(stripped)
        return True
    except SyntaxError:
        # Not valid Python syntax
        return False


def extract_code_blocks_from_rst(rst_content, filename):
    """
    Extract Python code blocks from RST content.

    Only extracts literal code blocks (::) that are NOT directives (.. name::)
    and that contain Python code (not bash scripts).

    Returns list of tuples: (block_number, code, filename)
    """
    code_blocks = []
    lines = rst_content.split("\n")

    i = 0
    block_count = 0

    while i < len(lines):
        line = lines[i]

        # Look for :: marker (code block indicator)
        # But skip if it's a directive (e.g., .. note::, .. math::)
        if line.rstrip().endswith("::") and not is_directive(line):
            i += 1

            # Skip blank lines after ::
            while i < len(lines) and not lines[i].strip():
                i += 1

            if i >= len(lines):
                break

            # Get base indentation
            base_indent = len(lines[i]) - len(lines[i].lstrip())
            code_lines = []

            # Collect indented code lines
            while i < len(lines):
                current_line = lines[i]

                if not current_line.strip():
                    code_lines.append("")
                    i += 1
                    continue

                current_indent = len(current_line) - len(current_line.lstrip())
                if current_indent >= base_indent:
                    code_lines.append(current_line[base_indent:])
                    i += 1
                else:
                    break

            # Remove trailing blank lines
            while code_lines and not code_lines[-1].strip():
                code_lines.pop()

            if code_lines:
                code = "\n".join(code_lines)
                # Only include Python code blocks, skip bash/shell
                if is_python_code(code):
                    block_count += 1
                    code_blocks.append((block_count, code, filename))
        else:
            i += 1

    return code_blocks


def collect_all_code_blocks(docs_path="docs"):
    """Collect all Python code blocks from RST files."""
    all_blocks = []

    # Walk through docs directory
    for root, dirs, files in os.walk(docs_path):
        for file in sorted(files):
            if file.endswith(".rst"):
                rst_file = os.path.join(root, file)
                try:
                    with open(rst_file, encoding="utf-8") as f:
                        content = f.read()
                    blocks = extract_code_blocks_from_rst(content, rst_file)
                    all_blocks.extend(blocks)
                except Exception as e:
                    print(f"Warning: Could not read {rst_file}: {e}")

    return all_blocks


# Collect all code blocks at module load time
CODE_BLOCKS = collect_all_code_blocks()


class TestDocumentationCodeBlocks:
    """Execute documentation code blocks as tests."""

    @pytest.mark.parametrize(
        "block_num,code,filename",
        CODE_BLOCKS,
        ids=[f"{Path(f).stem}_{n}" for n, _, f in CODE_BLOCKS],
    )
    def test_code_block_executable(self, block_num, code, filename):
        """Execute a code block from documentation."""
        try:
            exec(code, {})
        except Exception as e:
            pytest.fail(
                f"Code block from {filename} (block {block_num}) failed:\n"
                f"{type(e).__name__}: {e}\n\n"
                f"Code:\n{code}\n"
            )


def test_code_blocks_exist():
    """Verify code blocks were found."""
    assert len(CODE_BLOCKS) > 0, "No code blocks found in documentation"
