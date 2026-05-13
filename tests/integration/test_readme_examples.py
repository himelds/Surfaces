"""Test that README examples actually work.

Extracts Python code blocks from README.md and runs them.
This ensures documentation stays in sync with the actual API.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _slugify(text: str) -> str:
    """Convert a README section label into a stable pytest id."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "readme_example"


def _nearest_label(content: str, position: int) -> str:
    """Find the closest preceding README heading or details summary."""
    prefix = content[:position]
    candidates = []

    for match in re.finditer(r"^#{1,6}\s+(.+?)\s*$", prefix, re.MULTILINE):
        candidates.append((match.start(), match.group(1)))

    for match in re.finditer(r"<summary><b>([^<]+)</b></summary>", prefix):
        candidates.append((match.start(), match.group(1)))

    if not candidates:
        return "readme_example"
    return max(candidates, key=lambda item: item[0])[1]


def extract_python_blocks(readme_path: Path) -> list[tuple[str, str]]:
    """Extract all Python code blocks from README.md."""
    content = readme_path.read_text()

    examples = []
    seen_names: dict[str, int] = {}
    for match in re.finditer(r"```python\n(.*?)```", content, re.DOTALL):
        code = match.group(1)
        base_name = _slugify(_nearest_label(content, match.start()))
        count = seen_names.get(base_name, 0) + 1
        seen_names[base_name] = count
        name = base_name if count == 1 else f"{base_name}_{count}"
        examples.append((name, code))

    return examples


# Get README path relative to this file
README_PATH = Path(__file__).parent.parent.parent / "README.md"
EXAMPLES = extract_python_blocks(README_PATH) if README_PATH.exists() else []

REPO_ROOT = Path(__file__).parent.parent.parent
SRC_PATH = REPO_ROOT / "src"


@pytest.mark.parametrize("name,code", EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_readme_example(name: str, code: str):
    """Test that a README example runs without errors."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        temp_path = Path(f.name)

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{SRC_PATH}{os.pathsep}{env.get('PYTHONPATH', '')}"
        result = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=REPO_ROOT,
            env=env,
        )

        assert result.returncode == 0, (
            f"Example '{name}' failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    finally:
        temp_path.unlink()
