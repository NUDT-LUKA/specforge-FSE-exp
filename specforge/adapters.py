"""Small process adapter for independently installed proof-recovery tools."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile


class ExternalProofRecoverer:
    """Run a wrapper command with ``{input}`` and ``{output}`` placeholders."""

    def __init__(self, command: list[str], timeout: int = 900):
        if not command:
            raise ValueError("command cannot be empty")
        self.command = command
        self.timeout = timeout

    def recover(self, source: str) -> str | None:
        with tempfile.TemporaryDirectory(prefix="specforge_adapter_") as directory:
            root = Path(directory)
            input_path = root / "input.dfy"
            output_path = root / "output.dfy"
            input_path.write_text(source, encoding="utf-8")
            argv = [
                part.replace("{input}", str(input_path)).replace("{output}", str(output_path))
                for part in self.command
            ]
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if completed.returncode != 0 or not output_path.exists():
                return None
            return output_path.read_text(encoding="utf-8")

