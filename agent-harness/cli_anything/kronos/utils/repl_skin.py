"""Unified REPL skin for CLI-Anything harnesses."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


class ReplSkin:
    """Provides branded banner, prompt, help, and messages for CLI REPLs."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self._skill_path: Optional[str] = None

    def print_banner(self) -> None:
        w = max(52, len(self.name) + 20)
        border = "╔" + "═" * (w - 2) + "╗"
        title = f"  {self.name}  v{self.version}  —  agent-native CLI"
        sub = "  Type 'help' for commands · 'exit' to quit"
        print(border)
        print(f"║{title:<{w-2}}║")
        print(f"║{sub:<{w-2}}║")
        print("╚" + "═" * (w - 2) + "╝")
        # Show skill path hint if available
        skill = self._resolve_skill_path()
        if skill:
            print(f"\n  Skill: {skill}")

    def _resolve_skill_path(self) -> Optional[str]:
        """Resolve the canonical SKILL.md path for this harness."""
        # Try repo-root canonical path first
        candidates = [
            Path(__file__).resolve().parent.parent.parent.parent.parent / "skills" / f"cli-anything-{self.name}" / "SKILL.md",
            Path(__file__).resolve().parent.parent / "skills" / "SKILL.md",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return None

    def create_prompt_session(self):
        """Create a prompt_toolkit session with history and styling."""
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.styles import Style
        except ImportError:
            return None

        style = Style.from_dict({
            "prompt": "ansiblue bold",
            "suffix": "ansigreen",
        })
        history_path = Path.home() / ".cli-anything-kronos-history"
        session = PromptSession(
            style=style,
            history=FileHistory(str(history_path)),
            complete_while_typing=True,
        )
        return session

    def get_input(self, session, **kwargs) -> str:
        """Get a line of input from the user."""
        if session is None:
            return input(kwargs.get("prompt", f"{self.name}> "))
        return session.prompt(
            kwargs.get("prompt", f"{self.name}> "),
            **{k: v for k, v in kwargs.items() if k != "prompt"}
        )

    def help(self, commands: dict) -> None:
        """Print a formatted help listing of available commands."""
        print("\n  Available commands:")
        for group, cmds in commands.items():
            print(f"\n  [{group}]")
            for name, desc in cmds.items():
                print(f"    {name:<25s} {desc}")
        print()

    def success(self, msg: str) -> None:
        print(f"  ✓ {msg}")

    def error(self, msg: str) -> None:
        print(f"  ✗ {msg}", file=sys.stderr)

    def warning(self, msg: str) -> None:
        print(f"  ⚠ {msg}", file=sys.stderr)

    def info(self, msg: str) -> None:
        print(f"  ● {msg}")

    def status(self, key: str, value: str) -> None:
        print(f"  {key:<20s} {value}")

    def table(self, headers: list, rows: list) -> None:
        if not headers:
            return
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))
        fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*headers))
        print("  " + "  ".join("─" * w for w in widths))
        for row in rows:
            padded = list(row) + [""] * (len(headers) - len(row))
            print(fmt.format(*[str(c) for c in padded]))

    def progress(self, current: int, total: int, label: str = "") -> None:
        pct = current / total if total > 0 else 0
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r  {label} [{bar}] {current}/{total} ({pct:.0%})")
        sys.stdout.flush()
        if current >= total:
            print()

    def print_goodbye(self) -> None:
        print(f"\n  Goodbye from {self.name}!")


def make_skin(name: str = "kronos", version: str = "1.0.0") -> ReplSkin:
    return ReplSkin(name, version)
