"""Install the bundled Claude skill, its worked examples and the expert agent.

Console script ``discopt-speciation-install-skill``: copies the packaged
``SKILL.md`` and ``examples/*.py`` into a Claude skills directory
(auto-discovered by Claude Code) and the expert agent into the matching
``agents`` directory.
"""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

SKILL_NAME = "discopt-speciation"


def _skill_root():
    return resources.files("discopt.speciation").joinpath("skill")


def _copy(src, dest: Path, force: bool, installed: list, what: str) -> None:
    if dest.exists() and not force:
        print(f"{what} already installed: {dest} (use --force to overwrite)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"installed {what} -> {dest}")
    installed.append(dest)


def install_skill(claude_dir: Path, force: bool = False) -> list[Path]:
    """Install SKILL.md, examples/*.py and agents/*.md under ``claude_dir`` (a .claude dir)."""
    installed: list[Path] = []
    root = _skill_root()
    skill_dir = Path(claude_dir) / "skills" / SKILL_NAME

    _copy(root.joinpath("SKILL.md"), skill_dir / "SKILL.md", force, installed, "skill")
    for ex in sorted(root.joinpath("examples").iterdir(), key=lambda p: p.name):
        if ex.name.endswith(".py"):
            _copy(ex, skill_dir / "examples" / ex.name, force, installed, "example")
    for agent in sorted(root.joinpath("agents").iterdir(), key=lambda p: p.name):
        if agent.name.endswith(".md"):
            _copy(agent, Path(claude_dir) / "agents" / agent.name, force, installed, "agent")
    return installed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="discopt-speciation-install-skill",
        description="Install the discopt-speciation Claude skill, examples and expert agent.",
    )
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--user", action="store_true", help="install to ~/.claude (default)")
    scope.add_argument("--project", action="store_true", help="install to ./.claude (versioned with a repo)")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    args = p.parse_args(argv)

    claude_dir = Path(".claude") if args.project else Path.home() / ".claude"
    install_skill(claude_dir, force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
