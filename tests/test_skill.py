"""The bundled Claude skill: installer, its code blocks, and its worked examples."""

import math
import re
import runpy
from importlib import resources

import pytest

from discopt.speciation import PhreeqcDatabase
from discopt.speciation.install import SKILL_NAME, install_skill, main

SKILL = resources.files("discopt.speciation").joinpath("skill")
EXAMPLES = sorted(p.name for p in SKILL.joinpath("examples").iterdir() if p.name.endswith(".py"))
AGENTS = sorted(p.name for p in SKILL.joinpath("agents").iterdir() if p.name.endswith(".md"))


def test_install_writes_skill_examples_and_agents(tmp_path):
    installed = install_skill(tmp_path)
    skill_dir = tmp_path / "skills" / SKILL_NAME
    assert (skill_dir / "SKILL.md").read_text().startswith("---")
    assert sorted(p.name for p in (skill_dir / "examples").glob("*.py")) == EXAMPLES
    assert sorted(p.name for p in (tmp_path / "agents").glob("*.md")) == AGENTS == ["speciation-expert.md"]
    assert len(installed) == 1 + len(EXAMPLES) + len(AGENTS)


def test_install_does_not_overwrite_without_force(tmp_path, capsys):
    install_skill(tmp_path)
    skill = tmp_path / "skills" / SKILL_NAME / "SKILL.md"
    skill.write_text("locally edited")
    assert install_skill(tmp_path) == []
    assert skill.read_text() == "locally edited"
    assert "already installed" in capsys.readouterr().out
    install_skill(tmp_path, force=True)
    assert skill.read_text().startswith("---")


def test_main_project_scope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--project"]) == 0
    assert (tmp_path / ".claude" / "skills" / SKILL_NAME / "SKILL.md").exists()
    assert (tmp_path / ".claude" / "agents" / "speciation-expert.md").exists()


def test_frontmatter():
    for text, name in [(SKILL.joinpath("SKILL.md").read_text(), SKILL_NAME),
                       (SKILL.joinpath("agents", "speciation-expert.md").read_text(), "speciation-expert")]:
        head = text.split("---")[1]
        assert re.search(rf"^name: {name}$", head, re.M)
        assert "description:" in head


def test_skill_md_references_existing_examples():
    text = SKILL.joinpath("SKILL.md").read_text()
    assert sorted(set(re.findall(r"examples/(\w+\.py)", text))) == EXAMPLES


def test_skill_md_code_blocks_run():
    """The python blocks in SKILL.md run in order in one namespace (they must stay correct)."""
    text = SKILL.joinpath("SKILL.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(blocks) >= 3
    ns: dict = {}
    for block in blocks:
        exec(compile(block, "SKILL.md", "exec"), ns)
    assert ns["r"].present == ["Calcite"]
    assert ns["rg"].converged


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs(name, capsys):
    runpy.run_path(str(SKILL.joinpath("examples", name)), run_name="__main__")
    assert capsys.readouterr().out.strip()


def test_suppressed_solid_reports_saturation_index():
    """A suppressed solid cannot form but keeps its SI: equal to the SI when it is an undersaturated candidate."""
    db = PhreeqcDatabase.load("phreeqc.dat")
    s = db.system(["Ca", "C(4)"], phases=["Calcite", "Aragonite"])
    under = {"Ca": 1e-4, "C(4)": 1e-4}
    free = s.equilibrate(under, pH=7.0)
    sup = s.equilibrate(under, pH=7.0, suppress=["Calcite", "Aragonite"])
    assert free.present == [] and sup.converged
    for m in ("Calcite", "Aragonite"):
        assert free.SI[m] < 0
        assert sup.SI[m] == pytest.approx(free.SI[m], abs=1e-8)

    over = s.equilibrate({"Ca": 5e-3, "C(4)": 5e-3}, pH=8.5, suppress=["Calcite", "Aragonite"])
    assert over.solids["Calcite"] == 0.0
    assert over.SI["Calcite"] > 0 and over.SI["Aragonite"] < over.SI["Calcite"]
    assert math.isfinite(over.SI["Calcite"])
