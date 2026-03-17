"""Tests for the compilers subpackage."""

import sympy as sy
from sympy.abc import a, b

from flowprog.model_builder import ModelBuilder, Process
from flowprog.backends.markdown import compile_markdown

from .model_strategies import MObject


class TestMarkdownCompiler:
    """Tests for the markdown compiler."""

    def test_empty_model(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        m = ModelBuilder(processes, objects)

        md = m.describe()
        assert "# Model description" in md
        assert "No steps recorded" in md

    def test_structure_section(self):
        processes = [
            Process("P1", produces=["mid"], consumes=["in"]),
            Process("P2", produces=["out"], consumes=["mid"], has_stock=True),
        ]
        objects = [MObject("in"), MObject("mid", has_market=True), MObject("out")]
        m = ModelBuilder(processes, objects)
        m.add({m.X[0]: 1})

        md = m.describe()
        assert "**P1** (index 0)" in md
        assert "consumes [in]" in md
        assert "produces [mid]" in md
        assert "(has stock)" in md
        assert "(has market)" in md

    def test_step_with_label(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        m = ModelBuilder(processes, objects)

        m.add({m.X[0]: a}, label="my_step")
        md = m.describe()
        assert "### Step 1: my_step" in md

    def test_step_without_label(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        m = ModelBuilder(processes, objects)

        m.add({m.X[0]: a})
        md = m.describe()
        assert "(unlabelled)" in md

    def test_assignments_describe_processes(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        m = ModelBuilder(processes, objects)

        m.add({m.X[0]: a}, label="test")
        md = m.describe()
        assert "input activity of **M1**" in md

    def test_parameters_extracted(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        m = ModelBuilder(processes, objects)

        m.add({m.X[0]: a + b}, label="test")
        md = m.describe()
        assert "**Parameters:**" in md
        assert "`a`" in md
        assert "`b`" in md

    def test_no_model_symbols_in_parameters(self):
        """SympyModel indexed bases (S, U, X, Y) should not appear as parameters."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        m = ModelBuilder(processes, objects)

        m.add(m.pull_production("out", a), label="test")
        md = m.describe()
        params_line = [line for line in md.split("\n") if "**Parameters:**" in line][0]
        assert "`a`" in params_line
        # S, U etc. should not be listed
        for name in ["S", "U", "X", "Y"]:
            assert f"`{name}`" not in params_line.split("**Parameters:**")[1]

    def test_intermediates_shown(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        m = ModelBuilder(processes, objects)

        m.add(m.pull_production("out", a), label="demand")
        md = m.describe()
        assert "**Intermediates:**" in md
        assert "pull_process_output" in md

    def test_limit_transformation_shown(self):
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        m = ModelBuilder(processes, objects)

        unlimited = m.pull_production("out", a)
        m.add(unlimited)

        extra = m.pull_production("out", b)
        cap = sy.Symbol("cap", positive=True)
        limited = m.limit(extra, m.Y[0], cap)
        m.add(limited, label="limited")

        md = m.describe()
        assert "**Transformations:**" in md
        assert "Limit:" in md
        assert "`cap`" in md

    def test_multiple_steps(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        m = ModelBuilder(processes, objects)

        m.add({m.X[0]: a}, label="first")
        m.add({m.X[0]: b}, label="second")
        md = m.describe()
        assert "### Step 1: first" in md
        assert "### Step 2: second" in md

    def test_describe_returns_string(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        m = ModelBuilder(processes, objects)
        m.add({m.X[0]: 1})

        result = m.describe()
        assert isinstance(result, str)

    def test_can_call_compile_markdown_directly(self):
        """Test the compiler function directly, not just via describe()."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        m = ModelBuilder(processes, objects)
        m.add({m.X[0]: a}, label="direct")

        md = compile_markdown(m.structure, m._steps)
        assert "direct" in md
