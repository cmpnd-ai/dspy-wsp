"""The language-model task prompt must not change when decision-model support is added."""

import os
from pathlib import Path

import dspy
from conftest import exact_answer, scripted_writer

from dspy_wsp import WeightSpace

SNAPSHOT = Path(__file__).parent / "data" / "llm_task_prompt.txt"
MEASURE = 'Compare both models.\n```forward\n{"type": "generate", "model": "both", "problems": ["train_0"]}\n```'


def llm_task_prompt(backend, trainset, tmp_path) -> str:
    writer, _ = scripted_writer([MEASURE, "Done."])
    opt = WeightSpace(metric=exact_answer, writer_lm=writer, backend=backend, max_passes=1, sandbox="none", run_dir=tmp_path / "run", kernel_timeout=30)
    opt.compile(dspy.Predict("question -> answer"), trainset=trainset)
    return (tmp_path / "run" / "task_prompt.txt").read_text().replace(str(tmp_path.resolve()), "<TMP>").replace(str(tmp_path), "<TMP>")


def test_llm_task_prompt_unchanged(backend, trainset, tmp_path):
    text = llm_task_prompt(backend, trainset, tmp_path)
    if os.environ.get("WSP_WRITE_SNAPSHOT") == "1":
        SNAPSHOT.write_text(text)
    assert text == SNAPSHOT.read_text()
