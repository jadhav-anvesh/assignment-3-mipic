"""The Eq. (6) attention-KL term of SIA can be switched off without changing anything else."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from losses.sia_loss import SIALoss
from models.mipic import MIPICConfig, TinyBERT_MIPIC
from models.mrl_sia import MRLSIAConfig, TinyBERT_MRL_SIA
from models.mrl_sia_pic import MRLSIAPICConfig

ROOT = Path(__file__).resolve().parents[1]


def test_attention_kl_is_on_by_default() -> None:
    assert MRLSIAConfig().use_attention_kl
    assert MRLSIAPICConfig().use_attention_kl
    assert MIPICConfig().use_attention_kl


def test_config_reaches_the_sia_loss() -> None:
    # The wrappers only store the encoder at construction, so no TinyBERT download is needed.
    for flag in (True, False):
        mrl_sia = TinyBERT_MRL_SIA(encoder=nn.Identity(), config=MRLSIAConfig(use_attention_kl=flag))
        mipic = TinyBERT_MIPIC(encoder=nn.Identity(), config=MIPICConfig(use_attention_kl=flag))
        assert mrl_sia.sia_loss_fn.use_attention_kl is flag
        assert mipic.core.sia_loss_fn.use_attention_kl is flag


def test_checkpoints_have_the_same_parameters_with_and_without_kl() -> None:
    on = TinyBERT_MIPIC(encoder=nn.Identity(), config=MIPICConfig(use_attention_kl=True))
    off = TinyBERT_MIPIC(encoder=nn.Identity(), config=MIPICConfig(use_attention_kl=False))
    assert on.state_dict().keys() == off.state_dict().keys()


def test_disabling_kl_zeroes_only_the_attention_term() -> None:
    torch.manual_seed(0)
    hidden = [torch.randn(2, 12, 16, requires_grad=True) for _ in range(2)]
    mask = torch.ones(2, 12, dtype=torch.long)

    def build(flag: bool) -> SIALoss:
        torch.manual_seed(1)  # identical P_i initialization for both losses
        return SIALoss(16, (4, 8, 16), (1,), temperature=1.0, use_attention_kl=flag)

    on, off = build(True), build(False)
    out_on, out_off = on(hidden, mask), off(hidden, mask)

    assert out_on["attn_loss"] > 0
    assert out_off["attn_loss"] == 0
    torch.testing.assert_close(out_off["cka_loss"], out_on["cka_loss"])
    torch.testing.assert_close(out_off["sia_loss"], out_off["cka_loss"])

    # Without KL the P_i projections are unused, which is why DDP then needs static_graph.
    out_off["sia_loss"].backward()
    assert all(p.grad is None for p in off.attn_modules.parameters())


def test_runner_exposes_and_records_the_switch() -> None:
    runner = (ROOT / "run_experiments.py").read_text(encoding="utf-8")
    assert runner.count('_env_bool("MIPIC_ATTENTION_KL", True)') == 3  # mrl_sia, mipic, metadata
    assert '"attention_kl":' in runner
    readme = (ROOT / "kaggle" / "README.md").read_text(encoding="utf-8")
    assert "MIPIC_ATTENTION_KL" in readme
    assert "MIPIC_DDP_STATIC_GRAPH=1" in readme


def test_notebook_settings_cell_has_one_kl_switch() -> None:
    import json

    notebook = json.loads((ROOT / "MIPIC_Kaggle_FINAL.ipynb").read_text(encoding="utf-8"))
    cell = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "RUN_MODE = " in "".join(c["source"])
    )
    assert "ATTENTION_KL = True" in cell
    assert 'os.environ["MIPIC_ATTENTION_KL"] = "1" if ATTENTION_KL else "0"' in cell
