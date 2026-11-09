"""CPU-only checks for E13/E14 configuration, labels, routing, and checkpointing."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from carrepair import e13e14 as ex
from carrepair.e13e14_submit import job_order, job_key, needed_inputs


@pytest.fixture
def config(tmp_path):
    return {
        "manifest": str(tmp_path / "manifest.csv"),
        "data_root": str(tmp_path),
        "output": str(tmp_path / "outputs"),
        "checkpoint": str(tmp_path / "cardd.pt"),
        "device": "cpu", "batch_size": 2, "workers": 0,
        "seeds": [0, 1, 2], "epochs": 300, "head_lr": 0.001,
        "lr": 1e-4, "weight_decay": 1e-4,
        "early_stopping": {"patience": 30, "min_delta": 0.001,
                           "monitor": "val_macro_f1"},
        "expert": {
            "directory": str(tmp_path / "binary"),
            "source_checkpoint": str(tmp_path / "source.pt"),
            "source_sha256": "dummy", "image_size": 384,
            "expected_parts": 23,
        },
        "variants": {
            "e13b": {"kind": "cardd", "image_size": 384, "augmentation": "pad"},
            "e13c": {"kind": "cardd", "image_size": 384, "augmentation": "mild"},
            "e14a": {"kind": "hf", "family": "dinov3", "mode": "frozen",
                     "image_size": 224, "pretrained_dir": str(tmp_path / "hf"),
                     "pretrained_repo": "facebook/model", "unfreeze_last_blocks": 0},
            "e14b": {"kind": "hf", "family": "dinov3", "mode": "partial",
                     "image_size": 224, "pretrained_dir": str(tmp_path / "hf"),
                     "pretrained_repo": "facebook/model", "unfreeze_last_blocks": 4},
            "e14c": {"kind": "hf", "family": "dinov3", "mode": "partial",
                     "image_size": 224, "pretrained_dir": str(tmp_path / "hf"),
                     "pretrained_repo": "facebook/model", "unfreeze_last_blocks": 4},
            "e14d": {"kind": "hf", "family": "siglip2", "mode": "partial",
                     "image_size": 224, "pretrained_dir": str(tmp_path / "hf"),
                     "pretrained_repo": "google/model", "unfreeze_last_blocks": 4},
        },
    }


def test_all_six_variants_and_fixed_seeds(config):
    assert set(ex.VARIANTS) == set(config["variants"])
    for v in ex.VARIANTS:
        ex.verify_protocol(config, v, require_sources=False)
    bad = dict(config, seeds=[0, 1])
    with pytest.raises(ValueError, match="Seeds"):
        ex.verify_protocol(bad, require_sources=False)
    bad = dict(config, epochs=100)
    with pytest.raises(ValueError, match="300"):
        ex.verify_protocol(bad, require_sources=False)


def test_e13_pad_preserves_aspect_and_is_deterministic():
    image = Image.new("RGB", (120, 45), color=(25, 180, 10))
    f = ex.cardd_transform(384, train=False)
    a, b = f(image), f(image)
    assert a.shape == (3, 384, 384)
    assert torch.equal(a, b)
    assert tuple((a[:, 0, 0] * 255).round().int().tolist()) == (114, 114, 114)
    # Centered car colour retained rather than cropped out.
    assert tuple((a[:, 192, 192] * 255).round().int().tolist()) == (25, 180, 10)


def test_e13_mild_transform_still_returns_384_tensor():
    im = Image.new("RGB", (50, 200))
    x = ex.cardd_transform(384, train=True, augmentation="mild")(im)
    assert x.shape == (3, 384, 384)
    assert torch.isfinite(x).all()


def test_root_label_and_data_ids(config, tmp_path):
    Image.new("RGB", (60, 40)).save(tmp_path / "a.jpeg")
    frame = pd.DataFrame([{
        "id": "same-img-id", "path": "a.jpeg", "label": "bumper_scratch",
        "group": "g0", "split": "test"
    }])
    dataset = ex.RootDataset(frame, config, "e13b", train=False)
    image, label, row = dataset[0]
    assert label == ex.ROOT.index("scratch")
    assert image.shape == (3, 384, 384)
    assert row == 0


def test_checkpoint_contains_only_trainable_hf_weights():
    class Mock(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 4)
            self.head = nn.Linear(4, 6)
            self.encoder.requires_grad_(False)
    a, b = Mock(), Mock()
    with torch.no_grad():
        a.head.weight.fill_(3.5)
    state, kind = ex._trainable_state(a, "hf")
    assert kind == "trainable"
    assert set(state) == {"head.weight", "head.bias"}
    previous_encoder_weight = b.encoder.weight.clone()
    ex._load_trained_state(b, {"state": state, "state_kind": kind})
    assert torch.equal(b.head.weight, a.head.weight)
    assert torch.equal(b.encoder.weight, previous_encoder_weight)


def test_slurm_job_dag_and_e9_expert_same_seed(config):
    work = job_order(["e13b", "e14a"], [0, 1, 2])
    assert len(work) == 12
    assert work[0] == ("e13b", "train", 0)
    assert work[1] == ("e13b", "evaluate", 0)
    assert job_key("e14a", "evaluate", 2) == "e14a-evaluate-s2"
    paths = needed_inputs(config, ["e13b"])
    assert Path(config["checkpoint"]) in paths
    for seed in [0, 1, 2]:
        assert Path(config["expert"]["directory"]) / f"seed{seed}" / "best.pt" in paths


def test_normalized_soft_joint_and_hard_prediction():
    rp = np.array([[0.70, 0.10, 0.05, 0.05, 0.05, 0.05],
                   [0.10, 0.75, 0.05, 0.03, 0.04, 0.03]])
    pp = np.array([[0.70, 0.30], [0.20, 0.80]])
    probs = ex.joint_probabilities(rp, pp)
    hard = ex.hard_route(rp, pp)
    assert np.allclose(probs.sum(1), 1)
    assert ex.EIGHT[hard[0]] == "bumper_dent"
    assert ex.EIGHT[hard[1]] == "door_scratch"


def test_source_signature_fails_closed(config):
    with pytest.raises(FileNotFoundError):
        ex.source_signature(config, "e14a")
    with pytest.raises(FileNotFoundError):
        ex.source_signature(config, "e13b")


def test_fixed_partitions_reject_wrong_sizes(monkeypatch, config):
    three = [pd.DataFrame({"id": ["a"], "label": ["unknown"]}),
             pd.DataFrame({"id": ["b"], "label": ["unknown"]}),
             pd.DataFrame({"id": ["c"], "label": ["unknown"]})]
    monkeypatch.setattr(ex, "partitions", lambda cfg: tuple(three))
    with pytest.raises(ValueError, match="split sizes"):
        ex.fixed_partitions(config)


def test_cached_frozen_features_preserve_original_ids(config):
    class MockDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 5
        def __getitem__(self, ix):
            return torch.full((3, 4, 4), float(ix)), ix % 6, ix
    class MockModel(nn.Module):
        def embedding(self, x):
            return x.flatten(1).mean(1, keepdim=True)
    cached = ex.frozen_features(MockModel(), MockDataset(), config)
    assert len(cached) == 5
    assert [int(cached[i][2]) for i in range(5)] == list(range(5))
    assert cached[4][0].item() == pytest.approx(4.0)


@pytest.fixture
def tiny_hf_sources(tmp_path):
    hf = pytest.importorskip("transformers")
    dino = hf.DINOv3ViTModel(hf.DINOv3ViTConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=6,
        num_attention_heads=4, image_size=32, patch_size=16))
    dino_path = tmp_path / "dinov3"
    dino.save_pretrained(dino_path)
    hf.DINOv3ViTImageProcessorFast(size={"height": 32, "width": 32}).save_pretrained(dino_path)
    dinov2 = hf.Dinov2Model(hf.Dinov2Config(
        hidden_size=32, num_hidden_layers=6, num_attention_heads=4,
        image_size=28, patch_size=14))
    dinov2_path = tmp_path / "dinov2"
    dinov2.save_pretrained(dinov2_path)
    hf.BitImageProcessor(size={"shortest_edge": 36},
                         crop_size={"height": 32, "width": 32}).save_pretrained(dinov2_path)
    # Match Google's fixed-resolution full image/text checkpoint structure.
    siglip = hf.SiglipModel(hf.SiglipConfig(
        text_config={"hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 1, "num_attention_heads": 4,
                     "vocab_size": 32, "max_position_embeddings": 8},
        vision_config={"hidden_size": 32, "intermediate_size": 64,
                       "num_hidden_layers": 6, "num_attention_heads": 4,
                       "image_size": 32, "patch_size": 16}))
    siglip_path = tmp_path / "siglip2-fixed"
    siglip.save_pretrained(siglip_path)
    hf.SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(siglip_path)
    return {"dinov3": dino_path, "dinov2": dinov2_path, "siglip2": siglip_path}


@pytest.mark.parametrize("variant", ["e14a", "e14b", "e14c", "e14d"])
@pytest.mark.parametrize("dino_family", ["dinov3", "dinov2"])
def test_actual_hf_loading_gradients_processor_and_checkpoint_roundtrip(
        variant, dino_family, config, tiny_hf_sources, tmp_path):
    setting = config["variants"][variant]
    if variant != "e14d":
        setting["family"] = dino_family
    setting["pretrained_dir"] = str(tiny_hf_sources[setting["family"]])
    setting["image_size"] = 32
    image = tmp_path / "sample.jpeg"
    Image.new("RGB", (67, 49), (40, 180, 60)).save(image)
    frame = pd.DataFrame([{"id": str(i), "path": image.name,
                           "label": "bumper_scratch"} for i in range(2)])
    ds = ex.RootDataset(frame, config, variant)
    images, labels, _ = next(iter(ex.batches(ds, config)))
    assert images["pixel_values"].shape == (2, 3, 32, 32)
    model = ex.HFRoot(setting)
    model.train()
    logits = model(images)
    assert logits.shape == (2, 6)
    torch.nn.functional.cross_entropy(logits, labels).backward()
    assert model.head.weight.grad is not None
    if variant == "e14a":
        assert not model.encoder.training
        assert all(not p.requires_grad and p.grad is None for p in model.encoder.parameters())
    else:
        blocks, norm = ex._vision_layers_and_norm(model.encoder)
        assert len(blocks) == 6
        assert all(not p.requires_grad and p.grad is None for b in blocks[:2] for p in b.parameters())
        assert all(p.requires_grad and p.grad is not None for b in blocks[2:] for p in b.parameters())
        assert all(p.requires_grad and p.grad is not None for p in norm.parameters())
        allowed = {id(p) for b in blocks[2:] for p in b.parameters()} | {id(p) for p in norm.parameters()}
        assert {id(p) for p in model.encoder.parameters() if p.requires_grad} == allowed
    with torch.no_grad():
        model.head.weight.add_(0.01)
    model.eval()
    expected = model(images).detach()
    state, kind = ex._trainable_state(model, "hf")
    restored = ex.HFRoot(setting).eval()
    ex._load_trained_state(restored, {"state": state, "state_kind": kind})
    torch.testing.assert_close(restored(images), expected)
    from transformers.feature_extraction_utils import BatchFeature
    torch.testing.assert_close(restored(BatchFeature(images)), expected)
    if variant == "e14d":
        from transformers import SiglipModel
        original = SiglipModel.from_pretrained(setting["pretrained_dir"], local_files_only=True)
        assert torch.equal(model.encoder.vision_model.embeddings.patch_embedding.weight,
                           original.vision_model.embeddings.patch_embedding.weight)


def test_arbitrary_hf_architecture_is_rejected(config, tiny_hf_sources):
    setting = dict(config["variants"]["e14a"],
                   pretrained_dir=str(tiny_hf_sources["siglip2"]))
    with pytest.raises(ValueError, match="DINOv3 ViT snapshot"):
        ex.HFRoot(setting)


def test_ungated_template_preserves_e11_protocol_and_isolates_results():
    from carrepair.common import read_config
    cfg = read_config("configs/e14_ungated.yaml")
    original = read_config("configs/e13e14.yaml")
    assert cfg["output"] == "outputs/e14_ungated"
    for name in ("manifest", "data_root", "checkpoint", "expert", "seeds",
                 "epochs", "early_stopping"):
        assert cfg[name] == original[name]
    for variant, model in (("e14a", "small"), ("e14b", "small"), ("e14c", "base")):
        setting = cfg["variants"][variant]
        assert setting["family"] == "dinov2"
        assert setting["pretrained_repo"] == "facebook/dinov2-" + model
        assert len(setting["pretrained_revision"]) == 40
        ex.verify_protocol(cfg, variant, require_sources=False)
    assert cfg["variants"]["e14d"]["pretrained_repo"] == original["variants"]["e14d"]["pretrained_repo"]


def test_anonymous_downloader_pins_revision_and_deduplicates_models(tmp_path, monkeypatch):
    import yaml
    import huggingface_hub
    from scripts.download_e14_models import main
    from carrepair.common import read_config
    cfg = read_config("configs/e14_ungated.yaml")
    for variant in ("e14a", "e14b", "e14c", "e14d"):
        setting = cfg["variants"][variant]
        setting["pretrained_dir"] = str(tmp_path / setting["pretrained_repo"].split("/")[-1])
    config_path = tmp_path / "ungated.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    calls = []
    def fake_download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["local_dir"])
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"fixture")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    main(["--config", str(config_path), "--variants", "e14a", "e14b", "e14c", "e14d", "--anonymous"])
    assert len(calls) == 3
    assert all(call["token"] is False and len(call["revision"]) == 40 for call in calls)


def test_e11_config_paths_are_absolute_and_do_not_change_e11(tmp_path):
    from copy import deepcopy
    from scripts.create_e13e14_artemis_config import build_config
    old = tmp_path / "carABL-e11"
    new = tmp_path / "carABL-e13e14"
    previous = {"manifest": "data/manifest.csv", "data_root": "data/images",
                "checkpoint": "weights/cardd.pt", "output": "outputs/e11",
                "expert": {"directory": "../carABL-e9-es/outputs/e9/binary",
                           "source_checkpoint": "../carABL-e9-es/weights/source.pt"},
                "device": "cuda", "workers": 4}
    original = deepcopy(previous)
    template = {"output": "outputs/e13e14", "variants": {
        "e14a": {"kind": "hf", "pretrained_dir": "weights/hf/dino"}}}
    cfg = build_config(previous, template, old, new)
    assert cfg["manifest"] == str(old / "data/manifest.csv")
    assert cfg["expert"]["directory"] == str(tmp_path / "carABL-e9-es/outputs/e9/binary")
    assert cfg["reference_e11"] == str(old / "outputs/e11")
    assert cfg["output"] == str(new / "outputs/e13e14")
    assert cfg["variants"]["e14a"]["pretrained_dir"] == str(new / "weights/hf/dino")
    assert previous == original


def test_downloader_direct_invocation_uses_current_worktree():
    import os
    import subprocess
    import sys
    project = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(project / "scripts/download_e14_models.py"),
                             "--help"], cwd="/tmp", env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--variants" in result.stdout


def test_e11_patience_accumulates_small_gains(config):
    # Match E11: small consecutive gains accumulate against last significant F1.
    state = None
    for epoch, score in enumerate([0.5, 0.5006, 0.5012], 1):
        state, stop = ex.early_stopping_step(score, state, config, epoch)
    assert not stop and state["bad_epochs"] == 0
    assert state["last_significant_epoch"] == 3


@pytest.fixture
def completed_e11_inputs(config, tmp_path, monkeypatch):
    from carrepair import e13e14_preflight as check
    from carrepair.common import file_hash
    frames = []
    for split in ("train", "val", "test"):
        rows = []
        for index, label in enumerate(ex.EIGHT):
            name = f"{split}-{index}.jpeg"
            Image.new("RGB", (32, 32)).save(tmp_path / name)
            rows.append({"id": f"{split}-{index}", "path": name,
                         "label": label, "group": f"{split}-g{index}", "split": split})
        frames.append(pd.DataFrame(rows))
    pd.concat(frames).to_csv(config["manifest"], index=False)
    Path(config["checkpoint"]).write_text("original CarDD")
    Path(config["expert"]["source_checkpoint"]).write_text("original E9-B source")
    config["expert"]["source_sha256"] = file_hash(config["expert"]["source_checkpoint"])
    config["reference_e11"] = str(tmp_path / "reference-e11")
    for seed in config["seeds"]:
        expert = Path(config["expert"]["directory"]) / f"seed{seed}"
        expert.mkdir(parents=True)
        (expert / "best.pt").write_text("placeholder")
        (expert / "metrics.json").write_text(json.dumps({
            "seed": seed, "task": "binary", "classes": ex.BINARY}))
        (expert / "provenance.json").write_text(json.dumps({
            "manifest_sha256": file_hash(config["manifest"]),
            "pretrained_sha256": config["expert"]["source_sha256"]}))
        frames[2].iloc[:4].to_csv(expert / "test_manifest.csv", index=False)
        root = Path(config["reference_e11"]) / "six_class" / f"seed{seed}"
        cascade = Path(config["reference_e11"]) / "cascade" / f"seed{seed}"
        root.mkdir(parents=True)
        cascade.mkdir(parents=True)
        for directory in (root, cascade):
            (directory / "metrics.json").write_text(json.dumps({"seed": seed}))
        (root / "provenance.json").write_text(json.dumps({
            "manifest_sha256": file_hash(config["manifest"]),
            "cardd_v2_sha256": file_hash(config["checkpoint"])}))
        for name, frame in zip(("train", "val", "test"), frames):
            frame.to_csv(root / f"{name}_manifest.csv", index=False)
    monkeypatch.setattr(check, "fixed_partitions", lambda _: frames)
    return config, frames


def test_preflight_verifies_completed_e11_and_rejects_changed_samples(completed_e11_inputs):
    from carrepair.e13e14_preflight import validate_inputs
    cfg, frames = completed_e11_inputs
    validate_inputs(cfg, ["e13b"])
    saved = Path(cfg["reference_e11"]) / "six_class/seed0/train_manifest.csv"
    changed = pd.read_csv(saved)
    changed.loc[0, "group"] = "changed-vehicle-group"
    changed.to_csv(saved, index=False)
    with pytest.raises(ValueError, match="IDs/labels/groups/splits differ"):
        validate_inputs(cfg, ["e13b"])


def test_preflight_rejects_wrong_e9_source_and_missing_image(completed_e11_inputs):
    from carrepair.e13e14_preflight import validate_inputs
    cfg, frames = completed_e11_inputs
    source = Path(cfg["expert"]["source_checkpoint"])
    original = source.read_text()
    source.write_text("different weights")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_inputs(cfg, ["e13b"])
    source.write_text(original)
    (Path(cfg["data_root"]) / frames[0].iloc[0].path).unlink()
    with pytest.raises(FileNotFoundError, match="dataset image"):
        validate_inputs(cfg, ["e13b"])


def test_preflight_model_smoke_with_actual_hf_classes(
        completed_e11_inputs, tiny_hf_sources, monkeypatch):
    from carrepair import e13e14_preflight as check
    cfg, frames = completed_e11_inputs
    for variant in ("e14a", "e14b", "e14c", "e14d"):
        setting = cfg["variants"][variant]
        setting["pretrained_dir"] = str(tiny_hf_sources[setting["family"]])
        setting["image_size"] = 32
    checked_seeds = []
    def fake_load_expert(cpu, seed, test):
        assert cpu["device"] == "cpu" and cpu["workers"] == 0
        checked_seeds.append(seed)
        return nn.Linear(3, 2), "fixture"
    monkeypatch.setattr(check, "load_expert", fake_load_expert)
    check.model_smoke(cfg, ["e14a", "e14b", "e14c", "e14d"], frames)
    assert checked_seeds == [0, 1, 2]
    assert not Path(cfg["output"]).exists()


def test_config_generator_refuses_missing_e11_paths_before_writing(config, tmp_path):
    import yaml
    from scripts.create_e13e14_artemis_config import main
    source = tmp_path / "e11/configs/e11_artemis.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(yaml.safe_dump(config))
    output = tmp_path / "e13e14_artemis.yaml"
    with pytest.raises(FileNotFoundError, match="config was not written"):
        main(["--e11", str(source), "--output", str(output)])
    assert not output.exists()


def test_submitter_creates_real_dependency_ids_and_no_duplicates(config, monkeypatch, tmp_path):
    import yaml
    from types import SimpleNamespace
    from carrepair import e13e14_submit as submit
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    registry = tmp_path / "jobs.json"
    commands = []
    monkeypatch.setattr(submit, "validate_inputs", lambda *args: None)
    def fake_sbatch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=str(9000 + len(commands)))
    monkeypatch.setattr(submit.subprocess, "run", fake_sbatch)
    args = ["--config", str(cfg_path), "--variants", "e13b", "e13c",
            "--registry", str(registry), "--submit"]
    submit.main(args)
    assert len(commands) == 12
    for index in range(0, len(commands), 2):
        assert not any(x.startswith("--dependency") for x in commands[index])
        assert f"--dependency=afterok:{9001 + index}" in commands[index + 1]
        assert any(x.startswith("--chdir=") for x in commands[index])
    submit.main(args)
    assert len(commands) == 12


@pytest.mark.parametrize("variant", ["e14a", "e14b", "e14d"])
def test_training_early_stop_and_resume_finalize_without_retraining(
        variant, config, tiny_hf_sources, completed_e11_inputs, monkeypatch):
    cfg, frames = completed_e11_inputs
    setting = cfg["variants"][variant]
    setting["pretrained_dir"] = str(tiny_hf_sources[setting["family"]])
    setting["image_size"] = 32
    monkeypatch.setattr(ex, "fixed_partitions", lambda _: frames)
    monkeypatch.setattr(ex, "measure", lambda *args, **kw: {
        "loss": 1.0, "accuracy": 0.5, "macro_f1": 0.5})
    ex.train_one(cfg, variant, 0)
    out = ex.root_dir(cfg, variant, 0)
    report = json.loads((out / "metrics.json").read_text())
    assert report["epochs_completed"] == 31
    assert report["best_epoch"] == 1
    assert report["test_evaluated_during_training"] is False
    history = pd.read_csv(out / "history.csv")
    assert len(history) == 31
    before = (out / "last.pt").read_bytes()
    # Simulate interruption while writing final metrics/curves after early stop.
    (out / "metrics.json").unlink()
    ex.train_one(cfg, variant, 0, resume=True)
    assert (out / "last.pt").read_bytes() == before
    assert json.loads((out / "metrics.json").read_text())["epochs_completed"] == 31
    for name in ("loss_curve.png", "accuracy_curve.png", "f1_curve.png"):
        assert (out / name).stat().st_size > 0
