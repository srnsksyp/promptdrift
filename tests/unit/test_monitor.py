"""Behavioral diagnosis: evidence boundaries are as important as the happy path."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from promptdrift.cli import app
from promptdrift.engine.baseline import load_baseline, write_baseline
from promptdrift.engine.monitor import monitor_suite
from promptdrift.engine.runner import run_suite
from promptdrift.errors import (
    BaselineError,
    ConfigError,
    PromptDriftError,
    ProviderError,
    TemplateError,
)
from promptdrift.models import Config
from promptdrift.models.monitor import MonitorCase, MonitorReport, SampleResult
from promptdrift.models.result import ModelResponse
from promptdrift.models.test import Assertion
from promptdrift.providers.mock import MockProvider
from promptdrift.storage.history import (
    history_path,
    load_monitor_history,
    load_monitor_run_by_id,
    save_monitor_report,
)


@pytest.fixture
def monitored(tmp_path):
    path = tmp_path / "promptdrift.yaml"
    config = Config.model_validate(
        {
            "provider": {"type": "mock", "model": "echo"},
            "tests": [
                {
                    "id": "policy",
                    "prompt": "policy.txt",
                    "variables": {"days": 30},
                    "assertions": [{"type": "contains", "value": "30 days"}],
                }
            ],
        }
    )
    (tmp_path / "policy.txt").write_text("Refunds: {{ days }} days", encoding="utf-8")
    path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")
    write_baseline(tmp_path / config.baseline.path, run_suite(config, path))
    return config, path


def responses(monkeypatch, outputs, **metadata):
    iterator = iter(outputs)
    calls = []

    def complete(self, prompt, **kwargs):
        calls.append((prompt, kwargs))
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        return ModelResponse(output=value, latency_ms=1, model="echo", provider="mock", **metadata)

    monkeypatch.setattr(MockProvider, "complete", complete)
    return calls


def test_stable_full_suite_and_baseline_immutable(monitored):
    config, path = monitored
    baseline = path.parent / config.baseline.path
    before = baseline.read_bytes()
    report = monitor_suite(config, path)
    assert report.tests[0].diagnosis == "stable"
    assert report.exit_code == 0
    assert report.counts == {"PASS": 1, "WARN": 0, "FAIL": 0, "ERROR": 0}
    assert before == baseline.read_bytes()


@pytest.mark.parametrize(
    ("outputs", "diagnosis", "code"),
    [
        (["denied"] * 3, "observed_model_drift", 1),
        (["denied", "30 days", "denied"], "stochastic_behavior", 1),
        (["30 days please"] * 3, "output_changed", 0),
        (["denied"], "insufficient_evidence", 1),
        ([ProviderError("sensitive response")] * 3, "provider_error", 3),
        (["denied", ProviderError("secret"), "30 days"], "provider_error", 3),
    ],
)
def test_repeated_probe_diagnosis(monitored, monkeypatch, outputs, diagnosis, code):
    config, path = monitored
    calls = responses(monkeypatch, outputs)
    report = monitor_suite(config, path, samples=len(outputs))
    assert report.tests[0].diagnosis == diagnosis
    assert report.exit_code == code
    assert len(calls) == len(outputs)
    assert "sensitive response" not in report.model_dump_json()
    assert "secret" not in report.model_dump_json()


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("prompt", "prompt_changed"),
        ("variable", "prompt_changed"),
        ("temperature", "configuration_changed"),
        ("max_tokens", "configuration_changed"),
        ("contract", "contract_changed"),
        ("severity", "contract_changed"),
        ("model", "provider_changed"),
        ("endpoint", "provider_changed"),
    ],
)
def test_changed_inputs_block_model_only_attribution(monitored, monkeypatch, change, expected):
    config, path = monitored
    if change == "prompt":
        (path.parent / "policy.txt").write_text("new prompt", encoding="utf-8")
    elif change == "variable":
        config.tests[0].variables["days"] = 15
    elif change == "temperature":
        config.defaults.temperature = 1
    elif change == "max_tokens":
        config.defaults.max_output_tokens = 50
    elif change == "contract":
        config.tests[0].assertions[0].value = "unavailable"
    elif change == "severity":
        config.tests[0].assertions[0].severity = "warn"
    elif change == "model":
        config.provider.model = "other-model"
    elif change == "endpoint":
        config.provider.base_url = "https://example.invalid/v1"
    responses(monkeypatch, ["denied"] * 3)
    report = monitor_suite(config, path)
    assert report.tests[0].diagnosis == expected
    assert any("differs" in item for item in report.tests[0].evidence)


def test_simultaneous_changes_are_all_in_evidence(monitored, monkeypatch):
    config, path = monitored
    config.defaults.temperature = 0.5
    config.tests[0].variables["days"] = 90
    config.provider.model = "other"
    responses(monkeypatch, ["denied"] * 3)
    evidence = monitor_suite(config, path).tests[0].evidence
    assert sum("differs" in item for item in evidence) == 3


@pytest.mark.parametrize("field", ["legacy", "engine", "unhealthy"])
def test_insufficient_historical_evidence(monitored, monkeypatch, field):
    config, path = monitored
    baseline_path = path.parent / config.baseline.path
    baseline = load_baseline(baseline_path)
    if field == "legacy":
        baseline.schema_version = 2
        baseline.tests["policy"].provenance = None
    elif field == "engine":
        baseline.tests["policy"].provenance.engine_version = "older-version"
    else:
        baseline.tests["policy"].status = "FAIL"
    baseline_path.write_text(baseline.model_dump_json(), encoding="utf-8")
    responses(monkeypatch, ["denied"] * 3)
    assert monitor_suite(config, path).tests[0].diagnosis == "insufficient_evidence"


def test_provider_fingerprint_is_supporting_not_conclusive(monitored, monkeypatch):
    config, path = monitored
    baseline_path = path.parent / config.baseline.path
    baseline = load_baseline(baseline_path)
    baseline.tests["policy"].provenance.system_fingerprint_hash = "old"
    baseline_path.write_text(baseline.model_dump_json(), encoding="utf-8")
    responses(monkeypatch, ["Refunds: 30 days"] * 3, system_fingerprint="private-fingerprint")
    report = monitor_suite(config, path)
    assert report.tests[0].diagnosis == "stable"
    assert any("supporting signal" in item for item in report.tests[0].evidence)
    assert "private-fingerprint" not in report.model_dump_json()


def test_removed_and_new_cases_are_visible(monitored):
    config, path = monitored
    config.tests[0].id = "new-policy"
    report = monitor_suite(config, path)
    assert [(item.test_id, item.diagnosis) for item in report.tests] == [
        ("new-policy", "new_test"),
        ("policy", "insufficient_evidence"),
    ]
    assert report.tests[1].samples == 0


def test_operational_failure_is_not_model_drift(monitored, monkeypatch):
    config, path = monitored
    config.tests[0].thresholds = {"latency_ms": 0.5}
    responses(monkeypatch, ["30 days"] * 4)
    baseline_report = run_suite(config, path)
    # A deliberately healthy baseline to isolate failure classification at unchanged thresholds.
    baseline_report.tests[0].status = "PASS"
    baseline_report.tests[0].evaluations = []
    write_baseline(path.parent / config.baseline.path, baseline_report, force=True)
    result = monitor_suite(config, path)
    assert result.exit_code == 1
    assert result.tests[0].diagnosis == "insufficient_evidence"


@pytest.mark.parametrize("samples", [0, 21, -1])
def test_invalid_sample_count_does_not_call_provider(monitored, monkeypatch, samples):
    calls = responses(monkeypatch, [])
    with pytest.raises(ConfigError):
        monitor_suite(*monitored, samples=samples)
    assert not calls


def test_missing_baseline_does_not_call_provider(monitored, monkeypatch):
    config, path = monitored
    (path.parent / config.baseline.path).unlink()
    calls = responses(monkeypatch, [])
    with pytest.raises(BaselineError):
        monitor_suite(config, path)
    assert not calls


def test_late_missing_prompt_fails_before_any_probe(monitored, monkeypatch):
    config, path = monitored
    config.tests.append(
        config.tests[0].model_copy(update={"id": "missing", "prompt": "absent.txt"})
    )
    calls = responses(monkeypatch, [])
    with pytest.raises(TemplateError):
        monitor_suite(config, path)
    assert not calls


def test_no_raw_content_or_contract_values_in_history(monitored, monkeypatch):
    config, path = monitored
    responses(monkeypatch, ["PRIVATE provider output"] * 3)
    report = monitor_suite(config, path)
    save_monitor_report(report, history_path(path))
    history = load_monitor_history(history_path(path))
    content = json.dumps(history)
    for secret in ("PRIVATE provider output", "Refunds:", "30 days"):
        assert secret not in content
    assert history[0]["run_id"] == report.run_id
    assert history[0]["counts"]["FAIL"] == 1


def test_history_retention_order_and_read_only_empty(monitored, tmp_path):
    assert load_monitor_history(tmp_path / "missing.sqlite3") == []
    assert not (tmp_path / "missing.sqlite3").exists()
    config, path = monitored
    runs = [monitor_suite(config, path, samples=1) for _ in range(3)]
    for report in runs:
        save_monitor_report(report, history_path(path), retention=2)
    assert [run["run_id"] for run in load_monitor_history(history_path(path))] == [
        runs[2].run_id,
        runs[1].run_id,
    ]


def test_history_retention_keeps_exact_limit(monitored):
    config, path = monitored
    runs = [monitor_suite(config, path, samples=1) for _ in range(3)]

    for report in runs:
        save_monitor_report(report, history_path(path), retention=3)

    history = load_monitor_history(history_path(path))

    assert [run["run_id"] for run in history] == [
        runs[2].run_id,
        runs[1].run_id,
        runs[0].run_id,
    ]


@pytest.mark.parametrize("retention", [0, -1])
def test_history_rejects_invalid_retention(monitored, retention):
    config, path = monitored
    report = monitor_suite(config, path, samples=1)

    with pytest.raises(ValueError, match="retention must be positive"):
        save_monitor_report(report, history_path(path), retention=retention)


def test_history_schema_version_is_set(monitored):
    config, path = monitored
    report = monitor_suite(config, path, samples=1)
    history_db = history_path(path)

    save_monitor_report(report, history_db)

    with sqlite3.connect(history_db) as db:
        version = db.execute("PRAGMA user_version").fetchone()[0]

    assert version == 1


def test_history_migrates_legacy_schema(monitored):
    config, path = monitored
    history_db = history_path(path)

    # Create the legacy schema used before schema versioning.
    history_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(history_db) as db:
        db.execute("CREATE TABLE monitor_runs (run_id TEXT PRIMARY KEY, report_json TEXT NOT NULL)")

    report = monitor_suite(config, path, samples=1)
    save_monitor_report(report, history_db)

    with sqlite3.connect(history_db) as db:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        rows = db.execute("SELECT report_json FROM monitor_runs").fetchall()

    assert version == 1
    assert len(rows) == 1


def test_cli_history_and_no_history_option(monitored):
    _, path = monitored
    runner = CliRunner()
    result = runner.invoke(app, ["monitor", "-c", str(path), "--json", "--no-history"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["diagnosis_counts"] == {"stable": 1}
    assert not history_path(path).exists()
    result = runner.invoke(app, ["monitor", "-c", str(path), "--json"])
    run_id = json.loads(result.output)["run_id"]
    history = runner.invoke(app, ["history", "-c", str(path), "--json"])
    assert history.exit_code == 0
    assert json.loads(history.output)["runs"][0]["run_id"] == run_id


def test_cli_history_never_calls_provider(monitored, monkeypatch):
    config, path = monitored

    # Create history before blocking provider calls.
    report = monitor_suite(config, path, samples=1)
    save_monitor_report(report, history_path(path))

    def fail_provider_call(*args, **kwargs):
        raise AssertionError("Provider should not be called while reading history")

    monkeypatch.setattr(MockProvider, "complete", fail_provider_call)

    result = CliRunner().invoke(
        app,
        ["history", "-c", str(path), "--json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["runs"][0]["run_id"] == report.run_id


@pytest.mark.parametrize("failure", ["missing", "corrupt", "invalid_samples", "invalid_yaml"])
def test_cli_errors_are_json_and_do_not_publish_parser_payload(monitored, failure):
    config, path = monitored
    arguments = ["monitor", "-c", str(path), "--json"]
    if failure == "missing":
        (path.parent / config.baseline.path).unlink()
    elif failure == "corrupt":
        (path.parent / config.baseline.path).write_text(
            "PRIVATE malformed baseline", encoding="utf-8"
        )
    elif failure == "invalid_yaml":
        path.write_text("[PRIVATE invalid config", encoding="utf-8")
    else:
        arguments += ["--samples", "0"]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 2, result.output
    assert json.loads(result.output)["exit_code"] == 2
    assert "PRIVATE" not in result.output


def test_cli_provider_error_code_and_no_sensitive_exception(monitored, monkeypatch):
    _, path = monitored
    responses(monkeypatch, [ProviderError("PRIVATE body")] * 3)
    result = CliRunner().invoke(app, ["monitor", "-c", str(path), "--json"])
    assert result.exit_code == 3
    assert json.loads(result.output)["tests"][0]["provider_errors"] == 3
    assert "PRIVATE" not in result.output


def test_cli_contract_failure_code(monitored, monkeypatch):
    _, path = monitored
    responses(monkeypatch, ["denied"] * 3)
    result = CliRunner().invoke(app, ["monitor", "-c", str(path), "--json"])
    assert result.exit_code == 1
    assert json.loads(result.output)["diagnosis_counts"] == {"observed_model_drift": 1}


def test_project_relative_history_not_cwd(monitored):
    _, path = monitored
    result = CliRunner().invoke(app, ["monitor", "-c", str(path)])
    assert result.exit_code == 0
    assert history_path(path).is_file()
    assert not (Path.cwd() / ".promptdrift" / "history.sqlite3").exists()


def test_monitor_does_not_use_git_selection(monitored, monkeypatch):
    config, path = monitored
    (path.parent / "second.txt").write_text("Refunds: 30 days", encoding="utf-8")
    config.tests.append(config.tests[0].model_copy(update={"id": "second", "prompt": "second.txt"}))
    write_baseline(path.parent / config.baseline.path, run_suite(config, path), force=True)
    monkeypatch.setattr(
        "promptdrift.git.GitContext.get_changed_files", lambda *args, **kwargs: ["policy.txt"]
    )
    calls = responses(monkeypatch, ["denied"] * 6)
    report = monitor_suite(config, path)
    assert len(calls) == 6
    assert report.diagnosis_counts == {"observed_model_drift": 2}


def test_json_format_failures_count_as_behavioral(monitored, monkeypatch):
    config, path = monitored
    config.tests[0].assertions = []
    config.tests[0].output.format = "json"
    responses(monkeypatch, ['{"ok": true}'] + ["not json"] * 3)
    write_baseline(path.parent / config.baseline.path, run_suite(config, path), force=True)
    assert monitor_suite(config, path).tests[0].diagnosis == "observed_model_drift"


def test_history_write_failure_warns_without_masking_verdict(monitored, monkeypatch):
    from promptdrift.errors import PromptDriftError

    def unavailable(*args, **kwargs):
        raise PromptDriftError("Monitoring completed but local history could not be saved.")

    _, path = monitored
    monkeypatch.setattr("promptdrift.storage.history.save_monitor_report", unavailable)
    result = CliRunner().invoke(app, ["monitor", "-c", str(path), "--json"])
    assert result.exit_code == 0
    assert "history could not be saved" in json.loads(result.output)["warnings"][0]


def test_warning_contract_does_not_become_model_drift(monitored, monkeypatch):
    config, path = monitored
    config.tests[0].assertions.append(
        Assertion(type="not_contains", value="forbidden", severity="warn")
    )
    write_baseline(path.parent / config.baseline.path, run_suite(config, path), force=True)
    responses(monkeypatch, ["30 days forbidden"] * 3)
    report = monitor_suite(config, path)
    assert report.tests[0].status == "WARN"
    assert report.tests[0].diagnosis == "insufficient_evidence"
    assert report.exit_code == 0


def test_history_rejects_newer_schema(monitored):
    config, path = monitored
    history_db = history_path(path)

    history_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(history_db) as db:
        db.execute("CREATE TABLE monitor_runs (run_id TEXT PRIMARY KEY, report_json TEXT NOT NULL)")
        db.execute("PRAGMA user_version = 99")

    report = monitor_suite(config, path, samples=1)

    with pytest.raises(PromptDriftError, match="newer"):
        save_monitor_report(report, history_db)


def test_history_read_rejects_newer_schema(tmp_path):
    history_db = tmp_path / "history.sqlite3"

    with sqlite3.connect(history_db) as db:
        db.execute("CREATE TABLE monitor_runs (run_id TEXT PRIMARY KEY, report_json TEXT NOT NULL)")
        db.execute("PRAGMA user_version = 99")
        db.commit()

    with pytest.raises(PromptDriftError, match="newer"):
        load_monitor_history(history_db)


def test_history_rejects_incompatible_table_schema(tmp_path):
    history_db = tmp_path / "history.sqlite3"

    with sqlite3.connect(history_db) as db:
        db.execute("CREATE TABLE monitor_runs (run_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        db.commit()

    with pytest.raises(PromptDriftError, match="incompatible"):
        load_monitor_history(history_db)


def test_history_read_does_not_migrate_legacy_schema(tmp_path):
    history_db = tmp_path / "history.sqlite3"
    report_json = json.dumps({"run_id": "legacy-run", "counts": {"PASS": 1}})

    with sqlite3.connect(history_db) as db:
        db.execute("CREATE TABLE monitor_runs (run_id TEXT PRIMARY KEY, report_json TEXT NOT NULL)")
        db.execute(
            "INSERT INTO monitor_runs (run_id, report_json) VALUES (?, ?)",
            ("legacy-run", report_json),
        )
        db.commit()

    history = load_monitor_history(history_db)

    assert history == [{"run_id": "legacy-run", "counts": {"PASS": 1}}]

    with sqlite3.connect(history_db) as db:
        version = db.execute("PRAGMA user_version").fetchone()[0]

    assert version == 0


def test_history_corruption_has_clear_recovery_message(tmp_path, monkeypatch):
    history_db = tmp_path / "history.sqlite3"
    history_db.write_bytes(b"not a sqlite database")

    with pytest.raises(PromptDriftError, match="corrupted"):
        load_monitor_history(history_db)


def test_history_locked_database_has_clear_recovery_message(tmp_path, monkeypatch):
    history_db = tmp_path / "history.sqlite3"
    history_db.touch()

    def locked_connect(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(sqlite3, "connect", locked_connect)

    with pytest.raises(PromptDriftError, match="locked"):
        load_monitor_history(history_db)


def test_history_other_database_error_uses_generic_message(tmp_path, monkeypatch):
    history_db = tmp_path / "history.sqlite3"
    history_db.touch()

    def failing_connect(*args, **kwargs):
        raise sqlite3.Error("unexpected sqlite failure")

    monkeypatch.setattr(sqlite3, "connect", failing_connect)

    with pytest.raises(
        PromptDriftError,
        match="unreadable; back up or remove the local history database",
    ):
        load_monitor_history(history_db)


def test_history_save_locked_database_has_clear_recovery_message(monitored, monkeypatch):
    config, path = monitored
    history_db = history_path(path)
    report = monitor_suite(config, path, samples=1)

    def locked_connect(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(sqlite3, "connect", locked_connect)

    with pytest.raises(PromptDriftError, match="locked"):
        save_monitor_report(report, history_db)


def test_history_save_corrupted_database_has_clear_recovery_message(monitored, monkeypatch):
    config, path = monitored
    history_db = history_path(path)
    report = monitor_suite(config, path, samples=1)

    def corrupted_connect(*args, **kwargs):
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(sqlite3, "connect", corrupted_connect)

    with pytest.raises(PromptDriftError, match="corrupted"):
        save_monitor_report(report, history_db)


def test_cli_monitor_propagates_busy_timeout_to_storage(monitored, monkeypatch):
    config, path = monitored
    data = json.loads(path.read_text(encoding="utf-8"))
    data["storage"] = {"busy_timeout_ms": 12345}
    path.write_text(json.dumps(data), encoding="utf-8")
    passed_timeout = None

    def capture_save(report, history_file, **kwargs):
        nonlocal passed_timeout
        passed_timeout = kwargs.get("busy_timeout_ms")

    monkeypatch.setattr("promptdrift.storage.history.save_monitor_report", capture_save)
    result = CliRunner().invoke(app, ["monitor", "-c", str(path), "--json"])
    assert result.exit_code == 0
    assert passed_timeout == 12345


def test_cli_history_propagates_busy_timeout_to_storage(monitored, monkeypatch):
    config, path = monitored
    path.write_text(
        "version: 1\nprovider: {type: mock}\n"
        "storage:\n  busy_timeout_ms: 9876\n"
        "baseline:\n  path: promptdrift.baseline.json\n"
        "tests:\n  - id: hello\n    prompt: prompts/support.txt\n",
        encoding="utf-8",
    )
    passed_timeout = None

    def capture_load(history_file, **kwargs):
        nonlocal passed_timeout
        passed_timeout = kwargs.get("busy_timeout_ms")
        return []

    monkeypatch.setattr("promptdrift.storage.history.load_monitor_history", capture_load)
    result = CliRunner().invoke(app, ["history", "-c", str(path), "--json"])
    assert result.exit_code == 0
    assert passed_timeout == 9876


def test_cli_history_reports_invalid_config(tmp_path):
    path = tmp_path / "promptdrift.yaml"
    path.write_text(
        "version: 1\nprovider: {type: mock}\n"
        "storage:\n  busy_timeout_ms: -100\n"
        "tests:\n  - id: a\n    prompt: a.txt\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["history", "-c", str(path), "--json"])
    assert result.exit_code == 2
    assert "greater than or equal to 0" in result.output


class TestSampleResultModel:
    def test_sample_result_valid(self):
        sample = SampleResult(
            index=1,
            status="PASS",
            output_hash="abc12345",
            latency_ms=12.5,
            failed_assertions=[],
        )
        assert sample.index == 1
        assert sample.status == "PASS"
        assert sample.output_hash == "abc12345"
        assert sample.latency_ms == 12.5
        assert sample.failed_assertions == []

    def test_sample_result_defaults(self):
        sample = SampleResult(index=1, status="PASS")
        assert sample.output_hash is None
        assert sample.latency_ms is None
        assert sample.failed_assertions == []

    def test_sample_result_mutable_default_is_independent(self):
        s1 = SampleResult(index=1, status="PASS")
        s2 = SampleResult(index=2, status="PASS")
        s1.failed_assertions.append("failed_one")
        assert s2.failed_assertions == []

    @pytest.mark.parametrize("invalid_index", [0, -1, -5])
    def test_sample_result_rejects_non_positive_index(self, invalid_index):
        with pytest.raises(Exception):
            SampleResult(index=invalid_index, status="PASS")

    @pytest.mark.parametrize("invalid_status", ["UNKNOWN", "pass", "fail", "ERROR_CODE", ""])
    def test_sample_result_rejects_invalid_status(self, invalid_status):
        with pytest.raises(Exception):
            SampleResult(index=1, status=invalid_status)

    def test_sample_result_rejects_negative_latency(self):
        with pytest.raises(Exception):
            SampleResult(index=1, status="PASS", latency_ms=-1.0)

    def test_sample_result_rejects_extra_fields(self):
        with pytest.raises(Exception):
            SampleResult(index=1, status="PASS", raw_output="leak")

    def test_monitor_case_defaults_samples_detail_empty(self):
        case = MonitorCase(
            test_id="t1",
            status="PASS",
            diagnosis="stable",
            summary="ok",
            samples=1,
        )
        assert case.samples_detail == []


class TestMonitorPerSampleEvidence:
    def test_monitor_suite_mixed_stochastic_evidence(self, monitored, monkeypatch):
        import hashlib

        config, path = monitored
        # 3 samples: 1 passing, 1 failing, 1 provider error
        outputs = ["30 days", "denied", ProviderError("network timeout")]
        responses(monkeypatch, outputs)
        report = monitor_suite(config, path, samples=3)
        assert len(report.tests) == 1
        case = report.tests[0]

        # Aggregate diagnostics and counters are preserved
        assert case.diagnosis == "provider_error"
        assert case.status == "ERROR"
        assert case.samples == 3
        assert case.failures == 1
        assert case.provider_errors == 1
        assert len(case.samples_detail) == 3

        # Sample 1: PASS
        s1 = case.samples_detail[0]
        assert s1.index == 1
        assert s1.status == "PASS"
        assert s1.output_hash == hashlib.sha256(b"30 days").hexdigest()
        assert s1.latency_ms == 1
        assert s1.failed_assertions == []

        # Sample 2: FAIL
        s2 = case.samples_detail[1]
        assert s2.index == 2
        assert s2.status == "FAIL"
        assert s2.output_hash == hashlib.sha256(b"denied").hexdigest()
        assert s2.latency_ms == 1
        assert s2.failed_assertions == ["contains"]

        # Sample 3: ERROR
        s3 = case.samples_detail[2]
        assert s3.index == 3
        assert s3.status == "ERROR"
        assert s3.output_hash is None
        assert s3.latency_ms is None
        assert s3.failed_assertions == []

    def test_monitor_suite_warn_status_evidence(self, tmp_path, monkeypatch):
        import hashlib

        config = Config.model_validate(
            {
                "provider": {"type": "mock", "model": "echo"},
                "tests": [
                    {
                        "id": "warn_case",
                        "prompt": "prompt.txt",
                        "assertions": [
                            {"type": "contains", "value": "must_have", "severity": "warn"}
                        ],
                    }
                ],
            }
        )
        path = tmp_path / "promptdrift.yaml"
        (tmp_path / "prompt.txt").write_text("hello", encoding="utf-8")
        path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")
        write_baseline(tmp_path / config.baseline.path, run_suite(config, path))

        outputs = ["no match"]
        responses(monkeypatch, outputs)
        report = monitor_suite(config, path, samples=1)
        case = report.tests[0]

        assert len(case.samples_detail) == 1
        s = case.samples_detail[0]
        assert s.index == 1
        assert s.status == "WARN"
        assert s.output_hash == hashlib.sha256(b"no match").hexdigest()
        assert s.failed_assertions == ["contains"]

    def test_monitor_suite_missing_baseline_case_empty_samples_detail(self, tmp_path):
        # A test present in baseline but missing from active config
        active_config = Config.model_validate(
            {
                "provider": {"type": "mock", "model": "echo"},
                "tests": [
                    {
                        "id": "remaining_test",
                        "prompt": "prompt.txt",
                    }
                ],
            }
        )
        path = tmp_path / "promptdrift.yaml"
        (tmp_path / "prompt.txt").write_text("hello", encoding="utf-8")
        path.write_text(json.dumps(active_config.model_dump(mode="json")), encoding="utf-8")

        baseline_config = Config.model_validate(
            {
                "provider": {"type": "mock", "model": "echo"},
                "tests": [
                    {
                        "id": "remaining_test",
                        "prompt": "prompt.txt",
                    },
                    {
                        "id": "removed_test",
                        "prompt": "prompt.txt",
                    },
                ],
            }
        )
        write_baseline(tmp_path / active_config.baseline.path, run_suite(baseline_config, path))

        report = monitor_suite(active_config, path, samples=2)
        assert len(report.tests) == 2
        removed_case = next(c for c in report.tests if c.test_id == "removed_test")
        assert removed_case.samples == 0
        assert removed_case.samples_detail == []


class TestMonitorReportPrinting:
    def test_terminal_report_displays_samples_detail_for_non_stable(self):
        from io import StringIO

        from rich.console import Console

        from promptdrift.reports.monitor import print_monitor_report

        report = MonitorReport(
            provider="mock",
            model="echo",
            samples=2,
            baseline_path="baseline.json",
            tests=[
                MonitorCase(
                    test_id="drifted_case",
                    status="FAIL",
                    diagnosis="stochastic_behavior",
                    summary="Behavior varies",
                    samples=2,
                    failures=1,
                    evidence=["probe 1 passed, probe 2 failed"],
                    samples_detail=[
                        SampleResult(
                            index=1,
                            status="PASS",
                            output_hash="abcdef0123456789abcdef",
                            latency_ms=10.0,
                        ),
                        SampleResult(
                            index=2,
                            status="FAIL",
                            output_hash="9876543210fedcba987654",
                            latency_ms=12.0,
                            failed_assertions=["contains"],
                        ),
                    ],
                )
            ],
        )

        stream = StringIO()
        console = Console(file=stream, color_system=None, width=120)
        print_monitor_report(report, console)
        output = stream.getvalue()

        assert "Sample Evidence: drifted_case" in output
        assert "abcdef01" in output
        assert "abcdef0123456789abcdef" not in output
        assert "98765432" in output
        assert "1" in output
        assert "2" in output
        assert "PASS" in output
        assert "FAIL" in output

    def test_terminal_report_hyphen_when_hash_none(self):
        from io import StringIO

        from rich.console import Console

        from promptdrift.reports.monitor import print_monitor_report

        report = MonitorReport(
            provider="mock",
            model="echo",
            samples=1,
            baseline_path="baseline.json",
            tests=[
                MonitorCase(
                    test_id="error_case",
                    status="ERROR",
                    diagnosis="provider_error",
                    summary="Provider call failed",
                    samples=1,
                    provider_errors=1,
                    samples_detail=[
                        SampleResult(
                            index=1,
                            status="ERROR",
                            output_hash=None,
                            latency_ms=None,
                        ),
                    ],
                )
            ],
        )

        stream = StringIO()
        console = Console(file=stream, color_system=None, width=120)
        print_monitor_report(report, console)
        output = stream.getvalue()

        assert "Sample Evidence: error_case" in output
        assert "-" in output
        assert "ERROR" in output

    def test_terminal_report_does_not_display_table_for_stable(self):
        from io import StringIO

        from rich.console import Console

        from promptdrift.reports.monitor import print_monitor_report

        report = MonitorReport(
            provider="mock",
            model="echo",
            samples=1,
            baseline_path="baseline.json",
            tests=[
                MonitorCase(
                    test_id="stable_case",
                    status="PASS",
                    diagnosis="stable",
                    summary="All probes passed",
                    samples=1,
                    samples_detail=[
                        SampleResult(
                            index=1,
                            status="PASS",
                            output_hash="abcdef0123456789",
                            latency_ms=10.0,
                        ),
                    ],
                )
            ],
        )

        stream = StringIO()
        console = Console(file=stream, color_system=None, width=120)
        print_monitor_report(report, console)
        output = stream.getvalue()

        assert "Sample Evidence: stable_case" not in output

    def test_terminal_report_does_not_display_table_for_empty_samples_detail(self):
        from io import StringIO

        from rich.console import Console

        from promptdrift.reports.monitor import print_monitor_report

        report = MonitorReport(
            provider="mock",
            model="echo",
            samples=0,
            baseline_path="baseline.json",
            tests=[
                MonitorCase(
                    test_id="removed_case",
                    status="WARN",
                    diagnosis="insufficient_evidence",
                    summary="Removed case",
                    samples=0,
                    samples_detail=[],
                )
            ],
        )

        stream = StringIO()
        console = Console(file=stream, color_system=None, width=120)
        print_monitor_report(report, console)
        output = stream.getvalue()

        assert "Sample Evidence: removed_case" not in output


class TestMonitorBackwardCompatibilityAndPrivacy:
    def test_backward_compatibility_deserialization_without_samples_detail(self):
        legacy_json = json.dumps(
            {
                "test_id": "legacy_test",
                "status": "PASS",
                "diagnosis": "stable",
                "summary": "all ok",
                "samples": 2,
                "failures": 0,
                "provider_errors": 0,
                "output_changes": 0,
                "evidence": [],
                "output_hash": "hash123",
                "assertions": {"contains": True},
                "metrics": {"latency_ms": 50},
                "provenance": None,
            }
        )
        case = MonitorCase.model_validate_json(legacy_json)
        assert case.test_id == "legacy_test"
        assert case.samples_detail == []

    def test_sqlite_history_with_and_without_samples_detail(self, tmp_path):
        db_path = tmp_path / "history.sqlite3"
        # 1. Save legacy report without samples_detail
        legacy_report_dict = {
            "schema_version": 1,
            "run_id": "run-legacy",
            "generated_at": "2026-11-16T12:00:00Z",
            "provider": "mock",
            "model": "echo",
            "samples": 1,
            "baseline_path": "baseline.json",
            "warnings": [],
            "tests": [
                {
                    "test_id": "test1",
                    "status": "PASS",
                    "diagnosis": "stable",
                    "summary": "ok",
                    "samples": 1,
                    "failures": 0,
                    "provider_errors": 0,
                    "output_changes": 0,
                    "evidence": [],
                    "output_hash": "abc",
                    "assertions": {},
                    "metrics": {},
                    "provenance": None,
                }
            ],
        }
        # Save via raw SQLite insert into monitor_runs table
        with sqlite3.connect(db_path) as db:
            db.execute("PRAGMA user_version = 1")
            db.execute(
                "CREATE TABLE monitor_runs (run_id TEXT PRIMARY KEY, report_json TEXT NOT NULL)"
            )
            db.execute(
                "INSERT INTO monitor_runs (run_id, report_json) VALUES (?, ?)",
                ("run-legacy", json.dumps(legacy_report_dict)),
            )

        # Verify load_monitor_history can load it seamlessly
        history = load_monitor_history(db_path)
        assert len(history) == 1
        loaded_legacy_report = MonitorReport.model_validate(history[0])
        assert loaded_legacy_report.tests[0].samples_detail == []

        # 2. Save new report with samples_detail using save_monitor_report
        new_report = MonitorReport(
            provider="mock",
            model="echo",
            samples=1,
            baseline_path="baseline.json",
            tests=[
                MonitorCase(
                    test_id="test2",
                    status="FAIL",
                    diagnosis="stochastic_behavior",
                    summary="failed",
                    samples=1,
                    failures=1,
                    samples_detail=[
                        SampleResult(
                            index=1,
                            status="FAIL",
                            output_hash="hash2",
                            latency_ms=15.0,
                            failed_assertions=["assertion_x"],
                        )
                    ],
                )
            ],
        )
        save_monitor_report(new_report, db_path)

        # Verify both runs are present and valid
        history2 = load_monitor_history(db_path)
        assert len(history2) == 2
        # Load via load_monitor_run_by_id to test full MonitorReport deserialization
        loaded_legacy_run = load_monitor_run_by_id(db_path, "run-legacy")
        assert loaded_legacy_run.tests[0].samples_detail == []

        loaded_new_run = load_monitor_run_by_id(db_path, new_report.run_id)
        assert len(loaded_new_run.tests[0].samples_detail) == 1
        assert loaded_new_run.tests[0].samples_detail[0].index == 1
        assert loaded_new_run.tests[0].samples_detail[0].status == "FAIL"
        assert loaded_new_run.tests[0].samples_detail[0].failed_assertions == ["assertion_x"]

    def test_no_raw_prompt_or_output_leakage_in_samples_detail(self, monitored, monkeypatch):
        config, path = monitored
        secret_output = "SUPER_SECRET_MODEL_OUTPUT_12345"
        secret_prompt = "CONFIDENTIAL_USER_PROMPT_67890"

        # Update test prompt file with secret text
        (path.parent / "policy.txt").write_text(secret_prompt + " {{ days }}", encoding="utf-8")
        write_baseline(path.parent / config.baseline.path, run_suite(config, path), force=True)

        responses(monkeypatch, [secret_output])
        report = monitor_suite(config, path, samples=1)
        serialized = report.model_dump_json()

        assert secret_output not in serialized
        assert secret_prompt not in serialized
        assert len(report.tests[0].samples_detail) == 1
        assert report.tests[0].samples_detail[0].output_hash is not None
        # Verify that SampleResult schema itself doesn't contain prompt/output fields
        sample_dict = report.tests[0].samples_detail[0].model_dump()
        assert "prompt" not in sample_dict
        assert "output" not in sample_dict
        assert "raw_prompt" not in sample_dict
        assert "raw_output" not in sample_dict
