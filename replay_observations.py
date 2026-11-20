"""Verify a recorded session offline; never import bot startup or send orders."""

import argparse
import ast
from collections import Counter
import json
import logging
from pathlib import Path
from types import SimpleNamespace, MethodType

import numpy as np
import pandas as pd

from signal_scoring import causal_macd_scores
from strategy_decisions import entry_side, exit_decision, short_profit_stop
from strategy_inputs import feature_history_spec, prepare_strategy_inputs
from strategy_recording import (
    RecordingError,
    code_fingerprint,
    digest,
    json_value,
    read_blob,
    read_journal,
    runtime_versions,
)


def offline_signal_bot(config, source_file=None):
    if not (config["live_trade"] or config["is_backtest"]):
        raise RecordingError(
            "Paper DB rules must be captured before their replay can be certified"
        )
    source_file = (
        Path(source_file)
        if source_file
        else Path(__file__).with_name("crypto_bot_sub.py")
    )
    tree = ast.parse(source_file.read_text(encoding="utf-8-sig"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CryptoTradingBot"
    )
    names = {
        "build_features",
        "generate_signals",
        "generate_signals_with_sentiment",
        "_apply_rule_thresholds",
    }
    functions = [
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise RecordingError("Missing replay feature methods")
    # Compile only these existing methods, leaving startup, dotenv, DB setup,
    # exchange clients, notification modules and __main__ entirely untouched.
    namespace = {"pd": pd, "np": np, "causal_macd_scores": causal_macd_scores}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
            str(source_file),
            "exec",
        ),
        namespace,
    )
    bot = SimpleNamespace(
        logger=logging.getLogger("offline-replay"),
        exchange_settings_gmo={"live_trade": config["live_trade"]},
        is_backtest=config["is_backtest"],
        entry_thresholds=config.get("entry_thresholds", {}),
    )
    if config.get("trade_size") is not None:
        bot.TRADE_SIZE = config["trade_size"]
    if config.get("initial_capital") is not None:
        bot.initial_capital = config["initial_capital"]
    for name in names:
        setattr(bot, name, MethodType(namespace[name], bot))
    return bot


def restore_frame(value):
    frame = pd.DataFrame(value["rows"], columns=value["columns"])
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    return frame


def verify_session(path, source_file=None):
    events = read_journal(path)
    source_dir = Path(source_file).parent if source_file else None
    if events[0]["payload"]["code_fingerprint"] != code_fingerprint(source_dir):
        raise RecordingError(
            "Code fingerprint differs; use the matching source checkout"
        )
    if events[0]["payload"].get("runtime_versions") != runtime_versions():
        raise RecordingError(
            "Runtime versions differ; use the recorded Python/numpy/pandas versions"
        )
    counts = Counter()
    snapshots = {}
    linked_entries = 0
    passive = {
        "runtime_start",
        "ticker_response",
        "input_error",
        "funds_snapshot",
        "funds_blocked",
        "sr_snapshot",
        "order_result",
    }
    bots = {}
    for event in events[1:-1]:
        kind, payload = event["kind"], event["payload"]
        if kind == "feature_snapshot":
            if payload["history_spec"] != feature_history_spec():
                raise RecordingError("Feature history policy differs")
            key = digest(payload["config"])
            if key not in bots:
                bots[key] = offline_signal_bot(payload["config"], source_file)
            inputs = read_blob(path, payload["inputs"])
            signals, hourly = prepare_strategy_inputs(
                bots[key],
                payload["symbol"],
                restore_frame(inputs["intraday"]),
                restore_frame(inputs["hourly"]),
                event["observed_at"],
            )
            if (
                digest(signals) != payload["signals_hash"]
                or digest(hourly) != payload["hourly_features_hash"]
            ):
                raise RecordingError(
                    f"Feature replay mismatch at sequence {event['sequence']}"
                )
            snapshots[payload.get("record_id", event["hash"])] = (
                payload["symbol"],
                digest(signals.iloc[-2].to_dict()) if len(signals) >= 3 else None,
                digest(signals.iloc[-3].to_dict()) if len(signals) >= 3 else None,
            )
        elif kind == "entry_decision":
            direction = entry_side(
                payload["latest"],
                payload["previous"],
                event["observed_at"],
                payload["blocked_until"],
            )
            if direction != payload["direction"]:
                raise RecordingError(
                    f"Entry replay mismatch at sequence {event['sequence']}"
                )
            reference = payload.get("feature_snapshot")
            if reference is not None:
                if reference not in snapshots:
                    raise RecordingError("Entry references a missing feature snapshot")
                symbol, latest_hash, previous_hash = snapshots[reference]
                if symbol != payload.get("symbol") or latest_hash is None:
                    raise RecordingError(
                        "Entry feature snapshot does not match the symbol/history"
                    )
                if latest_hash != digest(payload["latest"]) or previous_hash != digest(
                    payload["previous"]
                ):
                    raise RecordingError(
                        f"Entry/feature snapshot mismatch at sequence {event['sequence']}"
                    )
                linked_entries += 1
        elif kind == "exit_decision":
            stop, armed = short_profit_stop(**payload["short_lock_input"])
            if (
                stop != payload["inputs"]["stop_loss"]
                or armed != payload["inputs"]["profit_lock_armed"]
            ):
                raise RecordingError(
                    f"Profit-lock replay mismatch at sequence {event['sequence']}"
                )
            if json_value(exit_decision(**payload["inputs"])) != payload["expected"]:
                raise RecordingError(
                    f"Exit replay mismatch at sequence {event['sequence']}"
                )
        elif kind not in passive:
            raise RecordingError(f"Unsupported observation kind: {kind}")
        counts[kind] += 1
    if not any(
        counts[kind] for kind in ("feature_snapshot", "entry_decision", "exit_decision")
    ):
        raise RecordingError("No replayable strategy observations in session")
    return {
        "complete_session": True,
        "code_matches": True,
        "checked": dict(counts),
        "snapshots_with_missing_history": sum(
            event["kind"] == "feature_snapshot"
            and any(
                item["missing_closed_intervals"]
                for item in event["payload"]["quality"].values()
            )
            for event in events
        ),
        "linked_entry_decisions": linked_entries,
        "verified_scope": [
            name
            for kind, name in (
                ("feature_snapshot", "recorded_features"),
                ("entry_decision", "entry_direction"),
                ("exit_decision", "exit_decision_and_profit_lock"),
            )
            if counts[kind]
        ],
        "not_verified": [
            "historical_sr_generation",
            "fills_and_accounting",
            "counterfactual_execution",
        ],
        "live_equivalent": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    args = parser.parse_args()
    try:
        result = verify_session(args.journal)
    except (RecordingError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"Replay failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
