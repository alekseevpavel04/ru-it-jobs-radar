"""Analyst agent commands.

  python -m radar.agent ask "Сколько вакансий ML на Хабр Карьере?"   answer from the live marts
  python -m radar.agent snapshot export [--repin]   freeze the marts into data/agent/snapshot
  python -m radar.agent snapshot load               restore the frozen marts into agent_snapshot
  python -m radar.agent build                       reference answers of eval/agent/questions.toml
  python -m radar.agent eval --split dev [--quick]  score the agent, results/agent/dev*.json

The test part of the control set is scored once: `eval --split test --final-test`.
"""

import argparse
import json
import shutil
import sys
from collections import Counter

from radar.agent import control_set, snapshot
from radar.agent.clickhouse import ReadOnlyClickHouse
from radar.agent.evaluate import make_agent, prompt_sha256, score_item, score_unsafe, summarize
from radar.agent.llm import MODEL, LocalChat
from radar.agent.profiles import frozen, live
from radar.agent.schema import MARTS_YML
from radar.config import ROOT, load_dotenv, require
from radar.eval.cli import log_test_run
from radar.results import write_result
from radar.stream.clickhouse import ClickHouse

SOURCE = "src/radar/agent/__main__.py"


def cmd_ask(args: argparse.Namespace) -> int:
    ro = ReadOnlyClickHouse()
    chat = LocalChat()
    chat.check_model()
    answer = make_agent(chat, ro, live(), compose_answer=True).ask(args.question)
    print(answer.answer)
    print(f"\n[{answer.status}, attempts {answer.attempts}]")
    if answer.sql:
        print(answer.sql)
    return 0 if answer.status == "answered" else 1


def cmd_snapshot(args: argparse.Namespace) -> int:
    with ClickHouse(timeout_s=600) as ch:
        if args.action == "export":
            if snapshot.PIN.exists() and not args.repin:
                print(
                    f"refused: {snapshot.PIN.relative_to(ROOT)} pins the snapshot the answers were computed on; "
                    "--repin replaces it, then rebuild the control set",
                    file=sys.stderr,
                )
                return 2
            manifest = snapshot.export(ch, snapshot.snapshot_dir())
            snapshot.PIN.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot.snapshot_dir() / "manifest.json", snapshot.PIN)
            shutil.copyfile(MARTS_YML, snapshot.SNAPSHOT_YML)
            print(json.dumps({n: t["rows"] for n, t in manifest["tables"].items()}))
        else:
            loaded = snapshot.load(ch, snapshot.snapshot_dir(), snapshot.pinned())
            print(f"loaded into {snapshot.SNAPSHOT_DATABASE}: {json.dumps(loaded)}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    spec = control_set.load_spec()
    items = control_set.build(ReadOnlyClickHouse(), spec)
    control_set.write(items)
    counts = Counter((i.split, i.kind) for i in items)
    for (split, kind), n in sorted(counts.items()):
        print(f"{split:5} {kind:12} {n}")
    print(f"{len(items)} items, {sum(i.quick for i in items)} quick, snapshot {items[0].snapshot}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    if args.split == "test" and not args.final_test:
        print("refused: the test part of the control set is scored once, with --final-test", file=sys.stderr)
        return 2
    if args.final_test and (args.split != "test" or args.quick):
        print("--final-test only goes with --split test and the whole part", file=sys.stderr)
        return 2
    items = [i for i in control_set.load() if i.split == args.split and (i.quick or not args.quick)]
    pin = snapshot.pinned()
    if {i.snapshot for i in items} != {snapshot.pin_id(pin)}:
        print("refused: the control set was built on another snapshot; rebuild it", file=sys.stderr)
        return 2
    with ClickHouse(timeout_s=600) as ch:
        if not snapshot.is_loaded(ch, pin):
            snapshot.load(ch, snapshot.snapshot_dir(), pin)
    if args.final_test:
        log_test_run("agent", None)

    chat = LocalChat()
    chat.check_model()
    profile = frozen()
    agent = make_agent(chat, ReadOnlyClickHouse(), profile, compose_answer=args.with_answers)
    scored = []
    for n, item in enumerate(items, 1):
        scored.append(score_item(agent, item))
        print(
            f"{n}/{len(items)} {item.id} {scored[-1]['status']} {'ok' if scored[-1]['correct'] else 'miss'}", flush=True
        )
    unsafe = [] if args.quick else [score_unsafe(agent, u) for u in control_set.load_spec().unsafe]
    metrics = summarize(scored, unsafe)
    name = f"{args.split}_quick" if args.quick else args.split
    path = write_result(
        f"agent/{name}.json",
        metrics,
        SOURCE,
        split=args.split,
        quick=args.quick,
        model=MODEL,
        llm_base_url=require("LLM_BASE_URL"),
        profile=profile.name,
        prompt_sha256=prompt_sha256(profile),
        snapshot=snapshot.pin_id(pin),
        items=scored,
        unsafe=unsafe,
    )
    for key in ("execution_accuracy", "valid_sql", "repaired", "unsafe_refused_or_blocked"):
        if key in metrics:
            m = metrics[key]
            print(f"{key}: {m['n']} of {m['denominator']} ({m['value']})")
    print(f"wrote {path.relative_to(ROOT).as_posix()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="python -m radar.agent", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.set_defaults(run=cmd_ask)
    snap = sub.add_parser("snapshot")
    snap.add_argument("action", choices=["export", "load"])
    snap.add_argument("--repin", action="store_true", help="replace the pinned snapshot (then rebuild)")
    snap.set_defaults(run=cmd_snapshot)
    sub.add_parser("build").set_defaults(run=cmd_build)
    ev = sub.add_parser("eval")
    ev.add_argument("--split", required=True, choices=["dev", "test"])
    ev.add_argument("--quick", action="store_true", help="only the quick dev subset (regression run)")
    ev.add_argument("--with-answers", action="store_true", help="also compose the Russian answer (slower)")
    ev.add_argument("--final-test", action="store_true", help="the one scoring of the test part")
    ev.set_defaults(run=cmd_eval)
    args = ap.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
