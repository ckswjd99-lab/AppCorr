#!/usr/bin/env python3
"""Terminal dashboard for the GPU0 eval queue: one screen, redrawn once a second, stdlib only.

Reads the state that already exists instead of asking the drivers to write anything new:
  * the chain script(s) -- each `... --arms ... > $L/<log>` line is one queue step (a minimal
    bash expander handles NAME=value, "$NAME"/"${NAME}" and one-level `for X in ...; do ... done`);
  * the step's log -- "Final Summary" lines mark finished arms, the driver's
    "[arm] N scored, running X% (Y samples/s)" line gives its own throughput figure;
  * the arm's output jsonl -- row count = progress, `"ok": 1` count = running accuracy; the
    dashboard's own rows/s is a 2-minute sliding window over the row count;
  * nvidia-smi for the two GPUs; the chain's stdout for the last event line.

  python tools/gpu_dash.py [chain.sh ...]      # default: the newest scratchpad chain_gpu0_*.sh
Keys: q quits (Ctrl-C works too).
"""
import glob, json, os, re, shlex, subprocess, sys, time
from collections import deque

SCRATCH = "/tmp/claude-3092/-NHNHOME-share-cjpark/a4974444-17dd-4d66-a897-140c17dbc4af/scratchpad"
EXPECTED = {"realworldqa": 765, "visdrone_det": 448, "visdrone_count": 2350, "textvqa": 5000,
            "refcoco": 8811, "chartqa": 2500, "vstar": 191, "gqa": 12578}
REFRESH = 1.0
WINDOW_S = 120

# ---------------------------------------------------------------- chain script -> queue steps
_ASSIGN = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)=(".*?"|\'.*?\'|\S*)\s*(#.*)?$')
_FOR = re.compile(r'^\s*for\s+(\w+)\s+in\s+(.*?);\s*do\s*$')


def _subst(s, env):
    def rep(m):
        name = m.group(1) or m.group(2)
        return env.get(name, m.group(0))
    return re.sub(r'\$\{(\w+)\}|\$(\w+)', rep, s)


def expand_chain(path):
    """Return the script's command lines with variables expanded and for-loops unrolled."""
    env, out = {}, []
    lines = open(path).read().split("\n")
    # join backslash continuations
    joined, buf = [], ""
    for l in lines:
        if l.rstrip().endswith("\\"):
            buf += l.rstrip()[:-1] + " "
            continue
        joined.append(buf + l)
        buf = ""
    i = 0
    while i < len(joined):
        l = joined[i]
        m = _ASSIGN.match(l)
        if m and not l.lstrip().startswith("export"):
            v = m.group(2)
            if v[:1] in "\"'":
                v = v[1:-1]
            env[m.group(1)] = _subst(v, env)
            i += 1
            continue
        m = _FOR.match(l)
        if m:
            var, items = m.group(1), shlex.split(_subst(m.group(2), env))
            body, depth = [], 1
            i += 1
            while i < len(joined):
                if re.match(r'^\s*done\s*$', joined[i]):
                    depth -= 1
                    if depth == 0:
                        break
                elif _FOR.match(joined[i]):
                    depth += 1
                body.append(joined[i])
                i += 1
            for it in items:
                for b in body:
                    out.append(_subst(b, {**env, var: it}))
            i += 1
            continue
        out.append(_subst(l, env))
        i += 1
    return env, out


def _opt(tokens, name, default=None, nargs=1):
    if name not in tokens:
        return default
    k = tokens.index(name)
    if nargs == 1:
        return tokens[k + 1] if k + 1 < len(tokens) else default
    vals = []
    for t in tokens[k + 1:]:
        if t.startswith("--"):
            break
        vals.append(t)
    return vals


def steps_from_chain(path):
    env, cmds = expand_chain(path)
    steps = []
    for c in cmds:
        if "--arms" not in c or ">" not in c or "accuracy.py" not in c:
            continue   # eval drivers only (not the gate comparison script)
        cmd, _, redir = c.partition(">")
        log = redir.split()[0] if redir.split() else ""
        try:
            toks = shlex.split(cmd)
        except ValueError:
            continue
        ds = _opt(toks, "--dataset")
        if ds is None:
            continue
        keep = float(_opt(toks, "--keep", "1.0"))
        arms = _opt(toks, "--arms", [], nargs=0)
        groups = _opt(toks, "--groups", "4")
        conc = _opt(toks, "--concurrency", None)
        n = int(_opt(toks, "--samples", "0"))
        tags = []
        for a in arms:
            t = a
            if a == "streaming":
                t += f"_g{groups}" + (f"_k{keep:.2f}" if keep < 1.0 else "")
            tags.append((a, t))
        steps.append(dict(dataset=ds, arms=tags, keep=keep, out=_opt(toks, "--out", "."),
                          log=log, n=n or EXPECTED.get(ds, 0), conc=conc,
                          engine="vllm" if "qwen_vllm_accuracy" in cmd else "hf",
                          gate="--contiguous" in toks and n and n < EXPECTED.get(ds, 10 ** 9)))
    cwd = None
    for c in cmds:
        m = re.match(r'^\s*cd\s+(\S+)', c)
        if m:
            cwd = m.group(1)
    return env, steps, cwd


# ---------------------------------------------------------------- live state
def read_log(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 20000))
            tail = f.read().decode("utf-8", "replace").replace("\r", "\n")
    except OSError:
        return None
    return tail


def jsonl_stats(path):
    try:
        data = open(path, "rb").read()
    except OSError:
        return 0, 0
    rows = data.count(b"\n")
    ok = data.count(b'"ok": 1')
    return rows, ok


def arm_file(step, tag):
    pats = glob.glob(os.path.join(step["out"], f"{step['dataset']}_*_{tag}*.jsonl"))
    # exact tag, with or without the _cN concurrency suffix; newest wins
    pats = [p for p in pats if re.search(rf"_{re.escape(tag)}(_c\d+)?\.jsonl$", p)]
    if not pats:
        return None
    return max(pats, key=os.path.getmtime)


def gpu_lines():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return ["nvidia-smi unavailable"]
    out = []
    for l in o.strip().split("\n"):
        try:
            i, used, tot, util = [x.strip() for x in l.split(",")]
            out.append(f"GPU{i} {int(used) / 1024:6.1f}/{int(tot) / 1024:.0f} GB  util {int(util):3d}%")
        except ValueError:
            out.append(l)
    return out


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def chain_pid(chain_path):
    """PID of a `bash <chain_path>` process, if any (via /proc; no pgrep)."""
    base = os.path.basename(chain_path)
    for d in glob.glob("/proc/[0-9]*"):
        try:
            cmd = open(os.path.join(d, "cmdline"), "rb").read().split(b"\0")
        except OSError:
            continue
        if len(cmd) >= 2 and cmd[0].endswith(b"bash") and cmd[1].decode(errors="ignore").endswith(base):
            return int(d.split("/")[-1])
    return None


def bar(frac, width=24):
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "█" * n + "░" * (width - n)


def fmt_eta(sec):
    if sec is None or sec != sec or sec < 0:
        return "  --:--"
    sec = int(sec)
    return f"{sec // 3600:2d}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"   {sec // 60:2d}:{sec % 60:02d}"


# ---------------------------------------------------------------- main loop
class Rate:
    def __init__(self):
        self.h = {}

    def push(self, key, rows):
        q = self.h.setdefault(key, deque())
        now = time.time()
        q.append((now, rows))
        while q and now - q[0][0] > WINDOW_S:
            q.popleft()
        if len(q) >= 2 and q[-1][0] - q[0][0] > 5:
            return (q[-1][1] - q[0][1]) / (q[-1][0] - q[0][0])
        return None


def render(chains, rate):
    lines = [f"\x1b[1mAppCorr GPU queue\x1b[0m  {time.strftime('%Y-%m-%d %H:%M:%S')}   " + "   ".join(gpu_lines())]
    for chain_path in chains:
        env, steps, cwd = steps_from_chain(chain_path)
        pid = chain_pid(chain_path)
        alive = pid is not None
        out_path = chain_path[:-3] + ".out" if chain_path.endswith(".sh") else chain_path + ".out"
        last = (read_log(out_path) or "").strip().split("\n")[-1][:110] if os.path.exists(out_path) else ""
        lines.append("")
        lines.append(f"\x1b[1m{os.path.basename(chain_path)}\x1b[0m  "
                     + (f"\x1b[32mrunning\x1b[0m pid {pid}" if alive else "\x1b[33mnot running\x1b[0m")
                     + (f"   last: {last}" if last else ""))
        running_seen = False
        for st in steps:
            log = os.path.join(cwd or ".", st["log"]) if not os.path.isabs(st["log"]) else st["log"]
            out_dir = st["out"] if os.path.isabs(st["out"]) else os.path.join(cwd or ".", st["out"])
            st = {**st, "out": out_dir}
            tail = read_log(log)
            finals = tail.count("Final Summary") if tail else 0
            crashed = bool(tail) and ("Traceback" in tail or "CUDA out of memory" in tail)
            label = f"{st['dataset']:14s} " + " ".join(t for _, t in st["arms"])
            if st["gate"]:
                label = f"gate n={st['n']} " + label
            if st["conc"]:
                label += f" c{st['conc']}"
            if tail is None:
                status = "\x1b[90m· queued \x1b[0m"
                lines.append(f"  {status} {label}")
                continue
            if finals >= len(st["arms"]) and not crashed:
                # accuracy over the whole output file (the driver's Final Summary counts only the
                # rows scored in that invocation, so a resumed run under-reports)
                accs = []
                for _, tag in st["arms"]:
                    f = arm_file(st, tag)
                    rows, ok = jsonl_stats(f) if f else (0, 0)
                    accs.append(f"{tag}={100 * ok / rows:.2f}" if rows else f"{tag}=?")
                lines.append(f"  \x1b[32m✔ done   \x1b[0m {label:48s} {'  '.join(accs)}")
                continue
            # in progress (or stopped midway)
            is_running = alive and not running_seen and not crashed
            running_seen = running_seen or is_running
            status = "\x1b[36m▶ running\x1b[0m" if is_running else (
                "\x1b[31m✘ crashed\x1b[0m" if crashed else "\x1b[33m■ stopped\x1b[0m")
            lines.append(f"  {status} {label}")
            for arm, tag in st["arms"]:
                f = arm_file(st, tag)
                rows, ok = jsonl_stats(f) if f else (0, 0)
                n = st["n"] or rows
                done_arm = f'"arm": "{tag}' in tail and finals > 0 and rows >= n
                r = rate.push(f or f"{st['log']}:{tag}", rows) if is_running else None
                drv = re.findall(rf"\[{re.escape(arm)}\] (\d+) scored, running ([\d.]+)%\s+\(([\d.]+) samples/s\)", tail)
                drv_rate = float(drv[-1][2]) if drv else None
                rate_s = f"{r:5.2f}/s" if r is not None else (f"{drv_rate:5.2f}/s*" if drv_rate else "   --  ")
                eta = fmt_eta((n - rows) / r) if (r and n > rows) else ("   done" if done_arm else "  --:--")
                acc = f"{100 * ok / rows:6.2f}%" if rows else "   --  "
                mark = "✔" if done_arm else ("▶" if is_running and not done_arm else " ")
                lines.append(f"      {mark} {tag:22s} {bar(rows / n if n else 0)} {rows:5d}/{n:<5d} {rate_s}  acc {acc}  eta{eta}")
    lines.append("")
    lines.append("\x1b[90m* = driver's own average since arm start; rows/s here is a 2-min window. q quits.\x1b[0m")
    return "\n".join(lines)


def main():
    chains = sys.argv[1:] or sorted(glob.glob(os.path.join(SCRATCH, "chain_gpu0_*.sh")), key=os.path.getmtime)[-1:]
    if not chains:
        sys.exit("no chain script given and none found in the scratchpad")
    rate = Rate()
    import select, termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd) if sys.stdin.isatty() else None
    try:
        if old:
            tty.setcbreak(fd)
        sys.stdout.write("\x1b[?25l")
        while True:
            frame = render(chains, rate)
            sys.stdout.write("\x1b[H\x1b[2J" + frame + "\n")
            sys.stdout.flush()
            if old and select.select([sys.stdin], [], [], REFRESH)[0]:
                if sys.stdin.read(1) in ("q", "Q"):
                    break
            elif not old:
                time.sleep(REFRESH)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[?25h\n")
        if old:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
