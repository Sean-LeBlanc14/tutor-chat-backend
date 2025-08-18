# tools/word_review.py
# Manage registry: approve/reject/set corrections, list pending/corrections

import os, json, argparse

REGISTRY_DIR = os.path.join("registry")
CORR_PATH    = os.path.join(REGISTRY_DIR, "corrections.json")
PEND_PATH    = os.path.join(REGISTRY_DIR, "pending.json")

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    if os.path.exists(path):
        os.replace(tmp, path)
    else:
        os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve", nargs="*", help="pending keys to approve (literal match)")
    ap.add_argument("--reject",  nargs="*", help="pending keys to reject (literal match)")
    ap.add_argument("--set", nargs=2, metavar=("KEY","REPLACEMENT"),
                    help="approve a custom KEY -> REPLACEMENT (literal)")
    ap.add_argument("--regex", action="store_true", help="when used with --set, treat KEY as regex")
    ap.add_argument("--list-pending", action="store_true", help="print pending keys")
    ap.add_argument("--list-corrections", action="store_true", help="print approved rules")
    args = ap.parse_args()

    corr = load_json(CORR_PATH, {"version":1, "rules":{}})
    pend = load_json(PEND_PATH, {"version":1, "candidates":{}})

    if args.list_pending:
        print("---- PENDING CANDIDATES ----")
        for k,v in pend["candidates"].items():
            print(f"- {k}  | sugg: {v.get('replacement')!r}  | count: {v.get('total_count',0)}")
        return

    if args.list_corrections:
        print("---- CORRECTIONS RULES ----")
        for k,v in corr["rules"].items():
            reg = " (regex)" if v.get("regex") else ""
            print(f"- {k}{reg}  -> {v.get('replacement')!r}  [{v.get('status')}]")
        return

    # approve from pending
    for key in (args.approve or []):
        item = pend["candidates"].get(key)
        if not item:
            print(f"[!] Not in pending: {key}")
            continue
        repl = item.get("replacement")
        if repl is None:
            print(f"[!] Pending '{key}' has no suggested replacement. Use --set \"{key}\" \"REPLACEMENT\"")
            continue
        corr["rules"][key] = {"status":"approved", "replacement": repl}
        print(f"[+] Approved '{key}' -> '{repl}'")
        # optional: remove from pending
        # del pend["candidates"][key]

    # reject
    for key in (args.reject or []):
        corr["rules"][key] = {"status":"rejected", "replacement": None}
        print(f"[-] Rejected '{key}'")

    # set explicit
    if args.set:
        key, repl = args.set
        corr["rules"][key] = {"status":"approved", "replacement": repl}
        if args.regex:
            corr["rules"][key]["regex"] = True
        print(f"[+] Set rule: {key} {'(regex) ' if args.regex else ''}-> '{repl}'")

    save_json(CORR_PATH, corr)
    save_json(PEND_PATH, pend)
    print(f"Updated {CORR_PATH} with {len(corr['rules'])} rules.")
    print(f"Pending count: {len(pend['candidates'])}")

if __name__ == "__main__":
    main()
