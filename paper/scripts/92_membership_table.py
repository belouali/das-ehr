#!/usr/bin/env python

"""Membership scores for every target and candidate pair in the held-out half."""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_s = importlib.util.spec_from_file_location("dn", HERE / "53_deploy_network.py")
dn = importlib.util.module_from_spec(_s)
_s.loader.exec_module(dn)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/external.yaml")
ap.add_argument("--half", choices=["development", "held_out"], default="held_out")
ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
args = ap.parse_args()
d = dn.load(args.config)
out_dir = d["res"] / "deploy_network"
out_dir.mkdir(parents=True, exist_ok=True)
members, targets, D, mt, mlook = dn.setting(d, args.half, 0)
tab = dn.membership_table(d, args.half, members, targets, out_dir, args.jobs)
print(f"{args.half}: {tab.target.nunique()} targets, {len(tab)} pairs, written to {out_dir}")
