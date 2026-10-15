#!/usr/bin/env python3
import json
import os

SEEDS = [1, 2, 3, 5]

def load_data(dir_path, prefix):
    trajs = {}
    for s in SEEDS:
        jp = os.path.join(dir_path, f"{prefix}_seed{s}.json")
        with open(jp) as f:
            d = json.load(f)
            for t in d['trajectories']:
                trajs[(s, t['env_id'])] = t
    return trajs

p5_data = load_data('evaluation_assets/phase5/audit', 'phase5')
exp_a_data = load_data('evaluation_assets/phase5_1/exp_a', 'exp_a')
exp_b_data = load_data('evaluation_assets/phase5_1/exp_b', 'exp_b')
exp_ab_data = load_data('evaluation_assets/phase5_1/combined_ab', 'exp_ab')

def eval_traj(t):
    clr = t['exact_clearance_cm']
    reached_20 = clr >= 20.0
    ab = t.get('recover_abort_count', 0)
    norm = t.get('normal_handoff_count', 0)
    to = t.get('timeout_handoff_count', 0)
    crash = t.get('crash', False)
    
    # Unified definition of full closed-loop success:
    # 1. clearance >= 20.0 cm
    # 2. No Recover Abort (ab == 0)
    # 3. Successful handoff completed (norm + to >= 1)
    # 4. No fatal fall (not crash)
    succ = reached_20 and (ab == 0) and (norm + to >= 1) and (not crash)
    return {
        'clr': clr,
        'reached_20': reached_20,
        'norm': norm,
        'to': to,
        'ab': ab,
        'crash': crash,
        'succ': succ
    }

print("=" * 110)
print(f"| Seed | Env | Phase 5 Base       | Exp A (Prep/Jump)  | Exp B (Recover)    | Combined A+B       | Trajectory Evolution |")
print("=" * 110)

p5_stats = {'reached': 0, 'norm': 0, 'to': 0, 'ab': 0, 'succ': 0}
ea_stats = {'reached': 0, 'norm': 0, 'to': 0, 'ab': 0, 'succ': 0}
eb_stats = {'reached': 0, 'norm': 0, 'to': 0, 'ab': 0, 'succ': 0}
eab_stats = {'reached': 0, 'norm': 0, 'to': 0, 'ab': 0, 'succ': 0}

for s in SEEDS:
    for e in range(4):
        p5 = eval_traj(p5_data[(s, e)])
        ea = eval_traj(exp_a_data[(s, e)])
        eb = eval_traj(exp_b_data[(s, e)])
        eab = eval_traj(exp_ab_data[(s, e)])
        
        for k, src, stat in [('reached_20', p5, p5_stats), ('reached_20', ea, ea_stats), ('reached_20', eb, eb_stats), ('reached_20', eab, eab_stats)]:
            if src[k]: stat['reached'] += 1
        for src, stat in [(p5, p5_stats), (ea, ea_stats), (eb, eb_stats), (eab, eab_stats)]:
            if src['norm'] > 0 and src['ab'] == 0: stat['norm'] += 1
            if src['to'] > 0 and src['ab'] == 0: stat['to'] += 1
            if src['ab'] > 0: stat['ab'] += 1
            if src['succ']: stat['succ'] += 1

        def fmt(res):
            st = "PASS" if res['succ'] else ("ABORT" if res['ab'] > 0 else "FAIL")
            return f"{res['clr']:>5.2f}cm ({st:<5})"

        evo = ""
        if s == 1 and e == 1:
            evo = "NEW REGRESSION (-7.8mm)"
        elif not p5['succ'] and eab['succ']:
            evo = "P5 FAIL -> FIXED"
        elif p5['succ'] and eab['succ']:
            evo = "CONSISTENT PASS"
        else:
            evo = "OTHER"

        print(f"| {s:>4} | {e:>3} | {fmt(p5):<18} | {fmt(ea):<18} | {fmt(eb):<18} | {fmt(eab):<18} | {evo:<20} |")

print("=" * 110)
print("\nSTATISTICAL SUMMARY:")
print(f"Phase 5:      Reached 20cm: {p5_stats['reached']}/16, Normal: {p5_stats['norm']}/16, Timeout: {p5_stats['to']}/16, Abort: {p5_stats['ab']}/16, Full Success: {p5_stats['succ']}/16 ({p5_stats['succ']/16*100:.1f}%)")
print(f"Exp A (Prep): Reached 20cm: {ea_stats['reached']}/16, Normal: {ea_stats['norm']}/16, Timeout: {ea_stats['to']}/16, Abort: {ea_stats['ab']}/16, Full Success: {ea_stats['succ']}/16 ({ea_stats['succ']/16*100:.1f}%)")
print(f"Exp B (Rec):  Reached 20cm: {eb_stats['reached']}/16, Normal: {eb_stats['norm']}/16, Timeout: {eb_stats['to']}/16, Abort: {eb_stats['ab']}/16, Full Success: {eb_stats['succ']}/16 ({eb_stats['succ']/16*100:.1f}%)")
print(f"Combined A+B: Reached 20cm: {eab_stats['reached']}/16, Normal: {eab_stats['norm']}/16, Timeout: {eab_stats['to']}/16, Abort: {eab_stats['ab']}/16, Full Success: {eab_stats['succ']}/16 ({eab_stats['succ']/16*100:.1f}%)")
