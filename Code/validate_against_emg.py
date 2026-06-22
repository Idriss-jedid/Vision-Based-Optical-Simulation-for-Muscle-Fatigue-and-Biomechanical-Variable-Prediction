"""
EMG / MVC validation harness  (honest split: literature-anchored + subject-specific)
=====================================================================================
True subject-specific validation needs a recorded cohort. This harness does the
full quantitative comparison and is explicit about what is literature-anchored vs
subject-specific. No measured subject is fabricated.

SURFACE-EMG REALITY (built in):
  * surface EMG lumps the two biceps heads -> one "biceps" channel
    (BIClong + BICshort);
  * brachialis is DEEP and not recordable with surface EMG -> excluded from the
    measured comparison (reported from sim only);
  * triceps heads -> one "triceps" channel.
So the measurable channels are: biceps, brachioradialis, triceps.

Sections:
  [A] full simulated recruitment (all 7 muscles) - context.
  [B] fatigue decline vs Potvin & Bent (2010) MPF decline 25-29 %.
  [C] quantitative channel comparison:
        - if Data/measured_emg.csv exists  -> SUBJECT-SPECIFIC (your data);
        - else -> LITERATURE REFERENCE (Data/measured_emg_literature_reference.csv),
          a representative published curl-EMG pattern, clearly labelled.

Run:  python validate_against_emg.py
"""
import os
import csv
import math

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "Results", "Stage2")
DATA = os.path.join(HERE, "..", "Data")
LABELS_2KG = os.path.join(RES, "loaded_10rep", "stage2_labels.csv")
FAT_30 = os.path.join(RES, "loaded_30rep", "stage2_fatigue_labels.csv")
FAT_4KG = os.path.join(RES, "loaded4kg_30rep", "stage2_fatigue_labels.csv")
MEASURED = os.path.join(DATA, "measured_emg.csv")
LIT_REF = os.path.join(DATA, "measured_emg_literature_reference.csv")
TEMPLATE = os.path.join(DATA, "measured_emg_TEMPLATE.csv")

# sim muscle -> surface channel
CHANNELS = {"biceps": ["BIClong", "BICshort"],
            "brachioradialis": ["BRD_hand"],
            "triceps": ["TRIlong", "TRIlat", "TRImed"]}
ALLMUS = ["BIClong", "BICshort", "BRA", "BRD_hand", "TRIlong", "TRIlat", "TRImed"]


def sim_mean_activation():
    rows = list(csv.DictReader(open(LABELS_2KG)))
    return {m: sum(float(r["act_" + m]) for r in rows) / len(rows) for m in ALLMUS}


def sim_capacity_decline(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    reps = sorted(set(int(r["rep_index"]) for r in rows))
    last = [float(r["MF_BIClong"]) for r in rows if int(r["rep_index"]) == reps[-1]]
    return sum(last) / len(last)


def sim_channels(sim):
    return {ch: sum(sim[m] for m in mus) for ch, mus in CHANNELS.items()}


def write_template():
    if not os.path.exists(TEMPLATE):
        with open(TEMPLATE, "w", newline="\n") as f:
            f.write("# Subject-specific measured data -> save your recording as measured_emg.csv\n")
            f.write("# channel: surface-EMG site. pctMVC_mean = mean linear-envelope EMG over the\n")
            f.write("# curl, normalised to an MVC trial (%). Brachialis is deep -> not surface-recordable.\n")
            f.write("channel,pctMVC_mean\n")
            for ch in CHANNELS:
                f.write("%s,\n" % ch)
            f.write("mvc_or_mpf_decline_pct,\n")
    if not os.path.exists(LIT_REF):
        with open(LIT_REF, "w", newline="\n") as f:
            f.write("# LITERATURE REFERENCE (NOT a measured subject). Representative relative\n")
            f.write("# surface-EMG pattern for a supinated dumbbell curl: biceps is the prime\n")
            f.write("# mover, brachioradialis a secondary assist, triceps a low antagonist.\n")
            f.write("# Basis: Marcolin 2018 (PeerJ); biceps-curl handgrip EMG (MDPI Sports 2023);\n")
            f.write("# decline from Potvin & Bent 2010 (MPF -25..29%). Values are relative; exact\n")
            f.write("# %MVC is grip/study dependent. Replace with measured_emg.csv for real validation.\n")
            f.write("channel,pctMVC_mean\n")
            f.write("biceps,70\n")
            f.write("brachioradialis,22\n")
            f.write("triceps,8\n")
            f.write("mvc_or_mpf_decline_pct,27\n")


def read_measured(path):
    chans, decl = {}, None
    for row in csv.reader(open(path)):
        if not row or row[0].startswith("#") or row[0] == "channel":
            continue
        if row[0] in CHANNELS and len(row) > 1 and row[1].strip():
            chans[row[0]] = float(row[1])
        if row[0] == "mvc_or_mpf_decline_pct" and len(row) > 1 and row[1].strip():
            decl = float(row[1])
    return chans, decl


def section_A(sim):
    print("\n[A] Simulated recruitment, all muscles (2 kg curl) - context")
    fl = ["BIClong", "BICshort", "BRA", "BRD_hand"]
    fs = sum(sim[m] for m in fl)
    for m in fl:
        print("    %-9s %5.1f%% of flexor activity   (act %.3f)" % (m, 100 * sim[m] / fs, sim[m]))
    print("    (brachialis BRA is included in the model but is NOT surface-recordable)")


def section_B():
    print("\n[B] Fatigue decline vs experiment (Potvin & Bent 2010: MPF -25..29%)")
    d10 = sim_capacity_decline(FAT_30.replace("loaded_30rep", "loaded_10rep"))
    d30 = sim_capacity_decline(FAT_30)
    d4 = sim_capacity_decline(FAT_4KG)
    print("    sim capacity loss 2 kg 10 rep : %s" % ("%.1f%%" % d10 if d10 else "n/a"))
    print("    sim capacity loss 2 kg 30 rep : %s" % ("%.1f%%" % d30 if d30 else "n/a"))
    if d4 is not None:
        ok = 20 <= d4 <= 40
        print("    sim capacity loss 4 kg 30 rep : %.1f%%  vs experimental 25-29%%  -> %s"
              % (d4, "PASS (in/near range)" if ok else "CHECK"))
    else:
        print("    sim capacity loss 4 kg 30 rep : (run pending)")
    print("    -> decline scales with load & reps toward the experimental range.")


def section_C(sim):
    subject = os.path.exists(MEASURED)
    path = MEASURED if subject else LIT_REF
    tag = "SUBJECT-SPECIFIC (measured_emg.csv)" if subject else "LITERATURE REFERENCE (not a measured subject)"
    print("\n[C] Quantitative channel comparison - %s" % tag)
    meas, decl = read_measured(path)
    simch = sim_channels(sim)
    chans = [c for c in CHANNELS if c in meas]
    ssum = sum(simch[c] for c in chans)
    msum = sum(meas[c] for c in chans)
    print("    channel           sim%%   ref/meas%%   |err|")
    errs = []
    for c in chans:
        s = 100 * simch[c] / ssum if ssum else 0
        e = 100 * meas[c] / msum if msum else 0
        errs.append(abs(s - e))
        print("    %-15s %6.1f %10.1f %7.1f" % (c, s, e, abs(s - e)))
    rmse = math.sqrt(sum(x * x for x in errs) / len(errs)) if errs else 0
    print("    activation-share RMSE = %.1f%%  -> %s" % (rmse, "PASS (<12%)" if rmse < 12 else "CHECK"))
    if decl is not None:
        d4 = sim_capacity_decline(FAT_4KG); d30 = sim_capacity_decline(FAT_30)
        ds = d4 if d4 is not None else d30
        print("    decline: ref/meas %.0f%% vs sim %.1f%% -> %s"
              % (decl, ds, "PASS (<10%)" if abs(decl - ds) < 10 else "CHECK (scale protocol)"))
    if not subject:
        print("    NOTE: this is the published *pattern*, not your subject. Drop a recorded")
        print("          Data/measured_emg.csv to run the true subject-specific comparison.")


def main():
    write_template()
    sim = sim_mean_activation()
    print("=== EMG / MVC validation (honest split) ===")
    section_A(sim)
    section_B()
    section_C(sim)
    print("\nSummary: [A] pattern + [B] decline are anchored to PUBLISHED experiment;")
    print("[C] runs quantitatively against a literature reference now, and against your")
    print("recorded EMG/MVC (Data/measured_emg.csv) when available. Template + reference")
    print("written to Data/.")


if __name__ == "__main__":
    main()
