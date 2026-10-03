"""
META2L_R3_BASELINES.py - comparisons requested by Reviewer 3, under identical conditions.

Uses META2L_SENSITIVITY4.py unchanged (the simulator behind Table V), default
Meta2L settings (as for the Meta2L row of Table V), N=64, mixed mobility,
200 episodes, statistics over the last 20 %, 10 seeds (42 + 13 s) shared by
every method.

Methods
  Meta2L                 run_meta2l with default settings
  Meta2L-C               same algorithm, element memory held in a controller
                         table (copied from the controller at every use)
  Previous-phase init    gradient ascent from phi*(t-1), no memory, no retrieval
  Nearest-neighbour      gradient ascent from the single best retrieved episode
                         (uniform tag weights; no Doppler interpolation, no
                         meta-policy, no mirror descent, no causal filter)
  Cold start             gradient ascent from a random phase
Adaptation: intervals to reach 95 % of the post-switch steady state after a
pedestrian -> vehicular switch (Fig. 7 rule), 5 seeds.
"""
import importlib.util, inspect, json, os, textwrap, time
import numpy as np
import torch
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rr", os.path.join(HERE, "r3_common.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
M = R.M
N_SEEDS, N_EP, TAIL = 10, 200, 0.8

# ---- Meta2L-C: element memory kept in a controller table ---------------------
src = textwrap.dedent(inspect.getsource(M.run_meta2l))
src = src.replace("def run_meta2l(", "def run_meta2l_central(", 1)
old_use = "gamma_mem * m_n)"
old_upd = """            m_n = (lambda_mem * m_n +
                   (1 - lambda_mem) * phi_r_opt)"""
assert src.count(old_use) == 1 and src.count(old_upd) == 1
src = src.replace(old_use, "gamma_mem * CTRL_TABLE['m'].copy())")
src = src.replace(old_upd, """            m_n = (lambda_mem * m_n +
                   (1 - lambda_mem) * phi_r_opt)
            CTRL_TABLE['m'] = m_n.copy()      # controller-side copy""")
src = src.replace("    m_n           = np.zeros(N, dtype=np.float32)",
                  "    m_n           = np.zeros(N, dtype=np.float32)\n    CTRL_TABLE['m'] = m_n.copy()")
M.__dict__["CTRL_TABLE"] = {}
exec(src, M.__dict__)


def seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def meta_stats(fn):
    sr, st, rt, traces = [], [], [], []
    for s in range(N_SEEDS):
        seed(R.SEED + 13 * s)
        t0 = time.time()
        m = fn(n_episodes=N_EP)
        rt.append((time.time() - t0) / N_EP * 1000)
        k = int(TAIL * N_EP)
        sr.append(np.mean(m["sumrate_meta2l"][k:]))
        st.append(np.mean(m["grad_steps"][k:]))
        traces.append(np.array(m["sumrate_meta2l"]))
    ad = []
    R.M.WirelessEnvironment = R.SwitchEnv
    try:
        for s in range(5):
            seed(R.SEED + 17 * s)
            m = fn(n_episodes=R.N_PRE + R.N_POST)
            ad.append(R.adaptation_intervals(m["sumrate_meta2l"][R.N_PRE:]))
    finally:
        R.M.WirelessEnvironment = R.BaseEnv
    return dict(sr=(float(np.mean(sr)), float(np.std(sr))), steps=(float(np.mean(st)), float(np.std(st))),
                rt=(float(np.mean(rt)), float(np.std(rt))), adapt=(float(np.mean(ad)), float(np.std(ad)))), traces


def simple_stats(method):
    sr, st, rt = [], [], []
    for s in range(N_SEEDS):
        seed(R.SEED + 13 * s)
        t0 = time.time()
        srl, stl = R.run_simple(method, N_EP, R.BaseEnv(N=M.P.N, v_profile="mixed"))
        rt.append((time.time() - t0) / N_EP * 1000)
        k = int(TAIL * N_EP)
        sr.append(np.mean(srl[k:]))
        st.append(np.mean(stl[k:]))
    ad = []
    for s in range(5):
        seed(R.SEED + 17 * s)
        R.M.WirelessEnvironment = R.SwitchEnv
        try:
            srl, _ = R.run_simple(method, R.N_PRE + R.N_POST, R.SwitchEnv(N=M.P.N))
        finally:
            R.M.WirelessEnvironment = R.BaseEnv
        ad.append(R.adaptation_intervals(srl[R.N_PRE:]))
    return dict(sr=(float(np.mean(sr)), float(np.std(sr))), steps=(float(np.mean(st)), float(np.std(st))),
                rt=(float(np.mean(rt)), float(np.std(rt))), adapt=(float(np.mean(ad)), float(np.std(ad))))


if __name__ == "__main__":
    out = {}
    out["Meta2L"], tr_d = meta_stats(M.run_meta2l)
    out["Meta2L-C (centralized memory)"], tr_c = meta_stats(M.run_meta2l_central)
    out["centralized_identical"] = bool(all(np.array_equal(a, b) for a, b in zip(tr_d, tr_c)))
    for name, key in [("Previous-phase initialization", "prev"),
                      ("Nearest-neighbour retrieval", "nn"),
                      ("Cold start (random initialization)", "cold")]:
        out[name] = simple_stats(key)
    for k, v in out.items():
        print(k, v, flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "r3_baselines.json"), "w"), indent=1)
