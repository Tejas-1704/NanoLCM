# ==== imports ====
import os, glob, gc, math, warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ==== config ====
ROOT = "/kaggle/input/unb-cic-iot-dataset/wataiData/csv"  # crawl this whole tree
PER_CLASS = 30000                                        # 30k benign + 30k attack
VAL_RATIO = 0.2
SEED = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# We’ll treat these labels as BENIGN (everything else => ATTACK)
BENIGN_TOKENS = {
    "BENIGN","Benign","BenignTraffic","Normal","normal","Background","Idle"
}

# ==== utils ====
def set_seed(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
set_seed(SEED)

def find_all_csvs(root):
    out = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if f.lower().endswith(".csv"):
                out.append(os.path.join(dp, f))
    out.sort()
    return out

def choose_feature_columns(csv_files):
    """
    Peek a few files until we find one with a 'label' column and numeric features.
    Returns: (feature_cols, usecols)
    """
    for f in csv_files[:50]:  # peek a handful; CICIoT2023 is consistent
        try:
            df = pd.read_csv(f, nrows=200, low_memory=False)
        except Exception:
            continue
        if "label" not in df.columns:
            continue
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if "label" in num_cols:
            num_cols.remove("label")
        if len(num_cols) == 0:
            continue
        return num_cols, (num_cols + ["label"])
    raise RuntimeError("Could not find a CSV with 'label' + numeric features to infer schema.")

# ==== balanced streaming loader across the whole tree ====
def load_balanced_binary_recursive(root, per_class=30_000, benign_tokens=BENIGN_TOKENS):
    csvs = find_all_csvs(root)
    if not csvs:
        raise FileNotFoundError(f"No CSV files under: {root}")

    feature_cols, usecols = choose_feature_columns(csvs)
    print(f"[i] feature_cols = {len(feature_cols)}  (example: {feature_cols[:8]} …)")
    print(f"[i] scanning {len(csvs)} CSV files under {root}")

    need_b, need_a = per_class, per_class
    ben_parts, atk_parts = [], []

    for f in tqdm(csvs, desc="Scanning CSVs"):
        if need_b <= 0 and need_a <= 0:
            break
        try:
            df = pd.read_csv(f, usecols=usecols, low_memory=False)
        except Exception as e:
            print(f"[warn] skipping {f}: {e}")
            continue

        labels = df["label"].astype(str).str.strip()
        m_b = labels.isin(benign_tokens)
        m_a = ~m_b

        if need_b > 0 and m_b.any():
            b_df = df.loc[m_b, feature_cols]
            if len(b_df) > need_b:
                b_df = b_df.sample(n=need_b, random_state=SEED)
            ben_parts.append(b_df)
            need_b -= len(b_df)

        if need_a > 0 and m_a.any():
            a_df = df.loc[m_a, feature_cols]
            if len(a_df) > need_a:
                a_df = a_df.sample(n=need_a, random_state=SEED)
            atk_parts.append(a_df)
            need_a -= len(a_df)

        del df, labels, m_b, m_a
        if (need_b <= 0 and need_a <= 0) or (len(ben_parts) + len(atk_parts)) % 20 == 0:
            gc.collect()

    if need_b > 0 or need_a > 0:
        print(f"[warn] missing rows -> benign={max(0,need_b)} attack={max(0,need_a)}")

    # assemble & exact-balance
    if len(ben_parts) == 0 and len(atk_parts) == 0:
        raise RuntimeError("Collected zero rows; check your ROOT and label tokens.")
    ben = pd.concat(ben_parts, axis=0, ignore_index=True) if ben_parts else pd.DataFrame(columns=feature_cols)
    atk = pd.concat(atk_parts, axis=0, ignore_index=True) if atk_parts else pd.DataFrame(columns=feature_cols)
    if len(ben) > per_class: ben = ben.sample(n=per_class, random_state=SEED)
    if len(atk) > per_class: atk = atk.sample(n=per_class, random_state=SEED)

    X = pd.concat([ben, atk], axis=0, ignore_index=True)
    y = np.array([0]*len(ben) + [1]*len(atk), dtype=np.int64)

    # shuffle
    idx = np.random.RandomState(SEED).permutation(len(X))
    X = X.iloc[idx].reset_index(drop=True)
    y = y[idx]

    # clean numerics
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

    print(f"[i] built balanced slice -> X={X.shape}, y={y.shape} (benign={np.sum(y==0)}, attack={np.sum(y==1)})")
    return X, y, feature_cols

# ==== build balanced dataset ====
X_df, y_np, feature_cols = load_balanced_binary_recursive(ROOT, per_class=PER_CLASS, benign_tokens=BENIGN_TOKENS)

# (optional) save the balanced slice for reuse
balanced_dir = "/kaggle/working"
X_df.assign(label=y_np).to_parquet(f"{balanced_dir}/CICIoT2023_balanced_30k_30k.parquet", index=False)
X_df.assign(label=y_np).to_csv(f"{balanced_dir}/CICIoT2023_balanced_30k_30k.csv", index=False)
print(f"[i] saved balanced dataset to:\n - {balanced_dir}/CICIoT2023_balanced_30k_30k.parquet\n - {balanced_dir}/CICIoT2023_balanced_30k_30k.csv")

# ==== tensors + split + standardize (fit on train only) ====
X = torch.tensor(X_df.values, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.long)
del X_df, y_np; gc.collect()

N = X.size(0)
perm = torch.randperm(N)
n_val = int(VAL_RATIO * N)
idx_val, idx_tr = perm[:n_val], perm[n_val:]
Xtr, ytr, Xval, yval = X[idx_tr], y[idx_tr], X[idx_val], y[idx_val]

mean = Xtr.mean(dim=0, keepdim=True)
std  = Xtr.std(dim=0, keepdim=True)
std  = torch.where(std == 0, torch.ones_like(std), std)
Xtr  = (Xtr - mean) / std
Xval = (Xval - mean) / std

# ==== quantum + model (Nano LCM) ====
import pennylane as qml

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def angle_encode(x: torch.Tensor):
    return torch.clamp(x, -6.0, 6.0) / 6.0 * math.pi

class QHilbertParam(nn.Module):
    def __init__(self, n_qubits=6, n_layers=3, out_dim=8, obs_pattern="zx"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.obs_pattern = obs_pattern
        self.out_dim = out_dim
        idx = torch.arange(out_dim) % n_qubits
        self.register_buffer("readout_idx", idx)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def _circuit(x_angles, theta_flat):
            for q in range(self.n_qubits):
                qml.RX(x_angles[q], wires=q)
                qml.RZ(0.5 * x_angles[q], wires=q)
            tf = theta_flat.reshape(self.n_layers, self.n_qubits, 3)
            for l in range(self.n_layers):
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q+1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
                for q in range(self.n_qubits):
                    thx, thy, thz = tf[l, q]
                    qml.RX(thx, wires=q); qml.RY(thy, wires=q); qml.RZ(thz, wires=q)
            meas = []
            for k in range(self.out_dim):
                q = int(self.readout_idx[k].item())
                if self.obs_pattern == "z":
                    meas.append(qml.expval(qml.PauliZ(q)))
                elif self.obs_pattern == "x":
                    meas.append(qml.expval(qml.PauliX(q)))
                else:
                    meas.append(qml.expval(qml.PauliZ(q) if (k % 2 == 0) else qml.PauliX(q)))
            return tuple(meas)
        self._pl_circuit = _circuit

    def forward(self, x: torch.Tensor):
        B, F = x.shape
        if F < self.n_qubits:
            pad = self.n_qubits - F
            x_packed = torch.cat([x, x[:, :pad]], dim=1)
        else:
            chunks = list(_chunks(list(range(F)), math.ceil(F / self.n_qubits)))
            pooled = []
            for idxs in chunks[:self.n_qubits]:
                pooled.append(x[:, idxs].mean(dim=1, keepdim=True))
            x_packed = torch.cat(pooled, dim=1)
        x_angles = angle_encode(x_packed)
        emb = []
        tf = self.theta
        for b in range(B):
            vals = self._pl_circuit(x_angles[b], tf)
            e = torch.stack(vals).to(device=x.device, dtype=x.dtype)
            emb.append(e)
        return torch.stack(emb, dim=0)

class HilbertParamPool(nn.Module):
    def __init__(self, n_heads=2, n_qubits=6, n_layers=3, head_dim=8, out_dim=16):
        super().__init__()
        self.heads = nn.ModuleList([
            QHilbertParam(n_qubits=n_qubits, n_layers=n_layers, out_dim=head_dim, obs_pattern="zx")
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(n_heads * head_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x):
        embs = [h(x) for h in self.heads]
        h = torch.cat(embs, dim=1)
        return self.norm(torch.tanh(self.proj(h)))

class QuantumContextGate(nn.Module):
    def __init__(self, ctx_dim, d_model):
        super().__init__()
        self.to_scale = nn.Sequential(nn.Linear(ctx_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.to_shift = nn.Sequential(nn.Linear(ctx_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
    def forward(self, tokens, h_ctx):
        s = self.to_scale(h_ctx).unsqueeze(1)
        t = self.to_shift(h_ctx).unsqueeze(1)
        return tokens * (1 + torch.tanh(s)) + t

class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model=96, n_heads=4, mlp_ratio=2.0, p=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.drop = nn.Dropout(p)
    def forward(self, x):
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

class NanoLCM(nn.Module):
    def __init__(self, in_feats, d_model=96, n_tokens=8, n_qubits=6, n_layers=3,
                 head_dim=8, hpp_dim=16, tf_layers=1, num_classes=1):
        super().__init__()
        self.n_tokens = n_tokens
        self.chunk_size = math.ceil(in_feats / n_tokens)
        self.embed = nn.Linear(self.chunk_size, d_model)
        self.hpp = HilbertParamPool(n_heads=2, n_qubits=n_qubits, n_layers=n_layers,
                                    head_dim=head_dim, out_dim=hpp_dim)
        self.qcg = QuantumContextGate(ctx_dim=hpp_dim, d_model=d_model)
        self.blocks = nn.ModuleList([TinyTransformerBlock(d_model=d_model, n_heads=4, mlp_ratio=2.0, p=0.1)
                                     for _ in range(tf_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))
    def forward(self, x):
        B, F = x.shape
        toks = []
        for idxs in _chunks(list(range(F)), self.chunk_size):
            t = torch.zeros(B, self.chunk_size, device=x.device, dtype=x.dtype)
            t[:, :len(idxs)] = x[:, idxs]
            toks.append(self.embed(t))
            if len(toks) == self.n_tokens: break
        tokens = torch.stack(toks, dim=1)
        h_ctx = self.hpp(x)
        tokens = self.qcg(tokens, h_ctx)
        for blk in self.blocks:
            tokens = blk(tokens)
        pooled = tokens.mean(dim=1)
        logits = self.head(pooled)
        return logits, h_ctx

def binary_metrics_from_logits(logits, y):
    prob = torch.sigmoid(logits.squeeze(-1))
    pred = (prob >= 0.5).long()
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}

# ==== train ====
EPOCHS = 3
BS     = 512
LR     = 2e-3
WD     = 2e-4

model = NanoLCM(
    in_feats=X.shape[1],
    d_model=96,
    n_tokens=8,
    n_qubits=6,
    n_layers=3,
    head_dim=8,
    hpp_dim=16,
    tf_layers=1,
    num_classes=1  # binary -> 1 logit
).to(DEVICE)

Xtr = Xtr.to(DEVICE); ytr = ytr.to(DEVICE)
Xval = Xval.to(DEVICE); yval = yval.to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
criterion = nn.BCEWithLogitsLoss()

model.train()
for epoch in range(EPOCHS):
    perm = torch.randperm(Xtr.size(0), device=DEVICE)
    Xe = Xtr[perm]; ye = ytr[perm]
    losses = []
    for i in range(0, Xe.size(0), BS):
        xb = Xe[i:i+BS]; yb = ye[i:i+BS]
        opt.zero_grad()
        logits, hctx = model(xb)
        loss = criterion(logits.squeeze(-1), yb.float()) + 1e-3 * (hctx.pow(2).mean())
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        logits, _ = model(Xval)
        m = binary_metrics_from_logits(logits, yval)
    print(f"Epoch {epoch+1}: loss={np.mean(losses):.4f}  acc={m['acc']:.3f}  f1={m['f1']:.3f}  "
          f"prec={m['precision']:.3f}  rec={m['recall']:.3f}")

print("\n=== Final (val) ===")
with torch.no_grad():
    logits, _ = model(Xval)
    m = binary_metrics_from_logits(logits, yval)
for k,v in m.items():
    print(f"{k}: {v:.4f}")
