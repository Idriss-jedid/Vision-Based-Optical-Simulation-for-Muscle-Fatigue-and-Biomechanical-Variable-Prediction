# Pipeline — Vision → OpenSim labels (descendant) + AI surrogates (montant)

```mermaid
flowchart TB
  classDef cls fill:#eaf2ff,stroke:#3a76c2,stroke-width:1px,color:#143;
  classDef ai  fill:#fff3e6,stroke:#e8821e,stroke-width:1px,color:#532;
  classDef out fill:#e9f7ef,stroke:#1a9a5a,stroke-width:2px,color:#063;
  classDef tst fill:#f6f8fa,stroke:#9aa,color:#555;

  subgraph CLASSIC["CLASSICAL PIPELINE  (videos to OpenSim labels)"]
    direction TB
    V["Multi-camera videos<br/>4 cams, 50 fps"]:::cls
    CAL["Calibration  t = -R*T<br/>-> Calib.toml"]:::cls
    P2D["2D pose · RTMPose<br/>-> keypoints JSON"]:::cls
    TRI["Triangulation<br/>-> 3D .trc"]:::cls
    FIL["Butterworth filtering<br/>-> filtered .trc"]:::cls
    ANG["Stage B · angles<br/>-> curl.mot"]:::cls
    MOD["Stage C · scaled Arm26<br/>-> .osim"]:::cls
    OS["OpenSim<br/>ID + SO + 3CC"]:::cls
    LAB["Biomechanics LABELS<br/>torque · forces · activations · fatigue"]:::out
    V --> CAL --> P2D --> TRI --> FIL --> ANG --> MOD --> OS --> LAB
  end

  T1["Test 1 — 2D pose<br/>reproj = 8.3 px (good)"]:::tst --- P2D
  T2["Test 2 — 3D recon vs Vicon<br/>27.8 mm · dir 5.5 deg (good)"]:::tst --- TRI
  T3["Test 3 — joint angles vs Vicon<br/>elbow r=0.993, MAE 4.2 deg<br/>shoulder MAE 1.2 deg"]:::tst --- ANG
  T4["Test 4 — Arm26 scaling<br/>x0.85 to x1.05 (good)"]:::tst --- MOD

  subgraph AIP["AI SURROGATES  (climb up: replace more of the chain)"]
    direction BT
    A1["L1 · ML from .mot + .osim<br/>(replaces OpenSim ID/SO/3CC)<br/>R2 = 0.95"]:::ai
    A2["L2 · ML from 3D joints<br/>(replaces OpenSim + angles + scaling)<br/>R2 = 0.90"]:::ai
    A3["L3 · ML from 2D keypoints (4 cams)<br/>(replaces triangulation + ...)<br/>R2 = 0.86"]:::ai
    A4["NEXT · ML directly from video ?<br/>end-to-end (future work)"]:::ai
    A1 --> A2 --> A3 --> A4
  end

  MOD -. tap .-> A1
  TRI -. tap .-> A2
  P2D -. tap .-> A3
  V   -. tap .-> A4

  A1 --> PRED["PREDICTED biomechanics<br/>(no OpenSim at inference)"]:::out
  A2 --> PRED
  A3 --> PRED
  A4 -. future .-> PRED
```

## How to read it
- **Left / descending** = the classical pipeline we built: videos -> calibration -> 2D pose -> triangulation -> filtering -> angles (.mot) -> scaled Arm26 (.osim) -> OpenSim (ID/SO/3CC) -> biomechanics **labels**. Each stage was **validated vs Vicon** (Tests 1-4).
- **Right / ascending** = the **AI part**. Each level taps the pipeline **higher up** and replaces **more** of the classical chain:
  - **L1 (0.95)** keeps .mot + .osim, replaces only the OpenSim computation.
  - **L2 (0.90)** drops OpenSim entirely, predicts from the **3D joints**.
  - **L3 (0.86)** drops triangulation too, predicts from the **2D keypoints**.
  - **Next** would predict **directly from the video** (true end-to-end).
- The trend **0.95 -> 0.90 -> 0.86** quantifies the accuracy cost of removing each classical step.
