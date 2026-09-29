# Step 2C-1B single-PS proxy A/B — 01021 16:10:44–16:15:48 (2026-09-28)

Isolated only: every measurement ran against throw-away artifact copies on the server's
127.0.0.1:8802–8805. Official 8801 was never contacted by any harness, browser or curl.

## The two proxies (identical container, only the rate control differs)

| | A (current, official) | B (candidate) |
|---|---|---|
| ffmpeg | `-c:v libx264 -preset veryfast -crf 18` | `-c:v libx264 -preset medium -b:v 5500k -maxrate 7500k -bufsize 15000k` |
| resolution / fps | 2560x1440 / 25 | 2560x1440 / 25 |
| pixel format | yuv420p | yuv420p |
| GOP | 10 s (250), 31 keyframes | 10 s (250), 31 keyframes |
| start / duration | 0.080 / 303.92 s | 0.080 / 303.92 s |
| samples | 7598 | 7598 |
| faststart | yes | yes |
| **size** | 339,916,149 B | 202,159,092 B |
| **mean bitrate** | 8.95 Mbps | 5.32 Mbps |
| p50 / p95 / max kB per second | 866 / 2404 / 2833 | 569 / 1646 / 1961 |
| 270 s / 275 s / 280 s kB per second | 2337 / 1787 / 2501 | 1333 / 630 / 1470 |

## Image set — column order is always SOURCE | A | B

| file | frame | time | what it shows |
|---|---|---|---|
| `PS4_f006750_ROI.png` | 6750 | 270.0 s | whole ROI before the item is taken, high-motion |
| `PS4_f006890_ROI.png` | 6890 | 275.6 s | the moment the marked item is removed |
| `PS4_f007000_ROI.png` | 7000 | 280.0 s | the 20 Mbps per-second peak of proxy A |
| `PS4_f006750_P1_796_771.png` | 6750 | 270.0 s | marked object 1, 2x zoom |
| `PS4_f006750_P4_847_686.png` | 6750 | 270.0 s | marked object 4, 2x zoom |
| `PS4_f006750_P6_863_443.png` | 6750 | 270.0 s | marked object 6, 2x zoom |
| `PS4_f006890_P1_796_771.png` | 6890 | 275.6 s | marked object 1 at the removal moment |

Frames are matched by **frame index**, not wall clock: the source PS carries
`start_time 41353.035`, so `-ss <seconds>` against it lands nowhere near the intended
frame (an early attempt produced near-flat frames, MAE ~100 against A).
