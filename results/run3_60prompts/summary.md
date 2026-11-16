# Real SD1.5 + ControlNet (canny): ControlNet step schedules

60 images analysed (0 skipped), 20 DDIM steps, 512x512, guidance 7.5. Same prompt, control image and seed for every condition. 95% bootstrap CIs over images.

## Control strength by metric (mean [95% CI])

| condition | ControlNet calls | Canny F1 (100/200) (up) | Canny F1 (3 thresholds) (up) | soft edge recall (up) | Chamfer dist (px) (down) | gradient correlation (up) | DINO structure dist (down) | CLIP |
|---|---|---|---|---|---|---|---|---|
| none | 0/20 | 0.406 [0.356, 0.455] | 0.407 [0.359, 0.456] | 0.453 [0.401, 0.505] | 16.867 [12.908, 21.219] | 0.071 [0.037, 0.102] | 0.162 [0.154, 0.170] | 35.782 [34.957, 36.642] |
| full | 20/20 | 0.846 [0.811, 0.879] | 0.829 [0.795, 0.862] | 0.885 [0.857, 0.906] | 1.514 [1.201, 1.952] | 0.755 [0.736, 0.773] | 0.091 [0.086, 0.098] | 35.541 [34.785, 36.334] |
| early | 6/20 | 0.611 [0.568, 0.652] | 0.606 [0.565, 0.646] | 0.694 [0.658, 0.727] | 4.131 [3.471, 4.933] | 0.400 [0.371, 0.428] | 0.119 [0.113, 0.125] | 35.643 [34.917, 36.312] |
| low_noise | 10/20 | 0.491 [0.430, 0.550] | 0.486 [0.429, 0.542] | 0.534 [0.464, 0.603] | 16.023 [11.619, 20.939] | 0.229 [0.179, 0.278] | 0.139 [0.131, 0.147] | 36.141 [35.436, 36.908] |
| combined | 10/20 | 0.621 [0.577, 0.664] | 0.613 [0.570, 0.655] | 0.692 [0.648, 0.732] | 5.302 [4.033, 6.823] | 0.407 [0.380, 0.434] | 0.117 [0.112, 0.123] | 36.195 [35.543, 36.815] |
| early_10 | 10/20 | 0.721 [0.682, 0.759] | 0.711 [0.673, 0.749] | 0.793 [0.768, 0.816] | 2.457 [2.135, 2.816] | 0.580 [0.555, 0.604] | 0.103 [0.097, 0.109] | 35.915 [35.113, 36.651] |
| low_noise_6 | 6/20 | 0.425 [0.372, 0.479] | 0.427 [0.375, 0.480] | 0.465 [0.407, 0.523] | 18.538 [14.033, 23.446] | 0.104 [0.066, 0.141] | 0.154 [0.147, 0.163] | 36.140 [35.345, 36.969] |

## Share of full ControlNet's gain over no control that each setting keeps (1 = as good as full, 0 = as bad as none)

| setting | Canny F1 (100/200) | Canny F1 (3 thresholds) | soft edge recall | Chamfer dist (px) | gradient correlation | DINO structure dist | pixel effect recovery |
|---|---|---|---|---|---|---|---|
| early | 0.467 [0.404, 0.529] | 0.472 [0.414, 0.531] | 0.557 [0.489, 0.629] | 0.830 [0.759, 0.883] | 0.481 [0.439, 0.521] | 0.611 [0.544, 0.671] | 0.404 [0.370, 0.440] |
| low_noise | 0.194 [0.137, 0.254] | 0.188 [0.137, 0.241] | 0.188 [0.113, 0.268] | 0.055 [-0.043, 0.164] | 0.231 [0.190, 0.275] | 0.321 [0.266, 0.378] | 0.101 [0.086, 0.117] |
| combined | 0.490 [0.426, 0.554] | 0.488 [0.428, 0.546] | 0.553 [0.478, 0.628] | 0.753 [0.642, 0.836] | 0.492 [0.451, 0.531] | 0.632 [0.573, 0.683] | 0.414 [0.384, 0.446] |
| early_10 | 0.717 [0.678, 0.755] | 0.721 [0.685, 0.757] | 0.786 [0.748, 0.821] | 0.939 [0.911, 0.961] | 0.744 [0.716, 0.770] | 0.839 [0.808, 0.867] | 0.579 [0.549, 0.610] |
| low_noise_6 | 0.044 [0.023, 0.068] | 0.047 [0.027, 0.068] | 0.027 [0.001, 0.057] | -0.109 [-0.174, -0.047] | 0.050 [0.037, 0.065] | 0.105 [0.083, 0.127] | 0.039 [0.033, 0.046] |

## Paired differences (A - B), mean [95% CI]; the sign depends on the metric direction

| A - B | Canny F1 (100/200) | Canny F1 (3 thresholds) | soft edge recall | Chamfer dist (px) | gradient correlation | DINO structure dist |
|---|---|---|---|---|---|---|
| early - low_noise | 0.120 [0.071, 0.170] | 0.120 [0.076, 0.166] | 0.159 [0.098, 0.224] | -11.893 [-16.749, -7.564] | 0.171 [0.114, 0.227] | -0.020 [-0.029, -0.013] |
| combined - early | 0.010 [-0.003, 0.024] | 0.007 [-0.004, 0.017] | -0.002 [-0.019, 0.014] | 1.171 [0.329, 2.204] | 0.007 [-0.007, 0.021] | -0.001 [-0.003, 0.000] |
| combined - low_noise | 0.130 [0.083, 0.178] | 0.127 [0.085, 0.170] | 0.158 [0.099, 0.217] | -10.722 [-15.540, -6.385] | 0.178 [0.124, 0.232] | -0.022 [-0.030, -0.015] |
| early - low_noise_6 | 0.186 [0.145, 0.227] | 0.179 [0.141, 0.218] | 0.229 [0.179, 0.279] | -14.407 [-19.227, -10.136] | 0.295 [0.250, 0.340] | -0.036 [-0.044, -0.028] |

## Runtime (interleaved repeats, same inputs; denoising loop only)

| condition | median s [IQR] | speedup vs full |
|---|---|---|
| none | 3.68 [3.55, 4.53] | 1.48x |
| full | 5.44 [5.15, 6.35] | 1.00x |
| early | 4.29 [3.95, 4.74] | 1.27x |
| low_noise | 4.61 [4.30, 5.53] | 1.18x |
| combined | 4.45 [4.27, 5.92] | 1.22x |
| early_10 | 4.64 [4.27, 5.27] | 1.17x |
| low_noise_6 | 4.44 [4.01, 5.91] | 1.23x |

Canny F1 compares output edges with the control edges within 2 px. Soft recall is mean exp(-d^2 / 2*3^2) over the distance d from each control edge to the nearest output edge, and Chamfer is that mean distance. Gradient correlation and DINO distance compare the output with the source image the control was extracted from.
