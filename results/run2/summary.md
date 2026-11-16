# Real SD1.5 + ControlNet(canny): ControlNet schedules

356 images analysed (0 skipped), 20 DDIM steps, 512x512, guidance 7.5. Paired design: same prompt, control image and seed for every condition. 95% bootstrap CIs over images.

## Control strength by metric (mean [95% CI])

| condition | ControlNet calls | Canny F1 (100/200) (up) | Canny F1 (3 thresholds) (up) | soft edge recall (up) | Chamfer dist (px) (down) | gradient correlation (up) | DINO structure dist (down) | CLIP |
|---|---|---|---|---|---|---|---|---|
| none | 0/20 | 0.452 [0.433, 0.471] | 0.452 [0.434, 0.471] | 0.488 [0.469, 0.507] | 13.633 [12.261, 15.246] | 0.071 [0.058, 0.083] | 0.156 [0.153, 0.159] | 35.314 [34.996, 35.647] |
| full | 20/20 | 0.859 [0.847, 0.870] | 0.843 [0.831, 0.854] | 0.876 [0.866, 0.886] | 1.982 [1.664, 2.360] | 0.746 [0.738, 0.755] | 0.088 [0.087, 0.090] | 35.164 [34.860, 35.471] |
| early | 10/20 | 0.738 [0.724, 0.752] | 0.729 [0.715, 0.743] | 0.782 [0.770, 0.794] | 3.031 [2.736, 3.353] | 0.574 [0.563, 0.586] | 0.100 [0.098, 0.102] | 35.451 [35.130, 35.771] |
| low_noise | 10/20 | 0.526 [0.504, 0.548] | 0.521 [0.500, 0.542] | 0.546 [0.521, 0.570] | 13.879 [12.256, 15.677] | 0.222 [0.205, 0.240] | 0.132 [0.130, 0.135] | 35.351 [35.039, 35.670] |
| combined | 10/20 | 0.651 [0.632, 0.669] | 0.643 [0.625, 0.660] | 0.694 [0.676, 0.711] | 5.623 [4.983, 6.343] | 0.396 [0.382, 0.410] | 0.116 [0.114, 0.118] | 35.709 [35.418, 36.016] |

## Share of full ControlNet's gain over no-control that each setting keeps (1 = as good as full, 0 = as bad as none)

| setting | Canny F1 (100/200) | Canny F1 (3 thresholds) | soft edge recall | Chamfer dist (px) | gradient correlation | DINO structure dist | pixel effect recovery |
|---|---|---|---|---|---|---|---|
| early | 0.701 [0.687, 0.715] | 0.708 [0.695, 0.722] | 0.759 [0.743, 0.775] | 0.910 [0.891, 0.930] | 0.745 [0.735, 0.756] | 0.828 [0.815, 0.838] | 0.565 [0.552, 0.578] |
| low_noise | 0.181 [0.158, 0.204] | 0.175 [0.155, 0.196] | 0.149 [0.120, 0.177] | -0.021 [-0.104, 0.063] | 0.224 [0.208, 0.241] | 0.347 [0.328, 0.367] | 0.105 [0.099, 0.110] |
| combined | 0.488 [0.465, 0.511] | 0.488 [0.467, 0.510] | 0.530 [0.501, 0.559] | 0.687 [0.630, 0.738] | 0.481 [0.462, 0.499] | 0.591 [0.568, 0.613] | 0.401 [0.390, 0.412] |

## Paired differences (A - B), mean [95% CI]; sign depends on the metric direction

| A - B | Canny F1 (100/200) | Canny F1 (3 thresholds) | soft edge recall | Chamfer dist (px) | gradient correlation | DINO structure dist |
|---|---|---|---|---|---|---|
| early - low_noise | 0.212 [0.195, 0.229] | 0.208 [0.193, 0.224] | 0.237 [0.215, 0.259] | -10.848 [-12.549, -9.349] | 0.352 [0.332, 0.373] | -0.032 [-0.035, -0.030] |
| combined - early | -0.087 [-0.095, -0.079] | -0.086 [-0.093, -0.079] | -0.089 [-0.099, -0.079] | 2.592 [2.156, 3.116] | -0.178 [-0.188, -0.169] | 0.016 [0.015, 0.017] |
| combined - low_noise | 0.125 [0.110, 0.142] | 0.122 [0.108, 0.137] | 0.148 [0.128, 0.169] | -8.256 [-9.868, -6.727] | 0.174 [0.153, 0.195] | -0.016 [-0.019, -0.014] |

## Runtime (interleaved repeats, same inputs; denoising loop only)

| condition | median s [IQR] | speedup vs full |
|---|---|---|
| none | 2.00 [1.72, 2.33] | 1.52x |
| full | 3.04 [2.52, 4.72] | 1.00x |
| early | 2.78 [2.10, 3.60] | 1.09x |
| low_noise | 2.19 [2.14, 2.89] | 1.39x |
| combined | 2.17 [2.10, 3.29] | 1.40x |

Metrics: Canny F1 = output edges vs control edges within 2 px; soft recall = mean exp(-d^2/2*3^2) of control-edge-to-nearest-output-edge distance; Chamfer = that mean distance; gradient correlation and DINO distance compare the output with the source image the control was extracted from.
