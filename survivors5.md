# 2026-3-18
[2026-03-18 15:32:05][Info] Steps:5,000 Alpha:0.2274 Entropy:1.1720 Q-Val:-0.8945 C-Loss:3.4685 A-Loss:0.7215 Rewards:-0.4080 Eaten:37 Killed:47 Collided:66 Starved:5
[2026-03-18 15:33:07][Info] Steps:10,000 Alpha:0.0585 Entropy:-0.0171 Q-Val:-2.0820 C-Loss:2.1050 A-Loss:1.8082 Rewards:-0.0852 Eaten:74 Killed:72 Collided:70 Starved:41
[2026-03-18 15:34:13][Info] Steps:15,000 Alpha:0.0157 Entropy:-1.2388 Q-Val:-0.0359 C-Loss:0.3998 A-Loss:-0.0095 Rewards:-0.0675 Eaten:106 Killed:75 Collided:73 Starved:76
[2026-03-18 15:35:21][Info] Steps:20,000 Alpha:0.0053 Entropy:-1.8528 Q-Val:-0.1603 C-Loss:0.2216 A-Loss:0.0962 Rewards:-0.0136 Eaten:193 Killed:76 Collided:77 Starved:118
[2026-03-18 15:36:21][Info] Steps:25,000 Alpha:0.0107 Entropy:-1.8300 Q-Val:-1.5617 C-Loss:2.5043 A-Loss:1.5753 Rewards:-0.0092 Eaten:302 Killed:76 Collided:81 Starved:171
[2026-03-18 15:37:17][Info] Steps:30,000 Alpha:0.0113 Entropy:-1.6674 Q-Val:-2.7903 C-Loss:0.4014 A-Loss:2.5765 Rewards:0.0482 Eaten:586 Killed:78 Collided:82 Starved:203
[2026-03-18 15:38:14][Info] Steps:35,000 Alpha:0.0260 Entropy:-1.8233 Q-Val:1.1976 C-Loss:1.2733 A-Loss:-1.4420 Rewards:0.0608 Eaten:1,130 Killed:83 Collided:83 Starved:213
[2026-03-18 15:39:06][Info] Steps:40,000 Alpha:0.0211 Entropy:-2.1349 Q-Val:2.1748 C-Loss:1.0421 A-Loss:-2.1865 Rewards:0.1149 Eaten:1,731 Killed:84 Collided:84 Starved:217
[2026-03-18 15:39:56][Info] Steps:45,000 Alpha:0.0237 Entropy:-2.2077 Q-Val:1.1425 C-Loss:1.9722 A-Loss:-1.3082 Rewards:0.1180 Eaten:2,491 Killed:85 Collided:85 Starved:219
[2026-03-18 15:40:46][Info] Steps:50,000 Alpha:0.0245 Entropy:-2.1599 Q-Val:1.6233 C-Loss:2.0819 A-Loss:-1.4331 Rewards:0.1591 Eaten:3,370 Killed:87 Collided:85 Starved:222
[2026-03-18 15:41:37][Info] Steps:55,000 Alpha:0.0338 Entropy:-2.0154 Q-Val:1.4615 C-Loss:3.0462 A-Loss:-1.5516 Rewards:0.1906 Eaten:4,308 Killed:89 Collided:85 Starved:223
[2026-03-18 15:42:28][Info] Steps:60,000 Alpha:0.0323 Entropy:-1.8641 Q-Val:-0.0898 C-Loss:3.0711 A-Loss:0.0005 Rewards:0.2139 Eaten:5,243 Killed:91 Collided:85 Starved:224
[2026-03-18 15:43:20][Info] Steps:65,000 Alpha:0.0353 Entropy:-1.9980 Q-Val:-0.4535 C-Loss:1.2023 A-Loss:0.7457 Rewards:0.1435 Eaten:6,305 Killed:91 Collided:85 Starved:225
[2026-03-18 15:44:19][Info] Steps:70,000 Alpha:0.0347 Entropy:-1.9214 Q-Val:-0.3200 C-Loss:4.1428 A-Loss:0.4437 Rewards:0.1162 Eaten:7,401 Killed:93 Collided:85 Starved:227
[2026-03-18 15:45:27][Info] Steps:75,000 Alpha:0.0372 Entropy:-1.6895 Q-Val:-3.1492 C-Loss:2.2691 A-Loss:2.9764 Rewards:0.1481 Eaten:8,505 Killed:94 Collided:85 Starved:227
[2026-03-18 15:46:37][Info] Steps:80,000 Alpha:0.0343 Entropy:-2.0259 Q-Val:-2.1182 C-Loss:1.9428 A-Loss:2.2443 Rewards:0.2197 Eaten:9,627 Killed:94 Collided:85 Starved:227
[2026-03-18 15:47:46][Info] Steps:85,000 Alpha:0.0346 Entropy:-2.3281 Q-Val:-1.7584 C-Loss:2.6120 A-Loss:1.5243 Rewards:0.1518 Eaten:10,842 Killed:95 Collided:85 Starved:228
[2026-03-18 15:48:45][Info] Steps:90,000 Alpha:0.0360 Entropy:-2.1046 Q-Val:-1.9571 C-Loss:3.0667 A-Loss:2.0423 Rewards:0.1796 Eaten:12,088 Killed:95 Collided:85 Starved:228
[2026-03-18 15:49:35][Info] Steps:95,000 Alpha:0.0357 Entropy:-2.2865 Q-Val:-2.5068 C-Loss:1.4280 A-Loss:2.5079 Rewards:0.1903 Eaten:13,182 Killed:96 Collided:85 Starved:228
[2026-03-18 15:50:26][Info] Steps:100,000 Alpha:0.0345 Entropy:-1.9657 Q-Val:-2.8836 C-Loss:1.5675 A-Loss:2.7990 Rewards:0.1685 Eaten:14,261 Killed:97 Collided:85 Starved:231

# 優化內容
將 alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
改成 alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
及反向傳播 Actor 時凍結 Critic 的參數
[2026-03-18 18:41:59][Info] Steps:5,000 Alpha:0.2273 Entropy:1.1603 Q-Val:5.3100 C-Loss:2.3717 A-Loss:-5.8554 Rewards:-0.1341 Eaten:29 Killed:29 Collided:37 Starved:20
[2026-03-18 18:42:52][Info] Steps:10,000 Alpha:0.0516 Entropy:1.1248 Q-Val:5.9500 C-Loss:0.2383 A-Loss:-6.0577 Rewards:-0.0483 Eaten:63 Killed:32 Collided:39 Starved:62
[2026-03-18 18:43:41][Info] Steps:15,000 Alpha:0.0135 Entropy:0.1047 Q-Val:3.5077 C-Loss:0.3791 A-Loss:-3.4253 Rewards:-0.0438 Eaten:77 Killed:33 Collided:42 Starved:121
[2026-03-18 18:44:32][Info] Steps:20,000 Alpha:0.0064 Entropy:-1.2902 Q-Val:0.8934 C-Loss:0.3162 A-Loss:-0.9401 Rewards:-0.0577 Eaten:96 Killed:33 Collided:43 Starved:191
[2026-03-18 18:45:22][Info] Steps:25,000 Alpha:0.0076 Entropy:-2.8442 Q-Val:-0.4899 C-Loss:7.2073 A-Loss:0.4811 Rewards:-0.0297 Eaten:133 Killed:33 Collided:46 Starved:262
[2026-03-18 18:46:11][Info] Steps:30,000 Alpha:0.0080 Entropy:-1.8072 Q-Val:-2.9106 C-Loss:0.7126 A-Loss:2.9137 Rewards:-0.0373 Eaten:156 Killed:33 Collided:49 Starved:338
[2026-03-18 18:46:59][Info] Steps:35,000 Alpha:0.0095 Entropy:-2.1245 Q-Val:-3.8677 C-Loss:0.4819 A-Loss:3.7099 Rewards:-0.0417 Eaten:172 Killed:33 Collided:53 Starved:420
[2026-03-18 18:47:48][Info] Steps:40,000 Alpha:0.0067 Entropy:-1.5910 Q-Val:-4.6778 C-Loss:0.1384 A-Loss:4.6102 Rewards:-0.0197 Eaten:251 Killed:33 Collided:55 Starved:473
[2026-03-18 18:48:37][Info] Steps:45,000 Alpha:0.0110 Entropy:-2.5245 Q-Val:-2.6129 C-Loss:2.2049 A-Loss:2.3731 Rewards:-0.0171 Eaten:497 Killed:36 Collided:55 Starved:513
[2026-03-18 18:49:26][Info] Steps:50,000 Alpha:0.0224 Entropy:-2.2495 Q-Val:1.9006 C-Loss:3.4183 A-Loss:-2.3804 Rewards:0.0479 Eaten:986 Killed:39 Collided:56 Starved:530
[2026-03-18 18:50:14][Info] Steps:55,000 Alpha:0.0329 Entropy:-1.8134 Q-Val:3.3674 C-Loss:1.4671 A-Loss:-3.2566 Rewards:0.1344 Eaten:1,583 Killed:40 Collided:57 Starved:541
[2026-03-18 18:51:06][Info] Steps:60,000 Alpha:0.0345 Entropy:-1.8595 Q-Val:3.3763 C-Loss:3.5477 A-Loss:-3.3476 Rewards:0.1312 Eaten:2,403 Killed:43 Collided:57 Starved:544
[2026-03-18 18:51:59][Info] Steps:65,000 Alpha:0.0376 Entropy:-1.6550 Q-Val:0.6798 C-Loss:3.7045 A-Loss:-0.8136 Rewards:0.2126 Eaten:3,281 Killed:43 Collided:57 Starved:546
[2026-03-18 18:52:53][Info] Steps:70,000 Alpha:0.0407 Entropy:-2.1207 Q-Val:-0.3619 C-Loss:2.9974 A-Loss:0.2870 Rewards:0.0516 Eaten:4,144 Killed:43 Collided:60 Starved:549
[2026-03-18 18:53:41][Info] Steps:75,000 Alpha:0.0330 Entropy:-2.0844 Q-Val:-2.2930 C-Loss:1.2662 A-Loss:2.2444 Rewards:0.1211 Eaten:5,001 Killed:43 Collided:60 Starved:551
[2026-03-18 18:54:32][Info] Steps:80,000 Alpha:0.0338 Entropy:-1.9737 Q-Val:-2.3071 C-Loss:4.9321 A-Loss:2.0631 Rewards:0.1642 Eaten:5,888 Killed:43 Collided:61 Starved:552
[2026-03-18 18:55:31][Info] Steps:85,000 Alpha:0.0399 Entropy:-1.7891 Q-Val:-3.4293 C-Loss:2.6762 A-Loss:3.5204 Rewards:0.1149 Eaten:6,926 Killed:43 Collided:61 Starved:553
[2026-03-18 18:56:32][Info] Steps:90,000 Alpha:0.0391 Entropy:-2.1115 Q-Val:-3.0238 C-Loss:1.7164 A-Loss:3.1507 Rewards:0.0672 Eaten:8,038 Killed:46 Collided:61 Starved:553
[2026-03-18 18:57:28][Info] Steps:95,000 Alpha:0.0376 Entropy:-2.0119 Q-Val:-2.9239 C-Loss:1.9581 A-Loss:3.1203 Rewards:0.0955 Eaten:9,086 Killed:47 Collided:61 Starved:553
[2026-03-18 18:58:22][Info] Steps:100,000 Alpha:0.0410 Entropy:-1.7426 Q-Val:-3.9922 C-Loss:1.6488 A-Loss:3.9831 Rewards:0.2172 Eaten:10,258 Killed:48 Collided:61 Starved:555

# 優化內容
讓 Critic 共享的感官層後的結果
[2026-03-18 19:49:58][Info] FPS:105.18 Steps:5,000 Alpha:0.2272 Entropy:0.9932 Q-Val:9.8996 C-Loss:13.5387 A-Loss:-10.4729 Rewards:-0.0476 Eaten:33 Killed:39 Collided:34 Starved:15
[2026-03-18 19:50:45][Info] FPS:104.05 Steps:10,000 Alpha:0.0518 Entropy:0.8855 Q-Val:10.4096 C-Loss:0.1457 A-Loss:-10.6648 Rewards:-0.0785 Eaten:77 Killed:41 Collided:36 Starved:57
[2026-03-18 19:51:30][Info] FPS:112.37 Steps:15,000 Alpha:0.0130 Entropy:-0.3276 Q-Val:7.3166 C-Loss:14.4474 A-Loss:-7.3910 Rewards:-0.0534 Eaten:97 Killed:43 Collided:36 Starved:102
[2026-03-18 19:52:14][Info] FPS:108.36 Steps:20,000 Alpha:0.0071 Entropy:-2.5780 Q-Val:5.3546 C-Loss:0.1642 A-Loss:-5.3745 Rewards:-0.0202 Eaten:167 Killed:45 Collided:41 Starved:158
[2026-03-18 19:53:01][Info] FPS:110.61 Steps:25,000 Alpha:0.0100 Entropy:-2.1539 Q-Val:3.9585 C-Loss:0.4605 A-Loss:-3.9099 Rewards:0.0128 Eaten:262 Killed:47 Collided:42 Starved:221
[2026-03-18 19:53:45][Info] FPS:112.21 Steps:30,000 Alpha:0.0103 Entropy:-1.8028 Q-Val:3.2176 C-Loss:0.3690 A-Loss:-3.2583 Rewards:-0.0270 Eaten:485 Killed:48 Collided:44 Starved:243
[2026-03-18 19:54:30][Info] FPS:112.36 Steps:35,000 Alpha:0.0114 Entropy:-1.9691 Q-Val:5.2149 C-Loss:2.7136 A-Loss:-5.6455 Rewards:0.0672 Eaten:1,033 Killed:53 Collided:46 Starved:253
[2026-03-18 19:55:14][Info] FPS:113.95 Steps:40,000 Alpha:0.0211 Entropy:-1.7749 Q-Val:12.8355 C-Loss:5.5802 A-Loss:-13.0051 Rewards:0.0956 Eaten:1,777 Killed:53 Collided:47 Starved:254
[2026-03-18 19:55:58][Info] FPS:98.82 Steps:45,000 Alpha:0.0307 Entropy:-2.3213 Q-Val:13.3815 C-Loss:3.2874 A-Loss:-13.3714 Rewards:0.1066 Eaten:2,501 Killed:54 Collided:47 Starved:256
[2026-03-18 19:56:43][Info] FPS:98.54 Steps:50,000 Alpha:0.0223 Entropy:-1.8322 Q-Val:12.1756 C-Loss:1.6714 A-Loss:-12.3639 Rewards:0.1377 Eaten:3,381 Killed:56 Collided:50 Starved:256
[2026-03-18 19:57:29][Info] FPS:111.16 Steps:55,000 Alpha:0.0281 Entropy:-2.0701 Q-Val:12.8880 C-Loss:1.6215 A-Loss:-12.7299 Rewards:0.1467 Eaten:4,350 Killed:57 Collided:52 Starved:258
[2026-03-18 19:58:14][Info] FPS:112.40 Steps:60,000 Alpha:0.0351 Entropy:-2.1251 Q-Val:10.8440 C-Loss:2.0796 A-Loss:-10.9445 Rewards:0.1481 Eaten:5,316 Killed:57 Collided:54 Starved:258
[2026-03-18 19:58:59][Info] FPS:110.90 Steps:65,000 Alpha:0.0382 Entropy:-1.8570 Q-Val:9.5221 C-Loss:3.6298 A-Loss:-9.5207 Rewards:0.1336 Eaten:6,368 Killed:57 Collided:55 Starved:259
[2026-03-18 19:59:44][Info] FPS:112.97 Steps:70,000 Alpha:0.0281 Entropy:-1.8069 Q-Val:8.1082 C-Loss:5.2002 A-Loss:-8.3633 Rewards:0.1467 Eaten:7,502 Killed:57 Collided:57 Starved:259
[2026-03-18 20:00:30][Info] FPS:110.08 Steps:75,000 Alpha:0.0309 Entropy:-1.9625 Q-Val:7.9213 C-Loss:3.0444 A-Loss:-7.8146 Rewards:0.2371 Eaten:8,634 Killed:58 Collided:57 Starved:259
[2026-03-18 20:01:16][Info] FPS:113.42 Steps:80,000 Alpha:0.0281 Entropy:-1.7219 Q-Val:6.7059 C-Loss:3.1825 A-Loss:-6.6057 Rewards:0.2773 Eaten:9,798 Killed:61 Collided:57 Starved:261
[2026-03-18 20:02:00][Info] FPS:117.23 Steps:85,000 Alpha:0.0340 Entropy:-2.1657 Q-Val:8.5215 C-Loss:2.2006 A-Loss:-8.5750 Rewards:0.1905 Eaten:11,008 Killed:65 Collided:57 Starved:261
[2026-03-18 20:02:44][Info] FPS:113.76 Steps:90,000 Alpha:0.0359 Entropy:-2.0727 Q-Val:7.3095 C-Loss:4.2746 A-Loss:-7.1926 Rewards:0.2096 Eaten:12,131 Killed:67 Collided:58 Starved:263
[2026-03-18 20:03:28][Info] FPS:115.11 Steps:95,000 Alpha:0.0380 Entropy:-2.0125 Q-Val:6.4317 C-Loss:5.7134 A-Loss:-6.6184 Rewards:0.2076 Eaten:13,385 Killed:70 Collided:59 Starved:266
[2026-03-18 20:04:13][Info] FPS:99.94 Steps:100,000 Alpha:0.0396 Entropy:-2.3464 Q-Val:7.8455 C-Loss:1.9620 A-Loss:-7.7125 Rewards:0.2702 Eaten:14,649 Killed:75 Collided:60 Starved:267

# 減枝內容
MAX_OBJ 減至 60
[2026-03-18 20:13:04][Info] FPS:118.26 Steps:5,000 Alpha:0.2279 Entropy:1.0044 Q-Val:4.5491 C-Loss:4.0233 A-Loss:-4.7741 Rewards:-0.2524 Eaten:25 Killed:40 Collided:79 Starved:6
[2026-03-18 20:13:46][Info] FPS:120.16 Steps:10,000 Alpha:0.0547 Entropy:0.2514 Q-Val:2.5716 C-Loss:3.0513 A-Loss:-2.5084 Rewards:-0.1991 Eaten:48 Killed:88 Collided:126 Starved:18
[2026-03-18 20:14:28][Info] FPS:114.93 Steps:15,000 Alpha:0.0156 Entropy:-1.1224 Q-Val:-0.7332 C-Loss:1.4928 A-Loss:0.6513 Rewards:-0.1352 Eaten:87 Killed:145 Collided:133 Starved:31
[2026-03-18 20:15:11][Info] FPS:118.86 Steps:20,000 Alpha:0.0048 Entropy:-2.1028 Q-Val:-1.7926 C-Loss:6.8021 A-Loss:1.7552 Rewards:-0.0826 Eaten:125 Killed:203 Collided:141 Starved:43
[2026-03-18 20:15:53][Info] FPS:116.61 Steps:25,000 Alpha:0.0051 Entropy:-1.7266 Q-Val:-3.2721 C-Loss:1.1930 A-Loss:3.1222 Rewards:-0.1308 Eaten:175 Killed:270 Collided:158 Starved:62
[2026-03-18 20:16:36][Info] FPS:122.00 Steps:30,000 Alpha:0.0059 Entropy:-2.0846 Q-Val:-6.2515 C-Loss:3.1651 A-Loss:6.1391 Rewards:-0.2244 Eaten:236 Killed:316 Collided:187 Starved:81
[2026-03-18 20:17:21][Info] FPS:118.87 Steps:35,000 Alpha:0.0115 Entropy:-2.2220 Q-Val:-10.1540 C-Loss:7.0963 A-Loss:10.0410 Rewards:-0.2140 Eaten:297 Killed:371 Collided:217 Starved:100
[2026-03-18 20:18:03][Info] FPS:119.06 Steps:40,000 Alpha:0.0161 Entropy:-2.1421 Q-Val:-13.6803 C-Loss:5.3570 A-Loss:13.6369 Rewards:-0.0797 Eaten:333 Killed:421 Collided:238 Starved:124
[2026-03-18 20:18:45][Info] FPS:114.52 Steps:45,000 Alpha:0.0132 Entropy:-2.2618 Q-Val:-13.5975 C-Loss:3.4279 A-Loss:13.4661 Rewards:-0.1982 Eaten:367 Killed:474 Collided:251 Starved:148
[2026-03-18 20:19:27][Info] FPS:121.03 Steps:50,000 Alpha:0.0083 Entropy:-1.9862 Q-Val:-15.0565 C-Loss:4.8456 A-Loss:14.7330 Rewards:-0.1169 Eaten:394 Killed:540 Collided:265 Starved:161
[2026-03-18 20:20:09][Info] FPS:120.87 Steps:55,000 Alpha:0.0106 Entropy:-1.9019 Q-Val:-12.6665 C-Loss:2.0027 A-Loss:12.5901 Rewards:-0.0939 Eaten:429 Killed:603 Collided:270 Starved:170
[2026-03-18 20:20:51][Info] FPS:118.77 Steps:60,000 Alpha:0.0081 Entropy:-2.3017 Q-Val:-11.0188 C-Loss:1.2818 A-Loss:10.8974 Rewards:-0.0289 Eaten:482 Killed:664 Collided:271 Starved:187
[2026-03-18 20:21:32][Info] FPS:121.93 Steps:65,000 Alpha:0.0069 Entropy:-2.7666 Q-Val:-9.6769 C-Loss:3.4187 A-Loss:9.4764 Rewards:-0.0504 Eaten:561 Killed:734 Collided:275 Starved:213
[2026-03-18 20:22:14][Info] FPS:118.90 Steps:70,000 Alpha:0.0077 Entropy:-1.8454 Q-Val:-8.9636 C-Loss:1.0763 A-Loss:8.8884 Rewards:-0.0759 Eaten:681 Killed:799 Collided:285 Starved:221
[2026-03-18 20:22:57][Info] FPS:118.49 Steps:75,000 Alpha:0.0136 Entropy:-1.7585 Q-Val:-5.7032 C-Loss:3.6789 A-Loss:5.7335 Rewards:-0.0182 Eaten:1,052 Killed:873 Collided:290 Starved:226
[2026-03-18 20:23:39][Info] FPS:120.02 Steps:80,000 Alpha:0.0210 Entropy:-2.0452 Q-Val:-1.3486 C-Loss:10.8476 A-Loss:1.1883 Rewards:-0.0996 Eaten:1,385 Killed:938 Collided:294 Starved:232
[2026-03-18 20:24:21][Info] FPS:117.91 Steps:85,000 Alpha:0.0151 Entropy:-1.7907 Q-Val:1.3942 C-Loss:1.1535 A-Loss:-1.4251 Rewards:-0.0577 Eaten:1,832 Killed:1,006 Collided:294 Starved:233
[2026-03-18 20:25:04][Info] FPS:117.39 Steps:90,000 Alpha:0.0139 Entropy:-2.0486 Q-Val:0.3874 C-Loss:11.2833 A-Loss:-0.4160 Rewards:0.0189 Eaten:2,353 Killed:1,063 Collided:295 Starved:238
[2026-03-18 20:25:49][Info] FPS:113.52 Steps:95,000 Alpha:0.0171 Entropy:-1.8781 Q-Val:1.0581 C-Loss:2.6070 A-Loss:-1.1779 Rewards:-0.0053 Eaten:3,013 Killed:1,121 Collided:300 Starved:240
[2026-03-18 20:26:36][Info] FPS:102.75 Steps:100,000 Alpha:0.0209 Entropy:-1.8260 Q-Val:2.7406 C-Loss:2.9126 A-Loss:-2.8099 Rewards:0.0552 Eaten:3,843 Killed:1,174 Collided:301 Starved:243

# 減枝內容
HIDDEN_FEAT_DIM = 32
HIDDEN_FC_DIM = 128
[2026-03-18 20:30:09][Info] FPS:123.15 Steps:5,000 Alpha:0.2277 Entropy:0.8670 Q-Val:6.6517 C-Loss:2.3635 A-Loss:-7.0783 Rewards:-0.1304 Eaten:18 Killed:51 Collided:65 Starved:6
[2026-03-18 20:30:50][Info] FPS:121.66 Steps:10,000 Alpha:0.0537 Entropy:0.6288 Q-Val:2.6905 C-Loss:5.1772 A-Loss:-2.7061 Rewards:-0.1617 Eaten:50 Killed:106 Collided:89 Starved:19
[2026-03-18 20:31:30][Info] FPS:123.24 Steps:15,000 Alpha:0.0141 Entropy:-1.1636 Q-Val:-3.3951 C-Loss:7.7491 A-Loss:3.2540 Rewards:-0.2213 Eaten:75 Killed:157 Collided:154 Starved:35
[2026-03-18 20:32:11][Info] FPS:123.19 Steps:20,000 Alpha:0.0095 Entropy:-1.7153 Q-Val:-8.3879 C-Loss:3.8543 A-Loss:8.1948 Rewards:-0.2323 Eaten:107 Killed:196 Collided:206 Starved:53
[2026-03-18 20:32:52][Info] FPS:117.12 Steps:25,000 Alpha:0.0064 Entropy:-1.7690 Q-Val:-10.5918 C-Loss:4.4644 A-Loss:10.5763 Rewards:-0.0760 Eaten:139 Killed:247 Collided:220 Starved:64
[2026-03-18 20:33:33][Info] FPS:120.51 Steps:30,000 Alpha:0.0072 Entropy:-2.2029 Q-Val:-10.0027 C-Loss:4.2012 A-Loss:9.9989 Rewards:-0.1455 Eaten:172 Killed:304 Collided:236 Starved:79
[2026-03-18 20:34:14][Info] FPS:121.86 Steps:35,000 Alpha:0.0046 Entropy:-1.7894 Q-Val:-11.0285 C-Loss:1.6511 A-Loss:10.9498 Rewards:-0.2053 Eaten:196 Killed:363 Collided:252 Starved:93
[2026-03-18 20:34:55][Info] FPS:119.92 Steps:40,000 Alpha:0.0049 Entropy:-1.8450 Q-Val:-11.4573 C-Loss:2.9839 A-Loss:11.3087 Rewards:-0.1077 Eaten:225 Killed:414 Collided:266 Starved:104
[2026-03-18 20:35:36][Info] FPS:125.14 Steps:45,000 Alpha:0.0049 Entropy:-2.0845 Q-Val:-12.3452 C-Loss:2.1997 A-Loss:12.3040 Rewards:-0.1724 Eaten:258 Killed:472 Collided:274 Starved:109
[2026-03-18 20:36:17][Info] FPS:121.32 Steps:50,000 Alpha:0.0083 Entropy:-1.6562 Q-Val:-12.6959 C-Loss:2.0773 A-Loss:12.5611 Rewards:-0.1414 Eaten:275 Killed:509 Collided:282 Starved:133
[2026-03-18 20:36:57][Info] FPS:122.83 Steps:55,000 Alpha:0.0043 Entropy:-2.6921 Q-Val:-12.7975 C-Loss:3.6100 A-Loss:12.6329 Rewards:-0.1233 Eaten:309 Killed:570 Collided:292 Starved:145
[2026-03-18 20:37:38][Info] FPS:124.00 Steps:60,000 Alpha:0.0028 Entropy:-2.2702 Q-Val:-13.2073 C-Loss:2.9414 A-Loss:13.1897 Rewards:-0.2021 Eaten:343 Killed:619 Collided:315 Starved:167
[2026-03-18 20:38:19][Info] FPS:121.73 Steps:65,000 Alpha:0.0054 Entropy:-1.6242 Q-Val:-13.1812 C-Loss:1.4143 A-Loss:13.1974 Rewards:-0.1472 Eaten:380 Killed:673 Collided:318 Starved:184
[2026-03-18 20:38:59][Info] FPS:124.21 Steps:70,000 Alpha:0.0056 Entropy:-1.6635 Q-Val:-12.4940 C-Loss:1.7328 A-Loss:12.4718 Rewards:-0.2055 Eaten:407 Killed:716 Collided:336 Starved:211
[2026-03-18 20:39:40][Info] FPS:123.75 Steps:75,000 Alpha:0.0057 Entropy:-2.6064 Q-Val:-13.5716 C-Loss:2.2839 A-Loss:13.5278 Rewards:-0.1901 Eaten:442 Killed:773 Collided:343 Starved:247
[2026-03-18 20:40:21][Info] FPS:120.32 Steps:80,000 Alpha:0.0055 Entropy:-1.4655 Q-Val:-13.9843 C-Loss:0.7619 A-Loss:14.0521 Rewards:-0.1383 Eaten:477 Killed:822 Collided:358 Starved:282
[2026-03-18 20:41:02][Info] FPS:119.40 Steps:85,000 Alpha:0.0036 Entropy:-1.7988 Q-Val:-13.8746 C-Loss:1.1491 A-Loss:13.7414 Rewards:-0.0917 Eaten:505 Killed:871 Collided:370 Starved:306
[2026-03-18 20:41:43][Info] FPS:123.49 Steps:90,000 Alpha:0.0056 Entropy:-2.3816 Q-Val:-13.8986 C-Loss:1.1525 A-Loss:13.7372 Rewards:-0.0877 Eaten:533 Killed:922 Collided:375 Starved:330
[2026-03-18 20:42:24][Info] FPS:120.79 Steps:95,000 Alpha:0.0029 Entropy:-1.9501 Q-Val:-14.1542 C-Loss:1.6823 A-Loss:14.1720 Rewards:-0.2903 Eaten:566 Killed:986 Collided:386 Starved:355
[2026-03-18 20:43:05][Info] FPS:122.36 Steps:100,000 Alpha:0.0082 Entropy:-3.0681 Q-Val:-14.6441 C-Loss:2.0493 A-Loss:14.6396 Rewards:-0.1728 Eaten:588 Killed:1,046 Collided:404 Starved:370

# 減枝內容
HIDDEN_FEAT_DIM = 32
[2026-03-18 20:50:18][Info] FPS:127.90 Steps:5,000 Alpha:0.2243 Entropy:1.2630 Q-Val:4.0312 C-Loss:1.7758 A-Loss:-4.4202 Rewards:-0.1024 Eaten:24 Killed:38 Collided:80 Starved:5
[2026-03-18 20:50:59][Info] FPS:122.58 Steps:10,000 Alpha:0.0549 Entropy:0.4278 Q-Val:4.8528 C-Loss:17.6879 A-Loss:-5.3417 Rewards:-0.0730 Eaten:48 Killed:67 Collided:88 Starved:40
[2026-03-18 20:51:40][Info] FPS:124.63 Steps:15,000 Alpha:0.0135 Entropy:-1.1628 Q-Val:3.3731 C-Loss:0.6063 A-Loss:-3.5193 Rewards:-0.0355 Eaten:65 Killed:68 Collided:92 Starved:88
[2026-03-18 20:52:21][Info] FPS:122.04 Steps:20,000 Alpha:0.0064 Entropy:-1.7487 Q-Val:2.5946 C-Loss:0.1146 A-Loss:-2.5888 Rewards:-0.0136 Eaten:116 Killed:71 Collided:98 Starved:139
[2026-03-18 20:53:02][Info] FPS:120.07 Steps:25,000 Alpha:0.0085 Entropy:-2.5193 Q-Val:1.0385 C-Loss:0.2721 A-Loss:-1.1572 Rewards:-0.0238 Eaten:209 Killed:71 Collided:104 Starved:191
[2026-03-18 20:53:46][Info] FPS:121.37 Steps:30,000 Alpha:0.0138 Entropy:-1.5868 Q-Val:0.6744 C-Loss:3.6972 A-Loss:-0.7988 Rewards:0.0052 Eaten:355 Killed:71 Collided:105 Starved:226
[2026-03-18 20:54:28][Info] FPS:122.07 Steps:35,000 Alpha:0.0153 Entropy:-1.8357 Q-Val:2.1561 C-Loss:1.3289 A-Loss:-2.3563 Rewards:-0.0067 Eaten:682 Killed:72 Collided:105 Starved:246
[2026-03-18 20:55:12][Info] FPS:120.81 Steps:40,000 Alpha:0.0306 Entropy:-2.0072 Q-Val:7.5309 C-Loss:1.5697 A-Loss:-7.7867 Rewards:0.0651 Eaten:1,186 Killed:73 Collided:106 Starved:254
[2026-03-18 20:55:58][Info] FPS:114.27 Steps:45,000 Alpha:0.0246 Entropy:-2.1511 Q-Val:9.3684 C-Loss:1.9914 A-Loss:-9.4059 Rewards:0.1045 Eaten:1,815 Killed:73 Collided:107 Starved:259
[2026-03-18 20:56:42][Info] FPS:117.69 Steps:50,000 Alpha:0.0267 Entropy:-2.1108 Q-Val:9.1122 C-Loss:2.8669 A-Loss:-9.3606 Rewards:0.1434 Eaten:2,648 Killed:74 Collided:108 Starved:259
[2026-03-18 20:57:28][Info] FPS:119.46 Steps:55,000 Alpha:0.0285 Entropy:-1.9566 Q-Val:8.5094 C-Loss:10.0829 A-Loss:-8.3961 Rewards:0.1397 Eaten:3,462 Killed:75 Collided:108 Starved:260
[2026-03-18 20:58:13][Info] FPS:100.76 Steps:60,000 Alpha:0.0310 Entropy:-2.3688 Q-Val:6.5162 C-Loss:2.8971 A-Loss:-6.7072 Rewards:0.1417 Eaten:4,255 Killed:75 Collided:109 Starved:261
[2026-03-18 20:58:57][Info] FPS:115.33 Steps:65,000 Alpha:0.0326 Entropy:-1.6491 Q-Val:5.1019 C-Loss:2.2287 A-Loss:-5.0693 Rewards:0.1833 Eaten:5,098 Killed:75 Collided:111 Starved:262
[2026-03-18 20:59:44][Info] FPS:106.60 Steps:70,000 Alpha:0.0384 Entropy:-2.0785 Q-Val:4.1971 C-Loss:2.1651 A-Loss:-4.2216 Rewards:0.2060 Eaten:6,071 Killed:77 Collided:111 Starved:262
[2026-03-18 21:00:28][Info] FPS:122.28 Steps:75,000 Alpha:0.0370 Entropy:-1.9564 Q-Val:4.0059 C-Loss:3.1435 A-Loss:-4.1178 Rewards:0.1051 Eaten:7,025 Killed:77 Collided:112 Starved:263
[2026-03-18 21:01:10][Info] FPS:116.77 Steps:80,000 Alpha:0.0383 Entropy:-2.1348 Q-Val:2.9332 C-Loss:2.0996 A-Loss:-3.0191 Rewards:0.1948 Eaten:7,979 Killed:77 Collided:112 Starved:266
[2026-03-18 21:01:53][Info] FPS:121.27 Steps:85,000 Alpha:0.0407 Entropy:-2.0285 Q-Val:3.7823 C-Loss:3.0193 A-Loss:-3.9493 Rewards:0.0655 Eaten:9,016 Killed:79 Collided:112 Starved:267
[2026-03-18 21:02:37][Info] FPS:114.18 Steps:90,000 Alpha:0.0388 Entropy:-1.5993 Q-Val:0.1791 C-Loss:2.2323 A-Loss:-0.3191 Rewards:0.1794 Eaten:9,981 Killed:80 Collided:114 Starved:269
[2026-03-18 21:03:20][Info] FPS:117.82 Steps:95,000 Alpha:0.0420 Entropy:-1.8694 Q-Val:0.6526 C-Loss:3.1281 A-Loss:-0.5744 Rewards:0.1134 Eaten:11,217 Killed:82 Collided:114 Starved:269
[2026-03-18 21:04:04][Info] FPS:115.72 Steps:100,000 Alpha:0.0424 Entropy:-1.9867 Q-Val:2.0674 C-Loss:1.5703 A-Loss:-2.3261 Rewards:0.0832 Eaten:12,322 Killed:83 Collided:115 Starved:270

# 2026-3-18 結論：
Critic 共享的感官層的表現目前居冠。
