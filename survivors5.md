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

如果在不影響模型表現的前提下，要對這套模型剪枝，應該怎麼做？
