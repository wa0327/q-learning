# MARL AI Survival Experiment: From Evolution to Intelligence
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![MARL](https://img.shields.io/badge/MARL-Reinforcement--Learning-green.svg)

這是一個關於 **多代理人強化學習 (MARL)** 的生存進化實驗。在這個模擬環境中，多個 AI 代理人必須學習如何在競爭與動態障礙中生存、獲取食物並避開捕食者。本專案展示了從基礎演化演算法到現代尖端強化學習架構 (SAC) 的技術迭代。

---

## 🧬 技術演化路徑 (Evolution Roadmap)

本專案透過五個核心檔案展示了 AI 「大腦」的進化過程：

### 1. 遺傳演算法時期 (`survivors.py`)
* **架構**: 基礎矩陣運算 (3層線性映射)
* **特性**: 採用 **遺傳演算法 (GA)**。每回合篩選表現前 10% 的優秀者保留，其餘 90% 透過繼承優秀基因並加入隨機變異產生。
* **狀態**: 這是實驗的起點，AI 僅具備初步的反射動作。

### 2. 混合感知時期 (`survivors2.py`)
* **架構**: `HybridBrain` (CNN-based 特徵提取 + MLP)
* **特性**: 引入 **1D 卷積層 (Conv1d)** 分別處理食物與捕食者的特徵，最後透過全連接層進行特徵融合。
* **改進**: AI 開始具備區分目標種類的能力，並能從多個目標中提取「最強特徵」(Global Max Pooling)。

### 3. 專注力演進時期 (`survivors3.py`)
* **架構**: `ActorAttentionPooling` & `Critic`
* **演算法**: **DDPG (Deep Deterministic Policy Gradient)**
* **特性**: 首次引入 **注意力機制 (Attention Mechanism)**。AI 不再只是看「最強」目標，而是學會自動分配權重給周遭不同的環境因子。
* **改進**: 行動變得更加流暢，開始產生具備預測性的生存策略。

### 4. 柔性智慧時期 (`survivors4.py`)
* **架構**: `SAC-based Actor-Critic` (具備均值與對數標準差輸出)
* **演算法**: **SAC (Soft Actor-Critic)**
* **特性**: 採用 **最大熵 (Maximum Entropy)** 強化學習。Actor 輸出動作分佈而非固定數值，並使用 **Reparameterization Trick** 進行採樣。
* **改進**: 具備強大的探索能力，能應對高度動態的環境，行為更具韌性且不易陷入局部最優解。

### 5. 終極優化版本 (`survivors5.py`)
* **架構**: 維持 SAC 高階架構，但進行深度環境優化與除錯。
* **核心改進**:
    * **獎勵函數精微化**: 移除食物靠近獎勵、減半前進獎勵、大幅提高死亡懲罰。
    * **感應邏輯調整**: 縮短障礙物警戒半徑，使 AI 行為更果斷。
    * **系統效能**: 實作 **批次化動力運算** 與 **異步錄影**，提升訓練吞吐量。
    * **動態環境**: 加入**動態牆面**機制，迫使 AI 學習處理更複雜的空間限制。
    * **可視化**: 增加離牆最近感應線，視覺化 AI 的避障邏輯。

---

## 🛠️ 技術關鍵字
* **MARL**: Multi-Agent Reinforcement Learning
* **Attention Mechanism**: 讓 AI 自動聚焦重要環境資訊。
* **SAC Algorithm**: 結合 Actor-Critic 架構與熵最大化，兼顧穩定與探索。
* **Vectorized Physics**: 批次處理物理運動，優化 CPU/GPU 使用率。

## 🚀 快速開始
1. 安裝環境: `pip install torch pygame opencv-python`
2. 執行實驗: `python survivors5.py`

---
*這部作品記錄了 AI 如何從隨機變異中，最終演化出令人驚嘆的生存智慧。*
