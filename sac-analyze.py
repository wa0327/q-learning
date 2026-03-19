import re
import os
import pandas as pd
from colorama import init, Fore, Style

# 初始化 Colorama (自動重置顏色)
init(autoreset=True)

def parse_sac_log(file_path):
    """讀取檔案並過濾掉 # 開頭的行，解析數據"""
    if not os.path.exists(file_path):
        print(f"{Fore.RED}錯誤：找不到檔案 {file_path}")
        return None

    # 定義正規表達式提取數據
    pattern = r"Steps:([\d,]+).*?Alpha:([\d\.-]+).*?Entropy:([\d\.-]+).*?Q-Val:([\d\.-]+).*?C-Loss:([\d\.-]+).*?A-Loss:([\d\.-]+).*?Rewards:([\d\.-]+).*?Eaten:([\d,]+).*?Killed:([\d,]+).*?Collided:([\d,]+).*?Starved:([\d,]+)"
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 忽略空白行或以 # 開頭的註解行
            if not line or line.startswith('#'):
                continue
                
            match = re.search(pattern, line)
            if match:
                # 處理數字中的逗號
                vals = [v.replace(',', '') for v in match.groups()]
                data.append([float(v) for v in vals])
    
    if not data:
        print(f"{Fore.YELLOW}警告：檔案 {file_path} 中沒有找到匹配的數據。")
        return None

    cols = ['Steps', 'Alpha', 'Entropy', 'Q-Val', 'C-Loss', 'A-Loss', 'Rewards', 'Eaten', 'Killed', 'Collided', 'Starved']
    return pd.DataFrame(data, columns=cols)

def get_trend_ui(val1, val2, higher_better=True):
    """根據指標好壞回傳顏色與符號"""
    if val1 == val2:
        return f"{Fore.WHITE}→ 平行", 0
    
    is_improving = (val2 > val1) if higher_better else (val2 < val1)
    diff_pct = ((val2 - val1) / abs(val1)) * 100 if val1 != 0 else 0
    
    if is_improving:
        return f"{Fore.GREEN}↑ 變好 ({diff_pct:+.2f}%)", 1
    else:
        return f"{Fore.RED}↓ 變差 ({diff_pct:+.2f}%)", -1

def run_analysis():
    file1 = "sac-analyze-1.log"
    file2 = "sac-analyze-2.log"

    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}SAC 訓練日誌對比工具 - 深度分析模式")
    print(f"{Fore.CYAN}{'='*60}\n")

    df1 = parse_sac_log(file1)
    df2 = parse_sac_log(file2)

    if df1 is None or df2 is None:
        return

    metrics = {
        'Rewards':  {'better': True,  'type': 'avg', 'name': '平均回饋 (Rewards)'},
        'Eaten':    {'better': True,  'type': 'inc', 'name': '進食增量 (Efficiency)'},
        'Killed':   {'better': False, 'type': 'inc', 'name': '被殺次數 (Survival)'},
        'Collided': {'better': False, 'type': 'inc', 'name': '碰撞次數 (Safety)'},
        'Starved':  {'better': False, 'type': 'inc', 'name': '餓死次數 (Persistence)'},
        'C-Loss':   {'better': False, 'type': 'avg', 'name': 'Critic 損失 (Stability)'},
        'Entropy':  {'better': None,  'type': 'avg', 'name': '策略熵 (Exploration)'}
    }

    score = 0
    print(f"{'指標名稱':<25} | {'Log-1':>12} | {'Log-2':>12} | {'趨勢狀態'}")
    print("-" * 80)

    for m, info in metrics.items():
        if info['type'] == 'inc':
            v1 = df1[m].iloc[-1] - df1[m].iloc[0]
            v2 = df2[m].iloc[-1] - df2[m].iloc[0]
        else:
            v1 = df1[m].mean()
            v2 = df2[m].mean()

        trend_str, point = get_trend_ui(v1, v2, info['better'])
        if info['better'] is not None:
            score += point
            
        print(f"{info['name']:<25} | {v1:>12.4f} | {v2:>12.4f} | {trend_str}")

    # 總結判斷
    print(f"\n{'='*60}")
    print(f"綜合判斷結論：", end="")
    if score > 0:
        print(f"{Fore.BLACK}{Fore.GREEN} 第二份日誌表現顯著優化！建議保留此版本。")
    elif score < 0:
        print(f"{Fore.BLACK}{Fore.RED} 第二份日誌表現退步。請檢查參數或隨機種子。")
    else:
        print(f"{Fore.YELLOW} 兩份日誌表現持平，模型進入瓶頸或無明顯變動。")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_analysis()