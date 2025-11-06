# Agent-Tradition: Game Logic, SDK, AI, and Evaluation

本仓库提供一个完整的双人回合制策略游戏开发与评测框架，包含四个清晰的模块：

- logic/ — 游戏逻辑（规则、状态更新、对局回放、胜负判定等）
- SDK/ — PettingZoo 环境封装、训练/评测脚本与策略工具
- AI/ — 选手 AI（统一接口 ai_{name}.py），可包含子目录与模型权重
- tests/ — 单元测试与集成测试（划分为 logic_sdk/ 与 ai/ 两类）
- replays/ — 统一保存所有对局回放（jsonl，每行一个状态/动作条目）

主目录仅保留游玩入口 `play.py`，支持手动操作（提供合法操作提示）、AI 对战。

---

## 1. Logic 讲解：规则与架构

### 1.1 游戏规则（简述）
- 棋盘大小 `row x col`（见 logic/constant.py），地形包括：平原、沼泽、山地。
- 双方各有一名主将（MainGenerals），盘面还会出现副将与农夫。
- 回合由先手/后手交替进行：
  - 玩家回合内可进行若干“军队移动”“将军移动/升级”“释放战法”“研发科技”“使用超级武器”“征召副将”等操作。
  - 每轮结束（双方都走完一回合）后触发：
    - 兵力增减（产兵、沼泽惩罚、每轮/每 10 轮增长等），
    - 技能与武器持续时间、冷却更新，
    - 重置移动步数，
    - 生成该轮回放记录。
- 胜负判定（logic/game_rules.py）：
  - 若任一方主将被消灭，另一方立即获胜；
  - 若回合数 ≤ 500，游戏未结束；
  - 超过 500 回合采用平局裁决：总兵力 > 占领格数 > 金币 的优先级比较决定胜负；完全相等则判先手胜（可配置）。

### 1.2 数据与模块架构

- 核心数据结构（logic/gamedata.py）
  - Cell, Generals（Main/Sub/Farmer）、SuperWeapon、枚举类型（方向、地形、技能、科技、武器）。
  - GameState（logic/gamestate.py）维护棋盘、将军列表、金币、科技、武器状态、剩余移动步数、回放文件路径等。

- 指令与执行（logic/ai2logic.py）
  - 统一入口 `execute_single_command(player, state, command_type, params)`：
    - 1 军队移动 → logic/movement.py
    - 2 将军移动 → logic/movement.py（含 BFS 合法性检查）
    - 3 将军升级（产量/防御/机动）→ logic/upgrade.py
    - 4 战法释放 → logic/general_skills.py
    - 5 科技研发 → logic/upgrade.py
    - 6 超级武器 → logic/super_weapons.py
    - 7 征召副将 → logic/call_generals.py
  - 每个子模块负责合法性检查、状态修改与回放输出。

- 战斗计算（logic/computation.py）
  - 计算攻击/防御：考虑周边将军的增益/减益、将军防御等级、超级武器影响。

- 回合与回放（logic/gamestate.py, logic/generate_round_replay.py）
  - `update_round(state)` 完成每轮产兵/扣兵、技能/武器/冷却衰减、移动步数重置，并记录本轮回放。
  - 回放为 jsonl：初始化一行，之后每次成功操作/轮末各追加一行，便于分析和重播。

- 胜负规则（logic/game_rules.py）
  - `is_game_over(state)` 返回 -1/0/1 表示未结束/先手胜/后手胜；`tiebreak_now(state)` 便于在限制回合后立即裁决。

- 本地对战运行器（logic/runner.py）
  - `run_match(ai0, ai1, seed, max_rounds, replay_dir, p0_name, p1_name)`：
    - 初始化/写入初始化回放
    - 交替调用双方 AI 产生命令 → `execute_single_command`
    - 每轮末调用 `update_round`
    - 默认回放保存到 `replays/YYYYmmdd_HHMMSS_p0-..._p1-..._seed-..._rounds-....jsonl`

---

## 2. SDK 环境：PettingZoo 封装与使用

### 2.1 环境说明（SDK/pettingzoo_env.py）
- AEC 环境（Alternating Environment）
  - `env(render_mode=None)` 返回环境实例
  - 代理命名：`player_0`, `player_1`，回合切换与逻辑一致
  - 终局奖励：胜者 +1，败者 -1，其他 0（非法动作 -0.01）

- 动作编码（MultiDiscrete, 长度 8）
  - idx 0: type（0..8）→ 0 EndTurn, 1 ArmyMove, 2 GeneralMove, 3 LevelUp, 4 Skill, 5 Tech, 6 SuperWeapon, 7 Call, 8 GiveUp
  - 其余槽位含义随 type 不同（已在 SDK/README.md 与源码注释中详细说明）

- 观测（Dict）
  - board_owner/board_army/board_terrain/generals_owner/coins/rest_moves/current_player

### 2.2 最小使用示例
```python
from SDK import env
e = env(render_mode="ansi")
e.reset()
for _ in range(20):
    agent = e.agent_selection
    a = e.action_space(agent).sample()
    a[0] = 0  # 偏置为 EndTurn 以推进游戏
    e.step(a)
    if any(e.terminations.values()) or any(e.truncations.values()):
        break
print(e.render())
```

### 2.3 训练与评测工具
- 策略工具（SDK/policies.py）：简单的共享小网络（reduced action set），含 `obs_to_tensor`、`logits_to_action` 等。
- 自对弈训练（SDK/train_selfplay.py）：REINFORCE 风格，自博弈，默认保存至 `AI/selfplay/model.pt`。
- 评测（SDK/eval_policy.py）：载入权重并统计胜/负/平。
- 批量评测（SDK/batch_eval.py）：命令行工具，支持先后手、对局数量、对手配置、扩展指标统计（见第 4 节）。

安装依赖（训练/评测用）：
```
pip install pettingzoo gymnasium torch
```

---

## 3. AI 编写与 RL 教程

### 3.1 AI 规范
- 文件位置：`AI/ai_{name}.py`
- 必须暴露函数：
```python
def policy(round_idx: int, my_seat: int, state: GameState) -> list[list[int]]:
    ...
```
- 返回值为“命令列表”，每条命令为整型列表，最后必须附加 `[8]` 表示结束回合。

示例（随机安全）：`AI/ai_random_safe.py`

### 3.2 使用 SDK 环境进行强化学习
1) 自对弈训练（REINFORCE）
```
python SDK/train_selfplay.py --episodes 200 --save AI/selfplay/model.pt
```
2) 使用训练好的 AI 对战
```
python play.py ai-vs-ai --ai0 selfplay --ai1 greedy
```
3) 评测
```
python SDK/eval_policy.py --model AI/selfplay/model.pt --episodes 50
```

### 3.3 自训练 AI 的权重组织与加载
- 训练默认将权重保存到 `AI/selfplay/model.pt`。
- 推理端 AI 文件 `AI/ai_selfplay.py`：
  - 默认从 `AI/selfplay/model.pt` 读取（可通过环境变量 `SELFPLAY_MODEL` 覆盖路径）。
  - 将 reduced action logits 映射为 `{EndTurn, 从主将格子向上下左右移动 1 个兵}`。

> 你也可以在 `AI/{your_name}` 目录下管理自己的权重与辅助代码，只需保证主入口文件 `AI/ai_{your_name}.py` 暴露 `policy(...)` 接口。

---

## 4. 运行与评测教程

### 4.1 快速测试
- 运行单局 AI 对战并生成回放（默认保存至 `replays/`）：
```
python play.py ai-vs-ai --ai0 greedy --ai1 random_safe --rounds 60 --seed 0
```
- 手动对战（含合法操作提示）：
```
python play.py manual-vs-ai --ai1 greedy
```

### 4.2 系统化评测（批量与可扩展指标）

我们提供了完整的批量评测脚本 `SDK/batch_eval.py`，支持：

- 指定先后手与交替先后手（`--swap_seats`）
- 指定对局数量（`--games`）与最大回合数（`--rounds`）
- 指定双方 AI（名称或 `module:function`）
- 默认回放统一保存到 `replays/`，文件名包含时间戳、对局双方、种子与回合数
- 自定义指标拓展：通过 `--metrics module:function` 注入评测函数，接收 `GameState` 返回 `dict`

示例：
```
python SDK/batch_eval.py --ai0 greedy --ai1 random_safe --games 50 --swap_seats --rounds 60
```

带自定义指标：
```
python SDK/batch_eval.py --ai0 selfplay --ai1 greedy --games 100 \
  --metrics yourpkg.metrics:collect_stats
```
自定义指标函数示例：
```python
def collect_stats(state: GameState) -> dict:
    # 统计最终总兵力与占领格数
    from logic.constant import row, col
    a0=a1=c0=c1=0
    for i in range(row):
        for j in range(col):
            cell = state.board[i][j]
            if cell.player == 0:
                a0 += cell.army; c0 += 1
            elif cell.player == 1:
                a1 += cell.army; c1 += 1
    return {"army0": a0, "army1": a1, "cells0": c0, "cells1": c1}
```

### 4.3 CI/CD（GitLab）
- `.gitlab-ci.yml`：
  - 使用 `python:3.11-slim`，安装 `requirements-dev.txt`
  - 运行 `pytest -q`
  - 将 `replays/` 作为构建产物保存（便于后续分析）

---

## 5. 测试与质量保障
- `tests/logic_sdk/`：覆盖移动/升级/技能/武器/回合/回放/规则/Runner/SDK 环境等核心逻辑。
- `tests/ai/`：验证 AI 接口、短局对战与回放输出路径（应在 `replays/`）。

运行测试：
```
pytest -q
```

---

## 6. 常见问题（FAQ）
- 回放文件为何是一行？
  - 每次成功操作与每轮结束都会追加一行；若 AI 在早期没有有效指令，只会看到初始化与轮末少量行。使用基线 AI 或实际策略可观察到多行回放。
- 手动模式如何查看合法提示？
  - 在 `play.py manual-vs-ai` 的输入中输入 `hint`；或直接查看提示输出的指令格式与当前可用资源。

