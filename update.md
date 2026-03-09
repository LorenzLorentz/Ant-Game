# 游戏逻辑更新说明

这个文档只说明规则层面的改动，供后续 AI / 仿真器 / rollout / 训练环境同步使用。

## 0. 总体原则

- 前端和协议没有扩字段。
- 操作码、塔类型编号、超级武器编号都保持不变。
- 但多种塔和蚂蚁的“行为语义”已经变了。
- 因此旧 AI 如果只按旧规则理解同样的 type id，大概率会模拟错误。

另外，这套规则里的随机过程仍然是“给定 seed 可复现”的；但因为引入了更多在线随机和隐藏内部状态，简单 if-else / alpha-beta 旧逻辑会更容易失配。

## 1. 信息素：`float -> int`

### 1.1 存储方式

- pheromone 不再以浮点数存储。
- 现在统一按定点整数存储，比例尺为：

```text
real_value = pheromone_int / 10000
```

- 常量：
  - `PHEROMONE_SCALE = 10000`
  - `LAMBDA_NUM = 97`
  - `LAMBDA_DENOM = 100`

### 1.2 初始化

- 初始 pheromone 仍然由 LCG 按 seed 生成。
- 现在初始化公式为整数版：

```text
pheromone_int = 80000 + ((lcg_value * 10000) >> 46)
```

- 对应真实值大致在 `8.0 ~ 12.0` 之间。

### 1.3 更新

- 成功到达：`+100000`
- 被击杀：`-50000`
- 老死：`-30000`

全局衰减改成整数公式：

```text
p_new = max(0, (97 * p_old + 3000 + 50) // 100)
```

等价于实数近似：

```text
p_new ~= 0.97 * p_old + 0.3
```

### 1.4 对 AI 的影响

- 底层 pheromone 更新已经不是 float。
- 但移动决策层仍然会把整数 pheromone 转成浮点评分参与 softmax。
- 不要把它错误实现成“纯整数 argmax”。

## 2. 蚂蚁新增内部状态

新增 5 种行为状态：

- `DEFAULT`
  - 默认型。
  - 仍朝敌方主基地前进，但不再是纯贪心，改为高温 softmax 采样。

- `CONSERVATIVE`
  - 保守型。
  - 使用旧式的确定性寻迹逻辑，按评分最大值走。

- `RANDOM`
  - 随机型。
  - 不看 pheromone，不看目标，直接在合法方向中随机采样。

- `BEWITCHED`
  - 蛊惑型。
  - 不再以敌方主基地为目标，而是朝内部 `target` 移动。

- `CONTROL_FREE`
  - 免控型。
  - 移动方式与 `CONSERVATIVE` 相同。
  - 但不会再被控制效果改写行为。

## 3. 蚂蚁出生分布

新生蚂蚁不再全是同一种行为。

当前出生概率：

- `DEFAULT`: `0.50`
- `CONSERVATIVE`: `0.20`
- `RANDOM`: `0.15`
- `CONTROL_FREE`: `0.15`

`BEWITCHED` 不会在出生时直接生成，只能通过控制效果进入。

## 4. 蚂蚁移动逻辑

### 4.1 候选方向

- 仍然只在 6 个相邻六边形里选。
- `DEFAULT / CONSERVATIVE / CONTROL_FREE` 默认不允许立即反向回头。
- `RANDOM / BEWITCHED` 允许回头。
- 如果非回头候选为空，则会退化为允许回头重新找候选。

### 4.2 `DEFAULT`

`DEFAULT` 不再直接选最大 pheromone 方向，而是：

1. 对每个候选格计算基础评分：

```text
raw = pheromone * weight
```

其中：

- 更接近敌方基地：`weight = 1.25`
- 距离不变：`weight = 1.0`
- 更远离敌方基地：`weight = 0.75`

2. 加入拥塞惩罚：

```text
score = raw - 1.25 * crowding_penalty
```

3. 用 softmax 采样，温度：

```text
DEFAULT_MOVE_TEMPERATURE = 4.0
```

### 4.3 `CONSERVATIVE`

- 评分仍按老式思路算 `raw = pheromone * weight`。
- 但最终不是采样，而是直接选 `raw` 最大的方向。
- 不吃拥塞惩罚。

### 4.4 `RANDOM`

- 在所有合法候选方向中均匀随机选一个。
- 不参考 pheromone。
- 不参考目标。

### 4.5 `BEWITCHED`

- 不再朝敌方主基地移动，而是朝自己的 `target` 移动。
- 对每个候选格计算：

```text
score = (current_distance_to_target - next_distance_to_target) * 4.0
        - 1.25 * crowding_penalty
```

- 再用 softmax 采样，温度：

```text
BEWITCH_MOVE_TEMPERATURE = 1.5
```

### 4.6 `CONTROL_FREE`

- 路径选择和 `CONSERVATIVE` 相同。
- 但控制技能不会再改写其行为状态。

## 5. 拥塞效应

引入了显式的拥塞惩罚。

对某个候选落点 `(x, y)`：

- 若该格已有己方蚂蚁：`+1.0`
- 若距离为 1 的邻近格有己方蚂蚁：每只 `+0.35`

然后乘上 `CROWDING_PENALTY = 1.25` 后，从移动评分中扣掉。

这个机制主要影响：

- `DEFAULT`
- `BEWITCHED`

不影响：

- `CONSERVATIVE`
- `CONTROL_FREE`
- `RANDOM`

## 6. 随机传送

新增周期性随机传送机制。

- 每 `10` 回合触发一次：

```text
ANT_TELEPORT_INTERVAL = 10
```

- 触发时，从“当前仍在场、且不是 `CONTROL_FREE`”的蚂蚁里随机抽取 `20%`：

```text
ANT_TELEPORT_RATIO = 0.2
```

- 这些蚂蚁会被传送到随机合法格。
- 合法格指可走路径格，且不包括双方主基地。

时序上，传送发生在当回合正常移动之后。

## 7. 状态退化 / 状态恢复

### 7.1 `RANDOM -> DEFAULT`

- `RANDOM` 状态不是永久的。
- 随机型蚂蚁在进入该状态后，持续 `5` 个行为回合会自动退化回 `DEFAULT`：

```text
RANDOM_ANT_DECAY_TURNS = 5
```

### 7.2 `BEWITCHED -> DEFAULT`

- 蛊惑型蚂蚁在到达自己的 `target` 后，会自动恢复为 `DEFAULT`。

### 7.3 冻结后的延迟状态切换

- `Ice` 不会立刻把蚂蚁改成 `RANDOM`。
- 当前实现是：
  - 先冻结 1 回合，导致该回合不能移动；
  - 同时记录一个 `pending_behavior = RANDOM`；
  - 下一轮攻击阶段开始前解冻，再把行为切到 `RANDOM`。

## 8. 防御塔语义修改

类型编号没变，但部分塔的语义已经重写。

### 8.1 `Ice`（编号 `12`）

- 造成正常伤害。
- 若目标不是免控蚂蚁，则：
  - 冻结 1 回合；
  - 解冻后切到 `RANDOM`。

### 8.2 `Cannon`（编号 `13`）

虽然协议里 type id 仍然叫 `Cannon`，但语义上它现在是 `Bewitch` 塔。

命中后：

- 若蚂蚁在“自己的半场”：
  - `target = 自己的主基地`
- 若蚂蚁已进入“对方半场”：
  - `target = 蚂蚁所属方自己半场内的随机合法位置`

之后把蚂蚁行为切到 `BEWITCHED`。

### 8.3 `Pulse`（编号 `32`）

- 造成伤害。
- 然后把目标切到 `RANDOM`。

也就是说，`Pulse` 现在不再只是传统范围伤害语义，而是带随机化控制效果。

## 9. 超级武器 / 区域状态修改

### 9.1 `EmergencyEvasion`

- 区域内蚂蚁仍然获得紧急回避效果。
- 当前实现仍会给蚂蚁 `2` 层 shield。
- 但新增一条规则：
  - 一旦蚂蚁脱离该状态，不论是因为离开区域还是效果结束，都会转成 `CONTROL_FREE`。

### 9.2 `Deflectors`

- 区域内蚂蚁仍然获得引力护盾效果。
- 但新增一条规则：
  - 一旦蚂蚁脱离该状态，不论是因为离开区域还是效果结束，都会转成 `CONTROL_FREE`。

### 9.3 `Lightning Storm`

- 闪电风暴区域不再固定。
- 每回合都会随机漂移一次。
- 候选位置为：
  - 原地不动
  - 六个相邻合法格

### 9.4 `EMP Blaster`

- EMP 区域同样不再固定。
- 每回合都会随机漂移一次。
- 候选位置规则与 `Lightning Storm` 相同。

这意味着：

- EMP 的禁建 / 禁升级 / 禁拆塔范围是动态变化的；
- Lightning 的伤害覆盖区域也是动态变化的。

## 10. 回合时序

当前一整轮的核心顺序是：

1. 塔与超级武器攻击
2. 蚂蚁移动
3. 周期性传送
4. 信息素更新
5. 结算成功 / 死亡 / 老死
6. 生成新蚂蚁
7. 增加年龄并处理行为退化
8. 更新金币与超级武器状态

其中：

- `Lightning Storm / EMP` 的漂移发生在回合末的 item 更新阶段；
- `Ice` 的冻结会直接影响同回合移动，因为攻击发生在移动之前。

## 11. 协议与可观测性

### 11.1 协议编号不变

以下编号都没变：

- 塔操作：`11 / 12 / 13`
- 超级武器：`21 / 22 / 23 / 24`
- 塔类型：`12 = Ice`, `13 = Cannon`, `32 = Pulse`

所以前端不用改，但 AI 不能再按旧语义理解这些编号。

### 11.2 新行为状态不是公开字段

当前公开给 AI 的蚂蚁字段仍然只有这些：

- `id`
- `player`
- `x, y`
- `hp`
- `level`
- `age`
- `status`

其中：

- `status` 里只能看到 `Alive / Success / Fail / TooOld / Frozen`
- 看不到 `DEFAULT / CONSERVATIVE / RANDOM / BEWITCHED / CONTROL_FREE`
- 也看不到 `BEWITCHED` 的内部 target
- 也看不到 `pending_behavior`

这意味着：

- 新规则下游戏已经变成带隐藏内部状态的环境；
- 如果 AI 想精确模拟，必须自己维护这些隐藏状态；
- 如果不维护，基于旧公共状态的硬编码策略会明显失准。

## 12. 对后续 AI 修正的直接建议

如果要适配当前规则，优先检查以下内容：

- 本地仿真器是否已经改成整数 pheromone
- 移动逻辑是否仍然保留 softmax，而不是退化成整数贪心
- 是否支持 5 种蚂蚁行为状态
- 是否支持 `Ice / Cannon(Bewitch) / Pulse` 的新控制语义
- 是否支持 `10` 回合一次的 `20%` 随机传送
- 是否支持拥塞惩罚
- 是否支持 `RANDOM` 的 5 回合退化
- 是否支持 `Deflectors / EmergencyEvasion` 结束后变 `CONTROL_FREE`
- 是否支持 `Lightning / EMP` 每回合漂移
- 是否错误假设这些内部状态能从公开状态直接读到

如果旧 AI 曾经：

- 把蚂蚁移动写成固定 if-else
- 把路线视为完全可预测
- 直接把 type `13` 当老 `Cannon`
- 直接把 type `32` 当老 `Pulse`
- 只用公共状态做 rollout，不维护内部行为状态

那么都需要重写。

