# Fast–Slow Embodied Agent Demo

## 1. Demo 名称

**When Should an Embodied Agent Think Slowly?**<br>
**基于快速决策与按需委托的动态 3D 具身 Agent Demo**

## 2. Demo 目标

本 Demo 展示一个具身 Agent 如何在 3D 家庭环境中，将高频、简单的行动交给轻量的 **Fast Brain**，并只在出现不确定性、记忆冲突、环境变化或高风险行动时，按需调用较慢但更强的 **Slow Brain**。

核心目标是验证以下直觉：

> 大模型不必负责 Agent 的每一个微动作。Agent 应当先快速判断当前是否足够确定；只有在需要额外推理或验证时，才消耗更多计算预算调用强模型。

Demo 不追求训练大型 Vision-Language-Action（VLA）模型，而是验证一个可观察、可解释、低计算成本的 **fast–slow delegation** 系统机制。

## 3. 用户可见的场景

环境为一个可交互的 3D 厨房或客厅，包含：

- 柜子、抽屉、桌子、餐桌等容器与家具；
- 杯子、钥匙、盘子等可交互物体；
- Agent 的第一人称视角；
- 可执行的移动、转向、观察、打开容器、拿取、放置等动作；
- 在 Agent 执行任务过程中发生的环境变化。

建议使用 **AI2-THOR** 或 ProcTHOR 提供房间、物体、第一人称渲染与交互接口。

## 4. 示例任务

### 用户指令

```text
找到蓝色马克杯，并把它放到餐桌上。
```

### 正常环境

```text
1. Agent 观察到蓝色马克杯位于左侧柜子。
2. Agent 在记忆中记录杯子位置与观察时间。
3. Agent 前往柜子、取走杯子、前往餐桌并放下。
```

### 动态变化测试

```text
1. Agent 已观察到蓝色马克杯位于左侧柜子。
2. Agent 执行其他移动动作。
3. 环境将马克杯从左侧柜子移动到右侧柜子，或将其遮挡。
4. Agent 返回左侧柜子时，发现观察结果与旧记忆冲突。
5. 系统决定：继续行动、重新观察、重新规划，或调用 Slow Brain。
```

## 5. 系统总体架构

```text
用户任务指令 + 第一人称观察 + 结构化记忆
                    │
                    ▼
        ┌───────────────────────────┐
        │ Fast Brain                │
        │ 轻量判别式决策模型         │
        │ 输出：下一步选择与置信度   │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │ Delegation Router          │
        │ 判断是否直接执行或升级推理  │
        └───────┬─────────────┬─────┘
                │             │
     高置信直接执行 │             │ 低置信/高风险/冲突
                ▼             ▼
       ┌──────────────┐   ┌────────────────┐
       │ 环境动作执行  │   │ Slow Brain     │
       │ move/open/...│   │ 强视觉/语言推理 │
       └──────┬───────┘   └───────┬────────┘
              │                   │
              └───────┬───────────┘
                      ▼
             记忆更新、重规划、继续执行
```

## 6. 核心模块

### 6.1 Environment：3D 具身环境

职责：

- 提供 3D 房间、物体与可交互容器；
- 提供 Agent 第一人称图像或状态观察；
- 执行动作并返回环境变化；
- 在指定条件下施加动态事件，例如移动目标、遮挡物体或关闭容器。

基础动作集合：

```text
move_forward
turn_left
turn_right
look_up / look_down
open(container)
close(container)
pick_up(object)
put_down(object, target)
inspect(object_or_area)
```

### 6.2 Memory：结构化时序记忆

Agent 不仅保存“物体在哪里”，还保存该记忆是否可能过期。

示例：

```json
{
  "object": "blue_mug",
  "last_known_location": "left_cabinet",
  "last_seen_step": 3,
  "confidence": 0.92,
  "environment_version": 1,
  "status": "possibly_stale"
}
```

记忆模块需要支持：

- 保存物体、位置、容器关系与观察时间；
- 根据时间、环境事件和观察冲突降低置信度；
- 标记已过期或需要验证的记忆；
- 提供 Fast Brain 使用的记忆新鲜度与冲突特征。

### 6.3 Candidate Generator：候选行动生成器

Fast Brain 不直接在无限动作空间中生成动作，而是从一个小而合法的候选集合中选择。

示例候选：

```text
1. 前往记忆中的左侧柜子
2. 打开当前面对的柜子
3. 检查附近区域
4. 重新观察目标位置
5. 根据记忆重新规划
6. 调用 Slow Brain 验证
7. 停止并请求帮助
```

候选生成器可以由规则、任务规划器或 Slow Brain 在高层阶段提供。

### 6.4 Fast Brain：快速选择模型

Fast Brain 是 Demo 的核心。它是一个小型判别式模型，而不是生成长文本的 LLM。

输入特征：

```text
- 当前第一人称视觉 embedding 或对象观察摘要
- 当前任务目标
- Agent 位姿与局部地图
- 结构化记忆
- 目标记忆的时间年龄（memory age）
- 当前观察与记忆是否冲突
- 每个候选行动的距离、信息增益与风险
- Slow Brain 剩余调用预算
- 最近动作历史与失败信号
```

输出：

```json
{
  "action": "delegate_to_slow_brain",
  "confidence": 0.38,
  "risk": "medium",
  "reason_features": [
    "memory_stale",
    "target_not_found_at_last_known_location"
  ]
}
```

Fast Brain 的动作模式：

| 模式 | 含义 |
|---|---|
| `act` | 高置信度地执行普通行动。 |
| `observe` | 优先获取更多低成本视觉信息。 |
| `verify` | 到记忆位置或关键区域核验旧信息。 |
| `replan` | 当前计划失效，重新生成行动候选。 |
| `delegate` | 调用 Slow Brain 处理模糊或复杂状态。 |
| `abstain` | 遇到高风险、无法安全判断的状态时停止。 |

第一版可以使用规则或分数函数验证系统流程；研究版本可替换为 Logistic Regression、LightGBM、两层 MLP 或经 teacher trajectory 蒸馏的轻量 Transformer。

### 6.5 Delegation Router：委托决策器

根据 Fast Brain 的置信度、风险与记忆状态决定是否调用 Slow Brain。

示例策略：

```text
若 confidence >= 0.80 且风险低：直接执行。
若 0.45 <= confidence < 0.80：优先 observe 或 verify。
若 confidence < 0.45、检测到记忆冲突或行动不可逆：delegate。
若任务明显超出当前能力或存在安全风险：abstain。
```

研究版本不应只使用固定阈值，而应学习在固定推理预算下最大化成功率与安全性的委托策略。

### 6.6 Slow Brain：慢速强推理模块

Slow Brain 只在 Fast Brain 明确请求帮助时运行。

职责：

- 识别模糊物体或容器内容；
- 从任务与记忆中生成/修正高层子目标；
- 处理记忆与当前观察之间的冲突；
- 生成新的候选行动；
- 判断某个高风险行动是否值得执行。

实现阶段：

```text
Demo 阶段：可使用环境 oracle 或规则规划器，仅验证整体控制流。
原型阶段：可接入 API VLM/LLM 或冻结的开源 VLM。
研究阶段：Slow Brain 只能使用 Agent 可获得的观察；环境真值仅用于训练标签和评估。
```

## 7. 关键交互流程

### 7.1 正常情况：快速执行

```text
任务：将蓝色马克杯放到餐桌。

Step 1：Slow Brain 初次规划子目标：进入厨房 → 找杯子 → 拿杯子 → 去餐桌。
Step 2：Agent 看到蓝杯在左柜；Memory 记录位置，confidence = 0.92。
Step 3：Fast Brain 选择前往左柜，confidence = 0.97。
Step 4：Fast Brain 选择打开左柜，confidence = 0.95。
Step 5：Fast Brain 选择拿取蓝杯，confidence = 0.96。
Step 6：Fast Brain 选择前往餐桌并放置。

结果：任务完成；Slow Brain 只在初始规划时被调用。
```

### 7.2 环境变化：按需升级推理

```text
任务：将蓝色马克杯放到餐桌。

Step 1：Agent 看见蓝杯在左柜；Memory confidence = 0.92。
Step 2–7：Fast Brain 执行普通移动。
Step 8：环境事件：蓝杯被移至右柜。
Step 9：Agent 打开左柜，未找到蓝杯。
Step 10：Memory 标记为 stale；confidence 降至 0.31。
Step 11：Fast Brain 发现“当前观察与旧记忆冲突”。
Step 12：Fast Brain 选择 delegate，调用 Slow Brain。
Step 13：Slow Brain 指出应搜索右侧柜子，并更新子目标。
Step 14–18：Fast Brain 接管短程移动、打开右柜、拿取杯子、前往餐桌。

结果：任务完成；只在真正需要时产生一次额外慢推理调用。
```

### 7.3 高风险情况：拒绝或请求帮助

```text
任务：将指定物体放入特定容器。

Fast Brain 检测到：
- 目标存在多个相似候选；
- 错误拿取会触发不可逆操作；
- Slow Brain 预算仍可用。

决策：delegate 或 abstain，而不是自信执行。
```

## 8. 可视化界面

Demo 页面或桌面窗口应同时显示：

| 区域 | 显示内容 |
|---|---|
| 主视图 | Agent 的第一人称 3D 环境画面。 |
| 任务栏 | 用户指令、当前子目标、任务进度。 |
| Memory 面板 | 已见物体、最后位置、时间年龄、置信度、是否过期。 |
| Fast Brain 面板 | 候选行动、被选行动、置信度、风险、决策延迟。 |
| Slow Brain 面板 | 调用次数、预算、最近一次推理/计划更新。 |
| 事件流 | 观察、动作、环境变化、委托和恢复轨迹。 |
| 指标栏 | 总动作数、总时延、Slow Brain 调用次数、成功状态。 |

推荐的可视化示例：

```text
Fast Brain
selected: delegate_to_slow_brain
confidence: 0.38
memory freshness: stale
latency: 4 ms

Slow Brain
calls: 2 / 10
new plan: inspect right cabinet
latency: 1.4 s
```

## 9. 对照系统

为证明 Demo 的设计价值，至少比较以下四种策略：

| 策略 | 行为 |
|---|---|
| Always Fast | 永不调用 Slow Brain，只使用快模型或规则。 |
| Always Slow | 每一步均调用 Slow Brain。 |
| Fixed-Frequency | 每固定 N 步调用一次 Slow Brain。 |
| Fast–Slow Router | 根据置信度、记忆冲突、风险与预算动态选择。 |

## 10. 评价指标

### 任务表现

- 任务成功率；
- 动态环境变化后的成功率；
- 平均完成步数；
- 失败恢复率；
- 重新规划次数。

### 计算效率

- 每个任务的 Slow Brain 调用次数；
- 总推理延迟；
- 平均单步决策延迟；
- API token 或估算计算成本；
- 在固定 Slow Brain 调用预算下的成功率。

### 可靠性与校准

- 高置信错误率；
- 记忆过期导致的错误率；
- 漏委托率：本应升级但仍直接行动的比例；
- 过度委托率：本可快速完成却调用慢模型的比例；
- 置信度校准误差（ECE / reliability curve）；
- 高风险行动前的 abstain 或 delegate 触发率。

## 11. MVP 范围

### 第一版：机制可视化 Demo

```text
- 一个 AI2-THOR 3D 房间；
- 一个“找杯子并放到桌上”的任务；
- 一个固定环境变化：在 Agent 观察后移动目标；
- 结构化记忆与记忆新鲜度；
- 基于规则/阈值的 Fast Brain；
- oracle 或规则 Slow Brain；
- 可视化动作、置信度、记忆和委托事件；
- Always Fast 与 Fast–Slow Router 的轨迹对比。
```

目标：在 1–3 天内录制一段清楚解释机制的演示视频。

### 第二版：轻量学习原型

```text
- 自动生成多个物体移动、遮挡、干扰物与容器状态变化场景；
- 从 oracle 或强 teacher agent 收集行动/委托标签；
- 训练小型 MLP 或树模型作为 Fast Brain；
- 与 Always Slow、Always Fast、Fixed-Frequency 比较；
- 输出成功率、延迟、预算与可靠性曲线。
```

目标：验证快模型能否学习非固定的委托策略。

### 第三版：研究版本

```text
- 部分可观测 3D 家庭任务；
- 多轮任务与长期时序记忆；
- 真实 VLM/LLM Slow Brain；
- 动态、按步骤的委托策略；
- 记忆冲突、行动可逆性、风险和剩余预算共同建模；
- 系统化的 OOD 与环境变化评测。
```

## 12. 研究定位

本项目不主张首次提出 fast–slow embodied control。已有研究已经表明：大 VLM 的低频语义理解配合小型高频控制策略，能提高实时性；也已有工作研究基于不确定性的 `Act / Think / Abstain` 或小模型向大模型委托。

本项目的具体切口是：

> 在动态、部分可观测的 3D 家庭环境中，Agent 如何结合记忆新鲜度、当前观察冲突、行动风险和有限推理预算，进行逐步（step-level）的校准式委托。

预期研究问题：

```text
1. 环境发生变化后，哪些特征最能预测“旧记忆不再可靠”？
2. 哪些状态应优先 observe / verify，而不是立即调用强模型？
3. 在相同的 Slow Brain 调用预算下，动态委托能否提升任务成功率？
4. 如何减少高置信错误和漏委托，同时避免过度调用慢模型？
```

## 13. 项目成功标准

Demo 成功不要求一开始超过所有 VLA 方法，而应满足：

1. 在 3D 交互环境中完成完整的观察—记忆—行动闭环；
2. 清楚展示环境变化导致普通快速策略失败；
3. 清楚展示 Fast–Slow Router 能识别冲突、升级推理并恢复任务；
4. 显示每次决策的置信度、记忆状态、调用预算与延迟；
5. 通过对照实验说明：在接近 Always Slow 成功率的同时，减少 Slow Brain 调用和总延迟；
6. 明确区分机制展示、学习原型与研究级评测，避免把 oracle 信息伪装成真实 Agent 能力。
