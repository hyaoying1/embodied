# Fast–Slow Embodied Agent Demo

一个可直接运行的 MVP，演示具身 Agent 如何在动态家庭任务中组合：

- **Fast Brain**：以透明的规则快速选择普通导航、检查、抓取和放置动作；
- **Memory**：保存目标位置、观察时间、置信度和 `fresh` / `unverified` / `contradicted` 状态；
- **Delegation Router**：在记忆被动作结果否定后，以本地轻量分类器决定委托或 abstain；
- **Slow Brain**：默认使用明确标注的 oracle planner 重新定位目标；可选 VLM 只用于 3D demo 的候选区域评估；
- **Dynamic Event**：Agent 初次观察目标后，环境会在首次抓取前将目标从左柜移动到右柜。

> 当前实现是可复现的机制 MVP，不是训练完成的 VLA。常规 CLI、dashboard 和单元测试运行在无外部依赖的 `HouseholdEnvironment` 上；`AI2ThorEnvironment` 已提供同一 AgentEnvironment 契约的 3D adapter，并支持显式配置的动态目标移动测试。既有 AI2-THOR demo 仍是独立的预编排 validation harness，尚未改为直接调用 `FastSlowAgent`。

## 运行

要求：Python 3.9 或更新版本。

运行终端版轨迹展示：

```zsh
PYTHONPATH=src python3 -m fast_slow_agent.cli
```

生成可直接在浏览器打开的 dashboard：

```zsh
PYTHONPATH=src python3 -m fast_slow_agent.dashboard
open artifacts/fast-slow-demo-dashboard.html
```

评测套件会固定运行四种情境：

1. `Always Fast` + 目标移动：持续依赖旧记忆，最终失败；
2. `Fast–Slow Router` + 目标移动：检测冲突并委托一次 Slow Brain，恢复成功；
3. `Fast–Slow Router` + 正常环境：不调用 Slow Brain，直接完成任务；
4. `Fast–Slow Router` + 零 Slow Brain 预算：检测冲突后安全 abstain，不重复错误。

## 核心 workflow

```text
观察目标 → 写入 fresh 记忆 → 动态事件移动目标 → 抓取失败
→ 记忆变为 contradicted → Router 计算 delegate_probability
→ delegate / 明确原因的 abstain → Slow Brain 写入 unverified 提示
→ 前往提示区域并强制 inspect 验证 → fresh 观察 → 抓取并放置
```

`unverified` 提示永远不会直接触发抓取：Router 会先检查该位置。若检查没有目标，记忆会被标为 `contradicted`，再决定委托或 abstain。abstain trace 会明确说明是预算耗尽还是委托概率未达阈值。

## 可选：AI2-THOR 3D 后端

项目在 `.venv` 中使用固定版本 `ai2thor==5.0.0`。如需在另一台机器安装：

```zsh
python3 -m venv .venv
.venv/bin/python -m pip install "ai2thor==5.0.0"
```

运行真实 3D target-move episode：

```zsh
PYTHONPATH=src .venv/bin/python -m fast_slow_agent.thor_demo
open artifacts/thor
```

运行由共享 `FastSlowAgent` 直接驱动的 adapter episode：

```zsh
PYTHONPATH=src .venv/bin/python -m fast_slow_agent.thor_agent_demo
```

该 harness 会建立一次初始观测、移动杯子、重放旧姿态制造抓取失败、验证委托分数，然后执行 oracle 或 VLM 区域选择。低层移动、抓取和放置仍由 AI2-THOR 的 `GetInteractablePoses` 与 `TeleportFull` bridge 完成；它不是学习到的视觉导航或抓取策略。

### Slow Brain 模式

```text
oracle：默认离线模式，直接使用 demo 内部的已知目标区域，适合确定性物理 smoke test。
vlm：将候选区域的第一人称 PNG 提供给配置的视觉模型，并在区域间选择最高置信度的可见目标。
```

启用 VLM：

```zsh
export FAST_SLOW_VLM_API_KEY="<your-api-key>"
export FAST_SLOW_VLM_MODEL="<your-vision-model>"
# 可选：兼容 Chat Completions 的自定义端点
export FAST_SLOW_VLM_ENDPOINT="https://api.openai.com/v1/chat/completions"

PYTHONPATH=src .venv/bin/python -m fast_slow_agent.thor_demo --slow-brain vlm
```

只有 `vlm` 模式会将 PNG 和任务描述发送到配置的 endpoint；密钥不会写入文件、日志或 dashboard。未配置 key/model 时，程序会在发送任何图片前失败。

每次委托最多发出 **3 个候选请求、每个候选最多 4 张图片**。3D harness 为覆盖诊断可在本地采样、保存每区最多 8 张视角，但只有前 4 张会上传给 VLM。视觉搜索会记录逐候选错误；若没有阳性结果或最佳结果低于默认 `0.60` 置信阈值，会 abstain 并输出错误/评估摘要，而不会将缺失选择变成无上下文的运行时错误。

## 无外部视觉模型的 3D 模式

```zsh
PYTHONPATH=src .venv/bin/python -m fast_slow_agent.thor_local_demo
open artifacts/thor/local-perception
```

此模式在移动后的绿色 Lettuce 任务上遍历候选台面视角，并读取 AI2-THOR 提供的局部 `visible` 元数据。它不上传图片、不使用 API，也不是现实世界可泛化视觉模型。

## 运行测试

```zsh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

AI2-THOR integration test 默认跳过；设置 `RUN_THOR_INTEGRATION=1` 才会启动模拟器。

## 当前功能与限制

- 固定的蓝色 Mug 家庭任务，以及绿色 Lettuce 本地 3D baseline；
- 结构化时序记忆和动态目标移动事件；
- Always Fast 与 Fast–Slow Router 对照、完整 trace 和 self-contained HTML dashboard；
- VLM candidate search 的请求、图像、响应格式和最小置信度约束；
- 普通 Fast 动作仍是规则，不是学习策略；`TinyDelegationModel` 仅用于矛盾记忆的委托 gate；
- `AI2ThorEnvironment` 让 2D 与 3D 后端共享同一 Agent loop；既有 `thor_demo` 仍保留为独立 oracle/VLM validation harness；
- `VisionSearchSlowPlanner` 是可注入的共享 loop 库接口，调用方负责候选图像采集；预编排 VLM CLI 仍由 `thor_demo --slow-brain vlm` 提供。

## 目录

```text
src/fast_slow_agent/
├── environment.py        # 确定性家庭环境和动态事件
├── memory.py             # 结构化时序记忆
├── router.py             # Fast Router 和委托判断
├── learning.py           # TinyDelegationModel
├── models.py             # 共享数据结构
├── planner.py            # 明确标记的 oracle Slow Brain
├── agent.py              # 两种 Agent 与控制循环
├── evaluation.py         # 固定对照实验
├── cli.py / dashboard.py # 终端和 HTML 入口
├── vision_slow_brain.py  # VLM 请求、解析和候选容错
├── thor_memory.py        # THOR 到共享记忆的桥接
├── thor_demo.py          # 3D oracle/VLM validation harness
├── thor_local_demo.py    # 本地 simulator-perception baseline
└── thor_diagnostics.py   # 候选图像覆盖率诊断
```

详细说明见 [fast-slow-embodied-demo.md](./fast-slow-embodied-demo.md)，实施计划见 [fast-slow-embodied-demo-implementation-plan.md](./fast-slow-embodied-demo-implementation-plan.md)。
