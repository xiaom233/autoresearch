# 盲识别隔离协议

## 问题

Agent 在实验过程中可以访问 GT 退化文件，导致：
1. 反思修正时同时修改了 `PARAMS_PATH` 和 `VAL_PARAMS_PATH`，R1 vs Spec 对比无效
2. Agent 可以直接对比 predicted_params 和 GT，"盲识别" 变成 "开卷考试"

## 三层隔离

### 1. 文件隔离 — GT 物理隐藏

```
expN/
├── .ground_truth/          # GT 退化（Agent 不得访问的任何路径）
│   └── D01_dual.json
├── degradation/            # Agent 盲识别输出 / 反思修正
│   └── D01_dual.json       # predicted_params
│   └── D01_dual_R1.json    # 反思修正（只改 PARAMS）
```

**规则**：
- Agent 不得 `cat`/`read`/`python open()` `.ground_truth/` 下的任何文件
- Agent 不得在训练命令中显式写出 `.ground_truth/` 路径
- Agent 使用 launcher 提交训练，launcher 自动注入 GT 路径

### 2. 进程隔离 — launcher 注入 GT

Agent 不直接写完整的训练命令，而是通过 launcher：

```bash
# Agent 提交（不包含 GT 路径）:
bash expN/scripts/launcher.sh D01_dual

# launcher 内部（Agent 看不到）:
AR_PARAMS_PATH=degradation/D01_dual.json         # Agent 的 predicted
AR_VAL_PARAMS_PATH=.ground_truth/D01_dual.json   # GT（强制注入）
AR_QUIET_PIPELINE=1                              # 抑制 pipeline 打印
uv run train.py
```

**关键**：
- `AR_QUIET_PIPELINE=1` 阻止 train.py 打印退化管线到日志
- launcher 的输出不包含 GT 路径
- Agent 只知道退化名称，不知道 GT 路径

### 3. 反思隔离 — VAL 永远指向 GT

反思修正时，只改 `PARAMS_PATH`，`VAL_PARAMS_PATH` 锁定为 GT：

```bash
# Spec: 盲识别训练 → GT 验证
bash launcher.sh D01_dual                     # PARAMS=predicted, VAL=GT

# R1: 修正训练 → 同一 GT 验证  
bash launcher.sh D01_dual_R1                  # PARAMS=修正后, VAL=GT (不变!)

# 对比才有意义：Spec 和 R1 考同一张试卷(GT)
```

**规则**：
- Agent 只改 degradation 文件名，不改验证路径
- launcher 始终将 VAL 指向 `.ground_truth/{base_name}.json`

## 防泄露检查清单

提交训练前确认：

| 检查项 | 正确 | 错误 |
|--------|------|------|
| VAL_PARAMS_PATH | 由 launcher 注入 | Agent 手动指定 |
| 训练命令 | `bash launcher.sh D01_dual` | `AR_VAL_PARAMS_PATH=degradation/...` |
| 反思修正 | 只创建 `D01_dual_R1.json` | 创建 `D01_dual_R1.json` 且修改 VAL |
| 日志 | `QUIET_PIPELINE=1` | pipeline 明文打印 |
| GT 访问 | Agent 绝不读取 | `cat .ground_truth/*.json` |

## 目录结构

```
expN/
├── .ground_truth/        ← chmod 000（启动器有权限，Agent 无权限）
├── degradation/          ← Agent 盲识别输出（可读写）
├── experiments/          ← 训练产物（checkpoint + results.json）
├── scripts/
│   └── launcher.sh       ← 训练启动器
├── results/              ← cross_test 评估结果
├── logs/
└── summarize/
```
