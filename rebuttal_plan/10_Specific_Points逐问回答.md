# Specific Points of Feedback for Rebuttal：逐问回答

与05主稿一致。各问题保留原文；Weakness回复仍单独保留在05中。

## Reviewer YRJq

#### Question 1

> How are x-restormer and the other baselines trained? Are they trained separately per scenario, jointly across scenarios, or using existing pretrained weights?

**Response.** All original baselines use their official releases; we do not retrain them separately for the 24 scenarios. X-Restormer uses the released Gaussian-denoising checkpoint, rather than a model trained for each composite scenario. AutoIR develops a scenario-specific restorer on external DIV2K clean images degraded by its estimated pipeline. The five support pairs guide diagnosis and model selection. The added DFPIR-ft experiment is explicitly separate: it fine-tunes a project-pretrained DFPIR model using the same final DC and adaptation-training GPU-time budget.

## Reviewer Tqi7

#### Question 1

> What distinctive reasoning or model-design mechanism does AutoIR contribute beyond its predefined toolkit? Please illustrate with an existing example.

**Response.** The RS case in the submitted supplement (blind_0018) directly illustrates evidence-guided restorer development. After verification supported the contrast-stretch (severity 4) + JPEG (severity 5) hypothesis, RS retained this estimated degradation chain and changed the network from plain Swin to Swin+SimpleGate+FiLM-GCM, activating global modulation for the contrast-related degradation. Both configurations used the same two-epoch training budget.

| Restorer configuration | DIV2K PSNR | LSDIR PSNR |
|---|---:|---:|
| Swin | 17.46 | 17.31 |
| Swin+SimpleGate+FiLM-GCM | 19.56 | 18.89 |
| Improvement | **+2.10** | **+1.58** |

Here the performance improvement accompanies a network-configuration change under the same estimated training degradation. The mechanism is to use diagnostic evidence and confidence to guide restorer configuration, train the resulting network and evaluate its behavior within the DC–RS–CLR development loop. The building blocks are predefined; the automated development decisions determine when and how to use them.

#### Question 2

> How does CLR distinguish an incorrect degradation hypothesis from an inadequate restorer, and why does it fail to correct the reported impulse-noise case?

**Response.** Composite degradation identification is ill-posed because coupled operators can imitate or obscure each other's signatures. CLR combines two kinds of evidence: support-pair simulation discrepancies inform DC revision, while training curves, divergence and checkpoint behavior inform model/training revision. Neither is a causal oracle. In the impulse case, strong corruption distorts brightness-related statistics and independent noise realizations undermine pixelwise verification. The underestimated noise and spurious brightness hypothesis persist, producing mismatched synthetic training data that architecture changes cannot repair. Reliable attribution is therefore a distinct research problem that can strengthen the DC–RS–CLR loop; the present failure remains unresolved.

## Reviewer bAkj

#### Question 1

> The specific contribution of the LLM agent is not sufficiently isolated from the handcrafted toolkits and predefined search space.

**Response.** We added a non-LLM rule controller with the same support pairs, diagnostic/restorer tools and three-round cap. AutoIR obtains 24.968/22.956 dB versus 24.272/22.132 dB and wins on 16/24 scenarios on each dataset. The paired mean CIs include zero; consumed training costs differ (101.95 versus 83.99 GPU-hours), and the rules incorporate historical scenario knowledge. This control quantifies the comparison within the toolkit. The supplement additionally records cross-stage LLM decisions, including revising a degradation hypothesis while retaining Swin+SG (21.75→27.20 dB). These complement the control without establishing that an LLM is uniquely necessary.

#### Question 2

> The evaluation is mainly conducted under a closed-world synthetic degradation setting.

**Response.** We added real SIDD noise and out-of-library UHDM moire tests with the original operator vocabulary. For SIDD, five real paired supports guide diagnosis/model selection; gradient training uses DIV2K training images with synthesized degradation, and testing uses 1,120 patches from 35 other scenes. AutoIR R0 reaches 30.533 dB, versus 32.014 dB for X-Restormer. The released DFPIR checkpoint's recorded 23.291 dB uses random text conditioning and requires corrected official-interface evaluation. UHDM produces an empty pipeline and an out-of-library flag, with no AutoIR specialist. These tests extend the evidence to real data while retaining the stated library and R0 limits.

#### Question 3

> The robustness of the agentic pipeline across repeated runs is insufficiently evaluated.

**Response.** Completed run-to-run variance measurements are not available in the current records. In particular, the repeated-LLM-agent experiment has not been completed; support resampling of a rule controller would not establish LLM-agent stability. The 24 degradation scenarios and scene-bootstrap confidence intervals also do not measure repeated-agent variability. We will explicitly identify this as an unmeasured limitation rather than claim robustness from those results.

## Reviewer osqx

#### Question 1

> Please refer to the 'Weakness' part.

**Response.** We address all five referenced Weaknesses individually above: (1) UHDM out-of-library behavior and retained performance; (2) reusable offline engineering versus per-scenario automation; (3) DC metric definitions and restoration evidence; (4) adaptation-training versus deployment/end-to-end cost; and (5) real-data applicability and the five-pair requirement. There is no additional standalone question in this section.

## Reviewer nLXc

#### Question 3

> Table 5 reports 31.2% degradation-type accuracy and 27.1% severity accuracy, but the evaluation unit is unspecified. Without this definition, the DC results are difficult to interpret quantitatively.

**Response.** The unit is a ground-truth operator position. Across 24 chains containing 48 ground-truth steps, 15/48 position-aligned type matches give 31.25%; requiring both type and severity gives 13/48 = 27.08%. Thus the latter is joint type–severity accuracy, not severity-only accuracy or whole-chain accuracy. We will state the denominator and rename the metric. New fixed-R2 exp41 results separately report whole-chain exact match: 37.50% for type and 29.17% for joint type–severity.

#### Question 4

> Every evaluation degradation is generated by operators that already exist in the DC degradation library. The current benchmark does not test how AutoIR behaves when the observed degradation is absent from its operator library, which is directly relevant to the claimed ability to autonomously characterize newly encountered degradations.

**Response.** We tested real UHDM moire without adding a moire operator. The diagnostic workflow returns an empty pipeline, no alternatives and POOR/out-of-library status. No AutoIR moire specialist is trained. On the remaining 495 pairs, the generic blind model and forced JPEG fallback give 17.307/17.373 dB, versus 17.100 dB for the input. This shows one conservative response to a missing operator and the current recovery limitation. It does not establish a general unknown-detection rate; post-training CLR was not scheduled in this test.

#### Question 5

> Table 1 combines task-specific, all-in-one, zero-shot, few-shot, and agent-based methods, but the adaptation protocol for each baseline is not reported in sufficient detail. In particular, access to the five support pairs, scenario-specific retraining, synthetic training data, and optimization budget should be stated for every method.

**Response.** The original baselines use official releases with no additional per-scenario retraining or supervised fine-tuning on our support pairs. AutoIR uses five supports for diagnosis and checkpoint/round selection; gradient training uses DIV2K training-set clean images with synthetic LQ generated by the estimated chain. Each training run uses two epochs (about 15k steps), with three training rounds in the new controlled comparison. Exp41 averages 4.248 training GPU-hours per scenario. The separate DFPIR-ft control receives the same final DC and adaptation-training GPU-time budget; its prior pretraining is excluded. Original methods retain their native inference/test-time procedures, which should not be confused with AutoIR's additional scenario-specific training.

#### Question 6

> Table 2 attributes only 0.34 dB on DIV2K and 0.37 dB on LSDIR to RS, compared with substantially larger gains from fine-grained DC and CLR. Since autonomous restorer development is presented as one of the three central components of AutoIR, the current ablation indicates a relatively modest contribution from architecture specialization itself.

**Response.** The 0.34/0.37 dB values isolate the incremental benefit of the current RS module choices under the Table 2 protocol. RS also executes the training experiment within the full automated development loop; that role is not measured by the module-selection increment alone. The supplement shows a fixed-estimated-degradation case where Swin→Swin+SG+GCM improves 17.46→19.56 dB, while CLR adds 1.32/1.11 dB on average in Table 2. We will distinguish the framework contribution from the bounded architecture-design capability and retain the modest average RS result.

#### Question 7

> Several implementation details differ between the main paper and the supplement. The LLM configuration appears as DeepSeek-V4, DeepSeek-V4-Pro, Qwen 3.7 Max, and Qwen 3.7 Plus in different sections; Section 3.3 also alternates between  and  for architecture-related operations.

**Response.** We will standardize the model names and toolkit symbols. The recorded Qwen baseline ID is `qwen3.7-plus-2026-05-26`, so Max will be corrected to Plus. DeepSeek's exact API identifier still needs verification before resolving V4 versus V4-Pro consistently. `T_rd` denotes restorer design, and `T_ds` denotes degradation simulation; the architecture-related `T_ds` occurrence is a notation error. We do not treat the differently written model names as interchangeable without verifying the actual backend.
