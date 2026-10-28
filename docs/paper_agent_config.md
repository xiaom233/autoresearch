\subsection{LLM Agent Configuration}

\subsubsection{Model and Infrastructure}
The autonomous research system is powered by \textbf{Claude Code} (Anthropic's CLI agent framework) with \textbf{DeepSeek-V4-Pro} as the backend LLM. The agent operates within a Linux environment with 8 NVIDIA GPUs, accessing a full filesystem including Python scripts, degradation libraries, and training pipelines.

\subsubsection{Agent Architecture: Single Orchestrator + Parallel Sub-Agents}

The system follows a \textbf{hierarchical multi-agent} design:

\begin{itemize}
    \item \textbf{Main Agent (Orchestrator)}: Responsible for task decomposition, sub-agent allocation, result aggregation, and Phase-level decision making (e.g., triggering reflection rounds).
    \item \textbf{Sub-Agents (Workers)}: Execute independent blind identification or reflection tasks on assigned subsets of challenges. Each sub-agent operates in an \textbf{isolated context} to prevent ground-truth leakage.
\end{itemize}

\textbf{Parallelism}: For 24 challenges, 8 sub-agents are launched concurrently, each processing 3 challenges. Reflection follows the same parallel pattern (Table~\ref{tab:agent_allocation}).

\begin{table}[h]
\centering
\caption{Sub-agent allocation by challenge count}
\label{tab:agent_allocation}
\begin{tabular}{c|c|c}
\toprule
Challenges & Sub-agents & Per Agent \\
\midrule
1--4  & 1 & 1--4 \\
5--8  & 4 & 2 \\
9--16 & 8 & 1--2 \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Context Isolation: Worktree Sandbox}

A critical design choice is \textbf{git worktree isolation} for sub-agents. When the main agent session has discussed ground-truth-related information (degradation types, per-challenge accuracy, etc.), sub-agents are launched with \texttt{isolation="worktree"}, creating independent git worktrees that prevent access to the main conversation context. This ensures sub-agents can only access project files and explicit prompt instructions, eliminating indirect GT leakage through conversation history.

\subsubsection{Prompt Design: Strictly Neutral}

All agent prompts enforce \textbf{degradation-agnostic} language:

\begin{itemize}
    \item Agent names must be neutral (e.g., ``Blind ID batch A''), never containing degradation step counts or type hints.
    \item Prompts forbid mentioning degradation count (``single-deg'', ``double-deg''), degradation type, or severity.
    \item Sub-agents must independently infer degradation steps, types, and severity purely from signal analysis of the target image.
    \item Reflection prompts use careful phrasing: ``re-examine signal analysis'' instead of ``fix incorrect predictions'' to avoid implying known GT.
    \item All prohibited file paths (GT mappings, degradation\_gt, training logs) are explicitly listed in each sub-agent prompt.
\end{itemize}

\subsubsection{Toolkit Organization}

The agent accesses a structured toolkit of scripts organized as a \textbf{Skill}:

\begin{itemize}
    \item \texttt{run\_full\_analysis.py}: Integrated signal report (compression, blur, noise, global degradation detection) from a single invocation.
    \item \texttt{test\_candidate.py}: Applies a candidate degradation pipeline to the clean image, computes PSNR/signal metrics vs.\ the target, and automatically updates \texttt{reflection.json}.
    \item \texttt{peeled\_noise\_check.py}: After stripping deterministic degradations, runs §B 6-step statistical noise diagnosis on the residual.
\end{itemize}

The full blind identification protocol is defined in \texttt{SKILL.md}, which sub-agents must re-read at the start of each challenge to prevent protocol drift during long-running sessions.

\subsubsection{Iteration Budget and Termination}

\textbf{Blind Identification}: Hard cap of $\leq 25$ PSNR tests per challenge, partitioned into three rounds:
\begin{enumerate}
    \item Round 1: Signal-driven coverage ($\leq 12$ tests) — test 3--4 candidates per degradation category.
    \item Round 2: Binary comparison ($\leq 8$ tests) — top-2 vs.\ cross-family alternatives.
    \item Round 3: Severity fine-tuning ($\leq 5$ tests) — $\pm 1$ severity sweep + masking inference for missed compression.
\end{enumerate}

\textbf{Reflection}: $\leq 15$ new candidates per round, with termination after 2 consecutive rounds without PSNR improvement.

\subsubsection{Randomness Control}

The system uses \textbf{deterministic degradation simulation} wherever possible. For noise-type degradations where PSNR is unreliable (seed mismatch causes pixel-level distortion), verification switches to statistical signal matching (6-step §B check) instead of PSNR. The blind challenge generation uses a fixed seed (\texttt{--seed 42}) for reproducibility.

\subsubsection{Memory and Long-Term State}

The agent maintains persistent state through:

\begin{itemize}
    \item \textbf{Protocol Memory}: \texttt{SKILL.md} — the canonical blind identification protocol, updated based on experimental findings (e.g., exp37 verdict per-degradation fix, exp34 noise PSNR deprecation).
    \item \textbf{Strategy Memory}: \texttt{finetune\_strategy.md} — architecture-degradation coupling rules derived from 400+ ablation experiments.
    \item \textbf{Reflection Memory}: \texttt{REFLECTION\_MECHANISM.md} — accumulated knowledge about when and how to trigger correction rounds.
    \item \textbf{Per-Challenge State}: \texttt{predicted\_params.json}, \texttt{reflection.json}, \texttt{thinking\_process.json} — full audit trail from signal analysis to final decision.
    \item \textbf{Cross-Experiment Memory}: Auto-memory system (\texttt{/memory/}) stores user preferences, feedback corrections, and project-level decisions across sessions.
\end{itemize}

\subsubsection{Configuration Summary}

\begin{table}[h]
\centering
\caption{LLM Agent configuration}
\label{tab:agent_config}
\begin{tabular}{l|l}
\toprule
Parameter & Value \\
\midrule
LLM Backend & DeepSeek-V4-Pro (1M context) \\
Agent Framework & Claude Code (Anthropic CLI) \\
Multi-Agent & Yes: 1 orchestrator + 8 parallel sub-agents \\
Context Isolation & Git worktree (prevents GT leakage) \\
Prompt Style & Degradation-agnostic, neutral naming \\
Blind ID Budget & $\leq 25$ PSNR tests/challenge, 3-round hard cap \\
Reflection Budget & $\leq 15$ candidates/round, 2-round no-improvement termination \\
Randomness & Fixed seed (42), deterministic degradation simulation \\
Memory & File-based: SKILL.md, finetune\_strategy.md, REFLECTION\_MECHANISM.md, auto-memory \\
\bottomrule
\end{tabular}
\end{table}
