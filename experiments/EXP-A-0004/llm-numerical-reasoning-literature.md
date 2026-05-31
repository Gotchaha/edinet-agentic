# LLM Numerical Reasoning in Long Contexts — Literature Review

**Date:** 2026-03-28
**Context:** Pre-EXP-A-0003 investigation into whether LLMs struggle with numerical data in long contexts, and whether tool-augmented approaches help with extraction vs. computation.

## Summary

The evidence strongly supports the claim that LLMs struggle to extract and use numbers from long, numerically dense contexts. This is not a single failure mode but a convergence of at least four well-documented problems: (1) positional retrieval degradation ("lost in the middle"), (2) fragility to numerical variation in reasoning, (3) poor accuracy on quantitative extraction from tables, and (4) magnitude/distractor sensitivity. Tool-augmented approaches show substantial improvements for computation but the evidence on whether they help with *extraction* (locating the right number) is more limited and mixed.

## Confidence Markers

| Claim | Confidence | Justification |
|-------|-----------|---------------|
| LLMs degrade on numerical extraction in long contexts | **High** | Multiple independent papers with quantitative results; consistent across model families |
| "Lost in the middle" applies to numerical data | **High** | Original Liu et al. 2023 finding is general (including key-value retrieval); follow-up GM-Extract paper confirms and quantifies |
| LLMs are fragile when numerical values change | **High** | GSM-Symbolic (Apple, ICLR 2025) provides direct causal evidence with controlled experiments |
| Tool augmentation helps with numerical *computation* | **High** | Raven (2024) shows 35% average improvement on financial math via calculator/SQL tools |
| Tool augmentation helps with numerical *extraction* (locating the right number) | **Medium** | RAG/retrieval-based approaches improve qualitative accuracy more than quantitative; AIE framework reaches only ~64-75% on strict numerical extraction; evidence is suggestive but less conclusive |
| These findings are directly relevant to EDINET-Bench failure modes (M2, M3, M7) | **Medium-High** | The M2 evidence-drift and M3 magnitude-insensitivity patterns in EXP-D-0001 are consistent with the documented numerical reasoning fragilities, but direct causal attribution requires further experimentation |

## Detailed Analysis

### 1. Do LLMs Make Errors Extracting Specific Numbers from Long Documents?

**Yes, substantially.** Several lines of evidence converge:

**Financial table extraction.** Balsiger et al. (2024) tested ChatGPT-4 and BARD on extracting numerical data from 46 Swiss annual reports. When magnitude errors are counted as incorrect, accuracy drops to 64-73%. ChatGPT-4 achieved 90% on balance sheet questions but only 53% on income statement questions, where cross-industry formatting variation is higher. The combined error rate for numerical questions was approximately 25%.

**Hybrid long document extraction.** The AIE framework (Yue et al., ICASSP 2025) tested extraction from financial documents averaging ~59,000 tokens. At the strictest precision level (RETA 1%, meaning the extracted number must be within 1% of ground truth), accuracy was only **63.89%**. Even at a relaxed 10% tolerance, accuracy was only 74.51%.

**Qualitative vs. quantitative gap.** Research consistently shows LLMs perform better on qualitative content than quantitative extraction. One study reports qualitative average accuracy of 66% vs. quantitative average accuracy of 55%. The Kim et al. (2024) financial statement analysis paper found GPT-4o calculation accuracy at only 44.7% while conceptual accuracy reached 79.5%. **[Note: this paper (arxiv 2407.17866) has been withdrawn due to "inconsistencies in the data and analyses"; treat these figures with caution.]**

### 2. Is There a "Lost in the Middle" Problem for Numerical Data?

**Yes, and it may be worse for numerical data than for textual facts.**

**Original finding.** Liu et al. (2023, TACL 2024) demonstrated the U-shaped performance curve: LLMs retrieve information best from the beginning and end of long contexts, with significant degradation in the middle. This was shown on both multi-document QA and key-value retrieval tasks -- the latter being essentially a numerical/symbolic extraction task.

**GM-Extract follow-up.** The GM-Extract study (2025) provides critical mechanistic insight. It distinguishes two failure modes: models often succeed at *semantic* retrieval (identifying "what" the answer is) but fail at *spatial* awareness ("where" in the document the information is located). As context density increases, spatial awareness degrades significantly. Long-context models (LLaMA-3.1-8B, 128K context) achieved 96-99% accuracy on 4K-token tasks but showed "significant degradation" at 12K tokens.

**Relevance to numerical data specifically.** Financial documents are particularly susceptible because they contain many structurally similar numbers (revenue figures, line items, ratios) that serve as distractors. The distractor effect is documented: as the number of numerical distractors increases, performance degrades, and simply using a larger context window is not the optimal solution when the noise-to-signal ratio is high.

### 3. Papers on LLM Numerical Reasoning Limitations with Financial/Tabular Data

Several key papers directly address this:

**GSM-Symbolic (Mirzadeh et al., Apple, ICLR 2025).** This is the strongest evidence for *numerical reasoning fragility*. When only numerical values in math word problems are changed (keeping the problem structure identical), all models show performance decline. Adding a single irrelevant but plausible numerical clause causes drops of **up to 65%**. The authors conclude that "current LLMs cannot perform genuine logical reasoning; they replicate reasoning steps from their training data."

**"How Well Do LLMs Reason over Tabular Data, Really?" (2025, ACL TRL Workshop).** This paper reveals that standard evaluation methods dramatically overestimate LLM tabular reasoning. When using open-form evaluation instead of multiple choice:
- Average calculations: **30% accuracy gap** between open-form and multiple-choice
- Subtraction tasks: **60% accuracy gap**
- Correlation calculations: only **8.65% accuracy**
- Entity lookup: 29-68% depending on model
- Aggregation (averaging): 11-32%

Models also poorly handle real-world data issues: duplicate entities caused accuracy drops of 50%+, and models acknowledged duplicates less than 27% of the time.

**"Evaluating Robustness of LLMs to Numerical Variations in Mathematical Reasoning" (Yang et al., ACL Insights 2025).** Studies the effect of varying numerical values in math word problems on LLM performance.

**"Cutting Through the Noise: Boosting LLM Performance on Math Word Problems" (Anantheswaran et al., ICLR 2025 Workshop).** LLMs are susceptible to distraction by numerical noise, showing approximately **26% average relative performance drop** when problems contain extraneous numerical information. Fine-tuning on adversarial instances recovers only ~8%. Models consistently incorporate irrelevant numbers into their calculations.

**Financial-domain papers:**
- Kim et al. (2024) found GPT-4 achieves 60.4% directional earnings prediction accuracy using only numerical financial statements -- better than random but far from reliable. **[Paper withdrawn; see note in Section 1.]**
- Hallucination rates for unassisted LLMs on financial data can reach 50%.
- Daloopa's analysis confirms models "correctly pull historical returns but miscalculate standard deviation" -- they can sometimes extract but fail to compute correctly.

### 4. Do Tool-Augmented Approaches Help with Numerical Extraction?

**For computation: clearly yes. For extraction: partially, with important caveats.**

**Computation gains (strong evidence).** The Raven model (Theuma & Shareghi, EACL 2024) equipped LLaMA-2 13B with a Python calculator and SQL engine, achieving a **35.2% average improvement** across four financial datasets. On multi-hop numerical reasoning, accuracy went from ~2% (base model) to **56.7%** with tools. This outperformed GPT-3.5 by 9.2% despite being a much smaller model.

**Extraction gains (weaker evidence).** The AIE framework uses a retrieval pipeline (segmentation, retrieval, summarization, extraction) that helps locate relevant segments, but numerical extraction accuracy still topped out at ~64-75%. RAG approaches generally boost qualitative retrieval more than quantitative retrieval (66% vs. 55% accuracy). The critical bottleneck is that the tool must first *find* the right number before it can compute with it.

**Practical implication for EDINET-Bench.** The EXP-D-0001 results show that the dominant failure modes (M7 conservatism at 58%, M2 evidence drift at 80% as primary/secondary, M3 magnitude insensitivity at 15%) are consistent with the literature findings. M2 (evidence drift) maps directly to the "irrelevant number incorporation" phenomenon documented in GSM-Symbolic and the distractor sensitivity literature. M3 (magnitude insensitivity) aligns with the tabular reasoning finding that models struggle with aggregation and comparison. A tool-augmented approach that provides *structured numerical extraction* (not just a calculator) could potentially address both by ensuring the model reasons over verified numbers rather than self-extracted ones.

## Sources and References

### Core Papers

- [GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models](https://arxiv.org/abs/2410.05229) — Mirzadeh et al., Apple, ICLR 2025. Up to 65% performance drop from irrelevant numerical clauses.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Liu et al., TACL 2024. U-shaped retrieval curve, middle-position degradation.
- [What Works for 'Lost-in-the-Middle' in LLMs? A Study on GM-Extract and Mitigations](https://arxiv.org/abs/2511.13900) — Gupte et al., 2025 (preprint). Spatial vs. semantic retrieval distinction; mitigation strategies.
- [How Well Do LLMs Reason over Tabular Data, Really?](https://arxiv.org/abs/2505.07453) — 2025, ACL TRL Workshop. 8.65% accuracy on correlation tasks; massive gaps in open-form evaluation.
- [Evaluating Robustness of LLMs to Numerical Variations in Mathematical Reasoning](https://aclanthology.org/2025.insights-1.16/) — Yang et al., ACL Insights 2025.

### Financial Domain

- [Assessing Large Language Models Used for Extracting Table Information from Annual Financial Reports](https://www.mdpi.com/2073-431X/13/10/257) — Balsiger et al., MDPI Computers, 2024. 64-73% accuracy on financial table extraction.
- [Financial Statement Analysis with Large Language Models](https://arxiv.org/abs/2407.17866) — Kim et al., 2024. **WITHDRAWN** — data/analysis inconsistencies identified by co-author. Previously cited: GPT-4 achieves 60.4% on earnings prediction.
- [Can Large Language Model Analyze Financial Statements Well?](https://aclanthology.org/2025.finnlp-1.19.pdf) — FinNLP 2025.
- [Can Large Language Model Analyze Financial Statements Well? (Daloopa summary)](https://daloopa.com/blog/analyst-best-practices/can-large-language-model-analyze-financial-statements-well) — Hallucination rates up to 50%; computation accuracy 44.7%.

### Tool Augmentation

- [Equipping Language Models with Tool Use Capability for Tabular Data Analysis in Finance](https://arxiv.org/html/2401.15328v1) — Theuma & Shareghi, EACL 2024. Raven: 35.2% improvement with calculator+SQL tools.
- [Extract Information from Hybrid Long Documents Leveraging LLMs: A Framework and Dataset](https://arxiv.org/abs/2412.20072) — Yue et al., ICASSP 2025. AIE framework: 63.89% accuracy at RETA 1%.
- [Integrating External Tools with Large Language Models to Improve Accuracy](https://arxiv.org/pdf/2507.08034) — 2025.

### Long Context and Retrieval

- [Cutting Through the Noise: Boosting LLM Performance on Math Word Problems](https://arxiv.org/abs/2406.15444) — Anantheswaran et al., ICLR 2025 Workshop. ~26% performance drop from numerical noise; ~8% recovery via adversarial fine-tuning.
- [LLMs Cannot Spot Math Errors, Even When Allowed to Peek into the Solution](https://aclanthology.org/2025.emnlp-main.553.pdf) — EMNLP 2025.

## Caveats and Limitations

1. **Publication bias.** Papers studying LLM limitations are more likely to report negative results. The most capable frontier models (GPT-5, Claude Opus 4.5, Gemini 2.5 Pro) may perform substantially better than the models tested in most of these papers (GPT-3.5, GPT-4, LLaMA-2 13B, Qwen 7B). The GSM-Symbolic results were shown for "all state-of-the-art models" at the time of publication (mid-2024), but reasoning models have improved since.

2. **Task specificity.** EDINET-Bench fraud detection involves Japanese financial statements (XBRL-parsed) with specific formatting. The papers reviewed primarily study English-language financial documents or synthetic math problems. Transfer of findings to the EDINET domain is plausible but not directly validated.

3. **Extraction vs. reasoning conflation.** Many studies conflate extraction errors (finding the wrong number) with reasoning errors (computing incorrectly with the right number). The EXP-D-0001 analysis suggests M2 (evidence drift) may involve both — the model may extract a real number but use it to support an unrelated conclusion, or it may be citing a number it hallucinated. Disentangling these would require tracing each cited number back to the source data.

4. **Tool augmentation evidence gap.** The strongest tool-augmentation results (Raven) are for *computation* tasks (multi-hop arithmetic). Whether tools help with the *extraction bottleneck* — reliably locating the right number among thousands of similar numbers in a long financial document — is much less studied. The AIE framework is the closest evidence, but it still shows only ~64% accuracy at strict precision, suggesting the extraction problem is not fully solved by retrieval pipelines.

5. **Rapidly evolving field.** These findings reflect the literature through early 2026. Model capabilities are improving rapidly; some documented limitations may be partially or fully addressed in newer model versions.
