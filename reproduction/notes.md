## From paper

**url:** http://arxiv.org/abs/2506.08762

EDINET --`edinet2dataset` --> EDINET-Corpus: ~40,000 documents (4,000 #companies x 10 #years)

- Accounting fraud detection:
  668/6,712 amendments --> fraudulent samples set.
  700/{#all non-frudulent companies}, one annual report for each --> non-frudulent sample set.
  Final: 534 frudulent and 555 non-frudulent samples. (for `test` split: 122 and 102, respectively)

- Earnings forecasting:
  1,000 companies --> pairs of consecutive annual reports spanning two fiscal years for each --> one pair for each company --> 1,000 instances (current year) in total (for `test` split: 451).

## EDA

**notebook:** `reproduction/notebooks/EDA.ipynb`
**subset:** `fraud_detection`
**split:** `test`

**Observations:**

1. Within the 12 columns: `edinet_code` is the company identifier while `doc_id` and `file_path` are unique keys for each sample; all fields except for the `label` (int64) are in string format, fields as input (`summary`, `bs`, `cf`, `pl`, `text`) are all not empty and are json parseable, as expected.
2. What `meta` provides?
Company Name, EDINET Code (the same as the `edinet_code` field), Fund Code (often being empty or '-'), Securities Code, Filing Document Type (being Annual Secuities Report), Accounting Standard, Fiscal Year Start/End Date, Consolidated Financial Statements Flag, Amendment Flag, Disclosure Item Correction Flag, XBRL Correction Flag. The last four are boolean field but stored in string.
So this field describes the sample, i.e. this annual report.
*I initially thought the last four flag fields would be related to the label, but it turned out that this is not the case, there are only two samples with Amendment Flag & Disclosure Item Correction Flag being true, and they both have `label` to be `1` with `explanation` filed not empty.*

3. Input columns:
   1. `summary`. Structured as expected, describes the finicial statement including profitability, cash flows, operational context, etc. in company-level for multiple fiscal years (prior 3 year to current year).
   *I noticed the '-', the missing indicator, this is one of the issues mentioned in the paper. Also the number of terms recorded is different in general, the same as the metric used (IFRS vs. non-IFRS), this may be an issue?*
   2. `bs`. Account-level, also for multiple years but usually for only prior and current year, sometimes with additional prior years. The schema is not uniforma across samples, union number is 83, and vary very differently across samples.
   3. `cf`. Same situation (multiple-year, non-uniform schema) as above, but in cash flow there is a core-vs-tail pattern that we have a stable core of high-coverage items like operating, investing, etc. with a long tail of company-specific items.
   *I found that `cf` is relatively more compact and consistent than `bs`*
   4. `pl`. Profit-and-loss, not schema-uniform as expected but its intersection is especially small -- only 1 key (pre-tax profit) appears in all samples. Same core-vs-tail pattern as in `cf`.
   *I noticed the reason of such non-uniformity in `pl` is that there are semantic substitution (for example, aggregate SG&A vs. separated selling/admin components).*
   5. `text`. Section-structured document field, not a numeric statement one.


## Model Abstraction Layer

**file:** `external/EDINET-Bench/src/edinet_bench/model.py`

**Observations:**

1. Implementation.
   A base model class `Model` and multiple subclasses for different providers. Minimum methods realized: `__init__` and `get_completion()` (with retry logic attached). It also provides a `MODEL_TABLE` for convenient instantiation call.
2. *Leaks.*
   1. Subclasses define their own `__init__` but do not call `super()` to use the base model method.
   2. `GenerationConfig` creates the default object once at function definition time and also there is special case where one model ignores these settings (`o4-mini`).
   *From further investigation, I know that for OpenAI resoning models (most o-series models), `temperature` is not supported (and this was once an issue: [#2468](https://github.com/browser-use/browser-use/pull/2468)), but for maximum output length, according to [official](https://help.openai.com/en/articles/5072518-controlling-the-length-of-openai-model-responses), it seems we should use `max_completion_tokens` (alias of `max_tokens`).*

## Core Pipeline

**file:** `external/EDINET-Bench/src/edinet_bench/predict.py`

**Observations:**

1. Implementation.
   The script reads in multiple args and use them first initialize the dataset, model and prompt. Then, it concurrently runs `process_example()` using `ThreadPoolExecutor`. The `process_example()` will first construct the final prompt for this example as the base prompt (task-specific), plus the "\n", then the features (called `sheet`) identified in the example, skipping the missing ones. Then it calls `predict()`. The prediction function will call the model to get and log the response of the final prompt, then extract json patterns in it to get three main fields (`prob`, `prediction` and `resoning`), these could be `None` if the json data itself is `None` or such key is missing in it. The three main fields will be used by the `process_example()` to form the `Result` (a dataclass defined for storing the results) instance. Finally these results will be written into a jsonl file.
   *Concurrency: I was initially not so familiar with the mechanics of it in Python. Then I learnt and got the idea that, the context manager will create a pool of (specific number of) workers, then the main thread will make each to `submit()` the work, here is the execution of `process_example()`, notice that this will immediately return a `Future` handle for this task, and in the code all these handles will be put into a list (whether they are finished or not), and the `as_completed()` will provide the completed handle so that we can get result through `result()`, and finally the context manager will shutdown all threads it created.*
2. *Flaws? (might just be “intentional” simplifications)*
   1. As I learnt of the concurrency mechanics, there is chance that one `future` raising an exception, in this case, it will re-raise at `future.result()` and the code does not do any caught so it will crash the script.
   2. `--temperature` is parsed, but never used when calling model. In model abstraction layer, `get_completion()` defaults to a config with `temperature=0.0`, so unless the caller passes a non-default config, `--temperature` is effectively ignored.

## Baseline design

**files:** [`external/EDINET-Bench/src/edinet_bench/naive_prediction.py`, `external/EDINET-Bench/src/edinet_bench/logistic.py`]

**Observations:**
1. Rationale.
   1. The `naive_prediction` is for earnings forecasting task and built during the construction of EDINET-Bench, using the Last Value strategy. So the code is to only extract this field from the HF dataset.
   *I noticed that from the naming convention and the EDA, this naive prediction info. is purely from the `summary` field of the dataset.*
   2. The `logistic` reads task name from args and prepare the the dataset: load the specific part from HF then construct the data list where each sample is a dict with `summary` field expanded at the top level and `label` field. Then it preprocesses this list to mainly expand further the content of `summary`, i.e. make "{key_year}-val" pairs (as we know from EDA that each key in `summary` contains values for multiple years), then creates the dataframe from the rows (*this makes the union of columns of all samples being the final column set, default behavior of `pd.DataFrame()`, filling `NaN` for missing keys*). Then it uses training mean to inpute, drops constant columns (*no info.*), reindexes the test set columns using training set's (*this hints that test already has all train columns or it will induce `NaN` problem since this is the final step, no imputation afterwards.*) and standardizes the features. Then, model fitting, prediction, evaluation and result-saving.
   *I am curious about how will the performace be using tree-based models, like XGBoost, under the same setting.* **TODO**

## Prompt engineering

**file:** [`external/EDINET-Bench/prompt/earnings_forecast.yaml`, `external/EDINET-Bench/prompt/fraud_detection.yaml`]

I think these prompts are very "baseline-like", the structure is clear and simple, and it requires a structured output using a markdown code fense (which aligns with the following JSON extraction pipeline). In prompt for `fraud detection` there is a line stating that:

> ... numerical values are consistent and correct from a calculation perspective. Therefore, please focus your analysis on non-numerical inconsistencies or logical red flags that could suggest fraud.

*It is like a "attention router" which may lead to a better result?*
And following this line, I noticed that there is no decision rubic or evidence checklist for both prompts. *Adding this may need some expertise in Finance, and if I want to make it possible extending prompt engineering, to the agent part, then trivially I need the model to have (or ba able to retrieve) the knowledge needed to make the decision, then search for such info. in the current sample annual report ... I will take a try*  **TODO**

## Utils - JSON extractor

**file:** `external/EDINET-Bench/src/edinet_bench/utils.py`
**function:** `extract_json_between_markers(llm_output: str)`

The function directly searches for the fenced code block in the llm output, and uses a fallback of searching JSON-like content (*but it would fail for nested structure*). Then for each pattern found, it tries to do JSON parse, with a fallback of removing invalid control characters then parsing. It will return `None` under the case that there is no required JSON pattern in the llm output or the JSON-like pattern found is broken and cannot be parsed.

## EXP-R-0002/EXP-R-0003 Observations

**helper code**: `reproduction/notebooks/Analyze.ipynb`

*(these are just my preliminary personal observations/ideas, which may need to be validated or better understood through experiments and/or a review of the literature.)*

### General

Let's first define the "aggressive/conservative" as the positive-call tendency, which means a model is more aggressive if it assigns the positive label to a larger fraction of instances, i.e., it has a higher predicted positive rate $\hat\pi_+$, where $\hat\pi_+$ is defined as $\frac{TP+FP}{N}$.Conversely, it is more conservative if $\hat\pi_+$ is smaller (it “rarely calls positive”).
Then the result
```
claude-haiku-4-5-20251001 @ EXP-R-0002: pi_hat_plus = 48/50 = 0.96
o4-mini-2025-04-16 @ EXP-R-0002: pi_hat_plus = 3/50 = 0.06
claude-haiku-4-5-20251001 @ EXP-R-0003: pi_hat_plus = 18/50 = 0.36
o4-mini-2025-04-16 @ EXP-R-0003: pi_hat_plus = 1/50 = 0.02
```
shows that:

1. Model "personality"?

`haiku` is generally way more aggressive than `o4-mini`. This is very significant in EXP-R-0002, where it assigned "positive" to nearly all the samples while `o4-mini` did the other way. 

If this reflects the "personality" of a model, I wonder what is the cause, maybe it is related to the pretraining data or the specific pretraining strategy? This is interesting because this is not the case in which one model is better than the other but it is that the two model fail in opposite ways.

2. The effect of new information ("Text").

With the inclusion of the new input feature, the "Text" field, both models shift their decision behavior toward the negative class, as evidenced by a reduction in the predicted-positive rate. This shift is substantially large for `Haiku` ($\Delta\hat\pi_+ = -0.60$) than for `o4-mini` ($\Delta\hat\pi_+ = -0.04$), indicating that "Text" changes `Haiku`'s operating point much more strongly. Importantly, this shift does not translate into better performance: for `Haiku`, the dominant error mode changes from being FP–heavy (FP=21, FN=0) to FN–heavy (FP=9, FN=18). `o4-mini` is affected mainly by becoming even more conservative (FP decreases from 1 to 0), while its overall behavior remains consistent throughout—high specificity with very low recall (FN-heavy).

### Sample Details

1. Sample within each experiment that has the most (absolute) difference "prob" across the two models.

```
===== EXP-R-0002 =====
models: claude-haiku-4-5-20251001 vs o4-mini-2025-04-16
max_abs_prob_diff = 0.67
-- case 1: doc_id=S1009C5F, abs_diff=0.67 (claude-haiku-4-5-20251001: 0.72, o4-mini-2025-04-16: 0.05)
claude-haiku-4-5-20251001 row:
{
  "doc_id": "S1009C5F",
  "edinet_code": "E02030",
  "label": 0,
  "prediction": 1,
  "prob": 0.72,
  "reasoning": "Analysis reveals several significant red flags warranting fraud investigation: (1) DRAMATIC REVENUE DECLINE: Sales dropped 19.9% from ¥32.5B to ¥26.0B, yet SG&A expenses increased 9.8% (¥7.38B to ¥8.11B), suggesting management failed to adjust cost structure proportionally - unusual for legitimate operational contraction. (2) PROFIT COLLAPSE DISPROPORTIONATE TO REVENUE: Operating profit plummeted 73.9% (¥6.14B to ¥1.60B) while gross margin deteriorated from 41.6% to 37.3%, indicating either severe pricing pressure, cost inflation, or potential inventory/revenue recognition issues. (3) CASH FLOW DETERIORATION: Operating cash flow crashed 77.1% (¥9.09B to ¥2.08B) despite only 19.9% revenue decline - this 3.9x multiplier is highly suspicious and suggests potential working capital manipulation or earnings quality issues. (4) WORKING CAPITAL ANOMALIES: Accounts receivable decreased ¥257M while sales fell ¥6.45B (implying improved collection), yet operating cash flow collapsed - inconsistent with normal business dynamics. (5) DIVIDEND INCREASE AMID CRISIS: Dividend payments more than doubled (¥585M to ¥1.29B) while earnings halved and cash flow collapsed - financially illogical and suggests potential cash preservation concerns or earnings manipulation to justify distributions. (6) ASSET IMPAIRMENT SIGNALS: Total assets declined 10.7% while fixed assets increased, combined with dramatic cash depletion (¥3.15B reduction) and increased investment securities, suggesting possible asset shuffling. (7) EQUITY DETERIORATION: Shareholders' equity declined ¥2.13B despite reported profits, with non-controlling interests dropping 22.5% - potential indicator of hidden losses or accounting adjustments. (8) FOREX ANOMALY: Prior year showed ¥196M forex gains; current year shows '-' (missing data) coinciding with ¥914M negative forex adjustment - potential concealment of significant losses. (9) TAX RATE SPIKE: Effective tax rate jumped from 6.5% to 40.1% despite lower profits - unusual and potentially indicative of prior-year tax benefits or current-year adjustments suggesting earnings quality issues.",
  "input_tokens": 4471,
  "output_tokens": 737,
  "elapsed_sec": 9.27
}
o4-mini-2025-04-16 row:
{
  "doc_id": "S1009C5F",
  "edinet_code": "E02030",
  "label": 0,
  "prediction": 0,
  "prob": 0.05,
  "reasoning": "A review of the narrative and qualitative items reveals no obvious indicators of fraud. The year-on-year declines in sales, profit margins and cash flow are explained by a downturn in operations rather than aggressive revenue recognition or one-off reversals. Cash flow from operations remains positive and broadly in line with the fall in reported profits, indicating earnings quality. The large dividend payment in the current year (¥1.286 billion) exceeds that year’s net income (¥0.524 billion), but this is transparently funded from retained earnings and does not appear to involve any off-balance-sheet financing or misleading reserves. Foreign-currency translation losses and missing (\"－\") items are consistent with usual FX translation accounting and immaterial line-item disclosures. All balances reconcile, CPA verification is noted, and there are no unexplained or mismatched line items, no evidence of channel-stuffing, nor any hidden liabilities. Accordingly, there is little objective reason to suspect fraudulent reporting.",
  "input_tokens": 4241,
  "output_tokens": 1321,
  "elapsed_sec": 10.81
}
-- case 2: doc_id=S100D1XI, abs_diff=0.67 (claude-haiku-4-5-20251001: 0.72, o4-mini-2025-04-16: 0.05)
claude-haiku-4-5-20251001 row:
{
  "doc_id": "S100D1XI",
  "edinet_code": "E00897",
  "label": 1,
  "prediction": 1,
  "prob": 0.72,
  "reasoning": "Analysis reveals several significant red flags suggesting potential fraudulent activity or severe operational distress:\n\n1. **Dramatic Cash Flow Deterioration**: Operating cash flow collapsed from +10.75B (Prior1Year) to -629M (CurrentYear) - a swing of 11.4B despite only modest revenue decline (0.7%). This is highly unusual and suggests either aggressive accounting practices or operational crisis.\n\n2. **Inventory and Receivables Manipulation**: Accounts receivable increased 10.4% (32.6B to 36.0B) while sales increased only 0.7%, indicating potential channel stuffing or fictitious sales. Simultaneously, inventory increased 26.3% (13.1B to 16.1B) despite flat sales - classic signs of demand fabrication.\n\n3. **Allowance for Doubtful Accounts Reversal**: Allowance decreased from 2.1B to 1.5B (28% reduction) while receivables increased, suggesting aggressive write-off reversals to inflate earnings. The cash flow statement shows -762M adjustment for this, contradicting the balance sheet improvement.\n\n4. **Profitability Collapse with Asset Growth**: Despite total assets increasing 3.3% and equity increasing 3.7%, operating profit fell 31.3% and net income fell 32.8%. This inverse relationship is suspicious.\n\n5. **Gross Margin Compression**: Gross margin declined from 32.3% to 28.3% (500 bps) while SG&A decreased only 6.7%, suggesting either pricing pressure, cost inflation, or inventory write-downs being masked.\n\n6. **Tax Rate Anomaly**: Effective tax rate dropped dramatically from 29.8% to 25.5% despite lower pre-tax income, unusual unless significant tax benefits were recognized.\n\n7. **Special Items Elimination**: Prior year included 113M investment securities gains and 29M losses; current year shows neither, suggesting potential earnings management through selective recognition.\n\n8. **Equity Ratio Improvement Despite Deterioration**: Self-equity ratio improved to 59.6% while profitability metrics (ROE) collapsed to 3.3%, indicating potential asset revaluation or liability reduction rather than operational improvement.\n\n9. **Working Capital Deterioration**: Despite operational cash flow collapse, the company increased short-term borrowing by 307M and reduced long-term debt, suggesting liquidity stress being masked by refinancing.\n\n10. **Missing Data**: Diluted EPS data marked as '-' for all periods is unusual for a listed company and may indicate data suppression.",
  "input_tokens": 4668,
  "output_tokens": 721,
  "elapsed_sec": 9.21
}
o4-mini-2025-04-16 row:
{
  "doc_id": "S100D1XI",
  "edinet_code": "E00897",
  "label": 1,
  "prediction": 0,
  "prob": 0.05,
  "reasoning": "Overall, the financial statements are internally consistent, reconciled by a CPA, and show no obvious non-numeric red flags of manipulation. The balance sheet totals tie, depreciation and construction-in-progress movements align, and there are no unexplained rounding anomalies. Although operating cash flow turned negative while net income remained positive—largely driven by higher receivables and inventory and a reversal of doubtful-account reserves—this appears linked to working-capital timing rather than fabricated earnings. Translation adjustments and one-off reclassifications (e.g. construction-in-progress) are explained by asset transfers. Missing diluted EPS figures and zero buyback entries reflect absence of such items, not concealment. In light of the lack of illogical disclosures or contradictory notes, the likelihood of fraud is very low.",
  "input_tokens": 4433,
  "output_tokens": 1547,
  "elapsed_sec": 12.47
}
-- case 3: doc_id=S100G8YC, abs_diff=0.67 (claude-haiku-4-5-20251001: 0.72, o4-mini-2025-04-16: 0.05)
claude-haiku-4-5-20251001 row:
{
  "doc_id": "S100G8YC",
  "edinet_code": "E01544",
  "label": 0,
  "prediction": 1,
  "prob": 0.72,
  "reasoning": "Analysis reveals several significant red flags warranting fraud investigation: (1) DRAMATIC CASH FLOW ANOMALY: Operating cash flow surged 383% YoY from 1.07B to 5.19B yen, the largest jump in the 5-year period, while revenue grew only 5.7%. This extreme disconnect is highly suspicious. (2) ACCOUNTS RECEIVABLE MANIPULATION: The cash flow statement shows a 3.35B yen positive adjustment from 'sales receivables decrease,' yet the balance sheet shows receivables actually DECREASED from 13.64B to 10.29B yen (25% drop). While this technically reconciles, the magnitude is extraordinary and suggests potential channel stuffing, side agreements, or aggressive collection tactics that may not be sustainable. (3) COMPREHENSIVE INCOME ANOMALY: Comprehensive income dropped 10% YoY (1.21B to 1.09B) despite net income rising 13%, indicating deteriorating unrealized gains/losses in securities (other comprehensive income fell from 868M to 761M). This divergence is unusual. (4) INVESTMENT SECURITIES INCONSISTENCY: The company suddenly acquired 300.67M yen in securities (Prior1Year shows '-'), while simultaneously reducing investment securities holdings by 443M yen. The timing and nature of these transactions lack clear business rationale. (5) CASH POSITION SPIKE: Cash increased 120% (3.4B to 7.5B) despite increased capex, share buybacks (+185M), and dividends (+63M). Combined with the operating cash flow spike, this suggests potential one-time events or accounting adjustments not fully explained. (6) SHARE BUYBACK ACCELERATION: Share buyback spending jumped from 45K to 185M yen (4,000x increase) in the same year as the cash flow anomaly, potentially masking dilution from questionable transactions. (7) MISSING DATA: 'Diluted EPS' marked as '-' for all periods is unusual for a listed company and may indicate data suppression. While the CPA verified numerical consistency, the logical relationships between line items show patterns consistent with earnings management or aggressive accounting.",
  "input_tokens": 4217,
  "output_tokens": 657,
  "elapsed_sec": 9.13
}
o4-mini-2025-04-16 row:
{
  "doc_id": "S100G8YC",
  "edinet_code": "E01544",
  "label": 0,
  "prediction": 0,
  "prob": 0.05,
  "reasoning": "A line‐by‐line consistency check shows no obvious red flags.  All major flows reconcile: the big jump in operating cash flow (¥1.07 bn to ¥5.19 bn) is fully explained by the large decrease in accounts receivable and routine depreciation charges.  Equity, net profit, cash, and dividend movements tie out with the changes in retained earnings and other comprehensive income.  No unusual one-off adjustments, round-number manipulations, or unexplained balance sheet discrepancies are evident.  The absence of dilutive instruments (all “–” in the EPS reconciliation) appears to reflect simple capital structure rather than a concealment.  Overall, the narratives and footnote data align, giving no indication of financial statement fraud.",
  "input_tokens": 3993,
  "output_tokens": 2361,
  "elapsed_sec": 18.13
}

===== EXP-R-0003 =====
models: claude-haiku-4-5-20251001 vs o4-mini-2025-04-16
max_abs_prob_diff = 0.63
-- case 1: doc_id=S100E0KV, abs_diff=0.63 (claude-haiku-4-5-20251001: 0.68, o4-mini-2025-04-16: 0.05)
claude-haiku-4-5-20251001 row:
{
  "doc_id": "S100E0KV",
  "edinet_code": "E33776",
  "label": 0,
  "prediction": 1,
  "prob": 0.68,
  "reasoning": "Analysis of this Japanese securities report reveals several significant red flags that warrant fraud investigation:\n\n1. **Extraordinary Asset Growth Inconsistency**: Total assets increased 158% (¥1.4B to ¥3.6B) while sales increased only 25% (¥4.1B to ¥5.1B). This disproportionate asset growth is unusual and suggests potential asset inflation or accounting irregularities.\n\n2. **Massive Capital Injection Anomaly**: Capital increased from ¥30M to ¥693M (2,210% increase) through equity issuance of ¥1.3B in a single period. This extraordinary capital raise relative to operational cash generation (¥89M operating CF) raises questions about the necessity and use of funds.\n\n3. **Suspicious Special Items**: The P&L shows extraordinary special gains of ¥823M and special losses of ¥821M that nearly offset each other, suggesting potential accounting manipulation to manage reported earnings. These items (補助金 subsidies and 固定資産圧縮損 asset compression losses) appear designed to neutralize each other.\n\n4. **Cash Flow Inconsistency**: Despite ¥45M net income, operating cash flow was only ¥89M, but cash increased ¥1.5B primarily from financing activities (¥1.9B). The company appears to be funding growth through debt/equity rather than operational performance.\n\n5. **ROE Deterioration**: Return on equity dropped from 13.4% to 4.9% despite reported profit growth, indicating the massive capital injection is not generating proportional returns.\n\n6. **Rapid Business Expansion**: Employee count increased 40% (275 to 386) and temporary workers increased 22% (1,557 to 1,895) in a single year, suggesting aggressive expansion that may not be sustainable or properly managed.\n\n7. **Founder Dependency**: 38% of shares held by founder (若濵久), with significant management concentration, creating potential governance risks and related-party transaction concerns.\n\n8. **Recent IPO Status**: Company went public in March 2018 (during this fiscal year), which often correlates with aggressive accounting to present favorable pre-IPO performance.\n\n9. **Subsidiary Profitability Concerns**: Main subsidiary (株式会社セリオ) shows ¥30.5M operating profit on ¥5.1B sales (0.6% margin), while parent company shows minimal operations, suggesting potential profit concentration or transfer pricing issues.\n\n10. **Missing Data Points**: Multiple entries marked with \"－\" (missing data) in historical comparisons, though stated as parsing errors, reduce transparency.\n\nWhile the CPA verified numerical consistency, the logical patterns suggest potential earnings management, aggressive accounting policies, or undisclosed related-party transactions typical of pre-IPO fraud schemes.",
  "input_tokens": 34370,
  "output_tokens": 697,
  "elapsed_sec": 37.33
}
o4-mini-2025-04-16 row:
{
  "doc_id": "S100E0KV",
  "edinet_code": "E33776",
  "label": 0,
  "prediction": 0,
  "prob": 0.05,
  "reasoning": "The record is coherent and consistent with a rapid expansion into childcare facilities under Japanese GAAP. The very large “special profit” (補助金収入) and matching “fixed-asset compression loss” (固定資産圧縮損) reflect government grants netting out in income, which is standard JGAAP treatment. Cash‐flow, balance‐sheet movements and financing (IPO proceeds, debt drawdowns) reconcile with stated capex and asset growth. Receivables, payables and operating cash flow behavior align with service‐industry norms. Corporate governance disclosures, related-party and insider holdings, segment reporting and auditor opinions show no obvious concealment or manipulation. No non-numerical inconsistencies or logical gaps indicative of fraud were identified.",
  "input_tokens": 30849,
  "output_tokens": 3129,
  "elapsed_sec": 23.76
}
```

I find that, the two models both are lack of verification, from a high level observation, they can draw very different (very aggressive/conservative) conclusions, but, they don't dig deeper to decompose the problem and analyze in more detail. Finding related info. and building the brige of reasoning are not seen in the LLM's one time conclusion. *(So maybe a baseline agent structure with a fundamental reflection loop can have a big improvement?)*

2. Sample within each model's result that has the most (absolute) difference "prob" across the two experiments.

```
===== claude-haiku-4-5-20251001 =====
experiments: EXP-R-0002 vs EXP-R-0003
max_abs_prob_diff = 0.44
-- case 1: doc_id=S100ET3W, abs_diff=0.44 (EXP-R-0002: 0.72, EXP-R-0003: 0.28)
EXP-R-0002 row:
{
  "doc_id": "S100ET3W",
  "edinet_code": "E03314",
  "label": 1,
  "prediction": 1,
  "prob": 0.72,
  "reasoning": "Analysis reveals several concerning patterns that warrant fraud investigation: (1) EXTREME VALUATION ANOMALY: P/E ratio exploded from 188.8 (Prior2Year) to 1790.6 (CurrentYear) despite minimal earnings recovery (net income only 10.8M yen), suggesting potential earnings manipulation or stock price inflation. (2) EQUITY STRUCTURE INCONSISTENCY: Pure equity increased 57.8% (5.5B to 8.7B yen) while net income remained negligible, driven primarily by capital increases (959M yen) and treasury stock reduction (978M yen) rather than operational performance. This suggests potential equity restructuring to mask poor fundamentals. (3) ASSET QUALITY CONCERNS: Total assets grew 7.8% while operating cash flow declined 11.2%, indicating asset growth not supported by cash generation. Fixed assets increased 2.0B yen (12.6%) with building/structure values up 24% despite depreciation accumulation, suggesting possible asset revaluation without corresponding operational justification. (4) PROFITABILITY DISCONNECT: Operating profit improved to 390.7M yen, but comprehensive income jumped 235% to 144.4M yen, driven by unrealized gains (other securities valuation difference increased 150.7M yen), indicating reliance on non-operational gains. (5) CASH FLOW DETERIORATION: Despite improved net income, operating cash flow declined and investment cash outflow increased 27%, while financing activities swung dramatically positive (2.35B yen), suggesting cash position artificially supported by debt rather than operations. (6) DEFERRED TAX ASSET DECLINE: Deferred tax assets fell 43% (233.6M to 133.6M yen) while tax expense increased, potentially indicating tax position deterioration or valuation allowance adjustments. (7) MISSING DATA PATTERN: Diluted EPS data missing for 3 consecutive years (Prior3-Prior1Year) then reappears, ROE missing for loss years, suggesting selective reporting of unfavorable metrics.",
  "input_tokens": 4148,
  "output_tokens": 496,
  "elapsed_sec": 7.44
}
EXP-R-0003 row:
{
  "doc_id": "S100ET3W",
  "edinet_code": "E03314",
  "label": 1,
  "prediction": 0,
  "prob": 0.28,
  "reasoning": "Analysis of this Japanese securities report reveals several concerning patterns that warrant fraud investigation, though the numerical consistency verified by CPAs limits definitive conclusions:\n\n1. **Dramatic Equity Volatility**: Pure equity surged 57.8% (¥5.5B to ¥8.7B) in one year, primarily through stock issuance (¥1.9B) and self-stock disposal (¥1.1B). While legitimate, this aggressive capital restructuring coincides with weak profitability (net income only ¥10.8M on ¥32.6B sales = 0.03% margin).\n\n2. **Profitability Inconsistency**: The company swung from ¥414.8M loss (prior year) to ¥10.8M profit (current year)—a ¥425.6M reversal. This dramatic turnaround is driven primarily by reduced special losses (¥514.8M → ¥72.9M impairment), not operational improvement. Operating profit only increased 24% while sales grew 4%, suggesting operational challenges masked by accounting adjustments.\n\n3. **Asset Quality Concerns**: \n   - Goodwill of ¥384.2M (1.3% of assets) with 10-year amortization suggests aggressive M&A valuations\n   - Significant impairment losses (¥72.9M current year, ¥514.8M prior year) indicate recurring asset write-downs\n   - Building/construction assets increased 25% (¥7.7B → ¥9.6B) while depreciation accelerated, suggesting potential overvaluation or aggressive capitalization\n\n4. **Cash Flow Deterioration**: Operating cash flow declined 11% (¥1.4B → ¥1.3B) despite profit improvement, indicating earnings quality issues. Investment cash outflows remained high (¥3.4B), suggesting continued capital intensity without proportional returns.\n\n5. **Leverage Concerns**: Debt-to-equity ratio of 2.5x (¥17.9B debt vs ¥8.7B equity) with 58.7% debt-to-assets ratio. The company reduced short-term debt (¥3.8B → ¥1.3B) while increasing long-term debt (¥12.6B → ¥13.9B), suggesting refinancing pressure.\n\n6. **Related Party Transactions**: Extensive related-party dealings with subsidiaries (100% ownership) and related companies. While disclosed, the complexity creates opacity. Subsidiary profitability appears inconsistent (e.g., ¥梅の花ｐｌｕｓ showing ¥3.3M loss).\n\n7. **Accounting Adjustments**: Large special items (¥23.5M gains, ¥172M losses) and significant tax adjustments (¥81.6M) suggest earnings management potential. The tax adjustment of 67% of pre-tax income is unusually high.\n\n8. **Missing Data**: Multiple fields marked with \"－\" in employee and financial metrics, though this may reflect reporting standards rather than fraud.\n\n9. **Positive Mitigating Factors**: \n   - CPA verification of numerical consistency\n   - Detailed disclosure of related parties and business segments\n   - Reasonable explanation for asset increases (new Kyoto central kitchen)\n   - Consistent revenue growth trajectory\n\nThe report shows signs of financial stress (losses, impairments, refinancing) managed through capital restructuring and accounting adjustments rather than operational improvement. However, the CPA verification and detailed disclosures reduce fraud probability.",
  "input_tokens": 36021,
  "output_tokens": 882,
  "elapsed_sec": 46.29
}

-- case 2: doc_id=S100IYSK, abs_diff=0.44 (EXP-R-0002: 0.72, EXP-R-0003: 0.28)
EXP-R-0002 row:
{
  "doc_id": "S100IYSK",
  "edinet_code": "E02547",
  "label": 1,
  "prediction": 1,
  "prob": 0.72,
  "reasoning": "Analysis of the financial data reveals several concerning inconsistencies and logical red flags:\n\n1. **CRITICAL: Revenue Decline with Profit Increase Anomaly** - Sales (売上高) declined 7.8% from Prior1Year (31.76B) to CurrentYear (29.26B), yet net profit (親会社株主に帰属する当期純利益) increased 4.2% from 1.63B to 1.70B. This inverse relationship is highly suspicious, as declining revenue typically correlates with declining profitability.\n\n2. **Gross Margin Expansion Despite Revenue Decline** - Gross profit margin improved from 19.3% (Prior1Year) to 20.4% (CurrentYear) despite lower sales. While possible, combined with profit growth during revenue decline, this raises concerns about cost accounting accuracy or revenue recognition practices.\n\n3. **Working Capital Manipulation Red Flag** - Accounts receivable decreased 9.6% (7.56B to 6.83B) while sales fell 7.8%. This disproportionate decrease suggests potential aggressive receivables collection or possible revenue recognition issues. The cash flow statement shows a 1.81B positive adjustment for sales debt reduction, which is unusually large.\n\n4. **Inventory Reduction Inconsistency** - Inventory decreased from 2.96B to 2.50B (15.5% reduction) while sales only declined 7.8%. This disproportionate inventory reduction, combined with the accounts receivable decrease, suggests potential manipulation of working capital to inflate operating cash flow.\n\n5. **Operating Cash Flow Reversal** - Operating cash flow swung dramatically from -460M (Prior1Year) to +3.77B (CurrentYear), a 9.2B swing. While the working capital adjustments explain this mathematically, the magnitude and direction are suspicious given declining revenues.\n\n6. **Investment Securities Anomaly** - Investment securities (投資有価証券) decreased 7.8% (3.07B to 2.83B), yet the company recorded a 180M gain on securities sales in the current year (vs. no such gain in Prior1Year). This suggests potential cherry-picking of securities sales to recognize gains.\n\n7. **Unrealized Gains Volatility** - Other comprehensive income items (その他有価証券評価差額金) collapsed from 236M to 56M, a 76% decline. Combined with the securities sale gain, this pattern suggests potential mark-to-market manipulation or selective realization of gains.\n\n8. **Financing Cash Flow Deterioration** - Despite improved profitability metrics, financing cash flow worsened significantly to -2.97B (from -326M), driven by short-term debt reduction of 1.87B. This suggests the company may be using operational improvements to pay down debt rather than organic cash generation.\n\n9. **Asset Quality Concerns** - Total assets decreased 3.5% while net assets increased 7.1%, indicating asset composition changes. The decrease in investment securities and accounts receivable, combined with increased intangible assets (117M to 226M), warrants scrutiny.\n\n10. **Employee Count Growth Inconsistency** - Employee count increased 5.2% (270 to 284) while revenue declined 7.8%, suggesting potential inefficiency or that headcount additions occurred late in the period without corresponding revenue generation.\n\nThe combination of declining revenue with increasing profits, dramatic working capital swings, and selective gain recognition creates a pattern consistent with earnings management rather than organic business improvement.",
  "input_tokens": 4346,
  "output_tokens": 845,
  "elapsed_sec": 11.09
}
EXP-R-0003 row:
{
  "doc_id": "S100IYSK",
  "edinet_code": "E02547",
  "label": 1,
  "prediction": 0,
  "prob": 0.28,
  "reasoning": "Analysis of this Japanese company's securities report reveals several areas of concern that warrant investigation, though the numerical consistency has been verified by a certified public accountant:\n\n1. **Revenue Decline with Profit Increase Anomaly**: Sales declined from ¥31.755B (Prior1Year) to ¥29.264B (CurrentYear), a 7.8% decrease. However, parent company net income increased from ¥1.630B to ¥1.698B (+4.2%), and operating profit only declined marginally from ¥2.029B to ¥1.974B (-2.7%). This disproportionate relationship is unusual - typically, significant revenue declines correlate with profit declines.\n\n2. **Suspicious Special Gains**: The company recorded ¥210M in special gains (CurrentYear) including ¥180M from investment securities sales, compared to only ¥75M in the prior year. This sudden increase in one-time gains coinciding with revenue decline raises questions about earnings quality.\n\n3. **Working Capital Manipulation Indicators**: \n   - Accounts receivable decreased by ¥726M despite lower sales\n   - Inventory decreased by ¥458M\n   - Accounts payable decreased by ¥563M\n   - These simultaneous decreases suggest potential aggressive working capital management\n\n4. **Cash Flow Volatility**: Operating cash flow swung dramatically from negative ¥460M (Prior1Year) to positive ¥3.770B (CurrentYear), a ¥4.23B swing. While explained by working capital changes, the magnitude is notable.\n\n5. **Incomplete Disclosure**: Multiple data fields marked as \"-\" (missing data) in diluted EPS and other metrics, though this may be due to parsing errors as noted.\n\n6. **Related Party Concentration Risk**: Heavy reliance on specific suppliers (Zircon sand from Iluka, Warman pumps from Taiheiyo Kiko) with limited diversification, creating vulnerability to supply disruptions.\n\n7. **Inconsistent Segment Performance**: The company reports 6 business segments but provides limited detailed breakdown of segment profitability, making it difficult to verify earnings quality by business line.\n\n8. **Asset Valuation Concerns**: Investment securities decreased from ¥3.071B to ¥2.831B, with other comprehensive income showing significant unrealized losses (¥236M to ¥56M), suggesting potential asset quality issues.\n\nHowever, mitigating factors include: verified numerical consistency by certified auditors, reasonable explanations for most variances in management discussion, stable equity base, and no obvious red flags in balance sheet structure.",
  "input_tokens": 28890,
  "output_tokens": 654,
  "elapsed_sec": 43.52
}


===== o4-mini-2025-04-16 =====
experiments: EXP-R-0002 vs EXP-R-0003
max_abs_prob_diff = 0.55
-- case 1: doc_id=S100T6RN, abs_diff=0.55 (EXP-R-0002: 0.2, EXP-R-0003: 0.75)
EXP-R-0002 row:
{
  "doc_id": "S100T6RN",
  "edinet_code": "E35294",
  "label": 1,
  "prediction": 0,
  "prob": 0.2,
  "reasoning": "Overall the numbers are consistent and the trends, margins and cash flows align with the disclosed P/L and B/S. No glaring numerical mismatches appear. However, there are a few non-numerical disclosure oddities: 1) 在「損益計算書」上は非支配株主に帰属する当期純利益（≈¥10.3 M）が計上されているのに、貸借対照表の「非支配株主持分」が当期で“–”とされており矛盾しています。2) 「繰延ヘッジ損益」が直前期には¥7.9 Mあるにもかかわらず当期で“–”と省略され、かつ為替換算調整勘定だけ大きく振れている点も不自然です。これらは開示漏れまたは科目間の移し替えミスの可能性はあるものの、証明監査人のサイン付きで数値計算上は整合しており、悪質な粉飾を示唆する十分な根拠は乏しいと判断します。",
  "input_tokens": 4306,
  "output_tokens": 1901,
  "elapsed_sec": 14.94
}
EXP-R-0003 row:
{
  "doc_id": "S100T6RN",
  "edinet_code": "E35294",
  "label": 1,
  "prediction": 1,
  "prob": 0.75,
  "reasoning": "Several non-numerical red flags appear in the narrative sections even though the numbers tie out: 1) The company discloses that “over multiple years, a consolidated subsidiary had fictitious transactions with outside contractors,” prompting an external special-investigation committee in early 2024 and confirmation of bogus receipts/payments—an explicit admission of fraud. 2) Governance shows extreme owner-control: the founding chairman (57.7% share) and his son as president/CEO, indicating weak board independence and potential conflicts. 3) Numerous related-party dealings and complex cross-shareholdings with 100% foreign and domestic subsidiaries increase opacity and the risk of undisclosed intra-group abuses. 4) A sudden elimination of all non-controlling interests suggests aggressive share take-overs, potentially to conceal past irregularities. Taken together, these qualitative inconsistencies and admitted subsidiary fraud point to serious internal-control failures and ongoing risk of wrongdoing.",
  "input_tokens": 36999,
  "output_tokens": 1314,
  "elapsed_sec": 13.71
}
```

Here the good thing is, both models make use of the "Text" to draw the conclusion, for example, the case `doc_id=S100T6RN`, o4-mini initially only gave 0.2 probability, but there is some strong evidence in the narrative section saying that the company has fictitious transactions, then the model noticed this and identified it as fraud. However, this example is kind of extreme in that it directly gives the answer in the "Text" part.

Also, look into more samples, we can see that it seems the two models just give too much priority to the narrative part, this is expectable because the signal is usually more strong in text than in number. But it should not be the golden evidence, for example, the case `doc_id=S100ET3W`, haiku initially found some red flags in the numerical part, and in the second experiment, it explains some of the flags using facts and evidence from the narrative part, but, it also ignored some important parts and just identified it as non-fraud.