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