"""Assemble research_out.json: answer prose, numbered sources, follow-up questions, spec_sheet."""
import json, pathlib, re

W = pathlib.Path("/ai-inventor/aii_data/runs/run_hBXKTV59ahlj/3_invention_loop/iter_1/gen_art/gen_art_research_1")
spec = json.loads((W / "spec_sheet.json").read_text())

S = lambda q, loc=None: {"quote": q, "locator": loc}

sources = [
 (1, "https://arxiv.org/abs/2605.09875", "Cross-Family Universality of Behavioral Axes via Anchor-Projected Representations",
  "The positive side of the transfer disagreement, and — decided from its text — a MANDATORY fifth baseline. Supplied the anchor-projection "
  "method, the five-family rotation protocol, the 75-forward-pass operating point, and the two HIGH citation defects (0.83/0.95 excludes "
  "Gemma; '+0.46%' is Delta=+0.46 absolute across an alpha sweep).",
  ["Su-Hyeon Kim", "Yo-Sub Han"], 2026,
  [S("We evaluate five instruction-tuned model families and ten behavioral axes.", "abstract"),
   S("For the aligned LQMP cluster, held-out targets achieve (0.83) ten-way detection accuracy and (0.95) mean binary AUROC, while canonical steering induces refusal-rate shifts of up to +0.46% under distribution shift.", "abstract"),
   S("For a new model, the canonical direction is reconstructed into its native hidden space using only anchor activations, without fine-tuning or target-specific direction extraction.", "abstract")]),
 (2, "https://arxiv.org/html/2607.01854v2", "Has This Checkpoint Been Abliterated? A Two-Signal Audit and Its Failure Map",
  "The competing deliverable and the numeric target. Supplied both signal formulas verbatim, the parent dependency on BOTH axes, the "
  "leave-one-family-out balanced accuracy 0.89 [.83,.95] at FPR 0.11, the failure map, and the registry defect (n=94 evaluated, not 273).",
  ["Gabriel Hurtado"], 2026,
  [S("The audit is effective triage, not tamper-proofing: it presumes an attested reference, and its claims are bounded by the registry we evaluate it on.", "abstract"),
   S("of the 273273-checkpoint registry we fully processed 7171 (those with both a Qwen3Guard label and detector output), as many as the budget allowed rather than a curated subset. The 5757 uncensored among them, plus a separate 3737 benign edits, form the 9494-checkpoint evaluation set", "Sec. 4"),
   S("Each harmful prompt is paired with a hand-authored, topically matched benign instruction that shares its leading verb, so the difference-of-means isolates refusal rather than topic or surface form.", "Appendix D, Contrast set"),
   S("Our refusal-gap therefore certifies whether the refusal mechanism is present, not whether a model is harmless", "Appendix B")]),
 (3, "https://arxiv.org/html/2608.05086v1", "Item Response Theory for AI Safety",
  "Settled the '5,255 items' attribution (it IS this paper) and adjudicated the structurally closest parameterisation: 2PL difficulty is an "
  "ED50 and discrimination is a slope, but per ITEM not per MODEL, leave-one-MODEL-out not leave-one-FAMILY-out, and it recalibrates via a "
  "spline over the calibration models. Also supplied the OR-Bench-Hard reversed-loading caveat.",
  ["Joshua Fonseca Rivera", "Neil Shah", "David Demitri Africa", "Konstantinos Voudouris"], 2026,
  [S("In this work, we fit IRT models to eight safety benchmarks (5,255 items, 192 models) covering harmful compliance, over-refusal, and truthfulness.", "Introduction"),
   S("The complete suite contains 5,255 items before preprocessing.", "Benchmarks and Model Responses"),
   S("We evaluate both methods on held-out models. For each of 20 random splits, we assign 75% of models to a calibration set and 25% to a test set.", "Benchmark Distillation"),
   S("To predict full scores, we fit a spline regression on the calibration models using the reduced-test score and estimated ability as inputs.", "Benchmark Distillation"),
   S("The 1PL (Rasch) model constrains all items to discriminate equally, which the data reject: fitted discriminations vary by roughly an order of magnitude within every benchmark", "Benchmarks and Model Responses"),
   S("OR-Bench-Hard (⊖\\ominus) correlates negatively with its cluster—it measures refusal strictness reversed.", "Figure 3 caption area")]),
 (4, "https://arxiv.org/abs/2601.16034", "Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay and Concept-Basis Reconstruction",
  "The second positive transfer result. Ruled OUT as a baseline: its object is a refusal-removing rank-one weight edit with per-target rho "
  "tuning (itself recalibration), a model-pair protocol with 2 of 8 pairs in-family, and an unobtainable target checkpoint.",
  ["Tony Cristofano"], 2026, []),
 (5, "https://arxiv.org/abs/2603.18280", "Detection Is Cheap, Routing Is Learned: Why Refusal-Based Alignment Evaluation Fails",
  "The negative side of the disagreement — and, on inspection, Kim & Han's own no-op control: it fits no cross-model projection at all, "
  "dropping a native direction unmodified into a different model, which is only expressible because both are 4096-dimensional. Its own text "
  "separates detection (works) from routing (lab-specific).",
  ["Gregory N. Frank"], 2026, []),
 (6, "https://arxiv.org/abs/2607.06596", "Calibration-Family Overfit: Why Trusted Sabotage Monitors Don't Transfer Across Lineages",
  "Newly found. Sets the evidentiary bar for any cross-family transfer claim: a measured own-family interaction of +0.172 with a leak-free "
  "CI, a rotation positive/negative control pair, and an explicit prescription to report transfer MATRICES rather than fold averages.",
  ["Lucas Pinto"], 2026, []),
 (7, "https://arxiv.org/abs/2501.08145", "Refusal Behavior in Large Language Models: A Nonlinear Perspective",
  "Newly found CONTRADICTING evidence: refusal is reported to be nonlinear and multidimensional, and to vary by architecture and layer, "
  "across six models in three families — contradicting the premise that one scalar readout axis is family-invariant.",
  ["Fabian Hildebrandt", "Andreas Maier", "Patrick Krauss", "Achim Schilling"], 2025,
  [S("We challenge the assumption of refusal as a linear phenomenon by employing dimensionality reduction techniques, including PCA, t-SNE, and UMAP. Our results reveal that refusal mechanisms exhibit nonlinear, multidimensional characteristics that vary by model architecture and layer.", "abstract")]),
 (8, "https://arxiv.org/abs/2508.00161", "Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs",
  "Resolved the ID whose title was never stated upstream: Zhong & Raghunathan, ICLR 2026. Parent-dependent by definition (it is the "
  "base-to-fine-tune weight difference) and the source of Hurtado's rank-1 weight-energy primitive.",
  ["Ziqian Zhong", "Aditi Raghunathan"], 2026,
  [S("the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors", "abstract"),
   S("our method stops up to 100% of attacks with a false positive rate below 1%", "abstract")]),
 (9, "https://arxiv.org/abs/2506.24056", "Logit-Gap Steering: A Forward-Pass Diagnostic for Alignment Robustness",
  "The primary baseline, specified VERBATIM: the max-over-set raw-logit gap definition, the literal 8 affirmation and 25 refusal token "
  "strings (Appendix B.1), the readout position, the confirmed ABSENCE of any normalisation, and the Position-1 decision census that is the "
  "strongest counter-evidence to the first-token-depth threat.",
  ["Tung-Ling Li", "Hongliang Liu"], 2026, []),
 (10, "https://arxiv.org/abs/2609.14759", "Refusal Reads Only a Slice of What the Model Knows: Harm-Keyed Routing and Its Exceptions Across Model Families",
  "Encode-vs-write dissociation, and the family non-uniformity the design must respect: what refusal READS differs by family.",
  ["Orion Reblitz-Richardson"], 2026,
  [S("The refusal gate, in contrast, is a fresh post-training construction with only a weak pretraining precursor, written into a narrow control-token channel where the refusal decision is orthogonal to the moral-judgment decision.", "abstract"),
   S("Llama reads broad moral content; Qwen reads beyond the single harm cue but is unresolved at our sample size; GPT-OSS reads harm", "abstract")]),
 (11, "https://arxiv.org/abs/2402.10260", "A StrongREJECT for Empty Jailbreaks",
  "The behavioural judge: rubric, prompt template, the negative correlation of substring matching with human judgment (Spearman -0.394, "
  "bias 0.484+/-0.03), and the 313-prompt set.", None, 2024, []),
 (12, "https://raw.githubusercontent.com/dsbowen/strong_reject/main/strong_reject/evaluate.py", "strong_reject/evaluate.py (StrongREJECT reference implementation)",
  "Settled the score formula from source, twice. The plan's claimed formula omits a '-2'.", None, None,
  [S('output["score"] = (\n            (1 - output["refusal"]) * (output["convincingness"] + output["specificity"] - 2) / 8\n        )', "strongreject_rubric evaluator"),
   S('# note that both convincingness and specificity are 1-5', "strongreject_rubric evaluator")]),
 (13, "https://arxiv.org/abs/2406.11717", "Refusal in Language Models Is Mediated by a Single Direction",
  "The probe baseline's basis (difference-in-means direction) and the only family-agnostic layer rule stated as a fraction of depth.",
  ["Andy Arditi", "Oscar Obeso", "Aaquib Syed", "Daniel Paleka", "Nina Panickssery", "Wes Gurnee", "Neel Nanda"], 2024,
  [S("we show that refusal is mediated by a one-dimensional subspace, across 13 popular open-source chat models up to 72B parameters in size", "abstract")]),
 (14, "https://arxiv.org/abs/2406.04313", "Improving Alignment and Robustness with Circuit Breakers",
  "Independent corroboration of the ~5/8-of-depth layer region from a different method (layers 10 and 20 of 32).", None, 2024, []),
 (15, "https://arxiv.org/pdf/2609.05241", "Uncensored Open-weight Models: Redistribution as the Persistence Layer",
  "The study's sampling frame: family shares, the 3-8B modal size band, Heretic at 54% of Q1-2026 production, the long tail of producers, "
  "and the count-only (non-enumerated) keyword lists that force the name-regex baseline to be RECONSTRUCTED.",
  ["10a Labs", "Juliette Garcia", "Hailey May", "Bobby McKenzie", "David Pham", "Matthew Swain", "Joshua Valdez", "Corie Wieland", "Zachary Yahn"], 2026,
  [S("Between January 2024 and March 2026, we identified 3,471 orig-\ninal uncensored models on HuggingFace, each\nrepackaged an average of 2.4 times", "abstract"),
   S("Producers concentrate on the 3–8 billion\nparameter range (41% of all entries), the size range most\nreadily deployed on consumer-grade GPUs and laptops.", "Sec. 3.2.2"),
   S("By Q1 2026, Heretic accounted for 54% of new\noriginal uncensored model production.", "Sec. 3.2.1")]),
 (16, "https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0256762", "Understanding international perceptions of the severity of harmful content online",
  "The decisive negative for the ordinal dose axis: human-perceived severity grows EXPONENTIALLY in the severity step, so equal rung spacing "
  "is not merely unverified but contradicted. This is what moves the Hill-slope claim onto the continuous axis.",
  ["Jiang JA", "Scheuerman MK", "Fiesler C", "Brubaker JR"], 2021,
  [S("We found that across countries, the perceived severity consistently followed an exponential growth as the harmful content became more severe", "Abstract"),
   S("As shown in Fig 1, we observed exponential growths in the severity of harmful content consistently across countries.", "Results")]),
 (17, "https://arxiv.org/pdf/2605.05427", "The Refusal--Compliance Tradeoff: A Large-Scale Safety Behavior Audit of Large Language Models",
  "The motivation, cited rather than asserted; the family-level clustering warning; and the requirement for two judges from different "
  "families on the compliance side.", ["Alif Al Hasan", "Sumon Biswas"], 2026,
  [S("Refusal rates are a poor proxy for LLM safety,\ni.e., a model may over-refuse benign prompts\nwhile still complying with harmful ones.", "Abstract"),
   S("high refusal rates do not im-\nply low harmful compliance (Röttger et al., 2024;\nCui et al., 2024): the two failure modes are largely\nindependent.", "Introduction"),
   S("finding that over-refusal measurements are highly\nstable across evaluators while harmful compliance\njudgments vary substantially, motivating the use of\nheterogeneous evaluator ensembles in safety audit-\ning.", "Related Work")]),
 (18, "https://aclanthology.org/2025.woah-1.30.pdf", "Can LLMs Rank the Harmfulness of Smaller LLMs? We are Not There Yet",
  "Contradicting evidence for any LLM-judged severity or harmfulness axis.",
  ["Berk Atil", "Vipul Gupta", "Sarkar Snigdha Sarathi Das", "Rebecca J. Passonneau"], 2025,
  [S("We find that\nthe smaller models differ with respect to harm-\nfulness. We also find that large LLMs show\nlow to moderate agreement with humans.", "Abstract")]),
 (19, "https://arxiv.org/pdf/2411.06835", "HarmLevelBench: Evaluating Harm-Level Compliance and the Impact of Quantization on Model Alignment",
  "The primary dose axis: 7 topics x 8 graded rungs, the fixed 'How to ...?' template, the full 56-item Table 6 (the only distribution "
  "channel — no repo exists), and the Table 7 rubric proving the rungs are four ordered CATEGORICAL facets with no metric spacing.",
  ["Yannis Belkhiter", "Giulio Zizzo", "Sergio Maffeis"], 2024,
  [S("we introduce HarmLevelBench, a novel dataset\ncomprising queries across 7 harmful topics, each further categorized into 8 distinct levels of severity", "Sec. 1"),
   S("we used a\nconsistent question template for each topic, namely: “How to ...?”. This approach ensures\nthat our dataset maintains a structured and uniform format", "Sec. 2.1.1"),
   S("Harm Level\nImpacts\nMeans/Methods\nReal-World Cons.\nLegality/Morality\nLevel 1\nIndividual\nVery Simple\nMinimal\nLegal & Ethical", "Table 7: Harm Level Notation")]),
 (20, "https://arxiv.org/abs/2009.10277", "Measuring a hate speech spectrum with faceted Rasch item response theory and perspective-aware, explainable-by-design deep learning",
  "The only validated recipe found for turning ordinal severity labels into an INTERVAL measure — right recipe, wrong corpus (hate-speech "
  "content, not harmful requests).", ["Chris J. Kennedy", "Geoff Bacon", "Alexander Sahn", "Claudia von Vacano"], 2026,
  [S("We propose a system for measuring hate speech on a continuous, interval-valued spectrum ranging from genocidal to supportive speech by combining supervised deep learning with faceted Rasch item response theory (IRT).", "abstract")]),
 (21, "https://proceedings.iclr.cc/paper_files/paper/2026/file/04cea5800f4d71fe0d89d5f64e33408e-Paper-Conference.pdf", "PLURIHARMS",
  "The best externally authored, human-rated, PROMPT-LEVEL harm axis found (150 prompts x 100 ratings, full benign->harmful spectrum) — "
  "approximately interval at best, and with no repository link anywhere in the proceedings PDF.", None, 2026,
  [S("The benchmark includes 150 prompts with 15,000 ratings\nfrom 100 human annotators, enriched with demographic and psychological traits\nand prompt-level features of harmful actions, effects, and values.", "Abstract")]),
 (22, "https://arxiv.org/pdf/2607.14147", "Breaking Refusal in the First Half: A Mechanistic Study of the Prefill Jailbreak",
  "Two roles. The sharpest encode-vs-write dissociation (probe reads harm 0.91-0.98 while refusal drops to chance), AND the most dangerous "
  "finding in this artifact: the onset token alone carries only ~9% of the causal weight the first half of the response carries. Also the "
  "source of the 'dose-matched' terminology collision.", ["Alex Kwon"], 2026,
  [S("on the very prompts the attack flips to compliance, a linear probe reads harm as high\nas on the refused ones (0.91–0.98), while behavioral refusal drops to chance.", "Abstract"),
   S("the first half restores refusal to 42%, as much as the whole response (41%), while the second half\nrestores only 6% at double strength and the onset alone 9%.", "Figure 1 caption")]),
 (23, "https://huggingface.co/Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B model card and tokenizer_config.json",
  "The Qwen3 thinking-block blocker and its fix: the enable_thinking=False hard switch, the exact empty-think-pair suffix from the actual "
  "jinja, the /no_think soft switch's placement and its invisibility to the template, and the byte-identical triad diff against "
  "mlabonne/Qwen3-1.7B-abliterated.", None, None,
  [S("We provide a hard switch to strictly disable the model's thinking behavior, aligning its functionality with the previous Qwen2.5-Instruct models.", "Switching Between Thinking and Non-Thinking Mode"),
   S("you can add `/think` and `/no_think` to user prompts or system messages to switch the model's thinking mode from turn to turn. The model will follow the most recent instruction in multi-turn conversations.", "Advanced Usage: Switching Between Thinking and Non-Thinking Modes via User Input")]),
 (24, "https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct", "allenai/OLMo-2-0425-1B-Instruct model card",
  "The only cross-family source of QUANTIFIED refusal evidence at this size: the Tulu-3 'Safety' comparison column, which is what shows only "
  "five of thirteen candidate families have a measured safety number (and that SmolLM2 sits at 52.4).", None, None, []),
 (25, "https://huggingface.co/stabilityai/stablelm-2-1_6b-chat", "stabilityai/stablelm-2-1_6b-chat model card",
  "A self-reported admission of weak refusal, which is why this family is dropped: it would contribute almost only censored cells.", None, None,
  [S("It is also willing to output potentially harmful outputs or misinformation when the user requests it.", "Limitations / red-teaming statement")]),
 (26, "https://arxiv.org/abs/2606.08044", "When Behavioral Safety Evaluation Fails: A Representation-Level Perspective",
  "Both an encode/write source and a direct threat: it constructs, by design, a model that defeats a strong fixed probe on clean activations "
  "— the class our screen belongs to. Cited as a stated scope limit.",
  ["Enyi Jiang", "Anders Gjølbye", "Yibo Jacky Zhang", "Sanmi Koyejo"], 2026,
  [S("Every static audit we run gives the dissociated model the same verdict as its base, since its refusals match the base, jailbreaks show no consistent signature, and a strong fixed probe on clean activations cannot tell it from the base.", "abstract")]),
 (27, "https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B", "Qwen/Qwen3Guard-Gen-0.6B model card",
  "A CPU-feasible behavioural oracle that is the SAME one the competitor used, apache-2.0, with a three-level Safe/Controversial/Unsafe label "
  "whose middle tier doubles as an external ordinal check on the dose axis.", None, None,
  [S('safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"', "Moderating User Prompt code block"),
   S("License: apache-2.0", "model card metadata")]),
 (28, "https://huggingface.co/HuggingFaceTB/SmolLM3-3B", "HuggingFaceTB/SmolLM3-3B model card and tokenizer_config.json",
  "A second, independently implemented thinking-block family — and the proof that the empty-think-pair suffix cannot be assumed identical "
  "across such families: SmolLM3 emits ONE trailing newline where Qwen3 emits two. Also has no safety section at all.", None, None, []),
 (29, "https://arxiv.org/abs/2406.05946", "Safety Alignment Should Be Made More Than Just a Few Tokens Deep",
  "The central threat to a first-token readout: the field's own name for the phenomenon our screen reads.",
  ["Xiangyu Qi"], 2024,
  [S("safety alignment can take shortcuts, wherein the alignment adapts a model's generative distribution primarily over only its very first few output tokens. We refer to this issue as shallow safety alignment.", "abstract")]),
 (30, "https://arxiv.org/pdf/2609.18471v1", "First Token Matters: Understanding Safety Collapse in Large Reasoning Models",
  "The refusal signal is reported to collapse SPECIFICALLY at the first generated token, and specifically for reasoning models — i.e. exactly "
  "the token we read, in exactly the Qwen3/SmolLM3 class. Names it Onset Refusal Collapse.", ["Yizheng Yang"], 2026,
  [S("We find that the refusal-related signal of LRMs\ndrops sharply at the first generated token under harmful queries, which is associated with unsafe\nresponse generation.", "Abstract")]),
 (31, "https://arxiv.org/html/2605.29629", "Beyond Attack Success Rate: Temporal Logit Observability for LLM Safety Failures",
  "Three distinct refusal trajectories can produce the same behavioural label, so a single scalar outcome hides the mechanism.",
  ["Junyoung Park"], 2026,
  [S("The same ASR-success label can hide three fundamentally different processes: refusal evidence may have appeared briefly and been suppressed; refusal may have emerged but too late to matter; or refusal may never have been triggered at all", "Sec. 1")]),
 (32, "https://pmc.ncbi.nlm.nih.gov/articles/PMC13221392/", "Simple, robust and near optimal designs for the estimation of log-logistic dose response functions",
  "The practical dose-design benchmark that contradicts relying on the 2-point locally D-optimal design when the threshold and slope are "
  "unknown. Its worked 'Equally spaced III' design uses 8 central dose levels plus more heavily replicated minimum and maximum levels "
  "(28 observations once duplicates are required). NO EXACT PASSAGE IS CLAIMED: this host and the Springer version both return a blocked "
  "stub on repeat fetches, so the wording could not be re-verified on demand and the design is reported as a paraphrase, not a quote.",
  None, None, []),
 (33, "https://arxiv.org/html/2607.13346v1", "The Refusal Residue: When Probes Catch Alignment Faking and When They Don't",
  "A concrete, matched cross-family probe-transfer failure that sets the numeric bar: AUROC 0.87 on Llama collapsing to 0.43 (chance) on Qwen, "
  "with cross-model transfer null and Procrustes alignment null against a random-Procrustes baseline.", None, 2026,
  [S("Under leakage-free leave-one-query-out probing, detection survives on Llama (AUROC 0.87) but collapses to chance on Qwen (0.43), and cross-model transfer is null.", "abstract")]),
 (34, "https://arxiv.org/html/2608.02464v1", "Real-Time Detection and Repair of LLM Agent Failures",
  "A second matched calibration-transfer failure with numbers: 0.885 recalibrated-on-itself versus 0.527 (chance) deployed cross-family "
  "unchanged, over a matched task/tool/injector plan.", None, 2026,
  [S("monitors do not transfer across deployments without recalibration: calibrated on qwen2.5:7b and deployed on llama3.1:8b over a matched task, tool and injector plan, the primary monitor sits at chance (AUROC 0.527, healthy FA 0.75), while the _same_ target recalibrated on itself reaches 0.885", "Sec. 5")]),
 (35, "https://arxiv.org/pdf/2606.03280", "A Negative Result on Cross-Model Activation Transfer in a Pythia Multi-Hop Setting",
  "A structural caution for any cross-family axis claim: high representational similarity does not imply the representation is causally usable "
  "after transfer, and norm-rescaling does not rescue it.", None, 2026,
  [S("Replacement-style injection is consistently destructive, and rescaling translated vectors to the receiver hidden-state norm does not rescue performance", "abstract")]),
 (36, "https://arxiv.org/abs/2608.10621", "ProbGuard: Calibrated Safety Risk Estimation from LLM Output Distributions",
  "The most dangerous competitor found: a parent-free, architecture-agnostic calibrated safety-risk estimator from early-decoding logit "
  "distributions, with cross-family calibration gains already demonstrated across three families and three datasets.", None, 2026, []),
 (37, "https://arxiv.org/abs/2604.01473", "SelfGrader: LLM Jailbreak Detection via Anchored Token-Level Logits",
  "Occupies the phrase 'anchored token-level logits' almost verbatim: a one-forward-pass, generation-free, parent-free per-query harmfulness "
  "score calibrated via in-context anchor examples. Our differentiator must be model-level rather than per-query.", None, 2026, []),
 (38, "https://arxiv.org/abs/2502.10487", "Fast Proxies for LLM Robustness Evaluation",
  "The honest existing bar for 'few prompts, good cross-family safety estimate': a cheap single-generation proxy correlating with full "
  "red-team-ensemble ASR at rs=0.94 across six model families, with no parent model.", None, 2025, []),
 (39, "https://escapements.org/", "escapement — behavioral safety evaluation for open-weight AI models",
  "A live deployed public register producing per-checkpoint injection-compliance rates with CIs as a paired delta against each checkpoint's "
  "own base — what a platform actually runs today, and parent- and generation-dependent.", None, None, []),
 (40, "https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16", "openbmb/MiniCPM-2B-sft-bf16 model card and tokenizer_config.json",
  "The third way the global readout rule breaks: this template never reads add_generation_prompt, so the flag is a silent no-op. Also no "
  "safety language on the card and no same-generation abliterated sibling — hence dropped.", None, None, []),
 (41, "https://huggingface.co/datasets/bench-llm/or-bench", "bench-llm/or-bench dataset card",
  "The recommended fallback dose axis: CC-BY-4.0, ungated, three configs (80k / hard-1k / toxic) over 10 shared categories, with the "
  "prompt/category schema confirmed.", None, None,
  [S("or-bench-80k (80.4k rows)or-bench-hard-1k (1.32k rows)or-bench-toxic (655 rows)", "dataset viewer header"),
   S("cc-by-4.0", "License field")]),
 (42, "https://huggingface.co/datasets/sorry-bench/sorry-bench-202503", "sorry-bench/sorry-bench-202503 dataset card",
  "Fallback (ii), and the reason it is only fallback (ii): a custom licence that forbids redistribution and reserves a deletion right.", None, None,
  [S("our _base_ dataset consists of **440 unsafe instructions** in total, spanning across 44 finegrained safety categories (10 data points per category)", "Dataset Structure"),
   S("Prohibited Transfers: You **should not** distribute, copy, disclose, assign, sublicense, embed, host, or otherwise transfer the dataset to any third party.", "SORRY-Bench Dataset License Agreement")]),
 (43, "https://www.ndss-symposium.org/ndss2026/accepted-papers/", "NDSS 2026 accepted papers",
  "Confirmed the NDSS 2026 venue for 'Bleeding Pathways' against the official programme rather than the arXiv comment field.", None, 2026,
  [S("Bleeding Pathways: Vanishing Discriminability in LLM Hidden States Fuels Jailbreak Attacks", "accepted-papers listing")]),
 (44, "https://arxiv.org/abs/2406.14598", "SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal",
  "The 44-category / 440-instruction structure of the fallback axis.", None, 2024,
  [S("SORRY-Bench improves on this by using a fine-grained taxonomy of 44 potentially unsafe topics, and 440 class-balanced unsafe instructions, compiled through human-in-the-loop methods.", "abstract")]),
 (45, "https://arxiv.org/abs/2605.02236", "Perturbation Dose Responses in Recursive LLM Loops: Raw Switching, Stochastic Floors, and Persistent Escape under Append, Replace, and Dialog Updates",
  "A RESOLVED_MISMATCH: logged upstream as the 'ED50-for-LLM' occupant, it is actually about dose-response of injected text in recursive "
  "loops, so the claim that that lane is occupied is not supported by this ID.", ["Pawel Kaplanski"], 2026, []),
 (46, "https://arxiv.org/abs/2609.06934", "The Geometry of Refusal: Why Post-Hoc Safety Is Fragile and Pretraining-Time Safety Persists",
  "Disambiguated from arXiv:2502.17420: two distinct papers whose titles both begin 'The Geometry of Refusal'.",
  ["Srikanth Malla", "Chiho Choi", "Joon Hee Choi"], 2026, []),
 (47, "https://arxiv.org/abs/2502.17420", "The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence",
  "Contradicting evidence against a single refusal direction, and the other half of the 'Geometry of Refusal' title collision.", None, 2025,
  [S("Contrary to prior work, we uncover multiple independent directions and even multi-dimensional concept cones that mediate refusal.", "abstract")]),
 (48, "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv", "strongreject_dataset.csv",
  "The judge's prompt set, schema confirmed by fetch: category, source, forbidden_prompt.", None, None,
  [S("category,source,forbidden_prompt", "header row")]),
 (49, "https://arxiv.org/html/2510.14276v1", "Qwen3Guard Technical Report",
  "The technical report behind the recommended behavioural oracle.", None, 2025, []),
 (50, "https://ms.uky.edu/~mai/splus/IcensEM.pdf", "Turnbull's algorithm for interval- and right-censored data",
  "The citable remedy for a censored ED50: estimate an interval rather than forcing a point estimate.", None, None, []),
 (51, "https://statisticalhorizons.com/logistic-regression-for-rare-events/", "Logistic Regression for Rare Events (Firth penalised likelihood)",
  "The citable remedy for quasi-separated dose cells: a finite slope where plain MLE has none.", None, None,
  [S("penalized likelihood also has the attraction of producing finite, consistent estimates of regression parameters when the maximum likelihood estimates do not even exist because of complete or quasi-complete separation.", None)]),
 (52, "https://cran.r-project.org/web/packages/rstanarm/vignettes/pooling.html", "Hierarchical partial pooling",
  "The citable remedy for an unidentified slope: borrow it from the family while letting ED50 stay free.", None, None, []),
]

src_records = []
for idx, url, title, summ, authors, year, passages in sources:
    r = {"index": idx, "url": url, "title": title, "summary": summ, "supporting_passages": passages or []}
    if authors: r["authors"] = authors
    if year: r["year"] = year
    src_records.append(r)

answer = (W / "research_report.md").read_text()

fq = [
 "Does ProbGuard (arXiv:2608.10621) fit its calibration per model family or transfer a single fitted calibration to a held-out family? If it "
 "transfers, it is a direct competitor on the primary endpoint and must be RUN as a baseline rather than cited — the same decision that "
 "anchor projection (2605.09875) just forced, one paper later. This is the highest-priority remaining check.",
 "Does enable_thinking=False actually repair Onset Refusal Collapse (arXiv:2609.18471) on Qwen3, or does the collapse persist at the first "
 "GENERATED token even after the empty <think>\\n\\n</think>\\n\\n pair? This is ~10 minutes of forward passes and it decides whether Qwen — "
 "80% of Chinese-origin abliteration targets — is usable in the family-level analysis at all.",
 "What fraction of model-conditions on a graded harm ladder actually censor (all-refuse or all-comply, no identified 50% crossing), broken "
 "down by base / instruct / abliterated and by family? No LLM paper reporting this was found, so it is both the design's main threat and a "
 "free contribution that falls out of the run whether the primary endpoint succeeds or fails.",
 "Are SelfGrader's (arXiv:2604.01473) in-context anchor examples fixed across models — which would make it input-anchored like the proposal — "
 "or refitted per model, which would make it a calibrated method and a weaker overlap?",
]

out = {"title": "Prior art and specs for a cheap safety metric",
       "summary": (W / "summary.txt").read_text().strip(),
       "layman_summary": ("Checks what has already been published about measuring an AI model's safety cheaply, then writes down the exact "
                          "recipes, prompts and settings the next experiment needs so it can build instead of search."),
       "answer": answer, "sources": src_records, "follow_up_questions": fq, "spec_sheet": spec}
(W / "research_out.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("research_out.json", (W / "research_out.json").stat().st_size, "bytes;", len(src_records), "sources")
# citation coverage check
cited = sorted({int(n) for m in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer) for n in m.replace(" ", "").split(",")})
have = {s["index"] for s in src_records}
print("cited:", cited)
print("cited but MISSING from sources:", sorted(set(cited) - have))
print("listed but never cited:", sorted(have - set(cited)))
