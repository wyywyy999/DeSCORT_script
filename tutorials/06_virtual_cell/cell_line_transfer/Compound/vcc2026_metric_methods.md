# Cell Painting evaluation: implementation-matched Methods

Audited on 2026-11-11 against `tutorials/06_vc/vcc2026-metrics.pdf`, the local `04_calculate_vcc2026_metric.sh/.py`, and `src/descort/vc/fast_cp_metrics.py`, `inference.py`, and `metric_cli.py`.

## Manuscript draft

### Evaluation of predicted morphological responses

We evaluated predicted Cell Painting profiles using perturbation discrimination, mean squared error (MSE), direction fidelity, direction reach, significant-feature Jaccard similarity, and normalized mean absolute effect error (NMAE). These metrics were adapted from the VCC2026 framework for continuous morphological features rather than transcript counts. Calculations used the supplied preprocessed feature values directly, without count normalization, logarithmic transformation, or low-expression filtering. Predicted and measured datasets were required to have identical feature names and order, perturbation labels, and sample identifiers and order. Exact duplicate control profiles were removed within each dataset before evaluation. Compounds were identified by their SMILES strings, with dimethyl sulfoxide (`CS(=O)C`) designated as the control. The evaluation used all supplied perturbation profiles and features without additional subsampling.

For each perturbation, predicted and measured profiles were summarized by arithmetic means across observations. Perturbation discrimination score (PDS) was computed by ranking the cosine distance between each predicted mean response and its corresponding measured response among distances to all measured perturbations. Responses were centered on the measured control mean. Ties received midranks, and the score was defined as one minus the zero-based rank divided by the number of perturbations minus one. MSE was the mean squared difference between predicted and measured mean profiles across features. Both metrics were averaged with equal weight across perturbations.

Differential morphological features were identified separately in predicted and measured data using two-sided Mann–Whitney U tests against the corresponding dataset's control profiles. Tests used the asymptotic normal approximation with tie correction and without continuity correction. Benjamini–Hochberg correction was applied separately across all features within each perturbation, and features with adjusted P < 0.05 were considered significant. Effect sizes were defined as arithmetic mean differences from control, retaining the sign of the morphological response, rather than as log fold changes. In the audited evaluation, predicted and measured files contained identical control profiles; consequently, both differential analyses used the same control observations after duplicate removal.

Direction fidelity was the number of predicted significant features with correctly predicted nonzero effect signs divided by the larger of the number of adjudicable predicted significant features and the number of measured significant features with finite, nonzero effects. A feature was adjudicable when its measured effect was finite and nonzero; a predicted zero effect counted as an incorrect directional call. A zero denominator was assigned a fidelity of zero. Direction reach assessed how deeply a prediction's confidence ranking retained at least 90% directional agreement within the measured significant-feature pool. Features were ranked by predicted significance, adjusted P, nominal P, absolute predicted effect, and feature name. The deepest qualifying prefix was divided by the number of measured significant features with finite, nonzero effects; perturbations with an empty directional reference budget were excluded from this metric's aggregate.

Significant-feature agreement was calculated as the Jaccard similarity between predicted and measured significant-feature sets, assigning a value of one when both sets were empty. NMAE was computed over measured significant features with finite effects as the sum of absolute differences between predicted and measured effects divided by the sum of absolute measured effects. Missing predicted effects were set to zero. NMAE was defined only when at least ten reference features met this criterion and the denominator was nonzero. Direction fidelity, direction reach, Jaccard similarity, and NMAE were averaged with equal weight over perturbations with finite metric values. Raw metric values were reported without the baseline–replicate rescaling or composite scoring used in the VCC2026 competition. Higher PDS, direction fidelity, direction reach, and Jaccard similarity indicate better performance; lower MSE and NMAE indicate better performance.

### Optional paragraph: normalized noise-corrected error

The evaluator additionally reported a normalized, sampling-noise-corrected squared error in continuous morphology space. For each measured or predicted perturbation group, the sampling correction was the sum of unbiased feature variances divided by the number of observations, with a correction of zero for groups containing fewer than two observations. The corrected prediction-error numerator subtracted the measured correction and the smaller of the measured and predicted corrections. The response-distance denominator subtracted the measured perturbation and measured control corrections from their squared mean-profile distance. Both quantities were divided by the feature count, and the final normalized error was the ratio of their sums over perturbations, rather than the mean of per-perturbation ratios. No panel-level predicted-spread factor or score clipping was applied. This quantity was reported separately from ordinary MSE.

## Equations and exact conventions

Let K be the number of non-control perturbations and F the number of morphological features. For perturbation p, let mu_p and muhat_p be measured and predicted mean vectors; c and chat are the control means in the measured and predicted files. Define delta_p = mu_p - c and deltahat_p = muhat_p - chat. PDS specifically uses muhat_p - c, irrespective of chat. In the audited data, c = chat.

1. PDS: d_pq = 1 - cosine(muhat_p - c, mu_q - c); k_p = #{q: d_pq < d_pp} + (#{q: d_pq = d_pp} - 1)/2; PDS_p = 1 - k_p/(K-1). For K = 1, the implementation returns 1. Aggregate: mean over perturbations.
2. MSE_p = (1/F) sum_f (muhat_pf - mu_pf)^2. Aggregate: mean over perturbations. This is the `mse` output, not `expr_mse_unbiased_capped_norm`.
3. Let R_p = {f: q_real,pf < 0.05}, P_p = {f: q_pred,pf < 0.05}. Let A_p = {f in P_p: delta_pf is finite and nonzero}; B_p = {f in R_p: delta_pf is finite and nonzero}. Fidelity_p = #{f in A_p: sign(deltahat_pf) = sign(delta_pf), deltahat_pf != 0} / max(|A_p|, |B_p|). If the denominator is zero, return 0, not missing.
4. Reach uses H_p = {f in R_p: delta_pf is finite} as the ranked pool, but |B_p| as the denominator. Sort predicted significance first, then q_pred ascending, nominal P_pred ascending, |deltahat| descending, and feature name ascending. Missing predicted q/P/effect are replaced with 1/1/0. A match requires nonzero predicted effect and equal effect signs. Take the deepest prefix whose cumulative match fraction is >= 0.9; use depth zero if no prefix qualifies. Divide by |B_p|, or return missing if |B_p| = 0. The implementation does not remove zero measured effects from H_p before calculating prefix length. This differs from the PDF's strictly adjudicable ranking convention when a significant zero-effect feature occurs. No such features were present in the audited reference DE cache.
5. Jaccard_p = |R_p intersect P_p| / |R_p union P_p|; empty union = 1.
6. For H_p as above, NMAE_p = sum_{f in H_p} |deltahat_pf - delta_pf| / sum_{f in H_p} |delta_pf|, requiring |H_p| >= 10 and a finite, nonzero denominator. Missing/NaN predicted effects are filled with zero. Input matrices containing NaN/Inf are rejected before testing.
7. For the optional normalized error, define C_p = sum_f s_pf^2/n_p, Chat_p similarly, and C_ctrl for measured controls. N_p = [||muhat_p-mu_p||^2 - min(Chat_p,C_p) - C_p]/F; D_p = [||mu_p-c||^2 - C_p - C_ctrl]/F. The reported aggregate is sum_p N_p / sum_p D_p, returning missing when the denominator sum is zero. Negative corrections/ratios are not clipped. Unlike the PDF's rule-version-3 definition, the implementation has no panel-spread correction factor rho.

DE metrics contain a helper that removes a feature only when its name exactly equals the perturbation label. In the audited compound run, perturbation labels are SMILES and feature names are Cell Painting measurements, so this excludes no features. PDS and the error metrics use the full morphology feature axis.

## PDF versus implementation

| Item | Local VCC2026 PDF | Current Cell Painting implementation |
|---|---|---|
| Input domain | Raw nonnegative integral RNA counts | Finite, signed continuous morphology values in the supplied preprocessing space |
| Mean profiles | Sum counts, normalize to 50,000, log-transform | Arithmetic mean of supplied features; no extra transformation |
| PDS rank | K-1 denominator and midrank ties | Same ranking convention; morphology-space distances; no panel target-transcript removal |
| Error metric | Noise-corrected normalized expression error | Same-named continuous-space counterpart plus an additional ordinary MSE |
| Sampling correction | Delete-one jackknife for transformed count profiles | Sum of unbiased feature variances divided by sample count for arithmetic means |
| Panel-spread safeguard | rho multiplies min(predicted correction, measured correction) | rho is absent; effectively fixed to 1 |
| DE test | Two-sided rank-sum on per-cell CPM | Two-sided tie-corrected asymptotic MWU directly on supplied features, no continuity correction |
| DE feature filter | Control mean >5 CPM before BH | No CPM/low-expression filter; all supplied features enter BH |
| DE effect | log2 fold change with epsilon = 1e-9 | Signed perturbation-minus-control arithmetic mean difference |
| FDR | BH within each perturbation, q < 0.05 | Same default; optional global scope exists but is not used by this preset |
| Prediction controls | Measured controls required for both DE analyses | Each input file's controls are used; audited input control matrices are identical, so equivalent here |
| Own target gene | Excluded; PDS removes all panel targets | No analogous morphology coordinate; exact-label DE exclusion has no effect for the audited compound data |
| Fidelity, zero denominator | Undefined; omitted | Returns 0 and includes it in the average |
| Reach, zero reference effects | Only adjudicable features count in prefixes | Finite zero-effect significant features stay in the ranked pool, but not its denominator |
| Final reporting | Six metrics calibrated using baseline and replicate anchors, then averaged, with specified clipping | Seven separate raw outputs; optional PM evaluation is not anchor calibration; no competition composite score |

## Audit of the existing 100-compound fine-tuning output

This is a check of existing files, not a new evaluation run. The 100-compound designation describes the fine-tuning set, not the evaluation panel size.

- Existing run manifest: `tutorials/06_vc_compound/result/cell_line_transfer/Morgan_few_shot/A549_to_U2OS/epoch_20/100_gene/DeSCORT/run_manifest.json`.
- Scored panel: 206 non-control compounds and 2,297 features.
- Each input originally has 1,691 control rows; exact-profile duplicate removal retains 482. There are 1,691 non-control rows, giving 2,173 evaluation rows per file after deduplication.
- Directly compared all 1,691 x 2,297 control values in the two input files: exactly equal.
- No morphology feature name equals any perturbation label.
- Valid counts in the stored results: PDS/MSE/fidelity/Jaccard = 206; reach = 180; NMAE = 177. The normalized error is an aggregate ratio, even though its output also contains per-perturbation intermediate ratios.
- The reference has 26 perturbations with no significant features and zero significant features with exactly zero mean effects.
- SHA-256 of the current `inference.py` matches the metric-source hash stored with that result's DE cache.
- Existing reported outputs: PDS 0.8358986502; ordinary MSE 10.6704039776; normalized corrected error 0.2859047967; fidelity 0.6034697095; reach 0.5801830801; Jaccard 0.3178885287; NMAE 0.7221264986.

## Chinese author notes and terminology

- 推荐写作定位：**Cell Painting metrics adapted from VCC2026**，不能声称完全复现官方 RNA competition score。
- 对形态特征使用 differential morphological features / differential-feature analysis；不要把形态特征称为 DE genes。
- `de_wilcoxon_lfc_nmae` 是保留的代码字段名；论文中应称 normalized mean absolute effect error，不应描述成 log2-fold-change error。
- 主图若读取 `mse`，应使用普通均值 MSE 的定义。只有确实报告 `expr_mse_unbiased_capped_norm` 时才加入上面的 optional paragraph。
- Methods 中不能写 pooled/global BH；当前 preset 是 per-perturbation BH。
- 当前 PM baseline 仅单独评分，不提供官方 baseline–replicate 两个锚点，不用于 s=(u-b)/(r-b) 校准。
- 本文的共同对照陈述已对上述 DeSCORT 文件核实；扩展到所有模型、数据划分前，需要核对各预测文件内实际 control 值，而不仅是 control 标签。
- 这里未更改指标代码、未重算生产指标。报告忠实描述现有实现，包括与 PDF 不一致的边界情况。

| Canonical manuscript term | Code field | Meaning |
|---|---|---|
| PDS | pds_cosine | Perturbation discrimination score |
| MSE | mse | Mean-profile mean squared error |
| Direction fidelity | de_wilcoxon_direction_fidelity_yield_raw | Coverage-penalized directional agreement |
| Direction reach | de_wilcoxon_direction_reach_raw | Deepest prefix maintaining 90% directional purity |
| Significant-feature Jaccard similarity | de_wilcoxon_sig_jaccard | Significant-feature set agreement |
| NMAE | de_wilcoxon_lfc_nmae | Normalized mean absolute effect error |
| Normalized corrected error | expr_mse_unbiased_capped_norm | Optional continuous-space ratio of corrected squared-distance sums |

Claim–evidence mapping: profile aggregation, PDS, ordinary MSE, DE definitions, FDR and aggregation are supported by `src/descort/vc/inference.py`; input handling, duplicate removal, fast MWU, seven-output selection and reporting are supported by `src/descort/vc/fast_cp_metrics.py`; preset choices are supported by `04_calculate_vcc2026_metric.sh`; official comparisons are supported by the supplied PDF (printed pp. 1–15). Final manuscript wording should cite the verified VCC2026 source; bibliographic authors/DOI are not supplied by this PDF and are not invented here.
