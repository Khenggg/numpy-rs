# Masked mean, variance and standard deviation parity

The reference is the checked-in upstream `numpy/ma/core.py`: `MaskedArray.mean`, `sum`, `var`, `_DomainedBinaryOperation`, `power`, and `sqrt`. These details were verified with the installed NumPy 2.5.4 oracle. They assume an explicit mask, as used by the Rust `MaskedArray`; NumPy's separate `nomask` identity can select a different implementation.

## Mean

For each reduction lane, count the original unmasked values, sum those values and divide by the count. A zero count is masked. A nonfinite result is masked when the result is an array; a scalar mean can retain an unmasked NaN or infinity. In particular, a `keepdims=True` mean of a nonscalar input has array shape, even when reducing all axes.

## Variance

1. Compute the mean with **keepdims=True**. The mean mask is true when there are no unmasked values, or when that kept-dimension mean is nonfinite. For scalar input, its mean is still scalar, so the nonfinite rule follows the scalar mean behavior above.
2. Form anomalies `x - mean` with masks `original_mask OR mean_mask`. For real inputs, square these anomalies without automatically masking overflow. For complex inputs, `abs(anomaly) ** 2` calls masked `power`; nonfinite squared magnitudes are masked and excluded.
3. Sum valid squared anomalies, filling masked terms with zero. Retain whether **all anomalies** were masked, separately from the original count.
4. The denominator is signed `original_count - ddof`. Do not clamp it to zero. Divide the numerator by that denominator.
5. The initial division mask is `all_anomalies_masked OR nonfinite(quotient) OR abs(numerator)*f64::MIN_POSITIVE >= abs(denominator)`. The last term reproduces `_DomainSafeDivide`.
6. For a scalar output, keep this division mask. For an array output, NumPy replaces it with `all_original_values_masked OR denominator <= 0`. This replacement can expose an underlying zero or infinity. When masked division restores underlying data in the array case, the data is the numerator; otherwise it is the quotient.

An array with an unmasked NaN can consequently have an unmasked zero variance, while the same values reduced to a scalar are masked. This follows the current reference implementation; applying a blanket nonfinite-result mask does not reproduce it.

## Standard deviation

First compute variance as above, then apply masked `sqrt`. Its mask combines the variance mask, a negative variance, and a nonfinite square root. Preserve the variance's underlying data where sqrt is masked; NumPy's unary masked operation restores its input at those positions. Negative zero is a valid square-root input.

## Regression vectors

All examples explicitly use `mask=False`:

| Input | Operation | Data | Mask |
| --- | --- | --- | --- |
| `[NaN, 1]` | mean() | NaN | false |
| `[[NaN,1],[NaN,2]]` | mean(axis=1) | NaN, NaN | true, true |
| `[[NaN,1],[NaN,2]]` | var(axis=0) | 0, 0.25 | false, false |
| `[[NaN,1],[NaN,2]]` | var(axis=1) | 0, 0 | false, false |
| `[[NaN,1],[NaN,2]]` | var() | unspecified masked data | true |
| `[[1e200,-1e200],[1e200,-1e200]]` | var(axis=1) | inf, inf | false, false |
| `[[1e200,-1e200],[1e200,-1e200]]` | std(axis=1) | inf, inf | true, true |
| `[1,3]` | var(ddof=3) | -2 | false |
| `[1,3]` | std(ddof=3) | unspecified masked data | true |
| `[1]` | var(ddof=3) | -0 | false |
| `[1]` | std(ddof=3) | -0 | false |
| `[[1e200+1j,-1e200+1j],[1+1j,2+1j]]` | var() | 0.40625 | false |
| `[[1e200+1j,-1e200+1j],[1+1j,2+1j]]` | var(axis=1) | 0, 0.25 | false, false |

Run the extended real-valued differential corpus with `run_review.py --only masked`. It checks shape, mask, and values only at unmasked positions. Differences in masked underlying data are not a correctness failure in that corpus.
