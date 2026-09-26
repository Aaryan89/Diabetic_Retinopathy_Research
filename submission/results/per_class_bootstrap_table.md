# Per-Class Bootstrap Performance (95% Confidence Intervals, 1000 resamples)

| Class Grade   | Model Variant                                 | Recall / Sensitivity    | Precision            | F1-Score                |
|:--------------|:----------------------------------------------|:------------------------|:---------------------|:------------------------|
| 0 (No DR)     | Variant A (Softmax Baseline)                  | 97.1% [94.9%, 98.9%]    | 96.7% [94.5%, 98.6%] | 0.9690 [0.9541, 0.9825] |
| 0 (No DR)     | Variant B (CORAL Ordinal)                     | 100.0% [100.0%, 100.0%] | 74.9% [70.7%, 79.3%] | 0.8560 [0.8281, 0.8843] |
| 0 (No DR)     | Variant C (Continuous Regression)             | 96.7% [94.4%, 98.6%]    | 96.0% [93.5%, 98.1%] | 0.9635 [0.9478, 0.9779] |
| 0 (No DR)     | Variant CORN (Conditional Ordinal + Soft-QWK) | 97.4% [95.4%, 99.2%]    | 95.7% [93.3%, 97.9%] | 0.9655 [0.9502, 0.9799] |
| 1 (Mild)      | Variant A (Softmax Baseline)                  | 55.5% [42.4%, 68.4%]    | 59.7% [46.8%, 73.3%] | 0.5728 [0.4602, 0.6777] |
| 1 (Mild)      | Variant B (CORAL Ordinal)                     | 1.8% [0.0%, 6.1%]       | 42.3% [0.0%, 100.0%] | 0.0344 [0.0000, 0.1132] |
| 1 (Mild)      | Variant C (Continuous Regression)             | 50.2% [37.7%, 63.6%]    | 50.9% [37.5%, 63.5%] | 0.5035 [0.3860, 0.6087] |
| 1 (Mild)      | Variant CORN (Conditional Ordinal + Soft-QWK) | 66.1% [52.6%, 79.2%]    | 47.6% [36.5%, 59.3%] | 0.5515 [0.4409, 0.6491] |
| 2 (Moderate)  | Variant A (Softmax Baseline)                  | 74.6% [67.3%, 81.9%]    | 76.3% [69.0%, 83.1%] | 0.7538 [0.6978, 0.8053] |
| 2 (Moderate)  | Variant B (CORAL Ordinal)                     | 11.9% [7.0%, 17.5%]     | 71.9% [52.9%, 88.9%] | 0.2039 [0.1241, 0.2857] |
| 2 (Moderate)  | Variant C (Continuous Regression)             | 68.5% [60.6%, 75.8%]    | 72.6% [64.4%, 80.0%] | 0.7044 [0.6370, 0.7638] |
| 2 (Moderate)  | Variant CORN (Conditional Ordinal + Soft-QWK) | 28.0% [21.1%, 35.3%]    | 81.0% [69.8%, 91.3%] | 0.4151 [0.3284, 0.5000] |
| 3 (Severe)    | Variant A (Softmax Baseline)                  | 41.7% [23.3%, 60.0%]    | 35.4% [20.0%, 51.6%] | 0.3791 [0.2221, 0.5313] |
| 3 (Severe)    | Variant B (CORAL Ordinal)                     | 0.0% [0.0%, 0.0%]       | 0.0% [0.0%, 0.0%]    | 0.0000 [0.0000, 0.0000] |
| 3 (Severe)    | Variant C (Continuous Regression)             | 38.4% [20.0%, 56.8%]    | 19.1% [9.2%, 29.0%]  | 0.2528 [0.1333, 0.3704] |
| 3 (Severe)    | Variant CORN (Conditional Ordinal + Soft-QWK) | 65.7% [48.3%, 82.9%]    | 17.3% [10.6%, 24.8%] | 0.2724 [0.1746, 0.3735] |
| 4 (PDR)       | Variant A (Softmax Baseline)                  | 52.3% [37.2%, 67.3%]    | 51.2% [36.6%, 65.1%] | 0.5147 [0.3830, 0.6377] |
| 4 (PDR)       | Variant B (CORAL Ordinal)                     | 86.3% [75.6%, 95.1%]    | 23.6% [17.2%, 30.2%] | 0.3692 [0.2857, 0.4502] |
| 4 (PDR)       | Variant C (Continuous Regression)             | 27.1% [14.3%, 41.7%]    | 54.6% [33.3%, 75.0%] | 0.3589 [0.2034, 0.5067] |
| 4 (PDR)       | Variant CORN (Conditional Ordinal + Soft-QWK) | 41.0% [26.2%, 55.1%]    | 52.8% [36.8%, 69.4%] | 0.4584 [0.3124, 0.5909] |