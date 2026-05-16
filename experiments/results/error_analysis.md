# Phase 4 — Error Analysis & Dataset Transparency

## 1. Europe Node Performance Gap

The **Europe** node consistently shows lower performance (F1 ~0.55) compared to the United States and China nodes. This is primarily due to a **distributional shift** in the synthetic data:

- **Label Balance**: While other nodes have a 55-60% suitability rate, the Europe node has only **38.1%**.
- **Model Bias**: The global federated model, being trained on a majority of nodes with high suitability rates, develops a bias towards predicting "Suitable" (Class 1).
- **Impact**: This leads to a high number of **False Positives** in the Europe node, as the model over-estimates suitability for European candidates based on global patterns that don't apply as strongly to the local European criteria.

## 2. Local Baseline: High Recall, Low Precision

Reviewers may notice that the **Local Baseline** often exhibits high recall but very low precision. 

- **The Majority Class Trap**: In nodes with smaller local datasets or skewed balances, the local model often defaults to predicting the majority class to quickly minimize training loss. 
- **Result**: If a node has 60% suitable candidates, a model that predicts "Suitable" for everyone achieves 60% accuracy and 100% recall, but suffers from 0% precision on the unsuitable class. FedAgent-Chain's global aggregation corrects this by providing a more generalized feature representation.

## 3. Subgroup Weaknesses

The **Disability Category** remains the most challenging attribute for fairness. 
- **Multiple Disabilities**: Candidates with "multiple" disabilities often have lower match scores due to the higher complexity of accommodation requirements.
- **Mitigation**: FedAgent-Chain's λ-penalty specifically targets this by increasing the aggregation weight for nodes that perform poorly on these minority subgroups, forcing the global model to "pay more attention" to these hard-to-classify edge cases.

## 4. Confusion Matrix Insights

Comparing the confusion matrices:
- **Standard FedAvg**: Shows a tendency to cluster predictions around the 0.5 threshold, leading to "noisy" classification near the decision boundary.
- **FedAgent-Chain**: Shows more distinct separation in high-confidence predictions, though it still struggles with the False Positive rate in the conservative Europe node.
