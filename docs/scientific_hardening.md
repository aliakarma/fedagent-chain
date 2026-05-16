# 🛡️ Threats to Validity

While FedAgent-Chain demonstrates strong performance in a controlled simulation, the following factors should be considered when interpreting these results for real-world deployment:

## 1. Synthetic Data Limitations
The datasets used in this study are synthetically generated using a rule-based oracle. While we have calibrated the distributions (e.g., disability rates, language prevalence) against WHO and World Bank statistics, synthetic data may fail to capture the complex, non-linear correlations and edge cases present in real-world employment records.

## 2. Institutional Node Count
Our current evaluation is restricted to $K=4$ regional nodes (Saudi Arabia, USA, China, Europe). In a global production environment, the number of nodes could be significantly higher, which may introduce greater heterogeneity and communication latency.

## 3. Local Model Simplicity
We utilize a Multilayer Perceptron (MLP) architecture to ensure efficient training on regional nodes. More complex architectures (e.g., Transformers for skill-text analysis) might offer higher accuracy but would increase the systems overhead profiled in Phase 3.

## 4. Fairness Metric Sensitivity
Our fairness-aware aggregation focuses on the **Disability Category** as the primary sensitive attribute. The framework's behavior under intersectional fairness (e.g., combining disability with age or gender) remains a subject for future investigation.

---

# ⚖️ Ethical Considerations

The deployment of AI in disability-inclusive recruitment carries significant ethical responsibilities:

## 1. Algorithmic Bias
Federated Learning does not inherently eliminate bias. If the local data at an institutional node contains historical human bias, the global model may inadvertently learn and propagate these patterns. Our λ-fairness penalty is a mitigation strategy, not a complete solution.

## 2. Privacy-Utility Tradeoff
The use of Differential Privacy (DP) introduces a tradeoff. Higher noise multipliers ($\sigma$) provide stronger privacy guarantees but can degrade the accuracy of suitability matching. We recommend a $\sigma=0.1$ setting as a baseline, but this must be calibrated by policy-makers based on local regulation (e.g., GDPR, NDMO).

## 3. Human-in-the-Loop Governance
The **Governance Agent** is designed to flag high-risk recommendations. We strictly advise against fully autonomous decision-making in high-stakes recruitment. Final hiring decisions should always involve a human reviewer who can interpret the "Audit Comment" provided by the Governance Agent.
