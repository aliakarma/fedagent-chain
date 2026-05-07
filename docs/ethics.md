# Ethical Considerations

## Overview

FedAgent-Chain is a research prototype for disability-inclusive employment AI.
This document outlines the ethical principles embedded in the framework design,
the limitations of the current prototype, and the requirements for any future
real-world deployment.

## Core Ethical Principles

### 1. Privacy by Design

- Raw disability data **never leaves** the local institutional node.
- Only cryptographic hashes of differentially private model updates are shared.
- The blockchain stores metadata and hashes only — no personal data on-chain.
- Consent validation is enforced programmatically: records with `consent_given=False`
  are excluded from training before any processing begins.

### 2. Fairness as a First-Class Objective

- The fairness disparity measure D_fair is explicitly minimized during training.
- Per-group performance is monitored across disability categories, language groups,
  work modes, and regional nodes throughout every federated round.
- Nodes with underrepresented groups receive higher aggregation weight,
  preventing majority-group dominance of the global model.

### 3. Human-in-the-Loop Governance

- The governance agent is **non-bypassable** for high-risk decisions.
- Any recommendation with risk score R_risk(d_i) > τ = 0.70 **must** be
  reviewed by a qualified human employment advisor before being acted upon.
- The system is designed as an assistive tool, never as an autonomous decision maker.

### 4. Disability Rights Alignment

- The framework is aligned with the UN Convention on the Rights of Persons with
  Disabilities (CRPD) Article 27 (Right to Work).
- Accommodation recommendations are grounded in the WHO-ICF framework.
- The system must not be used to restrict employment opportunities for any
  person with a disability.

## Prohibited Uses

The following uses of FedAgent-Chain are explicitly prohibited:

1. **Automated rejection**: The system must NEVER automatically reject a
   person with a disability's employment application without human review.
2. **Unsupervised deployment**: The system must not be deployed without
   qualified human oversight at the governance layer.
3. **Real data without IRB approval**: Using real disability-related data
   requires institutional ethics review board (IRB/REC) approval and
   informed consent from all participants.
4. **Discrimination amplification**: The system must not be used to screen
   out candidates based on disability category.

## Prototype Limitations

1. **Synthetic data**: All evaluation uses fictitious data. Performance on
   real-world data may differ significantly.
2. **Simulated blockchain**: The prototype uses an in-memory chain.
   Production deployment requires a hardened permissioned blockchain
   (e.g., Hyperledger Fabric).
3. **Four-node simulation**: Real deployments may involve dozens of institutions
   with more complex data heterogeneity than the simulation captures.
4. **Language coverage**: The multilingual encoder's performance on low-resource
   languages (e.g., minority languages within regional nodes) is not validated.

## Requirements for Real-World Deployment

Before any real-world deployment, the following are **mandatory**:

- [ ] Ethics Review Board (IRB/REC) approval at every participating institution
- [ ] Informed consent protocol reviewed by disability rights advocates
- [ ] Data sharing agreements compliant with GDPR (EU), PDPL (Saudi Arabia),
      PIPL (China), and applicable US state privacy laws
- [ ] Independent security audit of the blockchain and DP implementation
- [ ] Fairness audit on real data with domain experts and affected community representatives
- [ ] Human oversight protocols documented and trained
- [ ] Incident response plan for harmful or biased recommendations

## Contact

For ethical concerns about this research, contact: [ethics@fedagent-chain.org]
