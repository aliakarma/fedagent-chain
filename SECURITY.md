# Security Policy

## Supported Versions

| Version | Security Updates |
|---------|-----------------|
| 1.0.x   | ✅ Yes           |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities via GitHub Issues.**

Email security concerns privately to: **security@fedagent-chain.org**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- (Optional) Suggested fix

You will receive acknowledgement within 48 hours and a status update within 7 days.

## Scope

Security concerns relevant to this repository include:
- Vulnerabilities that could expose user data or model parameters
- Cryptographic weaknesses in the blockchain hashing implementation
- Dependencies with known CVEs in `requirements.txt`
- Code that could enable membership inference attacks on model updates

## Out of Scope

- Issues with the synthetic dataset (it contains no real personal data)
- Vulnerabilities in external services (MLflow, HuggingFace Hub)
- Issues requiring physical access to a deployment environment

## Privacy Note

FedAgent-Chain is a research prototype. The synthetic data contains no real
personal information. Any real-world deployment must undergo independent
security review before handling real disability-related data.
