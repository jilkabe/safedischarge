# <Project Name> — System Design Report

**Team:** <team name>
**Members:** <name> (<email>), …

Page budget (10 pages total, Arial 10pt): suggested split in brackets.

## 1. Introduction (~1 page)
- Problem and who it affects; why it matters (evidence, figures)
- Why existing approaches fall short
- Our solution in one paragraph; contributions

## 2. Application Description (~2 pages)
- Users and stakeholders
- Key user journeys / workflow
- System architecture diagram (data in → AI components → decisions/outputs → users)
- Data sources and what each is used for

## 3. Use of AI Technologies (~3 pages)
For each step of the workflow:
- Task (prediction, ranking, optimisation, generation, …)
- Technique chosen and why (vs alternatives)
- Inputs / outputs, how it is trained and evaluated
- Human-in-the-loop points

## 4. Responsible AI Practices Addressed (~3 pages)
Cover all three for the bonus; each must name a concrete mechanism in the design.
- **Explainability** — e.g. SHAP/LIME, counterfactual explanations, interpretable models, user-facing rationale
- **Fairness** — protected attributes, fairness metric (demographic parity, equal opportunity, …), mitigation (pre/in/post-processing), monitoring
- **Privacy preservation** — federated learning, differential privacy, secure aggregation, data minimisation, consent
- Trade-offs between them and with accuracy

## 5. Conclusion and Limitations (~0.5 page)

## References (~0.5 page)
