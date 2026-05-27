# Comprehensive Guideline for Writing Scientific and Technical Documents using the Nunamaker Methodology

## 1. General Principles

Scientific writing must be:
- precise,
- evidence-driven,
- reproducible,
- scientifically defensible,
- and logically structured.

A scientific document should continuously connect:
Motivation → Problem Statements → Research Questions → Research Objectives → Design → Implementation → Evaluation → Contributions.

The recommended methodology is the Nunamaker methodology:
1. Observation
2. Theory Building
3. System Development
4. Evaluation

---

# 2. Global Structure of the Scientific Document

| Chapter | Objective | Nunamaker Phase |
|---|---|---|
| Chapter 1 | Introduction and Anchoring | Problem Definition |
| Chapter 2 | State of the Art and Technology | Observation |
| Chapter 3 | Design and Modeling | Theory Building |
| Chapter 4 | Implementation / Proof of Concept | System Development |
| Chapter 5 | Evaluation | Experimentation & Evaluation |
| Chapter 6 | Conclusion and Future Work | Scientific Contribution |

---

# 3. Chapter 1 — Introduction and Anchoring

## 3.1 General Background and Trends

Introduce:
- the scientific domain,
- emerging technologies,
- and relevant trends.

Examples:
- AI,
- LLMs,
- NLP,
- Multi-Agent Systems,
- Information Retrieval,
- Knowledge Graphs,
- Speech Translation,
- Cloud Computing.

All claims must be supported with scientific references from:
- IEEE,
- ACM,
- Springer,
- Elsevier,
- ACL,
- EMNLP,
- NeurIPS,
- ICML.

---

## 3.2 Motivation

The motivation explains:
- why the research matters,
- existing societal or technical limitations,
- and the relevance of the topic.

Students should define:
- MS1,
- MS2,
- and optionally MS3.

The motivation should reference:
- prior projects,
- scientific literature,
- industrial challenges,
- and infrastructure limitations.

---

## 3.3 Overall Goal of the Research

After the motivation, the manuscript must include one concise sentence describing the overall goal of the research.

Example:

“The overall goal of this research is to develop an AI-powered multilingual platform for low-resource African languages operating under low-connectivity conditions.”

The subsequent Problem Statements define the scientific and technical challenges that must be solved to achieve this overall goal.

---

## 3.4 Problem Statements

Problem Statements (PS) define:
- unresolved limitations,
- scientific gaps,
- technical challenges,
- or missing systems.

Students should define:
- PS1,
- PS2,
- and optionally PS3.

Each PS should:
- correspond to a motivation statement,
- be concise,
- and be scientifically defensible.

---

## 3.5 Research Questions

The document should contain:
- minimum 2 research questions,
- maximum 3 research questions.

Each Research Question (RQ):
- must correspond to a Problem Statement,
- must be measurable,
- must target an informatics or computational outcome,
- and must not be a yes/no question.

---

## 3.6 Research Methodology

Introduce the Nunamaker methodology and explain:
- Observation,
- Theory Building,
- System Development,
- Evaluation.

---

## 3.7 Research Objectives (VERY IMPORTANT)

Each Research Question must produce exactly four primary Research Objectives.

| Research Question | Required Objectives |
|---|---|
| RQ1 | O1.1.O, O1.2.TB, O1.3.I, O1.4.E |
| RQ2 | O2.1.O, O2.2.TB, O2.3.I, O2.4.E |
| RQ3 (optional) | O3.1.O, O3.2.TB, O3.3.I, O3.4.E |

Types:
- O = Observation Goal
- TB = Theory Building Goal
- I = Development & Implementation Goal
- E = Experimentation & Evaluation Goal

---

## 3.8 Approach and Structure of the Work

Objectives are grouped by Nunamaker phase:

| Objective Type | Chapter |
|---|---|
| /O | Chapter 2 |
| /TB | Chapter 3 |
| /I | Chapter 4 |
| /E | Chapter 5 |

---

# 4. Chapter 2 — State of the Art and Technology

The objective is to:
- analyze literature,
- investigate technologies,
- identify limitations,
- and formulate Remaining Challenges (RCs).

Requirements:
- minimum 30 scientific references,
- 5–7 thematic subsections,
- scientific synthesis instead of isolated summaries.

Each subsection should:
- correspond to Observation Objectives,
- end with a Remaining Challenge (RCx),
- and formulate one or more corresponding requirements derived from the RC.

Example:

“However, existing multilingual systems remain insufficiently adapted to low-resource African languages, representing a major research challenge (RC1). Therefore, the proposed system must support multilingual low-resource translation using lightweight AI architectures.”

---

# 5. Chapter 3 — Design and Modeling

This chapter corresponds to Theory Building.

Objectives:
- define the system architecture,
- propose conceptual solutions,
- address Remaining Challenges.

Recommended methodologies:
- User-Centered System Design (UCSD),
- Rational Unified Process (RUP),
- UML.

Required diagrams:
- Use Case Diagram,
- Sequence Diagram,
- Activity Diagram,
- Component Diagram,
- ERD / Information Model,
- MVC Architecture,
- UI Mockups or API Definitions.

All diagrams must:
- be explained narratively,
- be referenced in the text,
- and be connected to RCs and Research Objectives.

---

# 6. Chapter 4 — Implementation / Proof of Concept

Objectives:
- realize the architecture technically,
- demonstrate feasibility through a prototype.

The chapter should explain:
- frontend,
- backend,
- APIs,
- AI pipelines,
- LLM integration,
- intelligent agents,
- synchronization mechanisms,
- cloud/offline support.

Include:
- 2–3 concise code snippets,
- workflow diagrams,
- UI mockups where relevant.

Avoid:
- full source code dumps,
- installation manuals,
- API documentation dumps.

---

# 7. Chapter 5 — Evaluation

The objective is to evaluate:
- effectiveness,
- usability,
- usefulness,
- scalability,
- performance,
- scientific contribution.

Not all projects require:
- surveys,
- cognitive walkthroughs,
- or user evaluations.

The evaluation methodology depends on:
- the research domain,
- the research questions,
- the system type,
- and the produced outputs.

---

## 7.1 Quantitative Evaluation

Examples:
- Accuracy,
- BLEU,
- F1-score,
- WER,
- Precision & Recall,
- latency,
- throughput,
- scalability,
- benchmark comparisons.

Suitable for:
- AI systems,
- NLP systems,
- retrieval systems,
- recommendation systems.

---

## 7.2 Qualitative Evaluation

Examples:
- Cognitive Walkthrough,
- Expert Review,
- Interviews,
- Focus Groups,
- Case Studies,
- Observational Studies.

---

## 7.3 Survey-Based Evaluation

Surveys are optional and recommended for user-facing systems.

Recommended tools:
- Google Forms,
- Microsoft Forms,
- Qualtrics.

The manuscript should report:
- methodology,
- participant demographics,
- charts,
- findings,
- interpretation.

---

## 7.4 Cognitive Walkthrough

Recommended for:
- dashboards,
- AI assistants,
- educational systems,
- web platforms.

Typical process:
1. define representative tasks,
2. observe users,
3. collect feedback,
4. identify usability issues,
5. discuss improvements.

---

## 7.5 Evaluation Reporting

For each Evaluation Objective (/E), report:
- methodology,
- setup,
- datasets or participants,
- expected outcomes,
- obtained results,
- visualizations,
- discussion of findings,
- limitations.

---

# 8. Chapter 6 — Conclusion and Future Work

Summarize:
- research contributions,
- implementation achievements,
- evaluation findings,
- limitations,
- future work.

Future work may include:
- larger datasets,
- improved AI models,
- multilingual expansion,
- mobile deployment,
- real-time inference,
- advanced agents,
- industrial deployment.

---

# 9. Figures, Tables, and Visual Elements

Figures, tables, UML diagrams, screenshots, and charts must:
- be referenced before appearing,
- be scientifically discussed,
- support scientific argumentation,
- and remain readable.

Students must never insert unexplained figures or tables.

---

## 9.1 Introducing Figures and Tables

The manuscript text must:
- explain what the figure/table represents,
- summarize important information,
- interpret findings,
- explain relevance.

The text must not simply state:
“Figure 3 shows the architecture.”

Instead, explain:
- workflows,
- interactions,
- trends,
- architectural decisions,
- results.

---

## 9.2 Captions

Captions must:
- appear below the figure or table,
- be concise,
- descriptive,
- scientifically meaningful.

Examples:
- Figure 3. Proposed Multi-Agent Architecture
- Table 5. Translation Accuracy Comparison

---

## 9.3 Referencing External Sources

If a figure or table originates from another publication, cite the source at the end of the caption.

Examples:
- Figure 2. Transformer Architecture (adapted from Vaswani et al.)
- Table 4. Comparison of Existing Systems

If modified:
- indicate “adapted from”,
- “inspired by”,
- or “based on”.

---

# 10. Literature and Citation Requirements

Use:
- IEEE,
- ACM,
- Springer,
- Elsevier,
- ACL Anthology,
- Google Scholar.

Avoid:
- Wikipedia,
- blogs,
- non-scientific websites.

All citations must:
- appear in the bibliography,
- use consistent citation styles,
- compile correctly in Overleaf.

---

# 11. Final Recommendations

A strong scientific document should:
- maintain narrative continuity,
- connect all chapters logically,
- justify scientific decisions,
- synthesize literature deeply,
- and continuously connect:
  - motivations,
  - problem statements,
  - research questions,
  - objectives,
  - architectures,
  - implementations,
  - evaluations,
  - and contributions.

The final document must demonstrate:
- scientific rigor,
- technical depth,
- reproducibility,
- innovation,
- usability,
- and practical relevance.
