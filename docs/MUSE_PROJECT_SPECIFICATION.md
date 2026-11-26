# MUSE --- BCI Project Specification

**Project type:** EEG-based Brain-Computer Interface (BCI)\
**Working name:** MUSE\
**Primary domain:** EEG signal processing, machine learning, neural
decoding, real-time human-computer interaction\
**Current starting point:** Motor-imagery intent classification\
**Long-term direction:** Neural intent → computer control → potentially
limited brain-to-text research

------------------------------------------------------------------------

## 1. Project Vision

MUSE is an end-to-end Brain-Computer Interface project intended to
decode useful information from EEG signals and translate that
information into computer actions.

The project should **not** begin by attempting to decode arbitrary
thoughts directly into text. Instead, it follows a staged progression
where every stage produces a measurable and usable result.

### Core progression

``` text
EEG signal
    ↓
Signal acquisition
    ↓
Preprocessing
    ↓
Neural / EEG feature extraction
    ↓
Intent classification
    ↓
Command generation
    ↓
Computer interaction
    ↓
Eventually: limited brain-to-text research
```

The project should be treated as an engineering + ML + research system
rather than a simple EEG classification notebook.

------------------------------------------------------------------------

# 2. Primary Initial Objective

The first concrete objective is:

> **Determine whether EEG recordings can be used to distinguish imagined
> left-hand movement from imagined right-hand movement.**

This is the first MUSE classification problem.

### Initial classes

``` text
LEFT
RIGHT
```

A `REST` / `NEUTRAL` class may also be introduced where useful,
especially for real-time control.

### Intended interpretation

``` text
Imagine LEFT
     ↓
EEG pattern
     ↓
ML decoder
     ↓
LEFT
     ↓
Computer action
```

and:

``` text
Imagine RIGHT
     ↓
EEG pattern
     ↓
ML decoder
     ↓
RIGHT
     ↓
Computer action
```

The first goal is **not** arbitrary thought decoding.

------------------------------------------------------------------------

# 3. Why Start With Motor Imagery?

Motor imagery provides a constrained and experimentally definable BCI
problem.

Instead of asking:

> "What is the user thinking?"

we ask:

> "Can the EEG signal contain a learnable difference between two
> predefined intentional mental tasks?"

This allows MUSE to develop systematically:

1.  EEG understanding
2.  Signal processing
3.  Feature extraction
4.  Classical ML
5.  Deep learning
6.  Evaluation methodology
7.  Real-time inference
8.  Computer control
9.  More ambitious neural decoding

------------------------------------------------------------------------

# 4. Overall Architecture

The intended MUSE architecture is:

``` text
                 EEG DATA
                    │
                    ▼
          ┌──────────────────┐
          │ Signal Acquisition│
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  Preprocessing   │
          │                  │
          │ • Filtering      │
          │ • Artifacts      │
          │ • Epoching       │
          │ • Normalization  │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Feature Extraction│
          │                  │
          │ • Frequency      │
          │ • Spatial        │
          │ • CSP            │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Neural Decoder   │
          │                  │
          │ LDA / SVM / etc. │
          │ Deep Learning    │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Intent Prediction│
          │                  │
          │ LEFT / RIGHT     │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Command Layer    │
          └────────┬─────────┘
                   │
                   ▼
        Computer / Cursor / UI
```

------------------------------------------------------------------------

# 5. Data Strategy

## Phase 1: Public EEG Dataset

MUSE should initially use public EEG datasets.

The main initial direction is the **PhysioNet EEG Motor Movement/Imagery
dataset**, which contains motor-movement and motor-imagery recordings.

The purpose of starting with public data is to:

-   validate the processing pipeline
-   establish a reproducible baseline
-   learn the EEG characteristics
-   compare models
-   avoid buying hardware before validating the methodology

Other established BCI benchmark datasets can be introduced later for
comparison.

------------------------------------------------------------------------

# 6. Experimental Protocol

For motor-imagery experiments, the conceptual trial structure is:

``` text
REST
  ↓
Instruction / cue
  ↓
Imagine LEFT or RIGHT
  ↓
Recording window
  ↓
REST
```

Repeated trials should be collected.

Example:

``` text
Trial 1 → LEFT
Trial 2 → RIGHT
Trial 3 → LEFT
Trial 4 → RIGHT
...
```

Later, personal recordings can be collected using an EEG headset.

The dataset should preserve:

-   class labels
-   trial boundaries
-   session information
-   subject information
-   recording metadata

This is important for valid evaluation.

------------------------------------------------------------------------

# 7. EEG Preprocessing

Raw EEG contains more than useful neural information.

Potential contamination includes:

-   eye movement artifacts
-   muscle activity
-   electrical noise
-   movement artifacts
-   physiological interference

The preprocessing pipeline should therefore investigate:

``` text
Raw EEG
   ↓
Filtering
   ↓
Artifact handling
   ↓
Epoching
   ↓
Normalization / scaling where appropriate
   ↓
Feature extraction
```

The exact preprocessing configuration should be treated as an
experimental variable and documented rather than assumed to be
universally optimal.

------------------------------------------------------------------------

# 8. Frequency-Domain Investigation

The project should investigate EEG frequency information relevant to
motor imagery.

Important bands to examine include:

-   Alpha / Mu: approximately 8--13 Hz
-   Beta: approximately 13--30 Hz

The goal is not simply to extract frequency numbers, but to investigate
whether changes in relevant frequency activity and spatial patterns help
distinguish the intended classes.

------------------------------------------------------------------------

# 9. Feature Extraction

## First major technique: Common Spatial Pattern (CSP)

CSP should be one of the first feature-extraction methods.

Conceptually:

``` text
Multi-channel EEG
       ↓
      CSP
       ↓
Spatial filters emphasizing class differences
       ↓
Feature vectors
       ↓
Classifier
```

CSP is particularly appropriate as an initial technique for a binary
motor-imagery problem.

------------------------------------------------------------------------

# 10. Initial ML Baseline

The first baseline should be:

``` text
EEG
 ↓
Preprocessing
 ↓
CSP
 ↓
LDA
 ↓
LEFT / RIGHT
```

### Why LDA?

It provides a simple, interpretable classical baseline against which
more complex models can be compared.

The project should not immediately jump to deep learning.

------------------------------------------------------------------------

# 11. Model Comparison

After establishing the baseline, compare multiple approaches.

Potential models:

### Classical ML

-   CSP + LDA
-   CSP + SVM
-   Random Forest
-   XGBoost

### Deep Learning

-   EEGNet
-   CNN-based approaches
-   Later temporal/spatial deep-learning architectures
-   Eventually transformer-style approaches if justified by the data

The goal is to determine what works for the dataset and experimental
setting, not to use the most complicated model by default.

------------------------------------------------------------------------

# 12. Evaluation

Evaluation is a major part of the project.

Initial metrics may include:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion matrix
-   ROC-AUC where appropriate

For real-time BCI, additional concerns include:

-   inference latency
-   prediction stability
-   false activations
-   command responsiveness

## Avoiding misleading evaluation

EEG data is highly correlated across time.

Therefore, blindly randomly splitting individual windows can produce
overly optimistic results.

MUSE should progressively evaluate:

``` text
Random / basic split
        ↓
Unseen trials
        ↓
Unseen sessions
        ↓
Unseen subjects
```

This progression is important.

------------------------------------------------------------------------

# 13. Subject Dependence

BCI systems can vary substantially between people.

A model that performs well on one person does not automatically
generalize to another person.

This creates a major future ML research problem:

> Can MUSE generalize across subjects?

For example:

``` text
Training:
Person A
Person B
Person C
Person D

          ↓
       MODEL
          ↓

Testing:
Person E
```

This introduces problems such as:

-   domain shift
-   subject variability
-   calibration requirements
-   distribution differences

Subject-independent decoding should therefore be treated as a research
extension, not assumed as a solved problem.

------------------------------------------------------------------------

# 14. Development Stages

## Stage 0 --- Foundations

Learn and establish:

-   EEG basics
-   motor imagery
-   MNE
-   signal processing
-   project environment
-   dataset handling

------------------------------------------------------------------------

## Stage 1 --- Classical BCI

Goal:

> Build the first LEFT vs RIGHT decoder.

Pipeline:

``` text
Dataset
 ↓
Load EEG
 ↓
Visualize
 ↓
Filter
 ↓
Epoch
 ↓
CSP
 ↓
LDA
 ↓
Evaluation
```

Deliverable:

**First quantitative classification result.**

------------------------------------------------------------------------

## Stage 2 --- ML Experimentation

Compare:

``` text
CSP + LDA
CSP + SVM
Random Forest
XGBoost
EEGNet
```

Investigate:

-   preprocessing choices
-   feature representations
-   model performance
-   subject variability
-   session variability

Deliverable:

**A documented model comparison.**

------------------------------------------------------------------------

## Stage 3 --- Personal EEG

Only after the public-data pipeline is understood:

``` text
Public dataset
      ↓
Validated methodology
      ↓
EEG hardware
      ↓
Personal recordings
      ↓
Personalized decoder
```

The exact hardware should be selected based on:

-   channel count
-   signal quality
-   software support
-   accessibility
-   budget
-   real-time streaming capability

OpenBCI Cyton is one candidate direction, but hardware selection should
remain an engineering decision.

------------------------------------------------------------------------

## Stage 4 --- Real-Time BCI

The offline model becomes a streaming system:

``` text
Live EEG
   ↓
Buffer
   ↓
Online preprocessing
   ↓
Sliding window
   ↓
Feature extraction
   ↓
Model inference
   ↓
LEFT / RIGHT
   ↓
Command
```

First demonstration:

> Use imagined LEFT/RIGHT intent to control a computer cursor or another
> simple interface.

This is a major project milestone.

------------------------------------------------------------------------

# 15. Multi-Command Expansion

Once binary classification is stable, expand the command vocabulary.

Potential commands discussed:

``` text
LEFT
RIGHT
UP
DOWN
STOP / REST
```

Possible imagined-movement mapping:

``` text
Imagine left hand → LEFT
Imagine right hand → RIGHT
Imagine feet       → DOWN
Imagine tongue     → UP
Rest               → STOP
```

These mappings are **candidate experimental classes**, not guaranteed
capabilities. They must be tested empirically.

The eventual goal is a 4--6 command BCI interface if signal quality and
classification performance support it.

------------------------------------------------------------------------

# 16. Typing Direction

MUSE should not initially claim:

> "Think any sentence and the computer types it."

That is substantially harder.

A more realistic intermediate step is a **BCI-controlled virtual
keyboard**.

Example:

``` text
EEG intent
   ↓
Navigation command
   ↓
Virtual keyboard
   ↓
Select / move / confirm / delete
   ↓
Typed text
```

For example:

``` text
LEFT  → move selection left
RIGHT → move selection right
UP    → move selection up
DOWN  → move selection down
STOP  → confirm / pause
```

The exact mapping can be redesigned after the command set is
experimentally validated.

------------------------------------------------------------------------

# 17. Long-Term Brain-to-Text Research

Brain-to-text is a long-term research direction.

Possible progression:

``` text
Motor imagery
      ↓
Multi-command BCI
      ↓
Virtual keyboard
      ↓
Limited vocabulary
      ↓
Imagined speech / neural decoding
      ↓
Potential brain-to-text research
```

MUSE should investigate this direction carefully.

It should **not** claim that consumer EEG can reliably decode arbitrary
private thoughts.

The long-term goal is to investigate whether useful neural
representations can be decoded under constrained experimental
conditions.

------------------------------------------------------------------------

# 18. Real-Time Software Architecture

A later implementation may look like:

``` text
EEG Headset
     ↓
Python BCI Engine
     ↓
Preprocessing
     ↓
Inference Model
     ↓
Intent
     ↓
FastAPI / WebSocket
     ↓
Frontend / Application
     ↓
Live visualization + command execution
```

A possible UI could show:

``` text
LIVE EEG
────────────────────────────

CH1  ~~~~~~~~~~~~~~~~~
CH2  ~~~~~~~~~~~~~~~~~
CH3  ~~~~~~~~~~~~~~~~~

Predicted Intent:
LEFT

Confidence:
87%

Command:
←
```

The UI is a later layer. The neural decoder remains the core.

------------------------------------------------------------------------

# 19. Technology Stack

## Core

-   Python
-   NumPy
-   SciPy
-   pandas
-   matplotlib

## EEG

-   MNE

## Classical ML

-   scikit-learn
-   XGBoost where useful

## Deep Learning

-   PyTorch

## Later real-time layer

Potentially:

-   FastAPI
-   WebSockets
-   React / Next.js

The stack should remain simple until the core decoder is validated.

------------------------------------------------------------------------

# 20. Suggested Repository Structure

``` text
muse/
│
├── data/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   └── realtime.py
│
├── experiments/
│
├── results/
│
├── README.md
│
└── requirements.txt
```

The structure can evolve as the project grows.

------------------------------------------------------------------------

# 21. Research Questions

MUSE should progressively answer:

1.  Can EEG distinguish imagined LEFT and RIGHT reliably?
2.  Which preprocessing pipeline works best?
3.  Which EEG frequency ranges contain useful information?
4.  How effective is CSP?
5.  Does CSP + LDA provide a strong baseline?
6.  Which classical ML model performs best?
7.  Can EEGNet improve over classical approaches?
8.  Does the model generalize to unseen trials?
9.  Does it generalize to unseen sessions?
10. How much does performance vary between people?
11. Can cross-subject models be developed?
12. Can the decoder operate reliably in real time?
13. Can predicted intents control a computer interface?
14. Can the command vocabulary be expanded?
15. Can the system eventually support constrained typing or
    limited-vocabulary neural decoding?

------------------------------------------------------------------------

# 22. Project Principles

These principles should remain consistent throughout development.

### 1. Build progressively

Do not jump directly to brain-to-text.

### 2. Establish baselines

Every advanced model should be compared against a simpler baseline.

### 3. Measure everything

Do not claim improvement without quantitative evaluation.

### 4. Avoid data leakage

EEG windows from the same recording/session are correlated.

### 5. Separate experimentation from claims

A candidate technique is not a proven capability until experiments
support it.

### 6. Prefer reproducibility

Record:

-   dataset version
-   preprocessing configuration
-   feature configuration
-   model configuration
-   random seeds where applicable
-   train/test methodology
-   results

### 7. Treat real-time performance separately

A model with high offline accuracy is not automatically a good real-time
BCI.

------------------------------------------------------------------------

# 23. What MUSE Should Eventually Demonstrate

The strongest demonstration would be:

``` text
User
 ↓
Wears EEG headset
 ↓
Performs defined mental task
 ↓
EEG is streamed
 ↓
MUSE preprocesses the signal
 ↓
Neural decoder predicts intent
 ↓
Intent is converted into command
 ↓
Computer responds
```

Example:

``` text
User imagines LEFT
        ↓
       EEG
        ↓
    MUSE decoder
        ↓
      LEFT
        ↓
Cursor moves left
```

This transforms MUSE from a dataset classification project into a
complete BCI prototype.

------------------------------------------------------------------------

# 24. Final Project Direction

MUSE should be understood as:

> **An EEG-based neural decoding framework that progresses from
> controlled motor-imagery classification to real-time intent-driven
> computer interaction, with brain-to-text as a long-term research
> direction.**

The project is intentionally broader than one ML model.

Its core areas are:

``` text
EEG
 +
Signal Processing
 +
Machine Learning
 +
Deep Learning
 +
Experimental Design
 +
Real-Time Systems
 +
Human-Computer Interaction
```

------------------------------------------------------------------------

# 25. Current Priority

Do not work on all phases simultaneously.

The immediate priority is:

``` text
1. Set up MUSE environment
2. Obtain public motor-imagery EEG data
3. Load and inspect EEG
4. Understand the trial labels
5. Filter the signals
6. Create epochs
7. Implement CSP
8. Train LDA
9. Evaluate LEFT vs RIGHT
10. Save the first reproducible result
```

Only after this baseline works should MUSE move toward deeper models or
hardware.

------------------------------------------------------------------------

# 26. Context for Future AI Assistants

If this document is provided to another AI assistant, use it as the
**project specification and current direction for MUSE**.

The assistant should:

-   preserve the staged development strategy
-   prioritize the current phase over future phases
-   avoid jumping directly to brain-to-text
-   distinguish experimental hypotheses from established results
-   avoid inventing performance numbers
-   recommend reproducible EEG/ML practices
-   track decisions and experiments
-   explain why a technical change is being made
-   keep the initial LEFT vs RIGHT motor-imagery objective as the
    baseline
-   treat real-time control and brain-to-text as later milestones

### Current state

**Project:** MUSE\
**Current technical milestone:** Public-dataset motor-imagery decoding\
**Current classification target:** LEFT vs RIGHT\
**First baseline:** CSP + LDA\
**Hardware:** Not required initially\
**Immediate output:** A reproducible EEG classification result\
**Next major milestone:** Compare ML models and evaluate generalization\
**Later milestone:** Real-time EEG → intent → computer command\
**Long-term research direction:** Limited-vocabulary typing /
imagined-speech / brain-to-text investigation

------------------------------------------------------------------------

## 27. Important Scope Statement

MUSE is an experimental BCI/ML research project.

It is **not** currently a system capable of reading arbitrary thoughts.

Any claim about neural decoding capability must be supported by
controlled experiments and appropriate evaluation.

The project should prioritize scientific validity, reproducibility,
measurable progress, and honest reporting of limitations.
