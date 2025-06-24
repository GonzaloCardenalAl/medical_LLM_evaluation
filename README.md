# 🏥 HIVMedQA: Benchmarking large language models for HIV medical decision support

Large language models (LLMs) are emerging as valuable tools to support clinicians in routine decision making. This study aims to evaluate the current state of LLMs for HIV management, examining their strengths and limitations. We developed HIVMedQA, a benchmark for evaluating open-ended medical question answering in the context of HIV patient management. We assessed seven general-purpose LLMs and three medically specialized LLMs, using prompt engineering to optimize performance.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Architecture & Workflow](#architecture--workflow)
- [Usage](#usage)
  - [Generate Model Answers](#get-model-answers)
  - [Obtain Evaluation Metrics (MedGPT & F1 score)](#obtain-metrics)
- [Contributing](#contributing)
- [License](#license)

## Overview

This study aims to evaluate the performance of current LLMs in the context of curbside consults for HIV care and provide actionable insights for their future development Specifically, we focus on:
- (1) Assessing the reliability of LLMs as judges 
- (2) Identifying the most effective lexical matching techniques for open-ended question evaluation
- (3) Comparing the performance of small-scale versus large-scale LLMs
- (4) Evaluating domain-specific (medical) models against generalized LLMs
- (5) Benchmarking clinical skills of LLMs across the key dimensions comprehension, reasoning, knowledge recall, bias, and harm.

## Getting Started

To set up the two separate environments needed to run inference and scoring components of the `medical_LLM_evaluation` project.

### 1. Purpose of Environments

- **`transformers_llm.yml`** — installs dependencies required for running inference with various large language models.
- **`scispacy_env.yml`** — sets up scispaCy and related tools necessary for computing biomedical F1 scores and entity extraction.

### 2. Creating the Environments

Run the following commands to create both Conda environments:

```bash
conda env create --file transformers_llm.yml
conda env create --file scispacy_env.yml
```

### 3. Installing scispaCy Models 
To install the scispacy model manually:
```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

## Architecture & Workflow
1.	Load the different questionnaire categories available on question_files and load the desired LLM.
2.	Run inference subsquently inputting the system prompt and current question. .
3.	Obtain the emperical maximum MedGPT and F1 by rephrasing the reference answer and obtained the scores
4.	Compute Metrics (MedGPT and F1 score) from the models answer and aggrete results.

## Usage

## Contributing 
Contributions are welcome!
Please open an issue or pull request. Include tests and follow best practices when extending functionality.

## License 
MIT License © [2025] [Gonzalo Cardenal Antolin]
See LICENSE file for details.
---
