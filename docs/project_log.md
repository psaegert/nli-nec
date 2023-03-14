# Project Log

#### 2023-02-10
*Paul, S*
<br>
Discuss favorite topics for the project. Share Slack and GitHub links.

---

#### 2023-02-14
*Paul, S, T, Prof*
<br>
Discuss ideas for the NEC topic.

---

#### 2023-02-18
*Paul, S, T*
<br>
Create draft for the project specification.

---

#### 2023-02-21
*Paul*
<br>
Covert draft to PDF and submit to Prof and Moodle.

---

#### 2023-02-22
*Paul*
<br>
Rewrite specification according to feedback from Prof and submit to Prof and Moodle again.

---

#### 2023-02-24
*Paul, S, T, Prof*
<br>
Discuss project specification

---

#### 2023-02-26
*Paul*
<br>
Prepare improvements for the project specification to discuss in the next group meeting.

---

#### 2023-02-27
*Paul, S, T*
<br>
Implement feedback from Prof and discuss the project specification.
Agree to implement utility functions for the project this week and do the experiments in the coming weeks.

---

#### 2023-02-28
*Paul*
<br>
Set up development environment and package structure.

---

#### 2023-03-01
*Paul*
<br>
Implement utility functions for loading and filtering data and test them in jupyter notebooks.

---

#### 2023-03-05
*Paul*
<br>
Implement utility functions for predicting types and test them in jupyter notebooks.
Merge PR #6.

---

#### 2023-03-06
*Paul, S, T*
<br>
Discuss utility functions.
Create plan for the granularity experiment.
Formulate and distribute jobs (Fine-Tuning: Sam, Prediction: Paul, Evaluation: T).
Agree on interfaces for the jobs to facilitate parallel work.
The fine-tuning job produces a model, the prediction job produces a table with entailment probabilities for each type for each sentence.

---

#### 2023-03-07
*Paul*
<br>
Create Issues to track progress on the tasks.
Refine prediction notebook.
Run predictions for the dev and test set, each with and without the premise.
Run predictions for the train set with only the hypothesis for bias analysis and later hard data filtering.

---

#### 2023-03-08
*Paul*
<br>
Begin Explorative Data Analysis.
Investigate target type distribution, entity distribution and target type distribution of the most frequent entities.

---

#### 2023-03-09
*Paul*
<br>
Identify and count the frequency of the most often misclassified entities in the `multi_nli` dataset.