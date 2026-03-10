Supervised learning: IID assumption, same distribution for training and test data.
Sequential imitation learning: breaks the IID assumption, **covariate shift** or the DAgger problem.

**Dataset Aggregation (DAgger)** is an imitation learning algorithm that addresses the covariate shift problem.

Repeat until convergence:
- Train a policy on the dataset.
- Roll out the learned policy to collect new states.
- Query an expert for the correct action at the new states.
- Aggregate the new labeled data into the dataset.

Inspired by DAgger: Data as demonstrator (DaD). It is a supervised learning method for sequential prediction models. It works by reusing training data to correct for prediction mistakes.

Non-Interactive Imitation Learning:
- **DART**: Inject noise to the supervisor's demonstrations during data collection.

**Student-Teacher Training** (in a simulator):
- Teacher trained with RL with privileged input.
- Student trained to imitate the teacher with actual sensor input.
- Execute the student policy, for each state visited by the student, ask the teacher to compute the expert action.

