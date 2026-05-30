Membership Inference Attacks and Defenses on Dynamic Graph-Based Models
======================================================================

This project studies membership inference attacks and privacy-preserving defense methods on dynamic graph-based neural network models.

The main purpose of this project is to analyze whether dynamic graph models may leak membership information of training nodes, and to explore possible defense strategies against such privacy attacks.

This code is intended for academic research and experimental analysis.


Project Overview
================

Dynamic graph data contains both structural information and temporal evolution patterns. Compared with static graph learning, dynamic graph-based models may expose more sensitive information because node features, graph structures, and temporal behaviors are jointly learned.

This project builds a general experimental framework for evaluating membership inference attacks on dynamic graph models.

The framework mainly includes:

1. Victim model training
2. Shadow model training
3. Attack model training
4. Membership inference evaluation
5. Defense method comparison


Main Files
==========

main.py
-------

main.py is the main control file of the project.

It manages the overall experimental process, including model selection, dataset loading, victim model training, shadow model training, attack model training, and final evaluation.

models.py
---------

models.py defines the dynamic graph-based models used in this project.

The project includes several commonly used dynamic graph neural network architectures, such as DCRNN, GConvGRU, TGCN and A3TGCN.

victim.py
---------

victim.py contains the training process of victim models.

The victim model is the target model attacked by the membership inference attack. Different defense strategies are also implemented in this file.

shadow.py
---------

shadow.py contains the training process of shadow models.

Shadow models are used to simulate the behavior of victim models and help construct training data for the attack model.

attack.py
---------

attack.py implements the attack model training process.

The attack model is trained to distinguish member samples from non-member samples.

layers.py
---------

layers.py contains self-attention related layers.

These layers are mainly used for spatial-temporal representation learning in dynamic graph defense models.


Supported Dynamic Graph Models
==============================

This project supports the following dynamic graph-based models:

1. DCRNN
2. GConvGRU
3. TGCN
4. A3TGCN

These models are used as victim models, shadow models, or attack-related dynamic graph encoders in the experimental framework.


Defense Methods
===============

The project contains several defense strategies for comparison:

1. Raw training
2. Relaxed loss based defense
3. Adversarial defense
4. Gaussian differential privacy defense
5. Laplace differential privacy defense
6. Spatial-temporal self-attention defense
7. Differentially private spatial-temporal self-attention defense

The goal of these defenses is to reduce the success rate of membership inference attacks while maintaining the utility of dynamic graph models.


General Workflow
================

The general workflow of this project is:

1. Train a victim model on a dynamic graph dataset.
2. Train a shadow model to simulate the behavior of the victim model.
3. Generate attack training data using the shadow model.
4. Train an attack model to distinguish members and non-members.
5. Apply the attack model to the victim model.
6. Evaluate privacy leakage and defense effectiveness.

This workflow is designed for experimental comparison among different dynamic graph models and defense methods.


Membership Inference Attack
===========================

Membership inference attack aims to infer whether a specific sample or node was used in the training process of a target model.

In this project, the attack is conducted in a dynamic graph learning scenario. The attacker observes model behaviors and attempts to identify whether certain nodes belong to the victim model's training set.

The attack model is trained based on the behavior of shadow models and then tested against the victim model.


Spatial-Temporal Defense
========================

Dynamic graphs contain both spatial and temporal information.

Spatial information describes graph structure and node relationships.

Temporal information describes how node features and graph structures evolve over time.

The spatial-temporal defense module is designed to improve robustness by learning more stable dynamic graph representations.


Differential Privacy Defense
============================

This project also considers differential privacy based defenses.

Differential privacy adds controlled noise during model training to reduce the influence of individual training samples.

The project includes both Gaussian noise based and Laplace noise based differential privacy strategies.

For the spatial-temporal defense model, privacy protection can be applied to spatial and temporal components separately.


Environment
===========

The project is implemented in Python and based on PyTorch.

The main dependencies include:

1. PyTorch
2. PyTorch Geometric
3. PyTorch Geometric Temporal
4. NumPy
5. SciPy
6. scikit-learn
7. NetworkX
8. tqdm

The specific versions may depend on the local CUDA and PyTorch environment.


Usage
=====

Configure the experimental settings in main.py, including dataset, model type and defense type.

Then run the main script:

python main.py

The detailed configuration may need to be adjusted according to the dataset and local environment.


Output
======

The program can output training results and attack evaluation results, such as:

1. Victim model performance
2. Shadow model performance
3. Attack model performance
4. Membership inference attack accuracy
5. Defense comparison results

These results can be used to analyze the privacy risks of dynamic graph-based models.


Notes
=====

1. This project is mainly designed for research and experimental evaluation.

2. The implementation details may need to be adjusted for different datasets or environments.

3. Some model parameters and privacy parameters can affect both model utility and defense performance.

4. Saved models may be reused during experiments.

5. To ensure fair comparison, the same dataset split and model settings should be used when comparing different defense methods.


Research Purpose
================

This project aims to explore the following questions:

1. Are dynamic graph-based models vulnerable to membership inference attacks?
2. How does temporal graph information affect privacy leakage?
3. How effective are existing defense methods in dynamic graph scenarios?
4. Can spatial-temporal representation learning improve robustness?
5. Can differential privacy further reduce membership leakage?


License
=======

This project is for academic research and educational purposes only.
