# Atari DQN
This is an implementation of the seminal DeepMind paper [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) which was published in Nature.

This was done for a Reinforcement Learning course at Wits University. Our agent learns to play Pong from experience, using Deep Q-Learning with the CNN architecture specified in the paper.

## Requirements
The agent is written in Python 3, and the following packages are required:
* `numpy`
* `gym`
* `torch`

It is also recommended to use a Cuda-compatible GPU device to train the DQN.

## Gameplay and training
The agent learns to pick up more reward as training progresses:

![Rewards over training](https://github.com/nishai/Atari-DQN/blob/master/Rewards.jpg)
