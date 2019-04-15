# AlphaZero Connect4
# From-scratch implementation of AlphaZero for Connect4

This repo demonstrates an implementation of AlphaZero framework for Connect4, using python and PyTorch.

We all know that AlphaGo, created by DeepMind, created a big stir when it defeated reigning world champion Lee Sedol 4-1 in the game of Go in 2016, hence becoming the first computer program to achieve superhuman performance in an ultra-complicated game. 

However, AlphaGoZero, published (https://www.nature.com/articles/nature24270) a year later in 2017, push boundaries one big step further by achieving a similar feat without any human data inputs. A subsequent paper (https://arxiv.org/abs/1712.01815) released by the same group DeepMind successfully applied the same reinforcement learning + supervised learning framework to chess, outperforming the previous best chess program Stockfish after just 4 hours of training.

Inspired by the power of such supervised reinforcement learning models, I initially created a repository to build my own chess AI program from scratch, closely following the methods as described in the papers above. 

However, I quickly realized that the cost/computational power of training the chess AI would be too much to bear, thus I decided to try to implement AlphaZero on Connect4, which has much reduced moves complexity and hence would be more gentle on computational power.
The point here, is to demonstrate that the AlphaZero algorithm works well to create a powerful Connect4 AI.

# Contents
In this repository, you will find the following core scripts:

1) MCTS_c4.py - implements the Monte-Carlo Tree Search (MCTS) algorithm based on Polynomial Upper Confidence Trees (PUCT) method for leaf transversal. This generates datasets (state, policy, value) for neural network training

2) alpha_net_c4.py - PyTorch implementation of the AlphaGoZero neural network architecture, with slightly reduced number of residual blocks (19) and convolution channels (256) for faster computation. The network consists of, in order:
- A convolution block with batch normalization
- 19 residual blocks with each block consisting of two convolutional layers with batch normalization
- An output block with two heads: a policy output head that consists of convolutional layer with batch normalization followed by logsoftmax, and a value head that consists of a convolutional layer with relu and tanh activation.

3) connect_board.py – Implementation of a Connect4 board python class with all game rules and possible moves

4) encoder_decoder_c4.py – list of functions to encode/decode Connect4 board class for input/interpretation into neural network

5) evaluator_c4.py – arena class to pit current neural net against the neural net from previous iteration, and keeps the neural net that wins the most games

6) train_c4.py – function to start the neural network training process

7) visualize_board_c4.py – miscellaneous function to visualize the board in a more attractive way

8) analyze_games_c4.py – miscellaneous script to visualize and save the Connect4 games

9) play_against_c4.py - run it to play a Connect4 game against AlphaZero! (change "best_net" to the alpha net you've trained)

# Iteration pipeline

A full iteration pipeline consists of:
1) Self-play using MCTS (MCTS_c4.py) to generate game datasets (game state, policy, value), with the neural net guiding the search by providing the prior probabilities in the PUCT algorithm

2) Train the neural network (train_c4.py) using the (game state, policy, value) datasets generated from MCTS self-play

3) Evaluate (evaluator_c4.py) the trained neural net (at predefined checkpoints) by pitting it against the neural net from the previous iteration, again using MCTS guided by the respective neural nets, and keep only the neural net that performs better.

4) Rinse and repeat. Note that in the paper, all these processes are running simultaneously in parallel, subject to available computing resources one has.

# How to play

1) Run the MCTS_c4.py to generate self-play datasets. Note that for the first time, you will need to create and save a random, initialized alpha_net for loading.

2) Run train_c4.py to train the alpha_net with the datasets.

3) At predetermined checkpoints, run evaluator_c4.py to evaluate the trained net against the neural net from previous iteration. Saves the neural net that performs better.

4) Repeat for next iteration.

# Results

Iteration 0:
alpha_net_0 (Initialized with random weights)
151 games of MCTS self-play generated

Iteration 1:
alpha_net_1 (trained from iteration 0)
148 games of MCTS self-play generated

Iteration 2:
alpha_net_2 (trained from iteration 1)
310 games of MCTS self-play generated

Evaluation 1:
After Iteration 2, alpha_net_2 is pitted against alpha_net_0 to check if the neural net is improving in terms of policy and value estimate. Indeed, out of 100 games played, alpha_net_2 won 83. 

Iteration 3:
alpha_net_3 (trained from iteration 2)
584 games of MCTS self-play generated

Iteration 4:
alpha_net_4 (trained from iteration 3)
753 games of MCTS self-play generated

Iteration 5:
alpha_net_5 (trained from iteration 4)
1286 games of MCTS self-play generated

Iteration 6:
alpha_net_6 (trained from iteration 5)
1670 games of MCTS self-play generated



![alt text](https://github.com/plkmo/AlphaZero_Connect4/blob/master/Loss_vs_Epoch0_iter0_2019-03-12.png) Typical Loss vs Epoch when training neural net (alpha_net_0)
