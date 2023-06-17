# Capstone Design 2023-1
# Python으로 훈련하는AI로 게임 놀기
* Student list:  
LI LIUYANG  
SONG HENGHUI

## Overview
* Motivation, Problem definition, Goal


## Environment configuration

> 1. Install and configure python3.9 environment
> 2. `pip install pipenv` installs `pipenv`
> 3. Use `pipenv` to install dependencies `pipenv install`

## Results

![test](game\Image\flappy_bird_demp.gif)

``` Python
GAME = 'bird' 
ACTIONS = 2 
GAMMA = 0.99 
EXPLORE = 2000000. 
FINAL_EPSILON = 0.0001 
INITIAL_EPSILON = 0.0001 
REPLAY_MEMORY = 50000 
BATCH = 32 
FRAME_PER_ACTION = 1 

def createNetwork():
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    s = tf.placeholder("float", [None, 80, 80, 4])

    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1
```

## Conclusion
* 이 프로젝트의 주요 목적은 딥러닝 기술을 사용하여 자동으로 Flappy Bird 게임을 플레이할 수 있는 AI 모델을 훈련시켜 게임 지능에 대한 딥러닝의 응용을 보여주는 것이다. 딥러닝 기술을 사용하면 게임 AI 성능을 효과적으로 향상시킬 수 있어 AI가 자체 훈련을 통해 자신의 의사 결정 전략을 지속적으로 최적화하여 인간 이상의 게임 성능을 달성할 수 있을 것으로 예상된다.

## References
* [1] Why Study Games ai : https://www.engati.com/blog/ai-in-gaming
* [2]Flappy Bird : https://namu.wiki/w/Flappy%20Bird
* [3]The World's Hardest Game : https://namu.wiki/w/The%20World's%20Hardest%20Game
* [4]Dqn algorithm : https://blog.csdn.net/Zhang_0702_China/article/details/12* 3423637
* [5]Genetic Algorithm : https://blog.csdn.net/LOVEmy134611/article/details/111639624
https://en.wikipedia.org/wiki/Genetic_algorithm
https://www.geeksforgeeks.org/genetic-algorithms/

## Reports
* [Report](Reports/Final.pdf) 
* [Paper](Reports/Paper.pdf)
* [Demo](Reports/Demo.mp4)
