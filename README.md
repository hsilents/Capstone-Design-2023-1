# Capstone Design 2023-1
# Python으로 훈련하는AI로 게임 놀기
* Student list  
LI LIUYANG  
SONG HENGHUI

## Overview
* Motivation：
  이 연구의 동기는 AI의 중요성과 게임을 통한 AI 연구의 잠재적 가치에 발견에 있습니다. AI는 4차 산업 혁명에서 핵심적인 역할을 하고 있으며, 스마트폰과 컴퓨터의 발전으로 게임의 다양성과 상업가치가 증가하고 있습니다. 또한 AI 기술은 사람들의 일상 생활에 점차적으로 보급되고 있습니다. 따라서 AI에 대한 이해와 더 많은 AI 개발이 중요한 일이 됐습니다. 게임을 통한 AI 연구는 알고리즘과 모델의 최적화에 집중하고, AI의 실용적 가치를 탐구하며, 다양한 분야에서의 교차 응용을 도모할 수 있는 장점을 가지고 있습니다.

* Problem definition:
  본 연구의 문제는 게임을 통해 AI를 훈련하고 성능을 개선하는 방법을 탐구하는 것입니다. 강화 학습을 적용하여 AI를 게임에서 높은 점수를 얻을 수 있도록 훈련시키는 것을 목표로 합니다. 또한 물리 엔진 기반의 게임을 선택하여 AI가 게임 규칙과 물리 모델을 이해하고 피해야 할 장애물을 피하기 위한 의사 결정을 내리는 능력을 연구합니다. 또한 AI의 일반화 능력을 탐구하기 위해 간단한 게임을 통해 AI를 훈련하고 다른 복잡한 게임에 적용하는 것을 목표로 합니다.


* Goal:
  1.강화 학습을 게임에 적용하는 방법을 배우고, AI가 게임에서 더 높은 점수를 얻을 수 있도록 훈련하는 것입니다. 이를 통해 딥 러닝과 강화 학습 알고리즘의 성능과 적용 효과를 게임에서 테스트합니다.
  2.게임 환경에서 AI의 의사 결정 능력을 연구합니다. 물리 엔진 기반의 게임을 선택하여 AI가 게임 규칙과 물리 모델을 이해하고, 장애물을 피하기 위해 게임 캐릭터를 제어하는 결정을 내리도록 합니다. 이를 통해 실제 결정 프로세스를 이해하고 시뮬레이션을 수행할 수 있습니다.
  3.게임 분야에서 AI의 일반화 능력을 탐색합니다. 간단한 게임 규칙과 물리 모델로 훈련된 AI가 다른 복잡한 게임의 표현을 연구하고 적용할 수 있는 일반화 능력을 탐구합니다.



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
* [4]Dqn algorithm : https://blog.csdn.net/Zhang_0702_China/article/details/123423637
* [5]Genetic Algorithm : https://blog.csdn.net/LOVEmy134611/article/details/111639624
https://en.wikipedia.org/wiki/Genetic_algorithm
https://www.geeksforgeeks.org/genetic-algorithms/

## Reports
* [Report](Reports/Final.pdf) 
* [Paper](Reports/Paper.pdf)
* [Demo](Reports/Demo.mp4)
