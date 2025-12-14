import os
import gym
import numpy as np
from ppo import Agent
import torch as T
import matplotlib.pyplot as plt

def plot_curve(x, scores, figure_file):
    plt.plot(x, scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = gym.make ("CartPole-v0")
    N = 2048
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape,
                  gamma=0.99, alpha=alpha, gae_lambda=0.95, policy_clip=0.2,
                  batch_size=batch_size, N=N, n_epochs=n_epochs)
    n_games = 300
    score_history = []

    fig_file = 'result/cartpole.png'
    os.makedirs('result', exist_ok=True)
    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        while not done:
            # Choose action based on current state
            action, prob, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n_steps += 1
            score += reward

            # Store the transition in memory
            agent.remember(observation, action, prob, val, reward, done)
            
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
              'avg score %.1f' % avg_score,
              'time_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_curve(x, score_history, fig_file)
