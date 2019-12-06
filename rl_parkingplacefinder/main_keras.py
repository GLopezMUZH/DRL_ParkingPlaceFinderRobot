from dqn_keras import Agent
from rl_parkingplacefinder import Park_Finder_Agent
from utils import plotLearning
import numpy as np


if __name__ == '__main__':
    env = Park_Finder_Agent()
    n_episodes = 500
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=12,
                  n_actions = 4, mem_size=1000000, batch_size=64, epsilon_end=0.1)

    #agent.load_model()
    scores = []
    eps_history = []

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        # observation = np.array(env.stateSpace)
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode ', i, 'score %.2f' %score, 'average score %.2f' % avg_score)

        if i % 10 == 0 and i > 0:
            agent.save_model()

    filename = 'dqn.png'
    x = [i+1 for i in range(n_episodes)]
    plotLearning(x, scores, eps_history, filename)