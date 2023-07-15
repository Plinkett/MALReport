import math
import sys
import numpy as np
import numpy.random as random
import matplotlib as mpl

mpl.use('pgf')
from numpy.random import choice
from matplotlib import pyplot as plt


# Utility functions for managing matrices and probability distributions
def column(A, j):
    return [A[i][j] for i in range(len(A))]


def transpose(A):
    return [column(A, j) for j in range(len(A[0]))]


def distr(weights, gamma=0.0):
    weight_sum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / weight_sum) + (gamma / len(weights)) for w in weights)


def draw(probability_distribution, arms):
    arm = choice(arms, size=1, p=probability_distribution, replace=False)[0]
    return arm


# Implementation of vanilla Exp3
def exp3(numActions, rewards, gamma, rewardMin=0, rewardMax=1):
    weights = [1.0] * numActions
    arms = np.array([i for i in range(numActions)])
    t = 0
    while True:
        probabilityDistribution = distr(weights, gamma)
        arm = draw(probabilityDistribution, arms)
        # Pull arm
        reward = rewards(arm, t)
        # Scaled to rewards between 0 and 1
        normalizedReward = (reward - rewardMin) / (rewardMax - rewardMin)
        estimatedReward = float(normalizedReward / probabilityDistribution[arm])
        # We update the weight of the chosen arm
        weights[arm] = weights[arm] * math.exp(estimatedReward * gamma / numActions)
        yield arm, reward, estimatedReward, weights
        t = t + 1


def runExp3Example():
    numActions = 10
    numRounds = 100000
    rewardVector = []
    with open("../rewards.txt", "r") as file:
        for line in file:
            line = line.strip().replace("[", "").replace("]", "")
            if line:
                row = [int(num) for num in line.split(",")]
                rewardVector.append(row)

    rewards = lambda arm, t: rewardVector[t][arm]
    cumulativeRewards = [sum([rewardVector[t][arm] for t in range(numRounds)]) for arm in range(numActions)]
    bestArm = max(range(numActions), key=lambda action: sum([rewardVector[t][action] for t in range(numRounds)]))

    # Exact value of Gmax
    gMax = cumulativeRewards[bestArm]

    # Optimal theoretical gamma
    gamma = math.sqrt(numActions * math.log(numActions) / ((math.e - 1) * gMax))

    cumulativeReward = 0
    bestArmCumulativeReward = 0

    # Upper bound of expected regret at each round
    regretUpperBound = (math.e - 1) * gamma * gMax + numActions * math.log(numActions) / gamma

    with open('exp3results.txt', 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        t = 0
        for (arm, reward, est, weights) in exp3(numActions, rewards, gamma):
            cumulativeReward += reward
            bestArmCumulativeReward += rewardVector[t][bestArm]

            weakRegret = (bestArmCumulativeReward - cumulativeReward)
            regretBound = 2 * math.sqrt(math.e - 1) * \
                          math.sqrt(bestArmCumulativeReward * numActions * math.log(numActions))
            print("regret: %d\tmaxRegret: %.2f\tweights: (%s)" % (
                weakRegret, regretBound, ', '.join(["%.3f" % weight for weight in distr(weights)])))
            t = t + 1
            if t >= numRounds:
                break
    sys.stdout = original_stdout
    print("Cumulative reward: ", cumulativeReward)
    print("Best arm reward: ", bestArmCumulativeReward)
    print("Regret:", cumulativeRewards[bestArm] - cumulativeReward)
    print("Regret upper bound:", regretUpperBound)
    print("Gamma: ", gamma)

def regretWeightsGraph(filename, title):
    with open(filename, 'r') as infile:
        lines = infile.readlines()
    lines = [[eval(x.split(": ")[1]) for x in line.split('\t')] for line in lines]
    data = transpose(lines)
    regret = np.array(data[0])
    regretBound = np.array(data[1])
    weights = np.array(transpose(data[2]))

    # Number of rounds
    xs = np.array(range(len(data[0])))
    ax1 = plt.subplot(211)
    plt.ylabel('Cumulative (weak) Regret')
    ax1.plot(xs, regret, label="Regret ")
    ax1.plot(xs, regretBound, label="Regret bound")
    plt.legend(loc="upper left")
    plt.title(title)

    ax2 = plt.subplot(212)
    plt.ylabel('Weight')

    for w in weights:
        ax2.plot(xs, w)
    # plt.show()
    # plt.legend(loc="upper left")
    plt.savefig('exp3optimal.png', dpi=200)
    # plt.savefig('exp3optimal.pgf', format='pgf')


if __name__ == '__main__':
    runExp3Example()
    regretWeightsGraph("exp3results.txt", "Exp3")
