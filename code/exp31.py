import math
import sys
import numpy as np
from numpy.random import choice
from matplotlib import pyplot as plt


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


def exp3_1(numEpochs, numRounds, numActions, rewards, bestArm=-1, rewardMin=0, rewardMax=1):
    epoch = 0  # number of epochs
    t = 0  # global number of rounds
    epochLimitReached = False
    timeLimitReached = False
    largerThanEstimate = False
    cumulativeReward = 0
    bestArmCumulativeReward = 0
    arms = np.array([i for i in range(numActions)])
    # Only for computing the regret, in real life we obviously do not know the best arm
    if bestArm == -1:
        bestArm = max(range(numActions), key=lambda action: sum(rewards(action, time) for time in range(numRounds)))
    # Iterate as long as there are samples
    with open('exp31results.txt', 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        while not timeLimitReached and not epochLimitReached:
            gMaxGuess = (numActions * math.log(numActions)) / (math.e - 1) * math.pow(4, epoch)
            gammaEpoch = min(float(1), math.sqrt((numActions * math.log(numActions)) / ((math.e - 1) * gMaxGuess)))

            rewardEstimates = [0.0] * numActions
            u = 0  # round for inner Exp3
            largerThanEstimate = False
            # Now start inner Exp3
            weights = [1.0] * numActions
            while not largerThanEstimate and not timeLimitReached:
                probabilityDistribution = distr(weights, gammaEpoch)
                arm = draw(probabilityDistribution, arms)
                reward = rewards(arm, t)
                normalizedReward = (reward - rewardMin) / (rewardMax - rewardMin)
                estimatedReward = float(normalizedReward / probabilityDistribution[arm])
                weights[arm] = weights[arm] * math.exp(estimatedReward * gammaEpoch / numActions)

                # Update some variables (reward, regret, etc)
                cumulativeReward = cumulativeReward + reward
                bestArmCumulativeReward = bestArmCumulativeReward + rewards(bestArm, t)

                weakRegret = (bestArmCumulativeReward - cumulativeReward)
                regretBound = 10.5 * math.sqrt(bestArmCumulativeReward * numActions * math.log(numActions)) + \
                              13.8 * numActions + 2 * numActions * math.log(numActions)

                rewardEstimates[arm] = rewardEstimates[arm] + estimatedReward

                print("regret: %d\tmaxRegret: %.2f\tweights: (%s)" % (
                    weakRegret, regretBound, ', '.join(["%.3f" % weight for weight in distr(weights)])))
                maxRewardEstimate = max(rewardEstimates[arm] for arm in range(numActions))
                t = t + 1
                u = u + 1
                if maxRewardEstimate > (gMaxGuess - numActions / gammaEpoch):
                    largerThanEstimate = True
                elif t >= numRounds:
                    timeLimitReached = True
            epoch = epoch + 1
            if epoch >= numEpochs:
                epochLimitReached = True
    sys.stdout = original_stdout
    print("Time limit reached with %d epochs!" % epoch)


def runExp3_1Example():
    numActions = 10
    numRounds = 100000
    numEpochs = 10

    rewardVector = []
    with open("../rewards.txt", "r") as file:
        for line in file:
            line = line.strip().replace("[", "").replace("]", "")
            if line:
                row = [int(num) for num in line.split(",")]
                rewardVector.append(row)

    rewards = lambda arm, t: rewardVector[t][arm]
    bestArm = max(range(numActions), key=lambda action: sum([rewardVector[t][action] for t in range(numRounds)]))

    exp3_1(numEpochs, numRounds, numActions, rewards, bestArm)


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
    ax1.plot(xs, regret, label="Regret")
    ax1.plot(xs, regretBound, label="Regret bound")
    ax1.legend()
    plt.title(title)

    ax2 = plt.subplot(212)
    plt.ylabel('Weight')
    for w in weights:
        ax2.plot(xs, w)
    plt.savefig('exp31.png', dpi=200)


if __name__ == '__main__':
    runExp3_1Example()
    regretWeightsGraph("exp31results.txt", "Exp3.1")
