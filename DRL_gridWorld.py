import math
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

MAX_LENGTH = 5
A = [0, 1]
A_PRIME = [4, 1]
B = [0, 3]
B_PRIME = [2, 3]

ACTIONS = [np.array([0, -1]),
		   np.array([-1, 0]),
		   np.array([0, 1]),
		   np.array([1, 0])]

def step(state, action):
	if state == A:
		return A_PRIME, 10
	if state == B:
		return B_PRIME, 5
	nextState = (np.array(state) + action).tolist()
	r, c = nextState
	reward = 0 # if inside grid
	if r < 0 or r >= MAX_LENGTH or c < 0 or c >= MAX_LENGTH: # if outside grid
		reward = -1.0
		nextState = state
	return nextState, reward

def alpha(t):
	return (1/math.ceil((t+1)/10))

def phi(state):
	stateIndex = state[0]*5 + state[1]
	phiState = np.zeros(25)
	phiState[stateIndex] = 1
	return phiState

def beta(t, c):
	return c*alpha(t)

def pickAction():
	return np.random.choice([0, 1, 2, 3])

def runExperiment(cValue):
	startStates = [[0, 0], [2, 2], [4, 4]]
	rAvgTrajectories = []
	for state in startStates:
		stateIndex = state[0]*5 + state[1]
		np.random.seed(2)
		wt = np.zeros(25)
		wtplus1 = np.zeros(25)
		numOfSteps = 5000
		rtAvg = 0
		rtplus1Avg = 0
		wtPhiPlot = []
		deltaPlot = []
		rAvgPlot = []
		for i in trange(numOfSteps):
			(nextRow, nextCol), reward = step([state[0], state[1]], ACTIONS[pickAction()])
			rtplus1Avg = rtAvg + beta(i, cValue)*(reward - rtAvg)
			deltat = reward - rtplus1Avg + np.sum(wt*phi([nextRow, nextCol])) - np.sum(wt*phi(state))
			wtplus1 = wt + alpha(i)*deltat*phi(state)

			# make necessary updates
			state = [nextRow, nextRow]
			wt = wtplus1
			rtAvg = rtplus1Avg

			# plots
			wtPhiPlot.append(wt[stateIndex])
			rAvgPlot.append(rtAvg)
			deltaPlot.append(deltat)
		labelStr = 'V(' + str(stateIndex) + ')'
		plt.figure(figsize = (10, 10))
		plt.plot(wtPhiPlot, label = labelStr)
		plt.legend()
		rAvgTrajectories.append(rAvgPlot)
	plt.show()

	for trajectory in range(0, len(rAvgTrajectories)):
		plt.subplot(3, 1, trajectory + 1)
		plt.plot(rAvgTrajectories[trajectory])
	plt.show()

if __name__ == '__main__':
	runExperiment(0.1)
	#runExperiment(1)
	#runExperiment(10)