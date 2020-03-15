import random
from tqdm import trange

gamma = 1 # not being used in actual calculation
policy = [['s'],['l'],['l'],['l','d'],['u'],['u','l'],['l','d'],['d'],['u'],['u','r'],['d','r'],['d'],['u','r'],['r'],['r'],['s']]
directionValues = {
	'l': -1,
	'd': 4,
	'u': -4,
	'r': 1
}
directionPositions = {
	'l': 0,
	'd': 1,
	'u': 2,
	'r': 3
}
terminalStates = [0,15]

def both_methods(numOfEpisodes):
	firstVisitedCounts = [0]*16
	firstGridValues = [0]*16
	everyVisitedCounts = [0]*16
	everyGridValues = [0]*16
	for episode in trange(numOfEpisodes):
		states, rewards = run_episode() # no need to use rewards because gamma is 1
		visited = [False]*16
		totalStates = len(states)
		for i, state in enumerate(states):
			everyVisitedCounts[state] += 1
			everyGridValues[state] += i - totalStates + 1
			if not visited[state]:
				visited[state] = True
				firstVisitedCounts[state] += 1
				firstGridValues[state] += i - totalStates + 1
	for i, value in enumerate(firstGridValues):
		firstGridValues[i] = value/firstVisitedCounts[i]
	for i, value in enumerate(everyGridValues):
		everyGridValues[i] = value/everyVisitedCounts[i]
	return firstGridValues, everyGridValues

def first_visit_method(numOfEpisodes):
	visitedCounts = [0]*16
	gridValues = [0]*16
	for episode in trange(numOfEpisodes):
		states, rewards = run_episode() # no need to use rewards because gamma is 1
		visited = [False]*16
		totalStates = len(states)
		for i, state in enumerate(states):
			if not visited[state]:
				visited[state] = True
				visitedCounts[state] += 1
				gridValues[state] += i - totalStates + 1
	for i, value in enumerate(gridValues):
		gridValues[i] = value/visitedCounts[i]
	return gridValues

def every_visit_method(numOfEpisodes):
	visitedCounts = [0]*16
	gridValues = [0]*16
	for episode in trange(numOfEpisodes):
		states, rewards = run_episode() # no need to use rewards because gamma is 1
		totalStates = len(states)
		for i, state in enumerate(states):
			visitedCounts[state] += 1
			gridValues[state] += i - totalStates + 1
	for i, value in enumerate(gridValues):
		gridValues[i] = value/visitedCounts[i]
	return gridValues

def select_action(state):
	return random.choice(policy[state])

def take_step(state, direction):
	probabilities = [0.1]*4
	probabilities[directionPositions[direction]] = 0.7
	step = random.choices(['l','d','u','r'],probabilities)[0]
	stepValue = directionValues[step]
	nextState = state + stepValue
	if nextState < 0 or nextState > 15:
		nextState = state
	return nextState, reward(nextState)

def reward(nextState):
	if nextState in terminalStates:
		return 0
	else:
		return -1

def run_episode():
	startState = random.choice(range(1,15))
	nextState = startState
	states = [startState]
	rewards = [-1]
	while nextState not in terminalStates:
		nextState, r = take_step(nextState, select_action(nextState))
		rewards.append(r)
		states.append(nextState)
	return states, rewards

if __name__ == "__main__":
	# For separate experiments aka different episode generation uncomment below lines
	#print(first_visit_method(100))
	#print(every_visit_method(100))

	# Same episode generation
	first, every = both_methods(100)
	print(first)
	print(every)