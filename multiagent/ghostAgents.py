# ghostAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from collections import defaultdict

import util
from game import Actions
from game import Agent
from game import Directions
from util import manhattanDistance


class GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index): dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()
        return dist


class AlphaBetaGhost(GhostAgent):
    def __init__(self, index, depth=1, bestProb=0.8):
        self.index = index
        self.depth = depth
        self.bestProb = bestProb

    def getDistribution(self, state):
        value, action = self.alphabeta(state, -float('inf'), float('inf'),
                                       self.depth * state.getNumAgents(), self.index)
        dist = util.Counter()
        dist[action] = self.bestProb
        legalActions = state.getLegalActions(self.index)
        for a in legalActions:
            dist[a] += (1 - self.bestProb) / len(legalActions)
        dist.normalize()
        return dist

    def alphabeta(self, state, alpha, beta, depth, agentIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return betterEvaluationFunction(state), None
        if agentIndex == 0:
            value = -float('inf')
            bestAction = None
            for action in state.getLegalActions(0):
                nextState = state.generateSuccessor(0, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                actionValue, _ = self.alphabeta(nextState, alpha, beta, depth - 1, nextAgent)
                if actionValue > value:
                    value = actionValue
                    bestAction = action
                if value > beta:
                    return value, bestAction
                alpha = max(alpha, value)
            return value, bestAction
        else:
            value = float('inf')
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                actionValue, _ = self.alphabeta(nextState, alpha, beta, depth - 1, nextAgent)
                if actionValue < value:
                    value = actionValue
                    bestAction = action
                if value < alpha:
                    return value, bestAction
                beta = min(beta, value)
            return value, bestAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    ghostDistances = []
    for i in range(1, currentGameState.getNumAgents()):
        ghostPos = currentGameState.getGhostPosition(i)
        ghostPos = (int(ghostPos[0]), int(ghostPos[1]))
        dist = aStarDist(currentGameState.getPacmanPosition(),
                         ghostPos,
                         currentGameState)
        ghostDistances.append(dist)

    score = 0
    score += currentGameState.getScore()

    score += sum(ghostDistances)

    score -= currentGameState.getNumFood()
    if currentGameState.getNumFood() > 0:
        score -= 0.1 * closestFoodDist(currentGameState)

    capsules = currentGameState.getCapsules()
    score -= 20 * len(capsules)
    if len(capsules) > 0:
        capsuleDistances = [aStarDist(currentGameState.getPacmanPosition(),
                                      c,
                                      currentGameState) for c in capsules]
        score -= 0.1 * min(capsuleDistances)
    return score


def closestFoodDist(state):
    visited = set()
    queue = util.Queue()
    queue.push((state.getPacmanPosition(), 0))
    while not queue.isEmpty():
        s, d = queue.pop()
        if state.hasFood(s[0], s[1]):
            return d
        visited.add(s)
        for u in getPossibleNeighbors(s, state):
            if u not in visited:
                queue.push((u, d + 1))


def aStarDist(start, goal, state):
    visited = set()

    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0

    f_score = defaultdict(lambda: float('inf'))
    f_score[start] = manhattanDistance(start, goal)

    queue = util.PriorityQueue()
    queue.push(start, f_score[start])
    while not queue.isEmpty():
        s = queue.pop()
        if s == goal:
            return g_score[s]
        visited.add(s)
        for u in getPossibleNeighbors(s, state):
            new_g = g_score[s] + 1
            if new_g < g_score[u]:
                g_score[u] = new_g
                f_score[u] = g_score[u] + manhattanDistance(u, goal)
                queue.push(u, f_score[u])
    print(start, goal)


def getPossibleNeighbors(pos, state):
    candidates = [(pos[0], pos[1] + 1), (pos[0] + 1, pos[1]),
                  (pos[0], pos[1] - 1), (pos[0] - 1, pos[1])]
    return [n for n in candidates if not state.hasWall(n[0], n[1])]
