# multiAgents.py
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

import random
from collections import defaultdict

import util
from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        value, action = self.minimax(gameState, self.depth * gameState.getNumAgents(), 0)
        return action

    def minimax(self, state, depth, agentIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        if agentIndex == 0:
            value = -float('inf')
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                actionValue, _ = self.minimax(nextState, depth - 1, nextAgent)
                if actionValue > value:
                    value = actionValue
                    bestAction = action
            return value, bestAction
        else:
            value = float('inf')
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                actionValue, _ = self.minimax(nextState, depth - 1, nextAgent)
                value = min(value, actionValue)
            return value, None


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value, action = self.alphabeta(gameState, -float('inf'), float('inf'), self.depth * gameState.getNumAgents(), 0)
        return action

    def alphabeta(self, state, alpha, beta, depth, agentIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
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
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                actionValue, _ = self.alphabeta(nextState, alpha, beta, depth - 1, nextAgent)
                value = min(value, actionValue)
                if value < alpha:
                    return value, _
                beta = min(beta, value)
            return value, None


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        value, action = self.expectimax(gameState, self.depth * gameState.getNumAgents(), 0)
        return action

    def expectimax(self, state, depth, agentIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        if agentIndex == 0:
            value = -float('inf')
            bestAction = None
            for action in state.getLegalActions(0):
                nextState = state.generateSuccessor(0, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                actionValue, _ = self.expectimax(nextState, depth - 1, nextAgent)
                if actionValue > value:
                    value = actionValue
                    bestAction = action
            return value, bestAction
        else:
            value = 0.
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                actionValue, _ = self.expectimax(nextState, depth - 1, nextAgent)
                value += actionValue
            value /= len(state.getLegalActions(agentIndex))
            return value, None


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

    # score += 0.05 * min(ghostDistances)

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
    f_score[start] = util.manhattanDistance(start, goal)

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
                f_score[u] = g_score[u] + util.manhattanDistance(u, goal)
                queue.push(u, f_score[u])
    print(start, goal)


def getPossibleNeighbors(pos, state):
    candidates = [(pos[0], pos[1] + 1), (pos[0] + 1, pos[1]),
                  (pos[0], pos[1] - 1), (pos[0] - 1, pos[1])]
    return [n for n in candidates if not state.hasWall(n[0], n[1])]


class MyAlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value, action = self.alphabeta(gameState, -float('inf'), float('inf'), self.depth * gameState.getNumAgents(), 0)
        return action

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
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                actionValue, _ = self.alphabeta(nextState, alpha, beta, depth - 1, nextAgent)
                value = min(value, actionValue)
                if value < alpha:
                    return value, _
                beta = min(beta, value)
            return value, None


# Abbreviation
better = betterEvaluationFunction
