# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:51:31 2017

@author: sam
"""

from board import Board, Move, Outcome
from ai import ProbBoard

		
		

class Node(object):
	MAYBE_MOVE_NODE=3
	MOVE_NODE=1
	BOARD_NODE=4
	
	def __init__(self, board, nodeType=BOARD_NODE, player=None, move=None, outcome=None, prob=1.0):
		self.board = board
		self.player = player
		self.move = move
		self.outcome = outcome
		self.prob = prob
		self.children=[]
		self.value = 0
		self.nodeType = nodeType
	
	def expand(self,player):
		preSplits,postSplits,doubles,nonattacking = self.createSplits(self.board,player)
		leafs = self.expandPreSplits(preSplits)
		for leaf in leafs:
			leaf.expandMoves(postSplits,doubles,nonattacking,player)
	
	def createSplits(self,board,player):
		""" Move types:
		1. Non-attacking
		2. Known
		3. Unknown defender
		4. Unknown attacker
		5. Double blind
		"""
		MOVE_ONLY=1
		KNOWN=2
		UNKNOWN_DEF=3
		UNKNOWN_ATT=4
		DOUBLE_BLIND=5
		FUTURE_KNOWN_DEF=6
		
		moves=board.getValidMoves(player)
		moveResults=[]
		for move in moves:
			attacker = board[move.fromPos]
			defender = board[move.toPos]
			if board.grid[move.toPos] == Board.EMPTY:
				moveType = MOVE_ONLY
				results=[(Outcome.MOVE,1.0,board.copy())]
				
			elif attacker.known and defender.known:
				moveType=KNOWN
				outcome = Outcome.fromRank(attacker.rank, defender.rank)
				results=[(outcome,1.0,board.copy())]
				
			elif attacker.known: # to unknown
				moveType = FUTURE_KNOWN_DEF if attacker.seen else UNKNOWN_DEF
				results = self.board.splitOnDefender(move.toPos, attacker.rank)
				
			elif defender.known:
				moveType = UNKNOWN_ATT
				results = self.board.splitOnAttacker(move.fromPos, defender.rank)
				
			else:
				moveType = DOUBLE_BLIND
				results = self.board.splitOnDefenderDoubleBlind(move.toPos, move.fromPos)
			
			moveResults.append([moveType,move,results])

		for moveType,move,results in moveResults:
			if len(results)==1 and results[0][0]==Outcome.LOSS:
				#print("Skipping losing move")
				continue

			#prenode = outcome known before taking the move.  post-node = outcome known after
			attackerKnowsSomethingWeDont = moveType in (UNKNOWN_ATT, FUTURE_KNOWN_DEF)
			if not attackerKnowsSomethingWeDont:
				# This is the case where the attacker has perfect knowledge, be we (the AI) don't
				# In this case, attacker will only take winning moves, but we have to guess if the move is winning
				# This is why losses are excluded and moves are given a probability depending on whether they are available
				# In this case every move, outcome pair is represented as a move node ,and the move node has only one child
				for outcome,prob,board in results:
					if outcome == Outcome.LOSS:
						continue
					moveNode = Node(self.board, self.MAYBE_MOVE_NODE, player, move, outcome, prob)
					self.children.append(moveNode)
					
					board.applyMove(move,outcome)
					board.addProbabilities()
					
					child = Node(board, self.BOARD_NODE, player, move, outcome, 1.0)
					moveNode.children.append(child)
			else:
				# This is the normal situation, where we know as much as the attacker
				# The move can have different outcomes with a prob distribution, which are
				# represented as child nodes of the move
				moveNode = Node(self.board, self.MOVE_NODE, player=player, move=move)
				self.children.append(moveNode)
				
				for outcome,prob,board in results:
					board.applyMove(move,outcome)
					board.addProbabilities()
					
					child = Node(board, self.BOARD_NODE, player, move, outcome, prob)
					moveNode.children.append(child)

	def getBoardValue(self, player):
		if self.nodeType != self.BOARD_NODE:
			raise Exception("getBoardValue must be called on board nodes")
		if len(self.children)==0:
			return self.board.heuristic()
		minimax = max if self.children[0].player==player else min
		for child in self.children:
			child.getMoveValue(player)
		
		candidates=list(self.children)
		prob=1.0
		bestMove=minimax(candidates, key=lambda node:node.value)
		if bestMove.nodeType == self.MAYBE_MOVE_NODE:
			value = prob*bestMove.value
			prob *= (1-bestMove.prob)
			candidates = [node for node in candidates if not (node.nodeType==self.MAYBE_MOVE_NODE and node.move.fromPos==bestMove.move.fromPos) ]
			if len(candidates)==0:
				raise Exception("What now?")
			bestMove=minimax(candidates, key=lambda node:node.value)
			# Loop here
		else:
			self.value = bestMove.value
		return self.value

	def getMoveValue(self,player):
		if not self.nodeType in (self.MOVE_NODE, self.MAYBE_MOVE_NODE):
			raise Exception("getMoveValue must be called on move nodes")
		self.value = sum(child.getBoardValue(player)*child.prob for child in self.children)
		return self.value

	"""	
	def getLeafs(self):
		if len(self.children)==0:
			return [self]
		else:
			arr=[]
			for child in self.children:
				arr.extend(child.getLeafs())
			return arr
	
	def getChildType(self):
		if len(self.children)==0:
			return self.LEAF
		else:
			return self.children[0].nodeType
			
	def getCumProd(self,prior=1.0):
		self.cumProb=self.prob*prior
		for child in self.children:
			child.getCumProd(self.cumProb)
	
	def getValue(self, player):
		childType=self.getChildType()
		if childType==self.LEAF:
			self.value = self.board.heuristic()
		elif childType==self.MOVE_NODE:
			movePlayer = self.children[0].player
			minimax = max if movePlayer==player else min
			self.value=minimax(child.getValue(player) for child in self.children)
		elif childType==self.PRE_NODE or childType==self.POST_NODE:
			self.value = sum(child.getValue(player)*child.prob for child in self.children)
		return self.value
	
	def greedySearch(self):
		childType=self.getChildType()
		if childType==self.LEAF or childType==self.POST_NODE:
			raise Exception("This should not happen")
		elif childType==self.PRE_NODE:
			raise Exception("This should not happen either - since pre-nodes are only for opponent moves")
			#node=max(self.children,key=lambda child:child.prob)
			#return node.greedySearch()
		elif childType==self.MOVE_NODE:
			node=max(self.children,key=lambda child:child.value)
			return node
	
	def __repr__(self):
		return "[NODE %s move=%s outcome=%s prob=%.2g value=%.1f]"%(self.nodeType,str(self.move),str(self.outcome),self.prob,self.value)
		
	def expandAll(self,player, level):
		queue=[self]
		curPlayer=player
		self.heuristic=self.board.heuristic()
		for i in range(level):
			print(len(queue))
			leafs=[]
			for node in queue:
				node.expand(curPlayer)
				node.getCumProd()
				node.getValue(player)
				node.moves=node.getMoves()
				for move in node.moves:
					move.heuristic = move.value
				leafs.extend([leaf for leaf in node.getLeafs() if leaf.cumProb>0.005])
			curPlayer=3-curPlayer
			queue=leafs
		self.getCumProd()
		self.getValue(player)
	
	def getMoves(self):
		if self.nodeType==self.MOVE_NODE:
			return [self]
		else:
			arr=[]
			for child in self.children:
				arr.extend(child.getMoves())
			return arr
	"""
			
		
"""
01 function alphabeta(node, depth, α, β, maximizingPlayer)
02      if depth = 0 or node is a terminal node
03          return the heuristic value of node
04      if maximizingPlayer
05          v := -∞
06          for each child of node
07              v := max(v, alphabeta(child, depth – 1, α, β, FALSE))
08              α := max(α, v)
09              if β ≤ α
10                  break (* β cut-off *)
11          return v
12      else
13          v := +∞
14          for each child of node
15              v := min(v, alphabeta(child, depth – 1, α, β, TRUE))
16              β := min(β, v)
17              if β ≤ α
18                  break (* α cut-off *)
19          return v
"""



def getBestMove(board,player):
	tree=Node(ProbBoard(board,player))
	tree.expand(player)
	node=tree.greedySearch()
	print(node.move,node.value,node.cumProb)
	return node.move

def doMove(board,player):
	board.applyMove(getBestMove(board,player))
	print(board)

def doMoves(board,n):
	for i in range(n):
		board.applyMove(getBestMove(board,1))
		board.applyMove(getBestMove(board,2))

#board=Board()
#board.placeRandom()
#print(board)
#doMoves(board,5)
#print(board)
#node=Node(ProbBoard(board,1))
#node.expand(1)

node=Node(ProbBoard(board2,2))
node.expandAll(2,2)

