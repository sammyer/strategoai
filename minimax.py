# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:51:31 2017

@author: sam
"""

from board import Board, Move
from ai import ProbBoard

		
		

class Node(object):
	PRE_NODE="pre"
	MOVE_NODE="move"
	POST_NODE="post"
	ROOT_NODE="root"
	LEAF="leaf"
	
	def __init__(self, board, nodeType=ROOT_NODE, player=None, move=None, outcome=None, prob=1.0):
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
		moves=board.getValidMoves(player)
		for move in moves:
			move.isAttack = board.grid[move.toPos] != Board.EMPTY
		attackingMoves = [move for move in moves if move.isAttack]
		nonattacking = [move for move in moves if not move.isAttack]
		preSplits=[]
		postSplits=[]
		doubles=[]
		
		if player==board.knownPlayer:
			for move in attackingMoves:
				postSplits.append(move)
		else:
			for move in attackingMoves:
				if board[move.fromPos].seen:
					if board[move.toPos].seen:
						pass
					else:
						postSplits.append(move)
				else:
					if board[move.toPos].seen:
						preSplits.append(move)
					else:
						doubles.append(move)
		
		return preSplits,postSplits,doubles, nonattacking

	def expandPreSplits(self,preSplits,idx=0):
		if idx == len(preSplits):
			return [self]
		move = preSplits[idx]
		
		attackerPos = move.fromPos
		defender = self.board[move.toPos]
		if not defender.known:
			raise Exception("Error: pre split defender must be known")
		results = self.board.splitOnAttacker(attackerPos, defender.rank)

		leafs=[]
		for outcome,prob,board in results:
			board.addProbabilities()
			child = Node(board, self.PRE_NODE,prob=prob)
			self.children.append(child)
			newLeafs = child.expandPreSplits(preSplits,idx+1)
			leafs.extend(newLeafs)
		return leafs

	def expandMoves(self,postSplits,doubles,nonattacking,player):
		moves=[]
		for move in doubles:
			results = self.board.splitOnDefenderDoubleBlind(move.toPos, move.fromPos)
			moves.append((move,results))
				
		for move in postSplits:
			defenderPos = move.toPos
			attacker = self.board[move.fromPos]
			if not attacker.known:
				print(move,attacker)
				raise Exception("Error: post split attacker must be known")
			results = self.board.splitOnDefender(defenderPos, attacker.rank)
			moves.append((move,results))
		
		for move,results in moves:
			if len(results)==1 and results[0][0]==ProbBoard.LOSS:
				#print("Skipping losing move")
				continue
			moveNode = Node(self.board, self.MOVE_NODE, player=player, move=move)
			self.children.append(moveNode)
			for outcome,prob,board in results:
				board.applyMove(move,outcome)
				board.addProbabilities()
				child = Node(board, self.POST_NODE, player, move, outcome, prob)
				moveNode.children.append(child)

		for move in nonattacking:
			board = ProbBoard(self.board, self.board.knownPlayer)
			board.applyMove(move,None)
			moveNode = Node(board, self.MOVE_NODE, player=player, move=move)			
			self.children.append(moveNode)
	
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

