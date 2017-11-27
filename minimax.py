# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:51:31 2017

@author: sam
"""

from board import Board, Move
from ai import ProbBoard


class Node:
	PRE_NODE="pre"
	MOVE_NODE="move"
	POST_NODE="post"
	ROOT_NODE="root"
	LEAF="leaf"
	
	def __init__(self, board, nodeType=ROOT_NODE, move=None, outcome=None, prob=1.0):
		self.board = board
		self.move = move
		self.outcome = outcome
		self.prob = prob
		self.children=[]
		self.value = 0
		self.nodeType = nodeType
	
	def expand(self,player):
		preSplits,postSplits,doubles = self.createSplits(self.board,player)
		leafs = self.expandPreSplits(preSplits)
		for leaf in leafs:
			leaf.expandMoves(postSplits,doubles)
		self.getCumProd()
		self.getValue()
	
	def createSplits(self,board,player):
		moves=board.getValidMoves(player)
		for move in moves:
			move.isAttack = board.grid[move.toPos] != Board.EMPTY
		attackingMoves=[move for move in moves if move.isAttack]
		preSplits=[]
		postSplits=[]
		doubles=[]
		
		if player==board.knownPlayer:
			for move in attackingMoves:
				if board[move.toPos].seen:
					# strightforward
					preSplits.append(move)
				else:
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
		
		return preSplits,postSplits,doubles

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

	def expandMoves(self,postSplits,doubles):
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
			moveNode = Node(self.board, self.MOVE_NODE, move=move)
			self.children.append(moveNode)
			for outcome,prob,board in results:
				board.applyMove(move,outcome)
				board.addProbabilities()
				child = Node(board, self.POST_NODE, move, outcome, prob)
				moveNode.children.append(child)
			
	
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
	
	def getValue(self):
		childType=self.getChildType()
		if childType==self.LEAF:
			self.value = self.board.heuristic()
		elif childType==self.MOVE_NODE:
			self.value=max(child.getValue() for child in self.children)
		elif childType==self.PRE_NODE or childType==self.POST_NODE:
			self.value = sum(child.getValue()*child.prob for child in self.children)
		return self.value
	
	def greedySearch(self):
		childType=self.getChildType()
		if childType==self.LEAF or childType==self.POST_NODE:
			raise Exception("This should not happen")
		elif childType==self.PRE_NODE:
			node=max(self.children,key=lambda child:child.prob)
			return node.greedySearch()
		elif childType==self.MOVE_NODE:
			node=max(self.children,key=lambda child:child.value)
			return node
	
	def __repr__(self):
		return "[NODE %s move=%s outcome=%s prob=%.2g value=%.1f]"%(self.nodeType,str(self.move),str(self.outcome),self.prob,self.value)
		

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

def searchTreeAux(node,depth,alpha,beta, maximizingPlayer):
	node.expandNode()

def searchTree(board,depth,player=1):
	# Note: player here denotes the player whose perspecitve it is for hidden pieces
	# This is not necesarrily whose player is making the move
	root=BoardNode(ProbBoard(board,player))
	searchTreeAux(root,depth,-np.inf,np.inf, True)


def makeBoardTree(board, player=1):
	return BoardNode(ProbBoard(board,player))

def makeBoard():
	board=Board()
	board.placeRandom()
	moves=board.getValidMoves(1)
	board.applyMove(moves[0])
	moves=board.getValidMoves(1)
	board.applyMove(moves[2])
	return board

def getBestMove(board,player):
	tree=Node(ProbBoard(board,player))
	tree.expand(player)
	node=tree.greedySearch()
	print(node.move,node.value,node.cumProd)
	return node.move

def doMove(board,player):
	board.applyMove(getBestMove(board,player))
	print(board)

def doMoves(board,n):
	for i in range(n):
		board.applyMove(getBestMove(board,1))
		board.applyMove(getBestMove(board,2))

#board2=makeBoard()

#tree=makeBoardTree(board2)
#tree.searchMoves(1)

board=Board()
board.placeRandom()
print(board)
#doMoves(board,5)
#print(board)
node=Node(ProbBoard(board,1))
node.expand(1)
