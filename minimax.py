# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:51:31 2017

@author: sam
"""

from board import Board, Move
from ai import ProbBoard

class BoardNode:
	def __init__(self,board,prob=1.0):
		self.board=board
		self.prob=prob
		self.localHeuristic=board.heuristic()
		self.heuristic=self.localHeuristic
		self.children=None #moves
	
	def expandNode(self, player):
		self.searchMoves(player,1)
	
	def searchMoves(self, player, depth=1):
		moves=self.board.getValidMoves(player)
		self.children = [self.makeMoveNode(move) for move in moves]
		isPlayer = player==self.board.knownPlayer
		self.children.sort(key=lambda x:x.heuristic, reverse=isPlayer)
		if depth>1:
			for child in self.children:
				child.searchMoves(3-player, depth-1)
		self.heuristic=self.children[0].heuristic
	
	def makeMoveNode(self,move):
		outcomes = self.board.applyMoveProbabilistic(move)
		boards = [BoardNode(board,prob) for board,prob in outcomes]
		return MoveNode(move, boards)
	
	def __getitem__(self,pos):
		return self.children[pos]

class MoveNode:
	def __init__(self,move,boardNodes):
		self.move=move
		self.children=boardNodes #boards
		self.localHeuristic = sum(board.prob * board.heuristic for board in boardNodes)
		self.heuristic=self.localHeuristic
	
	def searchMoves(self, player, depth):
		for child in self.children:
			child.searchMoves(player, depth)
		self.heuristic = sum(board.prob * board.heuristic for board in self.children)
	
	def __getitem__(self,pos):
		return self.children[pos]



def expandNode(board,player):
	moves=board.getValidMoves(player)
	for move in moves:
		move.isAttack = board.grid[move.toPos] != Board.EMPTY
	attackingMoves=[move for move in moves if move.isAttack]
	if player==board.knownPlayer:
		for move in attackingMoves:
			if board[move.toPos].isSeen:
				# strightforward
				preSplits.append(move)
			else:
				postSplits.append(move)
	else:
		for move in attackingMoves:
			if board[move.fromPos].isSeen:
				if board[move.toPos].isSeen:
					pass
				else:
					postSplits.append(move)
			else:
				if board[move.toPos].isSeen:
					preSplits.append(move)
				else:
					double.append(move)
				

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
	node=BoardNode(ProbBoard(board,player))
	node.searchMoves(player)
	move=node.children[0]
	print(move.move,node.heuristic,move.heuristic)
	return move.move

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
doMoves(board,5)
print(board)