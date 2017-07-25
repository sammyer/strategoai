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
		self.heuristic=board.heuristic()
		self.children=None #moves
	
	def searchMoves(self, player):
		moves=self.board.getValidMoves(player)
		self.children = [self.makeMoveNode(move) for move in moves]
		isPlayer = player==self.board.knownPlayer
		self.children.sort(key=lambda x:x.heuristic, reverse=isPlayer)
	
	def makeMoveNode(self,move):
		outcomes = self.board.applyMoveProbabilistic(move)
		boards = [BoardNode(board,prob) for board,prob in outcomes]
		return MoveNode(move, boards)

class MoveNode:
	def __init__(self,move,boardNodes):
		self.move=move
		self.children=boardNodes #boards
		self.heuristic = sum(board.prob * board.heuristic for board in boardNodes)
		

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

#board2=makeBoard()

tree=makeBoardTree(board2)
tree.searchMoves(1)

#board=ProbBoard(board2,1)
#print(board.heuristic())
#a10=board.applyMoveProbabilistic(Move(0,4,0,3))
#a11=board.getValidMoves(1)
#a12=[board.applyMoveProbabilistic(i) for i in a11]