# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:51:31 2017

@author: sam
"""

from board import Board, Move
from ai import ProbBoard



def moveHeuristic(boards):
	return sum(prob*board.heuristic() for board,prob in boards)

def makeBoard():
	board=Board()
	board.placeRandom()
	moves=board.getValidMoves(1)
	board.applyMove(moves[0])
	moves=board.getValidMoves(1)
	board.applyMove(moves[2])
	return board

#board2=makeBoard()
board=ProbBoard(board2,1)
print(board.heuristic())
a10=board.applyMoveProbabilistic(Move(0,4,0,3))
a11=board.getValidMoves(1)
a12=[board.applyMoveProbabilistic(i) for i in a11]