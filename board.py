# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 12:38:17 2017

@author: sam

TODO: make ProbabilisticPiece and ProbabilisticBoard
make heuristic probabilistic

"""

import numpy as np

def normalize(x):
	s=x.sum().astype(float)
	if s==0:
		return x
	else:
		return x/s

class Piece:
	FLAG=0
	SPY=1
	SCOUT=2
	MINER=3
	GENERAL=9
	MARSHALL=10
	BOMB=11
		
	def __init__(self,id,rank,player):
		self.id=id
		self.rank=rank
		self.player=player
		self.seen=False
		self.moved=False
		self.captured=False
	
	def char(self):
		p1="f1234567890b   X"
		p2="F1234567890B   X"
		if self.player==1:
			return p1[self.rank]
		else:
			return p2[self.rank]

	
	def defeats(self,defender):
		win=np.sign(self.rank - defender.rank)
		if self.rank==Piece.MINER and defender.rank==Piece.BOMB:
			win=1
		elif self.rank==Piece.SPY and defender.rank==Piece.MARSHALL:
			win=1
		return win
	
	@property
	def moveable(self):
		return self.rank not in (Piece.BOMB, Piece.FLAG)
	
	def isScout(self):
		return self.rank==self.SCOUT
	
	

class Board:
	RANK_COUNTS=[1,1,8,5,4,4,4,3,2,1,1,6]
	RANK_VALUES=np.array([100,2,3,6,4,5,6,7,9,12,15,8])
	EMPTY=-1
	LAKE=-2

	def __init__(self):
		self.resetBoard()

	def resetBoard(self):
		self.pieces=[]
		pieceId=0
		for player in (1,2):
			for rank,count in enumerate(self.RANK_COUNTS):
				for i in range(count):
					self.pieces.append(Piece(pieceId,rank,player))
					pieceId+=1
		
		self.grid=np.array((10,10),dtype=int)
		self.grid.fill(self.EMPTY)			
		self.grid[(2,3,6,7),(4,5)]=self.LAKE
		
	
	def placeRandom(self):
		playerPieces=[[piece for piece in self.pieces if piece.player==player] for player in (1,2)]
		playerPieces=[np.random.permutation(pieces) for pieces in playerPieces]
		for y in range(4):
			for x in range(10):				
				self[x,6+y]=playerPieces[0][y*10+x]
				self[x,3-y]=playerPieces[1][y*10+x]

		
	def isValidPos(self,pos):
		return np.min(pos)>=0 and np.max(pos)<10
	
	def isValidMove(self,move):
		fromPos,toPos=move
		if not self.isValidPos(fromPos) or not self.isValidPos(toPos):
			return False
			
		if self.grid[fromPos] in (self.EMPTY, self.LAKE):
			return False
		if self.grid[toPos]==self.LAKE:
			return False
		fromPiece=self[fromPos]
		toPiece=self[toPos]
		if not fromPiece.movable():
			return False
			
		if toPiece!=None and toPiece.player==fromPiece.player:
			return False
			
		dist=toPos-fromPos
		if np.count_nonzero(dist)!=1:
			return False
		if fromPiece.isScout():
			direction=np.sign(dist)
			pos=fromPos+direction
			while np.any(pos!=toPos):
				if not self[pos] is None:
					return False
				pos+=direction
		else:
			if np.max(np.abs(dist))!=1:
				return False

		return True
		
	def __getitem__(self,pos):
		pieceId=self.grid[pos]
		if pieceId<0:
			return None
		else:
			return self.pieces[pieceId]
	
#	def __setitem__(self,pos,piece):
#		self.grid[pos]=piece.id
	
	def getValidMoves(self,player):
		moves=[]
		directions=[[1,0],[-1,0],[0,1],[0,-1]]
		for x in range(10):
			for y in range(10):
				fromPos=np.array([x,y])
				fromPiece=self[x,y]
				if fromPiece==None or not fromPiece.player==player:
					continue
				if not fromPiece.movable():
					continue
				if fromPiece.isScout():
					for i in range(10):
						toPos=fromPos.copy()
						toPos[0]=i
						move=(fromPos,toPos)
						if self.isValidMove(move):
							moves.append(move)
						toPos=fromPos.copy()
						toPos[1]=i
						move=(fromPos,toPos)
						if self.isValidMove(move):
							moves.append(move)
				else:
					for direction in directions:
						toPos=fromPos+direction
						move=(fromPos,toPos)
						if self.isValidMove(move):
							moves.append(move)
		return moves

	def applyMove2(self,fromX,fromY,toX,toY):
		move=np.array([[fromX,fromY],[toX,toY]])
		self.applyMove(move)

	def applyMove(self,move):
		fromPos,toPos=move
		fromPiece=self[fromPos]
		fromPiece.moved=True
		if self.grid[toPos] == self.EMPTY:
			if np.max(toPos-fromPos)>1:
				# Now we know it's a scout
				fromPiece.seen=True
			self.grid[fromPos]=self.EMPTY # from pos is now empty
			self.grid[toPos]=fromPiece.id
		else:
			toPiece=self[toPos]
			win = fromPiece.defeats(toPiece)
			if toPiece.rank == Piece.FLAG: self.endgame=True
			fromPiece.seen=True
			
			self.grid[fromPos]=self.EMPTY # from pos is now empty
			if win==1:
				toPiece.captured=True
				fromPiece.seen=True
				self.grid[toPos]=fromPiece.id
			elif win==0:
				fromPiece.captured=True
				toPiece.captured=True
				self.grid[toPos]=self.EMPTY
			elif win==-1:
				fromPiece.captured=True
				toPiece.seen=True
	


	def __repr__(self):
		return '\n'.join([''.join([self.board[x][y].char() for x in range(10)]) for y in range(10)])						


def makeBoard():
	board=Board()
	board.placeRandom()
	moves=board.getValidMoves(1)
	board.applyMove(moves[0])
	moves=board.getValidMoves(1)
	board.applyMove(moves[2])
	return board

#board2=makeBoard()
board=board2.copy()
print(board.heuristic())
a10=board.applyMoveProb(0,4,0,3,1)
