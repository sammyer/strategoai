# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 12:38:17 2017

@author: sam
"""

import numpy as np

def normalize(x):
	s=x.sum().astype(float)
	if s==0:
		return x
	else:
		return x/s

class Piece:
	SPY=1
	SCOUT=2
	MINER=3
	GENERAL=9
	MARSHALL=10
	BOMB=11
	FLAG=12
	
	EMPTY=0
	LAKE=15
	
	def __init__(self,pid=0,player=0):
		self.pid=pid
		self.player=player
		self.seen=False
		self.moved=False
		self.captured=False
	
	def char(self):
		p1=" 1234567890bf  X"
		p2=" 1234567890BF  X"
		if self.player==1:
			return p1[self.pid]
		else:
			return p2[self.pid]
	
	def copy(self):
		piece=Piece(self.pid,self.player)
		piece.seen=self.seen
		piece.moved=self.moved
		piece.captured=self.captured
		return piece
	
	def isUnseen(self,byPlayer):
		return not (self.player==byPlayer or self.captured or self.moved)
	

class Board:

	def __init__(self,board=None,pieces=None):
		if pieces is None:
			self.resetBoard()
		else:
			self.board=board
			self.pieces=pieces

	def resetBoard(self):
		pieceCounts=[0,1,8,5,4,4,4,3,2,1,1,6,1]
		self.pieces=[]
		for player in (1,2):
			for pieceId,count in enumerate(pieceCounts):
				for i in range(count):
					self.pieces.append(Piece(pieceId,player))
		self.board=[[Piece() for i in range(10)] for j in range(10)]
		for x in (2,3,6,7):
			for y in (4,5):
				self.board[x][y]=Piece(Piece.LAKE,0)
		
	
	def placeRandom(self):
		playerPieces=[[piece for piece in self.pieces if piece.player==player] for player in (1,2)]
		playerPieces=[np.random.permutation(pieces) for pieces in playerPieces]
		for y in range(4):
			for x in range(10):				
				self.board[x][6+y]=playerPieces[0][y*10+x]
				self.board[x][3-y]=playerPieces[1][y*10+x]
	
	def isValidMove(self,move):
		fromPos,toPos=move
		if not self.isValidPos(fromPos) or not self.isValidPos(toPos):
			return False
		fromPiece=self[fromPos]
		toPiece=self[toPos]
		if fromPiece.pid in (0,11,12,15):
			return False
		if toPiece.pid==15:
			return False
		if toPiece.player==fromPiece.player:
			return False
			
		dist=toPos-fromPos
		if np.count_nonzero(dist)!=1:
			print(5)
			return False
		if fromPiece.pid==Piece.SCOUT:
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
		return self.board[pos[0]][pos[1]]
	
	def __setitem__(self,pos,piece):
		self.board[pos[0]][pos[1]]=piece
		
	def isValidPos(self,pos):
		return np.min(pos)>=0 and np.max(pos)<10
	
	def getValidMoves(self,player):
		moves=[]
		directions=[[1,0],[-1,0],[0,1],[0,-1]]
		for x in range(10):
			for y in range(10):
				fromPos=np.array([x,y])
				fromPiece=self.board[x][y]
				if not fromPiece.player==player:
					continue
				if fromPiece.pid in (0,11,12,15):
					continue
				if fromPiece.pid==Piece.SCOUT:
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

	def applyMove(self,move):
		fromPos,toPos=move
		fromPiece=self[fromPos]
		toPiece=self[toPos]
		fromPiece.moved=True
		if toPiece.pid==0:
			win=1
			if np.max(toPos-fromPos)>1:
				# Now we know it's a scout
				fromPiece.seen=True
		else:
			win=np.sign(toPiece.pid-fromPiece.pid)
			if fromPiece.pid==Piece.MINER and toPiece.pid==Piece.BOMB:
				win=1
			if toPiece.pid==Piece.FLAG:
				win=1
				self.endgame=True
			if fromPiece.pid==Piece.SPY and toPiece.pid==Piece.MARSHALL:
				win=1
			fromPiece.seen=True
			
		self[fromPos]=Piece() # from pos is now empty
		if win==1:
			toPiece.captured=True
			fromPiece.seen=True
			self[toPos]=fromPiece
		elif win==0:
			fromPiece.captured=True
			toPiece.captured=True
			self[toPos]=Piece()
		elif win==-1:
			fromPiece.captured=True
			toPiece.seen=True
	
	# AI	
	
	def addProbabilities(self,knownPlayer):
		immovableMask=np.zeros((15,))
		immovableMask[11:13]=1
		movableMask=1-immovableMask
		
		unmovedCount=0
		unseen=np.zeros((15,))
		for piece in self.pieces:
			if piece.isUnseen(knownPlayer):
				if not piece.moved:
					unmovedCount+=1
				unseen[piece.pid]+=1
			
		movedProb=normalize(unseen*movableMask)
		if unmovedCount==0:
			unmovedProb=np.zeros((15,))
		else:
			unmovedProb=(unseen*immovableMask)/float(unmovedCount)
			unmovedProb+=movedProb*(1-unmovedProb.sum())

		self.pieceProbs={}
		for piece in self.pieces:
			if piece.isUnseen(knownPlayer):
				if piece.moved:
					self.pieceProbs[piece]=movedProb
				else:
					self.pieceProbs[piece]=unmovedProb
		
	
	def applyMoveProbabilistic(self,move,knownPlayer):
		pass
			
	def heuristic(self,player=1):
		"""
		Points for captured pieces
		Points for revealed pieces
		Points for last miner
		Spy value depends on marshall existing
		Points for unprotected flag??
		Points for invincible pieces
		"""
		values=[0,10,4,6,4,5,6,7,9,12,15,8,100]
		points=0
		for piece in self.pieces:
			sign = 1 if piece.player==player else -1
			value=values[piece.pid]
			if piece.captured:
				pass
		

	def __repr__(self):
		return '\n'.join([''.join([self.board[x][y].char() for x in range(10)]) for y in range(10)])						

	def copy(self):
		board=Board()
		

board=Board()
board.placeRandom()
