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
	FLAG=0
	SPY=1
	SCOUT=2
	MINER=3
	GENERAL=9
	MARSHALL=10
	BOMB=11
	
	CHARS="F1234567890B"
		
	def __init__(self,id,rank,player):
		self.id=id
		self.rank=rank
		self.player=player
		self.seen=False
		self.moved=False
		self.captured=False
		self.char=('.' if self.player==2 else ' ')+self.CHARS[rank]

	
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
	
	def __repr__(self):
		return "[player=%d rank=%d seen=%d]"%(self.player,self.rank,self.seen)

class Move:
	def __init__(self,fromX,fromY,toX,toY):
		self.fromX=fromX
		self.fromY=fromY
		self.toX=toX
		self.toY=toY
	
	@classmethod
	def create(cls, fromPos, toPos):
		return cls(fromPos[0],fromPos[1],toPos[0],toPos[1])	
	
	@property
	def fromPos(self):
		return (self.fromX, self.fromY)
	
	@property
	def toPos(self):
		return (self.toX, self.toY)
	
	def dist(self):
		return (self.toPos[0]-self.fromPos[0],self.toPos[1]-self.fromPos[1])
	
	def __repr__(self):
		return "(%d,%d)-(%d,%d)"%(self.fromX,self.fromY,self.toX,self.toY)

class Board:
	RANK_COUNTS=[1,1,8,5,4,4,4,3,2,1,1,6]
	RANK_VALUES=np.array([0,2,3,6,4,5,6,7,9,12,15,8])
	PLAYERS=[1,2]
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
		
		self.grid=np.empty((10,10),dtype=int)
		self.grid.fill(self.EMPTY)
		self.grid[[2,3,6,7],4:6]=self.LAKE
		
	
	def placePlayerRandom(self,pieces,rows):
		bombs=[]
		for piece in pieces:
			if piece.rank==Piece.FLAG:
				flag=piece
			elif piece.rank==Piece.BOMB:
				bombs.append(piece)
		
		#Place flag at back surrounded by bombs
		flagX=np.random.randint(10)
		self.grid[flagX,rows[0]]=flag.id
		pieces.remove(flag)
		bombPositions=[(flagX-1,rows[0]),(flagX+1,rows[0]),(flagX,rows[1])]
		for bombPos in bombPositions:
			if bombPos[0]>=0 and bombPos[0]<10:
				bomb=bombs.pop()
				pieces.remove(bomb)
				self.grid[bombPos]=bomb.id
		
		np.random.shuffle(pieces)		
		
		for y in rows:
			for x in range(10):
				if self.grid[x,y]==Board.EMPTY:
					self.grid[x,y]=pieces.pop().id
	
	def placeRandom(self):
		self.placePlayerRandom([piece for piece in self.pieces if piece.player==1],(9,8,7,6))
		self.placePlayerRandom([piece for piece in self.pieces if piece.player==2],(0,1,2,3))

		
	def isValidPos(self,pos):
		return np.min(pos)>=0 and np.max(pos)<10
	
	def isValidMove(self,move):
		if not self.isValidPos(move.fromPos) or not self.isValidPos(move.toPos):
			return False
		
		if self.grid[move.fromPos]<0:
			return False
		if self.grid[move.toPos]==self.LAKE:
			return False
		fromPiece=self[move.fromPos]
		toPiece=self[move.toPos]
		if not fromPiece.moveable:
			return False
			
		if toPiece!=None and toPiece.player==fromPiece.player:
			return False
			
		dist=move.dist()
		if np.count_nonzero(dist)!=1:
			return False
		if fromPiece.isScout():
			direction=np.sign(dist)
			pos=move.fromPos+direction
			while np.any(pos!=move.toPos):
				if not self.grid[tuple(pos)]==Board.EMPTY:
					return False
				pos+=direction
		else:
			if np.max(np.abs(dist))!=1:
				return False

		return True
		
	def __getitem__(self,pos):
		pieceId=self.grid[tuple(pos)]
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
				if not fromPiece.moveable:
					continue
				if fromPiece.isScout():
					for i in range(10):
						toPos=fromPos.copy()
						toPos[0]=i
						move=Move.create(fromPos,toPos)
						if self.isValidMove(move):
							moves.append(move)
						toPos=fromPos.copy()
						toPos[1]=i
						move=Move.create(fromPos,toPos)
						if self.isValidMove(move):
							moves.append(move)
				else:
					for direction in directions:
						toPos=fromPos+direction
						move=Move.create(fromPos,toPos)
						if self.isValidMove(move):
							moves.append(move)
		return moves

	def applyMove(self,move):
		fromPos=move.fromPos
		toPos=move.toPos
		
		fromPiece=self[fromPos]
		fromPiece.moved=True
		if self.grid[toPos] == self.EMPTY:
			if max(toPos[0]-fromPos[0], toPos[1]-fromPos[1])>1:
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
	

	def charAt(self,x,y):
		id=self.grid[x,y]
		if id==self.EMPTY:
			return "  "
		elif id==self.LAKE:
			return " X"
		else:
			return self.pieces[id].char

	def __repr__(self):
		return '\n'.join([''.join([self.charAt(x,y) for x in range(10)]) for y in range(10)])						



