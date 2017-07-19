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
	
	EMPTY=14
	LAKE=15
	
	def __init__(self,pid=EMPTY,player=0):
		self.pid=pid
		self.player=player
		self.seen=False
		self.moved=False
		self.captured=False
		self.possibleIds=np.ones((12,)) # unknown piece
	
	def char(self):
		p1="f1234567890b   X"
		p2="F1234567890B   X"
		if self.player==1:
			return p1[self.pid]
		else:
			return p2[self.pid]
	
	def copy(self):
		piece=Piece(self.pid,self.player)
		piece.seen=self.seen
		piece.moved=self.moved
		piece.captured=self.captured
		piece.possibleIds=self.possibleIds.copy()
		return piece

	def updateIds(self):
		if self.seen or self.captured:
			self.possibleIds.fill(0)
			self.possibleIds[self.pid]=1
		if self.moved:
			self.possibleIds[self.BOMB]=0
			self.possibleIds[self.FLAG]=0
	
	def defeats(self,defender):
		win=np.sign(self.pid - defender.pid)
		if self.pid==Piece.MINER and defender.pid==Piece.BOMB:
			win=1
		elif self.pid==Piece.SPY and defender.pid==Piece.MARSHALL:
			win=1
		return win
	
	def getAttackWinMask(self):
		mask=np.zeros((12,),dtype=int)
		mask[:self.pid]=1
		if self.pid==Piece.MINER: mask[Piece.BOMB]=1
		if self.pid==Piece.SPY: mask[Piece.MARSHALL]=1
		return mask
	
	def getDefendWinMask(self):
		mask=np.zeros((12,),dtype=int)
		mask[:self.pid]=1
		return mask
	

class Board:
	PIECE_COUNTS=[1,1,8,5,4,4,4,3,2,1,1,6]
	PIECE_VALUES=np.array([100,2,3,6,4,5,6,7,9,12,15,8])
	EMPTY=Piece()
	LAKE=Piece(Piece.LAKE,0)

	def __init__(self,board=None,pieces=None):
		if pieces is None:
			self.resetBoard()
		else:
			self.board=board
			self.pieces=pieces
		self.hasProbabilities=False

	def resetBoard(self):
		self.pieces=[]
		for player in (1,2):
			for pieceId,count in enumerate(self.PIECE_COUNTS):
				for i in range(count):
					self.pieces.append(Piece(pieceId,player))
		self.board=[[self.EMPTY for i in range(10)] for j in range(10)]
		for x in (2,3,6,7):
			for y in (4,5):
				self.board[x][y]=self.LAKE
		
	
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
		if fromPiece.pid in (Piece.FLAG, Piece.BOMB, Piece.EMPTY, Piece.LAKE):
			return False
		if toPiece.pid==Piece.LAKE:
			return False
		if toPiece.player==fromPiece.player:
			return False
			
		dist=toPos-fromPos
		if np.count_nonzero(dist)!=1:
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
				if fromPiece.pid in (Piece.FLAG, Piece.BOMB, Piece.EMPTY, Piece.LAKE):
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

	def applyMove2(self,fromX,fromY,toX,toY):
		move=np.array([[fromX,fromY],[toX,toY]])
		self.applyMove(move)

	def applyMove(self,move):
		fromPos,toPos=move
		fromPiece=self[fromPos]
		toPiece=self[toPos]
		fromPiece.moved=True
		if toPiece.pid == Piece.EMPTY:
			if np.max(toPos-fromPos)>1:
				# Now we know it's a scout
				fromPiece.seen=True
			self[fromPos]=self.EMPTY # from pos is now empty
			self[toPos]=fromPiece
		else:
			win = fromPiece.defeats(toPiece)
			if toPiece.pid==Piece.FLAG: self.endgame=True
			fromPiece.seen=True
			
			self[fromPos]=self.EMPTY # from pos is now empty
			if win==1:
				toPiece.captured=True
				fromPiece.seen=True
				self[toPos]=fromPiece
			elif win==0:
				fromPiece.captured=True
				toPiece.captured=True
				self[toPos]=self.EMPTY
			elif win==-1:
				fromPiece.captured=True
				toPiece.seen=True
		fromPiece.updateIds()
		toPiece.updateIds()
	
	# AI	
	
	def addProbabilities(self,knownPlayer):
		pieces=[piece for piece in self.pieces if piece.player != knownPlayer]
		counts=np.array(self.PIECE_COUNTS)
		pieceMtx=np.array([piece.possibleIds for piece in pieces],dtype=float)
		probabilities=np.zeros((len(pieces),12))
		for i in range(len(pieces)):
			if np.count_nonzero(pieceMtx[i])==1:
				pieceId=np.where(pieceMtx[i]!=0)[0][0]
				probabilities[i,pieceId]=1
				pieceMtx[i]=0
				counts[pieceId]-=1
		for i in range(len(counts)):
			if counts[i]==0:
				pieceMtx[:,i]=0
		nonzeroRows=np.where(pieceMtx.sum(1)>0)[0]
		pieceMtx=pieceMtx[nonzeroRows]
		pieceMtx/=pieceMtx.sum(1,keepdims=True)
		pieceMtx/=pieceMtx.sum(0,keepdims=True)
		pieceMtx/=pieceMtx.sum(1,keepdims=True)
		probabilities[nonzeroRows]=pieceMtx
		for i in range(len(pieces)):
			pieces[i].probs=probabilities[i]
		#self.probabilities=probabilities
		self.hasProbabilities=True
	
	def applyMoveProb(self,fromX,fromY,toX,toY,knownPlayer):
		move=np.array([[fromX,fromY],[toX,toY]])
		return self.applyMoveProbabilistic(move,knownPlayer)
	
	def applyMoveProbabilistic(self,move,knownPlayer):
		if not self.hasProbabilities:
			self.addProbabilities(knownPlayer)
			
		fromPos,toPos=move
		fromPiece=self[fromPos]
		toPiece=self[toPos]
	
		# Handle case of moving without attacking
		boards=[]
		if toPiece.pid == Piece.EMPTY:
			newBoard=self.copy()
			fromPiece=newBoard[fromPos]
			fromPiece.moved=True
			if np.max(toPos-fromPos)>1:
				# Now we know it's a scout
				fromPiece.seen=True
			newBoard[fromPos]=self.EMPTY # from pos is now empty
			newBoard[toPos]=fromPiece
			boards.append((newBoard,1.0))

		elif fromPiece.player == knownPlayer:
			winMask = fromPiece.getAttackWinMask()
			loseMask = 1-winMask
			loseMask[fromPiece.pid] = 0
			tieMask=0*winMask
			tieMask[fromPiece.pid] = 1
			
			winProb = (winMask*toPiece.probs).sum()
			tieProb = toPiece.probs[fromPiece.pid]
			loseProb = (loseMask*toPiece.probs).sum()
			
			if winProb>0:
				newBoard = self.copy()
				fromPiece = newBoard[fromPos]
				toPiece = newBoard[toPos]
				fromPiece.moved = True
				fromPiece.seen = True
				toPiece.captured = True
				newBoard[fromPos]=self.EMPTY # from pos is now empty
				newBoard[toPos]=fromPiece
				fromPiece.updateIds()
				toPiece.possibleIds*=winMask
				boards.append((newBoard,winProb))
			if tieProb>0:
				newBoard = self.copy()
				fromPiece = newBoard[fromPos]
				toPiece = newBoard[toPos]
				fromPiece.moved = True
				fromPiece.captured = True
				toPiece.captured = True
				newBoard[fromPos]=self.EMPTY # from pos is now empty
				newBoard[toPos]=self.EMPTY
				fromPiece.updateIds()
				toPiece.possibleIds*=tieMask
				boards.append((newBoard,tieProb))
			if loseProb>0:
				newBoard = self.copy()
				fromPiece = newBoard[fromPos]
				toPiece = newBoard[toPos]
				fromPiece.moved = True
				fromPiece.captured = True
				toPiece.seen = True
				newBoard[fromPos]=self.EMPTY # from pos is now empty
				fromPiece.updateIds()
				toPiece.possibleIds*=loseMask
				boards.append((newBoard,loseProb))


		else:
			# Attacker lose = defender win
			loseMask = toPiece.getDefendWinMask()
			winMask = 1-loseMask
			winMask[fromPiece.pid] = 0
			tieMask=0*winMask
			tieMask[fromPiece.pid] = 1
			winMask[(Piece.BOMB, Piece.FLAG)]=0
			loseMask[(Piece.BOMB, Piece.FLAG)]=0
			
			winProb = (winMask*fromPiece.probs).sum()
			tieProb = fromPiece.probs[toPiece.pid]
			loseProb = (loseMask*fromPiece.probs).sum()
			
			if winProb>0:
				newBoard = self.copy()
				fromPiece = newBoard[fromPos]
				toPiece = newBoard[toPos]
				fromPiece.moved = True
				fromPiece.seen = True
				toPiece.captured = True
				newBoard[fromPos]=self.EMPTY # from pos is now empty
				newBoard[toPos]=fromPiece
				fromPiece.possibleIds*=winMask
				toPiece.updateIds()
				boards.append((newBoard,winProb))
			if tieProb>0:
				newBoard = self.copy()
				fromPiece = newBoard[fromPos]
				toPiece = newBoard[toPos]
				fromPiece.moved = True
				fromPiece.captured = True
				toPiece.captured = True
				newBoard[fromPos]=self.EMPTY # from pos is now empty
				newBoard[toPos]=self.EMPTY
				fromPiece.possibleIds*=tieMask
				toPiece.updateIds()
				boards.append((newBoard,tieProb))
			if loseProb>0:
				newBoard = self.copy()
				fromPiece = newBoard[fromPos]
				toPiece = newBoard[toPos]
				fromPiece.moved = True
				fromPiece.captured = True
				toPiece.seen = True
				newBoard[fromPos]=self.EMPTY # from pos is now empty
				fromPiece.possibleIds*=loseMask
				toPiece.updateIds()
				boards.append((newBoard,loseProb))
		
		return boards

			
	def heuristic(self,player=1):
		"""
		Points for captured pieces
		Points for revealed pieces
		Points for last miner
		Spy value depends on marshall existing
		Points for unprotected flag??
		Points for invincible pieces
		"""
		points=0

		piecesLeft=np.array([self.PIECE_COUNTS,self.PIECE_COUNTS])
		seenCount=np.zeros(piecesLeft.shape)
		for piece in self.pieces:
			row=0 if piece.player == player else 1
			if piece.captured:
				piecesLeft[row][piece.pid]-=1
			elif piece.seen:
				seenCount[row][piece.pid]+=1
		print(piecesLeft)
		print(seenCount)

		totalPoints=0		
		for row in (0,1):
			other=1-row
			
			# Points for pieces not captured
			points = (piecesLeft[row]*self.PIECE_VALUES).sum()
			# Deduct points for pieces seen
			points -= (seenCount[row]*self.PIECE_VALUES).sum()*0.4
			# Deduct points for no miners left
			if piecesLeft[row,Piece.MINER]==0:
				points -= 20
				
			# Points for having a marshall and spy captured
			if piecesLeft[row,Piece.MARSHALL]==1 and piecesLeft[other,Piece.SPY]==0:
				if piecesLeft[other,Piece.MARSHALL]==0:
					points += 16
				else:
					points += 8
			# Points for every piece which outranks all of the opponents pieces
			opponentHighestRank=0
			for rank in range(2,Piece.BOMB):
				if piecesLeft[other,rank]>0:
					opponentHighestRank=rank
			for rank in range(opponentHighestRank+1,Piece.MARSHALL):
				points += 16*piecesLeft[row,rank]
			
			# Add to total
			if row==0:
				totalPoints += points
			else:
				totalPoints -= points

		# Points for advancing pieces into opponents area
		for x in range(10):
			for y in range(10):
				piece=self.board[x][y]
				if piece.player==0:
					continue
				if piece.player==1:
					numSpaces = max(6-y, 0)
				elif piece.player==2:
					numSpaces = max(y-3, 0)
				if piece.player==player:
					totalPoints += numSpaces*0.1
				else:
					totalPoints -= numSpaces*0.1
					
		return totalPoints
		

	def __repr__(self):
		return '\n'.join([''.join([self.board[x][y].char() for x in range(10)]) for y in range(10)])						

	def copy(self):
		other=Board()
		other.pieces=[piece.copy() for piece in self.pieces]
		pieceMap={i:j for i,j in zip(self.pieces,other.pieces)}
		pieceMap[self.EMPTY]=other.EMPTY
		pieceMap[self.LAKE]=other.LAKE
		other.board=[[pieceMap[piece] for piece in row] for row in self.board]
		return other

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
