# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:50:32 2017

@author: sam
"""
from board import Piece, Board
import numpy as np

class ProbPiece(Piece):
	def __init__(self,piece, isKnown=False):
		self.id=piece.id
		self.player=piece.player
		self.captured=piece.captured
		self.seen=piece.seen
		if isKnown:
			self.rank=piece.rank
			
		if hasattr(piece,"possibleIds"):
			self.possibleIds=piece.possibleIds.copy()
		else:
			
			if self.seen or self.captured or isKnown:
				self.possibleIds=np.zeros((12,),dtype=int)
				self.possibleIds[piece.rank]=1
			else:
				self.possibleIds=np.ones((12,),dtype=int) # unknown piece
			if self.moved:
				self.possibleIds[Piece.BOMB]=0
				self.possibleIds[Piece.FLAG]=0			
	

	def updateIds(self):
		if self.seen or self.captured:
			self.possibleIds.fill(0)
			self.possibleIds[self.pid]=1
		if self.moved:
			self.possibleIds[self.BOMB]=0
			self.possibleIds[self.FLAG]=0
	
	def defeats(self,defender):
		raise Exception("function not applicable")
	
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
		
	@property
	def moveable(self):
		mask=np.ones((12,),dtype=int)
		mask[(Piece.BOMB, Piece.FLAG)]=0
		return (self.possibleIds*mask).sum()>0
	
	def isScout(self):
		notScoutProb=self.possibleIds.sum()-self.possibleIds[Piece.SCOUT]
		return notScoutProb==0
	
	def setMoved(self):
		self.possibleIds[self.FLAG]=0
		self.possibleIds[self.BOMB]=0
	
	def setRank(self,rank):
		self.possibleIds.fill(0)
		self.possibleIds[rank]=1
	
	def setSeen(self):
		self.seen=True
	
	def setCaptured(self):
		self.seen=True
		self.captured=True
	



class ProbBoard(Board):
	"""
	board - instance of Board
	knownPlayer - player whose perspective it is - all their pieces are seen
	"""
	def __init__(self,board,knownPlayer):
		self.grid=board.grid.copy()
		self.pieces=[ProbPiece(piece,piece.player==knownPlayer) for piece in board.pieces]
		self.knownPlayer=knownPlayer
		self.addProbabilities()
	
	
	def addProbabilities(self):
		pieces=[piece for piece in self.pieces if piece.player != self.knownPlayer]
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
		self.probabilities=probabilities
	
	def applyMoveProb(self,fromX,fromY,toX,toY,knownPlayer):
		move=np.array([[fromX,fromY],[toX,toY]])
		return self.applyMoveProbabilistic(move,knownPlayer)
	
	def applyMoveProbabilistic(self,move,knownPlayer):
		if not self.hasProbabilities:
			self.addProbabilities(knownPlayer)
			
		fromPos,toPos=move
		fromId = self.grid[fromPos]
		toId = self.grid[toPos]
		fromPiece=self[fromPos]
		player=self.pieces[fromId].player
	
		# Handle case of moving without attacking
		boards=[]
		if toId<0:
			newBoard=ProbBoard(self,self.knownPlayer)
			fromPieceNew=newBoard[fromPos]
			fromPieceNew.setMoved()
			fromPieceNew.setSeen()
			if np.max(toPos-fromPos)>1:
				# Now we know it's a scout
				fromPieceNew.setRank(Piece.SCOUT)
			newBoard.grid[fromPos]=Board.EMPTY # from pos is now empty
			newBoard.grid[toPos]=fromPiece.id
			boards.append((newBoard,1.0))

		# Player attacks
		elif player == knownPlayer:
			toPiece=self[toPos]
			
			winMask = fromPiece.getAttackWinMask()
			loseMask = 1-winMask
			loseMask[fromPiece.rank] = 0
			tieMask=0*winMask
			tieMask[fromPiece.rank] = 1
			
			winProb = (winMask*toPiece.probs).sum()
			tieProb = toPiece.probs[fromPiece.rank]
			loseProb = (loseMask*toPiece.probs).sum()
			
			if winProb>0:
				newBoard = ProbBoard(self)
				fromPieceNew = newBoard[fromPos]
				toPieceNew = newBoard[toPos]

				newBoard.grid[fromPos]=Board.EMPTY # from pos is now empty
				newBoard.grid[toPos]=fromId
				
				fromPieceNew.setSeen()
				toPieceNew.setCaptured()
				toPieceNew.possibleIds*=winMask
				
				boards.append((newBoard,winProb))
				
			if tieProb>0:
				newBoard = ProbBoard(self)
				fromPieceNew = newBoard[fromPos]
				toPieceNew = newBoard[toPos]

				newBoard.grid[fromPos]=Board.EMPTY 
				newBoard.grid[toPos]=Board.EMPTY
				
				fromPieceNew.setCaptured()
				toPieceNew.setCaptured()
				toPieceNew.possibleIds*=tieMask
				
				boards.append((newBoard,tieProb))
				
			if loseProb>0:
				newBoard = ProbBoard(self)
				fromPieceNew = newBoard[fromPos]
				toPieceNew = newBoard[toPos]

				newBoard.grid[fromPos]=Board.EMPTY
				
				fromPieceNew.setCaptured()
				toPieceNew.setSeen()
				toPieceNew.possibleIds*=loseMask
				
				boards.append((newBoard,loseProb))

		# Opponent attacks
		else:
			# Attacker lose = defender win
			loseMask = toPiece.getDefendWinMask()
			winMask = 1-loseMask
			winMask[toPiece.rank] = 0
			tieMask=0*winMask
			tieMask[toPiece.rank] = 1
			winMask[(Piece.BOMB, Piece.FLAG)]=0
			loseMask[(Piece.BOMB, Piece.FLAG)]=0
			
			winProb = (winMask*fromPiece.probs).sum()
			tieProb = fromPiece.probs[toPiece.rank]
			loseProb = (loseMask*fromPiece.probs).sum()
			
			if winProb>0:
				newBoard = ProbBoard(self)
				fromPieceNew = newBoard[fromPos]
				toPieceNew = newBoard[toPos]

				newBoard.grid[fromPos]=Board.EMPTY
				newBoard.grid[toPos]=fromId
				
				fromPieceNew.setMoved()
				fromPieceNew.setSeen()
				toPieceNew.setCaptured()
				toPieceNew.possibleIds*=winMask
				
				boards.append((newBoard,winProb))
				
			if tieProb>0:
				newBoard = ProbBoard(self)
				fromPieceNew = newBoard[fromPos]
				toPieceNew = newBoard[toPos]

				newBoard.grid[fromPos]=Board.EMPTY
				newBoard.grid[toPos]=Board.EMPTY
				
				fromPieceNew.setMoved()
				fromPieceNew.setCaptured()
				toPieceNew.setCaptured()
				
				fromPiece.possibleIds*=tieMask
				boards.append((newBoard,tieProb))
				
			if loseProb>0:
				newBoard = ProbBoard(self)
				fromPieceNew = newBoard[fromPos]
				toPieceNew = newBoard[toPos]

				newBoard.grid[fromPos]=Board.EMPTY
				
				fromPieceNew.setMoved()
				fromPieceNew.setCaptured()
				toPieceNew.setSeen()
				
				fromPiece.possibleIds*=loseMask
				boards.append((newBoard,loseProb))
		
		return boards


	def capturedBonus(self, piecesLeft, piecesLeftOpp):
		points=0
		
		# Deduct points for no miners left
		noMinersProb=max(0,1.0-piecesLeft[Piece.MINER])
		points -= 20*noMinersProb
		
		oppCapturedProb=1.0-piecesLeftOpp
		# Points for having a marshall and spy captured.
		marshallNoSpy = piecesLeft[Piece.MARSHALL]*oppCapturedProb[Piece.SPY]
		points += marshallNoSpy*8
		# Extra if other marshall is also captured, so our marshall is invincible
		invincibleMarshall = marshallNoSpy*oppCapturedProb[Piece.MARSHALL]
		points += invincibleMarshall*8
		
		# Points for invincible general
		invincibleGeneral = piecesLeft[Piece.GENERAL]*oppCapturedProb[Piece.MARSHALL]*oppCapturedProb[Piece.GENERAL]
		points += 16*invincibleGeneral
		return points
	
			
	def heuristic(self):
		"""
		Points for captured pieces
		Points for revealed pieces
		Points for last miner
		Spy value depends on marshall existing
		Points for unprotected flag??
		Points for invincible pieces
		"""
		points=0

		multipliers=np.empty((len(self.pieces),))
		playerPiecesLeft=np.array(Board.RANK_COUNTS,dtype=float)
		opponentPiecesLeft=np.array(Board.RANK_COUNTS,dtype=float)
		
		for i,piece in enumerate(self.pieces):
			if piece.captured:
				value = 0
				if piece.player == self.knownPlayer:
					playerPiecesLeft -= self.probabilities[i]
				else:
					opponentPiecesLeft -= self.probabilities[i]
			elif piece.seen:
				value = 0.6
			else:
				value = 1.0
			if piece.player != self.knownPlayer:
				value = -value
			multipliers[i]=value
		
		pieceValues=np.array(Board.RANK_VALUES).reshape((-1,1))
		pointsPerPiece=np.matmul(self.probabilities,pieceValues)*multipliers
		points=pointsPerPiece.sum()
		
		points += self.capturedBonus(playerPiecesLeft, opponentPiecesLeft)
		points -= self.capturedBonus(opponentPiecesLeft, playerPiecesLeft)


		# Points for advancing pieces into opponents area
		playerSpacesAdvanced=0
		opponentSpacesAdvanced=0
		for x in range(10):
			for y in range(10):
				pieceId=self.grid[x][y]
				if pieceId<0:
					continue
				player=self.pieces[pieceId].player
				if player==1:
					numSpaces = max(6-y, 0)
				elif player==2:
					numSpaces = max(y-3, 0)
				if player==self.knownPlayer:
					playerSpacesAdvanced += numSpaces
				else:
					opponentSpacesAdvanced += numSpaces
		
		points += 0.1*(playerSpacesAdvanced-opponentSpacesAdvanced)
					
		return points
		

