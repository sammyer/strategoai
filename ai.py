# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:50:32 2017

@author: sam
"""
from board import Piece, Board, Outcome
import numpy as np

DEBUG_MODE=False

def log(*a):
	if DEBUG_MODE:
		print(a)

class ProbPiece(Piece):
	def __init__(self,piece, isKnown=False):
		self.id=piece.id
		self.player=piece.player
		self.captured=piece.captured
		self.seen=piece.seen
		self.char=piece.char
		self.rank=None
			
		if hasattr(piece,"possibleIds"):
			self.possibleIds=piece.possibleIds.copy()
			self.known = np.count_nonzero(self.possibleIds)==1
			if self.known:
				self.rank=np.argmax(self.possibleIds)
		else:
			self.known = isKnown or self.seen or self.captured
			
			if self.known:
				self.possibleIds=np.zeros((12,),dtype=int)
				self.possibleIds[piece.rank]=1
				self.rank = piece.rank
			else:
				self.possibleIds=np.ones((12,),dtype=int) # unknown piece
				if piece.moved:
					self.possibleIds[Piece.BOMB]=0
					self.possibleIds[Piece.FLAG]=0			
	
	
	def defeats(self,defender):
		raise Exception("function not applicable")
	
	def getAttackWinMask(self):
		mask=np.zeros((12,),dtype=int)
		mask[:self.rank]=1
		if self.rank==Piece.MINER: mask[Piece.BOMB]=1
		if self.rank==Piece.SPY: mask[Piece.MARSHALL]=1
		return mask
	
	def getDefendWinMask(self):
		mask=np.zeros((12,),dtype=int)
		mask[:self.rank]=1
		return mask
		
	@property
	def moveable(self):
		mask=np.ones((12,),dtype=int)
		mask[Piece.BOMB]=0
		mask[Piece.FLAG]=0
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
	
	def __repr__(self):
		return "[Piece player=%d seen=%d rank=%s captured=%d"%(self.player,self.seen,str(self.rank),self.captured)



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
		for player in Board.PLAYERS:
			# Number of hidden pieces per rank
			hiddenCounts=np.array(Board.RANK_COUNTS)
			playerPieces=[piece for piece in self.pieces if piece.player==player]
			pieceMtx=np.array([piece.possibleIds for piece in playerPieces],dtype=float)
			probabilities=np.zeros((len(playerPieces),12))
			
			# Exclude pieces
			for i,piece in enumerate(playerPieces):
				# Enemy piece is known if it has only one possible id (i.e. piece.possibleIds is 1-hot)
				# In this case, subtract
				if np.count_nonzero(pieceMtx[i])==1:
					# Find index where possibleIds==1
					rank=np.where(pieceMtx[i]!=0)[0][0]
					# Set probability to 1
					probabilities[i,rank]=1.0
					pieceMtx[i]=0 # Exclude from further calculations
					hiddenCounts[rank]-=1 # Mark that rank as found for further calcaulations
					
			# For every rank which is completely found, remove that rank as a possiblility for hidden pieces
			for rank in range(len(hiddenCounts)):
				if hiddenCounts[rank]==0:
					pieceMtx[:,rank]=0
		
	
			# Calcualte probability of hidden rows only			
			hiddenRows=np.where(pieceMtx.sum(1)>0)[0]
			
			pieceMtx=pieceMtx[hiddenRows]
			pieceMtx/=pieceMtx.sum(1,keepdims=True)
			pieceMtx/=np.maximum(pieceMtx.sum(0,keepdims=True),1e-7)
			pieceMtx*=hiddenCounts
			pieceMtx/=pieceMtx.sum(1,keepdims=True)
			probabilities[hiddenRows]=pieceMtx
			
			for i,piece in enumerate(playerPieces):
				piece.probs=probabilities[i]
			
		self.probabilities=np.array([piece.probs for piece in self.pieces])


	"""
	Note : for attacks there are 7 possible situations
	1. Player attacks, opponent known (complete information)
	2. Player attacks, opponent unknown
	3. Player attacks in the future, opponent unkonwn now, but will be known at time of attack
	4. Opponent attacks, complete information (all pieces known to all parties)
	5. Opponent attacks with information advantage (we dont know opponent, but it knows us)
	6. Opponent attacks with information disadvantage (we know opponent, but it doesn't know us)
	7. Opponent attacks, double blind
	"""
	
	def splitOnDefender(self,defenderPos,attackerRank):
		""" For an attacking move where the attacker is known but the defender is unknown, 
		creates boards for each possible outcome, with probablity of that outcome based on all known information.

		The defender is unknown, meaning the attacker is taking a chance and does not know for sure the best decision.
		As the AI, we know the same amount as the attacker, therefore we can calculate the expected value
		This is "post-split", meaning we make a decision based on the expected value.
		E(x)=sum(prob(y)*E(y)) for y in outcomes
		
		RETURNS:
			an array of:
			outcome - one of WIN,TIE,LOSS,IMMOVABLE
			prob - probability of board configuration (all probs add to one)
			board - new board
		"""
		rankOutcomes = Outcome.getDefenderMask(attackerRank)
		defender = self[defenderPos]
		assert(not defender is None)
		
		boards=[]
		for outcome in (Outcome.WIN, Outcome.TIE, Outcome.LOSS):
			mask = rankOutcomes==outcome
			prob=np.dot(mask, defender.probs)
			if prob>0:
				board = self.maskBoard(defenderPos,mask)
				boards.append((outcome,prob,board))
		return boards
	
	def splitOnAttacker(self,attackerPos,defenderRank):
		""" For an attacking move where the defender is known but the attacker is unknown, 
		creates boards for each possible outcome, with probablity of that outcome
		based on all known information
		
		The attacker is unknown to us, but the attacker knows who they are.
		Therefore, the attacker has perfect information and will always make the best decision
		However, as the AI we don't know what that decision is
		This is "pre-split", meaning we have to consider every possibile outcome, calculated a different expected value
		for each possibility, and take the max path based on the outcome.
		"""
		rankOutcomes = Outcome.getAttackerMask(defenderRank)
		attacker = self[attackerPos]
		assert(not attacker is None)
		
		boards=[]
		for outcome in (Outcome.WIN, Outcome.TIE, Outcome.LOSS):
			# Combine immovable and loss, since in either case, we won't be taking the move
			if outcome == Outcome.LOSS:
				mask = (rankOutcomes==Outcome.LOSS)|(rankOutcomes==Outcome.IMMOVABLE)
			else:
				mask = rankOutcomes==outcome
			prob=np.dot(mask, attacker.probs)
			if prob>0:
				board = self.maskBoard(attackerPos,mask)
				boards.append((outcome,prob,board))
		return boards
	
	def splitOnDefenderDoubleBlind(self,defenderPos,attackerPos):
		""" For an attacking move where neither defender or attacker are known, 
		creates boards for each possible outcome, with probablity of that outcome
		
		This is also "post-split" because we take a shortcut here and calculate
		an expected value for wins, ties, losses based on both the attacker and defender probs
		"""
		attacker = self[attackerPos]
		defender = self[defenderPos]
		assert(not attacker is None)
		assert(not defender is None)

		# Matrix of size (outcomes, ranks)
		outcomes=[Outcome.WIN, Outcome.TIE, Outcome.LOSS]
		rankOutcomeProbs = np.zeros((3,12))
		
		for rank,rankProb in enumerate(attacker.probs):
			if rankProb>0:
				mask = Outcome.getDefenderMask(rank)
				for idx,outcome in enumerate(outcomes):
					rankOutcomeProbs[idx] += (mask==outcome)*rankProb
				
		outcomeProbs = (rankOutcomeProbs*defender.probs).sum(1)
		
		boards=[]
		for outcome,prob in zip(outcomes,outcomeProbs):
			if prob>0:
				board = self.maskBoard(defenderPos,mask)
				boards.append((outcome,prob,board))
		return boards
	
	
	def maskBoard(self,piecePos,rankMask):
		# Creates a new board where a single pieces rank is masked by rankMask
		# rankMask can either be a vector mask, or just a single integer which gets changed into a 1-hot mask
		if type(rankMask)==int:
			rank=rankMask
			rankMask=np.zeros((12,),dtype=int)
			rankMask[rank]=1
			
		newBoard=ProbBoard(self,self.knownPlayer)
		piece=newBoard[piecePos]
		
		piece.possibleIds*=rankMask
		return newBoard

	def applyMove(self,move,outcome):
		if outcome==Outcome.WIN:
			self.applyWin(move.fromPos,move.toPos)
		elif outcome==Outcome.LOSS:
			self.applyLoss(move.fromPos,move.toPos)
		elif outcome==Outcome.TIE:
			self.applyTie(move.fromPos,move.toPos)
		else:
			self.applyNonattack(move.fromPos,move.toPos)

	def applyWin(self,fromPos,toPos):
		#fromPiece.setMoved()
		self[fromPos].setSeen()
		self[toPos].setCaptured()

		self.grid[toPos]=self.grid[fromPos]
		self.grid[fromPos]=Board.EMPTY
		

	def applyTie(self,fromPos,toPos):
		self[fromPos].setCaptured()
		self[toPos].setCaptured()
		
		self.grid[fromPos]=Board.EMPTY
		self.grid[toPos]=Board.EMPTY

	def applyLoss(self,fromPos,toPos):
		self[fromPos].setCaptured()
		self[toPos].setSeen()
		
		self.grid[fromPos]=Board.EMPTY
	
	def applyNonattack(self,fromPos,toPos):
		self[fromPos].setMoved()

		self.grid[toPos]=self.grid[fromPos]
		self.grid[fromPos]=Board.EMPTY

	def copy(self):
		return ProbBoard(self,self.knownPlayer)

	def capturedBonus(self, piecesLeft, piecesLeftOpp):
		points=0
		
		# Deduct points for flag capture		
		flagCapturedsProb=1.0-piecesLeft[Piece.FLAG]
		points -= 40*flagCapturedsProb
		
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
		
		log(flagCapturedsProb, noMinersProb, marshallNoSpy, invincibleMarshall, invincibleGeneral, points)
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
		
		pieceValues = np.array(Board.RANK_VALUES).reshape((-1,1))
		pointsPerPiece = np.matmul(self.probabilities,pieceValues).flatten()
		pointsPerPiece *= multipliers
		points=pointsPerPiece.sum()
		
		log(pointsPerPiece)
		log(multipliers.flatten())
		log(np.matmul(self.probabilities,pieceValues).flatten())
		log(playerPiecesLeft)
		log(opponentPiecesLeft)
		log(points)
		
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
					numSpaces = max(9-y, 0)
				elif player==2:
					numSpaces = max(y, 0)
				if player==self.knownPlayer:
					playerSpacesAdvanced += numSpaces
				else:
					opponentSpacesAdvanced += numSpaces
		
		points += 0.1*(playerSpacesAdvanced-opponentSpacesAdvanced)
					
		return points
		
