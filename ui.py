# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:27:03 2017

@author: sam
"""

import tkinter as tk
from board import Board, Move
import minimax


def getPieceChar(player,rank):
	#white=[0x2690]+list(range(0x2460,0x246a))+[0x1f4a3,0x25cb]
	#black=[0x2691]+list(range(0x2776,0x277f))+[0x1f4a3,0x25cf]
	white=[70]+list(range(0x2460,0x246a))+[0x2739,0x25cb]
	black=[70]+list(range(0x2776,0x2780))+[0x2739,0x25cf]
	chars=[None,white,black]
	if rank<0 or rank>=12:
		rank=-1
	player=1 if player==1 else 2
	unicode_int=chars[player][rank]
	return chr(unicode_int)

class BoardUI(tk.Canvas):
	def __init__(self, root, board, size=300):
		tk.Canvas.__init__(self,root,width=size,height=size)
		self.size=size
		self.bind("<Button-1>", self.onclick)
		self.board=board
		self.selectedSquare=None
	
	def drawBoard(self,hideUnseen=None):
		sqsize = self.size//10
		font = ("clearlyu",sqsize*2//3)
		for i in range(10):
			for j in range(10):
				pieceId=board.grid[i,j]
				bgcolor="white"
				char=None
				if pieceId==Board.EMPTY:
					pass
				elif pieceId==Board.LAKE:
					bgcolor="blue"
				else:
					if self.selectedSquare==(i,j):
						bgcolor="yellow"
					piece=board.pieces[pieceId]
					player=piece.player
					rank=piece.rank
					if player==hideUnseen and not piece.seen:
						rank=-1
					char=getPieceChar(player,rank)
				x=i*sqsize
				y=j*sqsize
				canvas.create_rectangle(x, y, x+sqsize, y+sqsize, fill=bgcolor, outline="black")
				if not char is None:
					canvas.create_text(x+sqsize//2, y+sqsize//2, font=font, width=sqsize, text=char,justify=tk.CENTER)

	def onclick(self,event):
		sqsize = self.size//10
		print("Clicked",event.x,event.y)
		bx = int(event.x/sqsize)
		by = int(event.y/sqsize)
		if bx<0 or bx>=10 or by<0 or by>=10:
			return
		self.selectSquare(bx,by)
	
	def selectSquare(self,x,y):
		if self.selectedSquare is None:
			self.selectedSquare=(x,y)
		elif self.selectedSquare==(x,y):
			self.selectedSquare=None
		else:
			move=Move(self.selectedSquare[0],self.selectedSquare[1],x,y)
			self.selectedSquare=None
			self.onBoardMove(move)
		self.delete("all")
		self.drawBoard()
	
	def onBoardMove(self,move):
		self.board.applyMove(move)

board=Board()
board.placeRandom()

root=tk.Tk()
root.title("Stratego")

canvas = BoardUI(root,board)
canvas.pack()

def onMove():
	move=minimax.getBestMove(board,2)
	board.applyMove(move)
	canvas.drawBoard()
moveBtn = tk.Button(root,text="Move",command=onMove)
moveBtn.bind()
moveBtn.pack()

def onDump():
	pass
dumpBtn = tk.Button(root,text="Dump",command=onDump)
dumpBtn.bind()
dumpBtn.pack()

canvas.drawBoard(board)

root.mainloop()