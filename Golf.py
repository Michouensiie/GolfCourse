# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:51:07 2022

@author: micci
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

class State:
    
    ##Initialisation de la classe State
    def __init__(self,player,M):
        self.M = M #dimension
        
        #création aléatoire du board
        self.board = np.zeros(((M,M)))
        ball = (np.random.randint(0,M),np.random.randint(0,M))
        flag = (np.random.randint(0,M),np.random.randint(0,M))
        while(ball == flag):
            ball = (np.random.randint(0,M),np.random.randint(0,M))
            flag = (np.random.randint(0,M),np.random.randint(0,M))
        self.ball = [ball[0],ball[1]]
        self.flag = [flag[0],flag[1]]
        self.board[ball[0],ball[1]] = 1
        self.board[flag[0],flag[1]] = 2
        
        self.player = player
        self.isEnd = False
        self.boardHash = None
        self.nbShoot = 0
        self.parShoot = max(abs(self.ball[0]-self.flag[0]),abs(self.ball[1]-self.flag[1])) #nombre de coup optimal
        self.nbShoots = [] #nombre de shoot pour chaque partie lors de l'entrainement
        self.diffShoot = [] #difference nombre coup optimal, nombre de coup
        self.time = [] #temps d'execution
        
    ## Permet de placer le board sous la forme d'une vecteur linéaire, que l'on stockera dans un dictionnaire avec une valeur
    def getHash(self):
        self.boardHash = str(self.board.reshape(pow(self.M,2)))
        return self.boardHash
    
    ## On distingue ici toutes les positions où la balle peut se rendre
    def avaiblePositions(self):
        avaiablePos = []
        x = self.ball[0]
        y = self.ball[1]
        
        avaiablePos.append([x+1,y])
        avaiablePos.append([x+1,y-1])
        avaiablePos.append([x+1,y+1])
        avaiablePos.append([x-1,y])
        avaiablePos.append([x-1,y-1])
        avaiablePos.append([x-1,y+1])
        avaiablePos.append([x,y+1])
        avaiablePos.append([x,y-1])
            
        return avaiablePos
    
    ## On met ici a jour le board de jeu
    def updateBoard(self,action):
        
        self.board[self.ball[0],self.ball[1]] = 0
        self.ball[0] = action[0]
        self.ball[1] = action[1]
        

        if action == self.flag:
            self.board[action[0],action[1]] = 3 #cas où l'on a gagné une partie
        else:
            self.board[action[0],action[1]] = 1 #cas où l'on a déplacé la balle
        
        
    ## Fonction qui nous permet de savoir si la partie est terminée
    def checkIsEnd(self):
        if self.ball == self.flag: 
            self.isEnd = True
            self.player.win = True
        if self.nbShoot>=3*self.M:
            self.isEnd = True
            
    ## On récompense ici notre agent, 1 s'il gagne, 0 sinon
    def giveRewards(self):
        if self.player.win:
            reward = 1
        else:
            reward = 0
        self.player.feedReward(reward) 
        
    ## Remise à zéro des paramètres afin de pouvoir en débuter une nouvelle
    def reset(self):
        M = self.M
        self.board = np.zeros(((M,M)))
        ball = (np.random.randint(0,M),np.random.randint(0,M))
        flag = (np.random.randint(0,M),np.random.randint(0,M))
        while(ball == flag):
            ball = (np.random.randint(0,M),np.random.randint(0,M))
            flag = (np.random.randint(0,M),np.random.randint(0,M))
        self.ball = [ball[0],ball[1]]
        self.flag = [flag[0],flag[1]]
        self.board[ball[0],ball[1]] = 1
        self.board[flag[0],flag[1]] = 2
        self.isEnd = False
        self.boardHash = None
        self.nbShoot = 0
        self.parShoot = max(abs(self.ball[0]-self.flag[0]),abs(self.ball[1]-self.flag[1]))
    
    
    ## On check si le coup choisi par l'agent correspond à une case de la grille, ie s'il est ou non out of bounds
    def checkActionIntoBounds(self,action):
        x = action[0]
        y = action[1]
        if x>=0 and x<=self.M-1:
            if y>=0 and y<=self.M-1:
                return True
        return False
    
    ## Fonction d'entrainement de l'agent
    ## Dans un premier temps on récupère les positions envisageables, puis on choisis une action à l'aide de la fonction 
    ## chooseAction de l'agent. On test ensuite si elle est envisageableou on. Dans le cas où elle l'est, on update de le board.
    ## On enregistre ensuite le hashed Board dans la list de l'agent prévue à cet effet
    def play(self,rounds=100):
        startTime = time.time() #nous permet de chronométrer le temps d'éxecution
        for i in range(rounds):
            if i%10000 == 0:
                print("Rounds {}".format(i)) #état des lieux du nombre de parties
            while self.isEnd == False:
                self.nbShoot += 1
                avaiblePositions = self.avaiblePositions()
                action = self.player.chooseAction(avaiblePositions,self.board,self.flag,self.ball)
                if self.checkActionIntoBounds(action):
                    self.updateBoard(action)
                boardHash = self.getHash()
                self.player.states.append(boardHash)
                self.checkIsEnd()
            
            self.nbShoots.append(self.nbShoot)
            self.diffShoot.append(self.nbShoot - self.parShoot)
            self.time.append(time.time()-startTime)
            self.giveRewards()
            self.player.reset()
            self.reset()

    ## Fonction qui nous permet à nous de jouer         
    def playHuman(self):
        while self.isEnd==False:
            self.showBoard()
            self.nbShoot +=1
            avaiblePositions = self.avaiblePositions()
            print(avaiblePositions)
            print('choisir en fonction de la position où l"on veut aller')
            i = int(input())
            action = [self.ball[0],self.ball[1]]
            for k in range(len(avaiblePositions)):
                if k == i:
                    action = avaiblePositions[k]
            if self.checkActionIntoBounds(action):
                self.updateBoard(action)
            self.checkIsEnd()
            
            
        if self.player.win == True:
            print("Winner")
        else:
            print('Looser')
            
    ## Fonction permettant à l'agent de nous montrer ce qu'il a appri
    def playAlone(self):
        while self.isEnd == False:
            self.showBoard()
            self.nbShoot += 1
            avaiblePositions = self.avaiblePositions()
            action = self.player.chooseAction(avaiblePositions,self.board,self.flag,self.ball)
            if self.checkActionIntoBounds(action):
                self.updateBoard(action)
                boardHash = self.getHash()
                self.player.states.append(boardHash)
                self.checkIsEnd()
        if self.player.win == True:
            print("Winner")
        else:
            print('Looser')
        print("Le plus petit nombre de coup pour gagner est de ",self.parShoot)
        print("La partie a été gagné en ",self.nbShoot," coup")
        self.reset()
        
    ## Affichage du board
    def showBoard(self):
        for i in range(self.M):
            print('--------------------------')
            out = '|'
            for j in range(self.M):
                elem = self.board[i][j]
                if elem == 1:
                    token = ' O '
                elif elem == 2:
                    token = ' X '
                else:
                    token = '   '
                out += token + '|'
            print(out)
        print('--------------------------')
        
        
        
class Player:
    ## Fonction d'initialisation avec un exploration rate, un gamma et un learning rate
    def __init__(self, name, exp_rate=0.3,decayGamma=0.7,lr = 0.1):
        self.name = name
        self.win = False
        self.states = [] #on récupère le chemin
        self.exp_rate = exp_rate
        self.learningRate = lr
        self.decayGamma = decayGamma
        self.statesValue= {}
    
    ## Reset des paramètre du joueur
    def reset(self):
        self.states = []
        self.win = False
    
    ## Update du board lors du choix d'une action
    def updateBoard(self,board,action,flag,ball):
        board[ball[0],ball[1]] = 0
        if action == flag:
            board[action[0],action[1]] = 3
        else:
            board[action[0],action[1]] = 1
    
    ## Hashage du board
    def getHash(self, board):
        boardHash = str(board.reshape(pow(len(board),2)))
        return boardHash
    
    ## On fourni à l'agent une récompense lorsqu'il gagne la partie
    def feedReward(self,reward):
        for state in reversed(self.states):
            if self.statesValue.get(state) is None:
                self.statesValue[state] = 0
            self.statesValue[state] += self.learningRate*(self.decayGamma*reward - self.statesValue[state])
            reward = self.statesValue[state]
        
    ## On check si le coup est out of bounds
    def checkActionIntoBounds(self,action,M):
        x = action[0]
        y = action[1]
        if x>=0 and x<=M-1:
            if y>=0 and y<=M-1:
                return True
        return False
    
    ## Fonction qui permet à l'agent de choisir l'action qu'il va réaliser
    def chooseAction(self, avaiblePositions,board,flag,ball):
        #exploration de l'environnement par l'agent
        if np.random.uniform(0,1) <= self.exp_rate:
            action = avaiblePositions[np.random.choice(len(avaiblePositions))]
        #exploitation de l'environnement par l'agent
        else:
            # Ici on va tester toutes les actions envisageable et sélectionner celle qui maximise notre fonction de valeure
            # Pour cela, pour chaque avaible positions, on récupère le nouveau board et on va chercher sa value dans notre 
            # dictionaire, puis on sélectionne la meilleure possible
            
            values = [0 for i in range(8)]
            for i in range(8):
                p = avaiblePositions[i]
                nextBoard = board.copy()
                if self.checkActionIntoBounds(p,len(board)):
                    self.updateBoard(nextBoard,p,flag,ball)
                nextBoardHash = self.getHash(nextBoard)
                values[i] = 0 if self.statesValue.get(nextBoardHash) is None else self.statesValue.get(nextBoardHash)
            if values == [0,0,0,0,0,0,0,0]:
                action = avaiblePositions[np.random.choice(len(avaiblePositions))]
            else:
                idx = values.index(max(values))
                action = avaiblePositions[idx]
        return action
    
    ## Enregistrement de la politique
    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.statesValue, fw)
        fw.close()
    
    ## Chargement de la politique
    def loadPolicy(self, file):
        fr = open(file,'rb')
        self.statesValue = pickle.load(fr)
        fr.close()
        
        
N= 50000
P = Player("Luca")
St = State(P,7)

St.play(N)

P.savePolicy()


x = [p for p in range(N)]
rollingMean = [np.mean(St.nbShoots[:i]) for i in range(N)] #moyenne des coups en fonction du temps
timeTraining = St.time
deltaTho = St.diffShoot


plt.figure(figsize=(10,5))
plt.title("Nombre de coup moyen en fonction du temps",size=18)
plt.xlabel("Nombre de Partie",size=18)
plt.ylabel("Nombre moyen de coup",size=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.plot(x,rollingMean,"or")
plt.show()


plt.figure(figsize=(10,5))
plt.title("Temps de convergence",size=18)
plt.xlabel("Nombre de Partie",size=18)
plt.ylabel("Temps",size=18)
plt.plot(x,timeTraining,"or")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


plt.figure(figsize=(10,5))
plt.title("Nombre de coup supplémentaire par rapport à l'optimalité")
plt.xlabel("Nombre de Partie")
plt.ylabel("Différence nombre coup/nombre optimal de coup")
plt.plot(x,deltaTho,"ob")
plt.show()

K = 1
while rollingMean[K] > 5.5:
    K+=1
print("On atteint convergence au bout de ",K," partie, pour un temps d'execution de ",timeTraining[K]," secondes.")


N= 15000
gamma = np.linspace(0.5,1.0,6)
rollingMean = []
timeTraining = []
for g in gamma:
    P = Player("Luca",decayGamma= g)
    St = State(P,7)
    St.play(N)
    rollingMean.append([np.mean(St.nbShoots[:i]) for i in range(N)])
    timeTraining.append(St.time)
    
    
x = [p for p in range(N)]
for i in range(6):
    print("avec un gamma de",gamma[i], ", on trouve :",rollingMean[i][N-1])
    
N= 15000
alpha = np.linspace(0.0,0.3,6)
rollingMean = []
timeTraining = []

for a in alpha:
    P = Player("Luca",lr=a)
    St = State(P,7)
    St.play(N)
    rollingMean.append([np.mean(St.nbShoots[:i]) for i in range(N)])
    
x = [p for p in range(N)]
for i in range(6):
    print("avec un alpha de",alpha[i], ", on trouve :",rollingMean[i][N-1])
    
    
alpha = 0.12
gamma = 0.7
M = 7
N = 15000

P = Player("computer",lr=alpha,decayGamma=gamma)
E = State(P,M)

E.play(N)
P.savePolicy()


testPlayer = Player("test",exp_rate=0)
testPlayer.loadPolicy("policy_computer")

TestState = State(testPlayer, 7)


TestState.playAlone()

TestState.playAlone()

TestState.playAlone()