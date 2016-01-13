import numpy as np
import os
#import matplotlib.pyplot as plt
import string
from warnings import catch_warnings
ff=open("dot4.txt")
st=ff.readlines()
i=0
tcc=[]
loc=[]
for tc in st:
    tc=st[i].strip()
    tc=tc.split('\t')
    tc.remove('')
    ii=0
    for ttc in tc:
        try:
            tc[ii]=float(tc[ii])
        except:
            tc[ii]=int(tc[ii])
        ii+=1
    st[i]=tc
    if tc==[0,0]:
        loc.append(i)
    i+=1
print (len(loc))    
tpl=0
line=[]
for l in loc:
    x=[]
    y=[]
    line.append(st[tpl:l])
    tpl=l+1
print(line[0:1])
def rmvDump():
    Pt=lastPt=np.array([0,0])
    for it in line:
        for idd,itt in enumerate(it):
            if len(it)<3:
                break
            Pt=np.array(itt)
            dist=Pt-lastPt
            lastPt=Pt
            dist=np.sqrt(dist.dot(dist))
            if dist<0.004:
                if len(it)<4:
                    del it[1]
                elif idd==len(it)-1:
                    it[-2]=it[idd]
                    del it[idd]
                else:
                    del it[idd]
rmvDump()
linehead=[[hd[0],hd[-1],[idx]] for idx,hd in enumerate(line)]
class JoncPt:
    "co pt"
    def __init__(self, coHead=[], coTail=[]):
        self.coHead = coHead
        self.coTail = coTail
        self.ncoHead = len(coHead)
        self.ncoTail = len(coTail)
def cmptAngle(pt):
    vec=np.array(pt)
    vec1=vec[0]-vec[1]
    vec2=vec[2]-vec[1]
    vecm=vec1.dot(vec2)
    L1=np.sqrt(vec1.dot(vec1))
    L2=np.sqrt(vec2.dot(vec2))
    if L1*L2==0:
        angle=180
    else:
        cosangle=vecm/(L1*L2)
        angle=np.arccos(cosangle)*180/np.pi
    return angle
   
def FindConn(index, Pt,Ls=[] ):
    coTail=[]
    coHead=[]
    for ii in range(0,len(Ls)):
        if Ls[ii]==[]:
            continue
        if (abs(Pt[0][0]-Ls[ii][0][0])<0.01 and abs(Pt[0][1]-Ls[ii][0][1])<0.01 and ii!=index):
            tpLine=[line[index][1]]+[line[index][0]]+[line[ii][1]]
            angle=cmptAngle(tpLine)
            coHead.append([ii,-1,angle])
           
            pass
        elif(abs(Pt[0][0]-Ls[ii][1][0])<0.01 and abs(Pt[0][1]-Ls[ii][1][1])<0.01 and ii!=index):
            tpLine=[line[index][1]]+[line[index][0]]+[line[ii][-2]]
            angle=cmptAngle(tpLine)
            coHead.append([ii,1,angle])          
            pass
        elif(abs(Pt[1][0]-Ls[ii][0][0])<0.01 and abs(Pt[1][1]-Ls[ii][0][1])<0.01 and ii!=index):
            tpLine=[line[index][-2]]+[line[index][-1]]+[line[ii][1]]
            angle=cmptAngle(tpLine)
            coTail.append([ii,-2,angle])
           
        elif(abs(Pt[1][0]-Ls[ii][1][0])<0.001 and abs(Pt[1][1]-Ls[ii][1][1])<0.001 and ii!=index):
            tpLine=[line[index][-2]]+[line[index][-1]]+[line[ii][-2]]
            angle=cmptAngle(tpLine)
            coTail.append([ii,-3,angle])
           
            pass
    else:
        cId=JoncPt(coHead,coTail)
        return cId
for idx in range(0,len(linehead)):
    if linehead[idx]==[]:
        continue
    CoHead=JoncPt()
    CoHead=FindConn(idx,linehead[idx],linehead)
    linehead[idx][2].append(CoHead)
    subLineNo=idx
    while CoHead.ncoHead==1 or CoHead.ncoTail==1:
        if CoHead.ncoTail==1:
            JointID=CoHead.coTail        
            tempseg=line[JointID[0][0]]
            if JointID[0][1]==-2:
                line[idx].extend(tempseg)
                subLineNo=JointID[0][0]
                linehead[idx]=[line[idx][0],line[idx][-1],[idx]]
                linehead[subLineNo]=[]
                linehead[idx]=[line[idx][0],line[idx][-1],[idx]]
                CoHead=FindConn(idx,linehead[idx],linehead)
                linehead[idx][2].append(CoHead)
                line[subLineNo]=[[],[]]
            elif JointID[0][1]==-3:
                tempseg=tempseg[::-1]
                line[idx].extend(tempseg)
                subLineNo=JointID[0][0]
                linehead[idx]=[line[idx][0],line[idx][-1],[idx]]
                linehead[subLineNo]=[]
                linehead[idx]=[line[idx][0],line[idx][-1],[idx]]
                CoHead=FindConn(idx,linehead[idx],linehead)
                linehead[idx][2].append(CoHead)
                line[subLineNo]=[[],[]]
        elif CoHead.ncoHead==1:
            JointID=CoHead.coHead        
            tempseg=line[JointID[0][0]]
            if JointID[0][1]==1:
                tempseg.extend(line[idx])
                line[idx]=tempseg
                subLineNo=JointID[0][0]
                linehead[idx]=[line[idx][0],line[idx][-1],[idx]]
                linehead[subLineNo]=[]
                linehead[idx]=[line[idx][0],line[idx][-1],[idx]]
                CoHead=FindConn(idx,linehead[idx],linehead)
                linehead[idx][2].append(CoHead)
                line[subLineNo]=[[],[]]
            elif JointID[0][1]==-1:
                tempseg=tempseg[::-1]
                tempseg.extend(line[idx])
                line[idx]=tempseg
                subLineNo=JointID[0][0]
                linehead[idx]=[line[idx][0],line[idx][-1],[idx]]
                linehead[subLineNo]=[]
                linehead[idx]=[line[idx][0],line[idx][-1],[idx]]
                CoHead=FindConn(idx,linehead[idx],linehead)
                linehead[idx][2].append(CoHead)
                line[subLineNo]=[[],[]]
    else:
        CoHead=[[],[]]          
    pass
print(len(line))
def cleandata():
    while [] in linehead:
        linehead.remove([])
    while [[],[]] in line:
        line.remove([[],[]])
    for idx,hd in enumerate(linehead):
        linehead[idx][2]=[[idx]]
        CoHead=FindConn(idx,linehead[idx],linehead)
        if CoHead != []:
            linehead[idx][2].append(CoHead)
       
cleandata()  
for idx in range(0,len(line)):
    if len(line[idx][0])>1:
        x.extend([tp[0] for tp in line[idx]])
        y.extend([tp[1] for tp in line[idx]])
        x.extend([0.0])  
        y.extend([0.0])
#plt.plot(x,y,'*-')
##fl=open('ss.txt','w')
##for i in range(len(x)):
##    fl.write(str(x[i]))
##    fl.write('\t')
##    fl.write(str(y[i]))
##    fl.write('\n')
##fl.close()
print('4')
#------------------------------------------------------
#
#
##
#--------------------------------------------------------
       
#==================================================================================================
import random
from math import *
import sets
BestTour = []                   # store the best path
NodeSet = sets.Set()            # Node set
NodeList = []                   # Node list
PheromoneTrailList = []         # pheromone trail list
PheromoneDeltaTrailList = []    # delta pheromone trail list
NodeDistanceList = []           # Node distance list
AntList = []                    # ants
markVol = 1000.0
stopTm  = 3e-4
NodeList=linehead
NodeSet=sets.Set(range(1,len(NodeList)+1))  
class BACA:
    "implement basic ant colony algorithm"
    # following are some essential parameters/attributes for BACA
    def __init__(self, NodeCount=51, antCount=34, q=80, alpha=2, beta=5, rou=0.4, nMax=10):
        self.NodeCount = len(NodeList)
        self.AntCount = int(self.NodeCount*1.5)
        self.Q = q
        self.Alpha = alpha
        self.Beta = beta
        self.Rou = rou
        self.Nmax = nMax
        self.Shortest = 10e6
        # set random seed
        random.seed()      
        # init global data structure
        for nNode in range(self.NodeCount):
            BestTour.append(0)
           
        for row in range(self.NodeCount):
            pheromoneList = []
            pheromoneDeltaList = []
            for col in range(self.NodeCount):
                pheromoneList.append([100,100,100,100])               # init pheromone list to const 100
                pheromoneDeltaList.append([0,0,0,0])            # init pheromone delta list to const 0
            PheromoneTrailList.append(pheromoneList)
            PheromoneDeltaTrailList.append(pheromoneDeltaList)
       
    def ReadNodeInfo(self):        
        #print NodeDistanceList
        for row in range(self.NodeCount):
            distanceList = []
            for col in range(self.NodeCount):
                distanceHH = sqrt(pow(NodeList[row][0][0]-NodeList[col][0][0],2)+pow(NodeList[row][0][1]-NodeList[col][0][1],2))+0.0003
                distanceHT = sqrt(pow(NodeList[row][0][0]-NodeList[col][1][0],2)+pow(NodeList[row][0][1]-NodeList[col][1][1],2))+0.0003
                distanceTH = sqrt(pow(NodeList[row][1][0]-NodeList[col][0][0],2)+pow(NodeList[row][1][1]-NodeList[col][0][1],2))+0.0003
                distanceTT = sqrt(pow(NodeList[row][1][0]-NodeList[col][1][0],2)+pow(NodeList[row][1][1]-NodeList[col][1][1],2))+0.0003
                distanceList.append([distanceHH,distanceHT,distanceTH,distanceTT])
            NodeDistanceList.append(distanceList)
           
        for itx in range(len(NodeList)):
            if len(NodeList[itx][2])<1:
                continue
            for hd in NodeList[itx][2][1].coHead:
                if hd[2]<150:
                    dis=markVol*stopTm
                    loc=hd[1]
                    target=hd[0]
                    if loc==-1:
                        NodeDistanceList[itx][target][0]=dis
                    elif loc == 1:
                        NodeDistanceList[itx][target][1]=dis
            for hd in NodeList[itx][2][1].coTail:
                if hd[2]<150:
                    dis=markVol*stopTm
                    loc=hd[1]
                    target=hd[0]
                    if loc==-2:
                        NodeDistanceList[itx][target][2]=dis
                    elif loc == -3:
                        NodeDistanceList[itx][target][3]=dis
                   
           
    def PutAnts(self):
        """randomly put ants on cities"""
        for antNum in range(self.AntCount):
            Node = random.randint(1, self.NodeCount)
            pos=int(random.uniform(0,1.3))
            ant = ANT([Node,pos])
            AntList.append(ant)
            #print ant.CurrNode
    def outRout(self,tour):
        x=[]
        y=[]
        for [Node,pos] in tour:
            if pos==1:
                seg=line[Node-1][::-1]
            else:
                seg=line[Node-1]
            if len(line[Node-1][0])>1:
                x.extend([tp[0] for tp in seg])
                y.extend([tp[1] for tp in seg])
                x.extend([0.0])  
                y.extend([0.0])
        fl=open('ss.txt','w')
        for i in range(len(x)):
            fl.write(str(x[i]))
            fl.write('\t')
            fl.write(str(y[i]))
            fl.write('\n')
        fl.close()

    def Search(self):
        """search solution space"""
        for iter in range(self.Nmax):
            self.PutAnts()
            for ant in AntList:
                for ttt in range(len(NodeList)):
                    ant.MoveToNextNode(self.Alpha, self.Beta)
                ant.UpdatePathLen()
            tmpLen = AntList[0].CurrLen
            tmpTour = AntList[0].TabuNodeList
            for ant in AntList[1:]:
                if ant.CurrLen < tmpLen:
                    tmpLen = ant.CurrLen
                    tmpTour = ant.TabuNodeList
            if tmpLen < self.Shortest:
                self.Shortest = tmpLen
                BestTour = tmpTour
            print (iter,":",self.Shortest,":",BestTour)
            if iter == self.Nmax-1:
                BestTour = tmpTour[:]
                self.outRout(BestTour)
            self.UpdatePheromoneTrail()
    def UpdatePheromoneTrail(self):    #0======TH 1 TT
        for ant in AntList:
            for Node in ant.TabuNodeList[0:-1]:
                idx = ant.TabuNodeList.index(Node)
                nextNode = ant.TabuNodeList[idx+1]
                rout=[Node[1],nextNode[1]]
                if rout==[0,0]:
                    PheromoneDeltaTrailList[Node[0]-1][nextNode[0]-1][2] = self.Q/ant.CurrLen
                elif rout==[0,1]:
                    PheromoneDeltaTrailList[Node[0]-1][nextNode[0]-1][3] = self.Q/ant.CurrLen
                elif rout==[1,0]:
                    PheromoneDeltaTrailList[Node[0]-1][nextNode[0]-1][0] = self.Q/ant.CurrLen
                elif rout==[1,1]:
                    PheromoneDeltaTrailList[Node[0]-1][nextNode[0]-1][1] = self.Q/ant.CurrLen
            lastNode = ant.TabuNodeList[-1]
            firstNode = ant.TabuNodeList[0]
            rout=[firstNode[1],lastNode[1]]
            if rout==[0,0]:
                PheromoneDeltaTrailList[firstNode[0]-1][lastNode[0]-1][2]\
                        =PheromoneDeltaTrailList[lastNode[0]-1][firstNode[0]-1][0] = self.Q/ant.CurrLen
            elif rout==[0,1]:
                PheromoneDeltaTrailList[firstNode[0]-1][lastNode[0]-1][3]\
                       =PheromoneDeltaTrailList[lastNode[0]-1][firstNode[0]-1][1] = self.Q/ant.CurrLen
            elif rout==[1,0]:
                PheromoneDeltaTrailList[firstNode[0]-1][lastNode[0]-1][0]\
                       =PheromoneDeltaTrailList[lastNode[0]-1][firstNode[0]-1][2] = self.Q/ant.CurrLen
            elif rout==[1,1]:
                PheromoneDeltaTrailList[firstNode[0]-1][lastNode[0]-1][1]\
                      =PheromoneDeltaTrailList[lastNode[0]-1][firstNode[0]-1][3] = self.Q/ant.CurrLen
        for (seg1,seg1,seg1No) in NodeList:
            for (seg2,seg2,seg2No) in NodeList:
                Node1=seg1No[0][0]
                Node2=seg2No[0][0]
                #HH HT TH TT
                PheromoneTrailList[Node1-1][Node2-1][0] = ((1-self.Rou)*PheromoneTrailList[Node1-1][Node2-1][0] +
                                                    PheromoneDeltaTrailList[Node1-1][Node2-1][0])                               
                PheromoneTrailList[Node1-1][Node2-1][1] = ((1-self.Rou)*PheromoneTrailList[Node1-1][Node2-1][1] +
                                                    PheromoneDeltaTrailList[Node1-1][Node2-1][1])                                       
                PheromoneTrailList[Node1-1][Node2-1][2] = ((1-self.Rou)*PheromoneTrailList[Node1-1][Node2-1][2] +
                                                    PheromoneDeltaTrailList[Node1-1][Node2-1][2])                                   
                PheromoneTrailList[Node1-1][Node2-1][3] = ((1-self.Rou)*PheromoneTrailList[Node1-1][Node2-1][3] +
                                                    PheromoneDeltaTrailList[Node1-1][Node2-1][3])                                                 
                for  ii in range(4):
                    PheromoneDeltaTrailList[Node1-1][Node2-1][ii] = 0
   
class ANT:
    "implement ant individual"
    def __init__(self, currNode = [0,0]):
        # following are some essential attributes for ant
        self.TabuNodeSet = sets.Set()            # tabu Node set
        self.TabuNodeList = []                   # tabu Node list
        self.AllowedNodeSet = sets.Set()         # AllowedNodeSet = NodeSet - TabuNodeSet
        self.TransferProbabilityList = []        # transfer probability list
        self.CurrNode = [0,0]                       # Node which the ant current locate
        self.CurrLen = 0.0                       # current path len
        self.AddNode(currNode)
        pass
    def SelectNextNode(self, alpha, beta):
        """select next Node to move to"""
        #MAXLEN = 1e6
        if len(self.AllowedNodeSet) == 0:
            return (0)
        sumProbability = 0.0
        #HH HT TH TT
        for Node in self.AllowedNodeSet:
            segProbability=0
            currentPos=self.CurrNode[1]
            if currentPos==1:# it is right for tabu seg Pt
                banRout=[2,3]
            else:
                banRout=[0,1]
            for HT in range(4):
                if HT in banRout:
                    continue
                segProbability+=(pow(PheromoneTrailList[self.CurrNode[0]-1][Node-1][HT], alpha)
                                               * pow(1.0/NodeDistanceList[self.CurrNode[0]-1][Node-1][HT], beta))            
            sumProbability = sumProbability + segProbability
        self.TransferProbabilityList = []
        for Node in self.AllowedNodeSet:
            currPos = self.CurrNode[1]
            if currPos==0:
                transferProbability1 = (pow(PheromoneTrailList[self.CurrNode[0]-1][Node-1][2], alpha)
                                    * pow(1.0/NodeDistanceList[self.CurrNode[0]-1][Node-1][2], beta))/sumProbability
                transferProbability2 = (pow(PheromoneTrailList[self.CurrNode[0]-1][Node-1][3], alpha)
                                    * pow(1.0/NodeDistanceList[self.CurrNode[0]-1][Node-1][3], beta))/sumProbability
            else:
                transferProbability1 = (pow(PheromoneTrailList[self.CurrNode[0]-1][Node-1][0], alpha)
                                    * pow(1.0/NodeDistanceList[self.CurrNode[0]-1][Node-1][0], beta))/sumProbability
                transferProbability2 = (pow(PheromoneTrailList[self.CurrNode[0]-1][Node-1][1], alpha)
                                    * pow(1.0/NodeDistanceList[self.CurrNode[0]-1][Node-1][1], beta))/sumProbability                  
            self.TransferProbabilityList.append([Node,transferProbability1,transferProbability2])
        # determine next Node
        select = 0.0
        for Node,NodeProb1,NodeProb2 in self.TransferProbabilityList:
            if NodeProb1 > select or NodeProb2 >select:
                if NodeProb1>NodeProb2:
                    select=NodeProb1
                else:
                    select=NodeProb2
        threshold = select * random.random()
        for (NodeNum, NodeProb1,NodeProb2) in self.TransferProbabilityList:
            if NodeProb1 >= threshold or NodeProb2 >= threshold:
                if NodeProb1>NodeProb2:
                    return [NodeNum,0]
                else:
                    return [NodeNum,1]

        return (0)
    def MoveToNextNode(self, alpha, beta):
        """move the ant to next Node"""
        nextNode = self.SelectNextNode(alpha, beta)
        if nextNode > 0:
            self.AddNode(nextNode)
    def ClearTabu(self):
        """clear tabu list and set"""
        self.TabuNodeList = []
        self.TabuNodeSet.clear()
        self.AllowedNodeSet = NodeSet - self.TabuNodeSet
    def UpdatePathLen(self):
        """sum up the path length"""
        for Node in self.TabuNodeList[0:-1]:
            nextNode = self.TabuNodeList[self.TabuNodeList.index(Node)+1]
            rout=[Node[1],nextNode[1]]
            distPos=4
            if rout==[0,0]:
                distPos=2
            elif rout==[0,1]:
                distPos=3
            elif rout==[1,0]:
                distPos=0
            elif rout==[1,1]:
                distPos=1
            self.CurrLen = self.CurrLen + NodeDistanceList[Node[0]-1][nextNode[0]-1][distPos]
        lastNode = self.TabuNodeList[-1]
        firstNode = self.TabuNodeList[0]
        self.CurrLen = self.CurrLen + NodeDistanceList[lastNode[0]-1][firstNode[0]-1][lastNode[1]+2]
    def AddNode(self,Node):
        """add Node to tabu list and set"""
        if Node[0] <= 0:
            return
        self.CurrNode = Node
        self.TabuNodeList.append(Node)
        self.TabuNodeSet.add(Node[0])
        self.AllowedNodeSet = NodeSet - self.TabuNodeSet  
if __name__ == "__main__":
    theBaca = BACA()
    theBaca.ReadNodeInfo()
    theBaca.Search()
    os.system("pause")
   
