import numpy as np
import matplotlib.pyplot as plt
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
            if len(it)<20:
                break
            Pt=np.array(itt)
            dist=Pt-lastPt
            lastPt=Pt
            dist=np.sqrt(dist.dot(dist))
            if dist<0.004:
                if idd==len(it)-1:
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
plt.plot(x,y,'*-')
fl=open('ss.txt','w')
for i in range(len(x)):
    fl.write(str(x[i]))
    fl.write('\t')
    fl.write(str(y[i]))
    fl.write('\n')
fl.close()
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
CitySet = sets.Set()            # city set
CityList = []                   # city list
PheromoneTrailList = []         # pheromone trail list
PheromoneDeltaTrailList = []    # delta pheromone trail list
CityDistanceList = []           # city distance list
AntList = []                    # ants
markVol = 2000.0
stopTm  = 3e-4
class BACA:
    "implement basic ant colony algorithm"
    # following are some essential parameters/attributes for BACA
    def __init__(self, cityCount=51, antCount=34, q=80, alpha=2, beta=5, rou=0.3, nMax=100):
        self.CityCount = len(linehead)
        self.AntCount = int(self.CityCount*0.8)
        self.Q = q
        self.Alpha = alpha
        self.Beta = beta
        self.Rou = rou
        self.Nmax = nMax
        self.Shortest = 10e6
        # set random seed
        random.seed()       
        # init global data structure
        for nCity in range(self.CityCount):
            BestTour.append(0)
           
        for row in range(self.CityCount):
            pheromoneList = []
            pheromoneDeltaList = []
            for col in range(self.CityCount):
                pheromoneList.append([100,100])               # init pheromone list to const 100
                pheromoneDeltaList.append(0)            # init pheromone delta list to const 0
            PheromoneTrailList.append(pheromoneList)
            PheromoneDeltaTrailList.append(pheromoneDeltaList)
       
    def ReadCityInfo(self):
        CityList=linehead[:]
        CitySet=sets.Set(range(len(CityList)))          
        #print CityDistanceList
        for row in range(self.CityCount):
            distanceList = []
            for col in range(self.CityCount):
                distanceHH = sqrt(pow(CityList[row][0][0]-CityList[col][0][0],2)+pow(CityList[row][0][1]-CityList[col][0][1],2))
                distanceHT = sqrt(pow(CityList[row][0][0]-CityList[col][1][0],2)+pow(CityList[row][0][1]-CityList[col][1][1],2))
                distanceTT = sqrt(pow(CityList[row][1][0]-CityList[col][0][0],2)+pow(CityList[row][1][1]-CityList[col][0][1],2))
                distanceTH = sqrt(pow(CityList[row][1][0]-CityList[col][1][0],2)+pow(CityList[row][1][1]-CityList[col][1][1],2))
                distanceList.append([distanceHH,distanceHT,distanceTH,distanceTT])
            CityDistanceList.append(distanceList)
        print (len(CityDistanceList),len(CityDistanceList[1]))
        for itx in range(len(CityList)):
            if len(CityList[itx][2])<1:
                continue
            for hd in CityList[itx][2][1].coHead:
                if hd[2]<150:
                    dis=markVol*stopTm
                    loc=hd[1]
                    target=hd[0]
                    if loc==-1:
                        CityDistanceList[itx][target][0]=dis
                    elif loc == 1:
                        CityDistanceList[itx][target][1]=dis
            for hd in CityList[itx][2][1].coTail:
                if hd[2]<150:
                    dis=markVol*stopTm
                    loc=hd[1]
                    target=hd[0]
                    if loc==-2:
                        CityDistanceList[itx][target][2]=dis
                    elif loc == -3:
                        CityDistanceList[itx][target][3]=dis
                   
           
    def PutAnts(self):
        """randomly put ants on cities"""
        for antNum in range(self.AntCount):
            city = random.randint(1, self.CityCount)
			pos=random.randint(0,1)
            ant = ANT([city,pos])
            AntList.append(ant)
            #print ant.CurrCity
    def Search(self):
        """search solution space"""
        for iter in range(self.Nmax):
            self.PutAnts()
            for ant in AntList:
                for ttt in range(len(CityList)):
                    ant.MoveToNextCity(self.Alpha, self.Beta)
                ant.UpdatePathLen()
            tmpLen = AntList[0].CurrLen
            tmpTour = AntList[0].TabuCityList
            for ant in AntList[1:]:
                if ant.CurrLen < tmpLen:
                    tmpLen = ant.CurrLen
                    tmpTour = ant.TabuCityList
            if tmpLen < self.Shortest:
                self.Shortest = tmpLen
                BestTour = tmpTour
            print (iter,":",self.Shortest,":",BestTour)
            self.UpdatePheromoneTrail()
##            for ant in AntList:
##                city = ant.TabuCityList[-1]
##                ant.ClearTabu()
##                ant.AddCity(city)
    def UpdatePheromoneTrail(self):    #0======TH 1 TT
        for ant in AntList:
            for city in ant.TabuCityList[0:-1]:
                idx = ant.TabuCityList.index(city)
                nextCity = ant.TabuCityList[idx+1]
                PheromoneDeltaTrailList[city-1][nextCity-1] = self.Q/ant.CurrLen
                PheromoneDeltaTrailList[nextCity-1][city-1] = self.Q/ant.CurrLen
            lastCity = ant.TabuCityList[-1]
            firstCity = ant.TabuCityList[0]
            PheromoneDeltaTrailList[lastCity-1][firstCity-1] = self.Q/ant.CurrLen
            PheromoneDeltaTrailList[firstCity-1][lastCity-1] = self.Q/ant.CurrLen
        for (seg1,seg1,seg1No) in CityList:
            for (seg2,seg2,seg2No) in CityList:
                city1=seg1No[0][0]
                city2=seg2No[0][0]
                PheromoneTrailList[city1-1][city2-1][0] = ((1-self.Rou)*PheromoneTrailList[city1-1][city2-1][0] +
                                                    PheromoneDeltaTrailList[city1-1][city2-1])
                PheromoneTrailList[city1-1][city2-1][1] = ((1-self.Rou)*PheromoneTrailList[city1-1][city2-1][1] +
                                                    PheromoneDeltaTrailList[city1-1][city2-1])
                PheromoneDeltaTrailList[city1-1][city2-1] = 0
   
class ANT:
    "implement ant individual"
    def __init__(self, currCity = [0,0]):
        # following are some essential attributes for ant
        self.TabuCitySet = sets.Set()            # tabu city set
        self.TabuCityList = []                   # tabu city list
        self.AllowedCitySet = sets.Set()         # AllowedCitySet = CitySet - TabuCitySet
        self.TransferProbabilityList = []        # transfer probability list
        self.CurrCity = [0,0]                       # city which the ant current locate
        self.CurrLen = 0.0                       # current path len
        self.AddCity(currCity)
        pass
    def SelectNextCity(self, alpha, beta):
        """select next city to move to"""
        #MAXLEN = 1e6
        if len(self.AllowedCitySet) == 0:
            return (0)
        sumProbability = 0.0
        #
        for city in self.AllowedCitySet:
            sumProbability = sumProbability + (pow(PheromoneTrailList[self.CurrCity-1][city-1][0], alpha)
                                               * pow(1.0/CityDistanceList[self.CurrCity-1][city-1][2], beta))
            +(pow(PheromoneTrailList[self.CurrCity-1][city-1][1], alpha)
              * pow(1.0/CityDistanceList[self.CurrCity-1][city-1][3], beta))
        self.TransferProbabilityList = []
        for city in self.AllowedCitySet:
            transferProbability1 = (pow(PheromoneTrailList[self.CurrCity-1][city-1][0], alpha)
                                * pow(1.0/CityDistanceList[self.CurrCity-1][city-1][2], beta))/sumProbability
            transferProbability2 = (pow(PheromoneTrailList[self.CurrCity-1][city-1][1], alpha)
                                * pow(1.0/CityDistanceList[self.CurrCity-1][city-1][3], beta))/sumProbability
            self.TransferProbabilityList.append([city,transferProbability1,transferProbability2])
        # determine next city
        select = 0.0
        for city,cityProb1,cityProb2 in self.TransferProbabilityList:
            if cityProb1 > select or cityProb2 >select:
                if cityProb1>cityProb2:
                    select=cityProb1
                else:
                    select=cityProb2
        threshold = select * random.random()
        for (cityNum, cityProb1,cityProb2) in self.TransferProbabilityList:
            if cityProb1 >= threshold or cityProb2 >= threshold:
                if cityProb1>cityProb2:
                    return [cityNum,0]
                else:
                    return [cityNum,1]

        return (0)
    def MoveToNextCity(self, alpha, beta):
        """move the ant to next city"""
        nextCity = self.SelectNextCity(alpha, beta)
        if nextCity > 0:
            self.AddCity(nextCity)
    def ClearTabu(self):
        """clear tabu list and set"""
        self.TabuCityList = []
        self.TabuCitySet.clear()
        self.AllowedCitySet = CitySet - self.TabuCitySet
    def UpdatePathLen(self):
        """sum up the path length"""
        for city in self.TabuCityList[0:-1]:
            nextCity = self.TabuCityList[self.TabuCityList.index(city)+1]
            self.CurrLen = self.CurrLen + CityDistanceList[city-1][nextCity[0]-1][nextCity[1]+2]
        lastCity = self.TabuCityList[-1]
        firstCity = self.TabuCityList[0]
        self.CurrLen = self.CurrLen + CityDistanceList[lastCity-1][firstCity-1]
    def AddCity(self,city):
        """add city to tabu list and set"""
        if city[0] <= 0:
            return
        self.CurrCity = city
        self.TabuCityList.append(city)
        self.TabuCitySet.add(city[0])
        self.AllowedCitySet = CitySet - self.TabuCitySet  
		
		
		
if __name__ == "__main__":
    theBaca = BACA()
    theBaca.ReadCityInfo()
    theBaca.Search()`11	
    os.system("pause")
   
