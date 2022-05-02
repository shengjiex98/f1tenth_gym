import os,sys
import random
import numpy.random as nrd
import time


class RandSampling:
    def __init__(self,H=150,schedPol="HoldSkip-Next",distro="K-Miss",K_miss=-1):
        self.H=H
        self.schedPol=schedPol
        self.distro=distro
        self.K_miss=K_miss
        self.l=self.buildLengthDict()

    def getARandSampleKMissesRGA(self):
        q=0
        seqn=[]
        for i in range(1,self.H+1):
            d=self.l[q,self.H-i+1]
            if q==self.K_miss+1:
                prob_one=self.l[q,self.H-i]/d # probability of choosing one
            else:
                prob_one=self.l[0,self.H-i]/d # probability of choosing one
            rand_bit=nrd.binomial(1,prob_one)
            #print(prob_one)
            seqn.append(rand_bit)
            if rand_bit==1:
                if q==self.K_miss+1:
                    q=q
                else:
                    q=0
            else:
                if q==self.K_miss+1:
                    q=q
                else:
                    q=q+1

        return seqn

    def buildLengthDict(self):
        l={}
        for q in range(self.K_miss+2):
            if q==self.K_miss+1:
                # dummy state
                l[(q,0)]=0
            else:
                # all other states
                l[(q,0)]=1

        for i in range(1,self.H+1):
            for q in range(self.K_miss+2):
                if q==self.K_miss+1:
                    z=l[(q,i-1)]
                    o=l[(q,i-1)]
                else:
                    z=l[(q+1,i-1)]
                    o=l[(0,i-1)]

                l[(q,i)]=z+o

        return l

