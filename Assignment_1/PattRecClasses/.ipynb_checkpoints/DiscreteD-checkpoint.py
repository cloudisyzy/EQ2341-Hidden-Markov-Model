import numpy as np

class DiscreteD:
    """
    DiscreteD - class representing random discrete integer.
    
    A Random Variable with this distribution is an integer Z
    with possible values 1,...,length(ProbMass).
    
    Several DiscreteD objects may be collected in an array
    """
    def __init__(self, x):
        self.pseudoCount = 0
        self.probMass = x/np.sum(x)
        
    def rand(self, nData):
        """
        R=rand(nData) returns random scalars drawn from given Discrete Distribution.
        
        Input:
        nData= scalar defining number of wanted random data elements
        
        Result:
        R= row vector with integer random data drawn from the DiscreteD object
           (size(R)= [1, nData]
        """
        
        #*** Insert your own code here and remove the following error message 
        
        print('Not yet implemented')
        
        
    def init(self, x):
        """
        initializes DiscreteD object or array of such objects
        to conform with a set of given observed data values.
        The agreement is crude, and should be further refined by training,
        using methods adaptStart, adaptAccum, and adaptSet.
        
        Input:
        x=     row vector with observed data samples
        
        Method:
        For a single DiscreteD object: Set ProbMass using all observations.
        For a DiscreteD array: Use all observations for each object,
               and increase probability P[X=i] in pD(i),
        This is crude, but there is no general way to determine
               how "close" observations X=m and X=n are,
               so we cannot define "clusters" in the observed data.
        """
        if len(np.shape(x))>1: 
            print('DiscreteD object can have only scalar data')
            
        x = np.round(x)
        maxObs = int(np.max(x))
        # collect observation frequencies
        fObs = np.zeros(maxObs) # observation frequencies

        for i in range(maxObs):
            fObs[i] = 1 + np.sum(x==i)
        
        self.probMass = fObs/np.sum(fObs)

        return self


    def entropy(self):
        pass

    def prob(self):
        pass
    
    def double(self):
        pass

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
