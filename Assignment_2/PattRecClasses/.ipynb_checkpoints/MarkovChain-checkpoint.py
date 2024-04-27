import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


        self.nStates = transition_prob.shape[0]
        self.endState = self.nStates + 1

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD


    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
#     def rand(self, tmax):
#         """
#         S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
#         Input:
#         tmax= scalar defining maximum length of desired state sequence.
#            An infinite-duration MarkovChain always generates sequence of length=tmax
#            A finite-duration MarkovChain may return shorter sequence,
#            if END state was reached before tmax samples.
        
#         Result:
#         S= integer row vector with random state sequence,
#            NOT INCLUDING the END state,
#            even if encountered within tmax samples
#         If mc has INFINITE duration,
#            length(S) == tmax
#         If mc has FINITE duration,
#            length(S) <= tmaxs
#         """
        
#         #*** Insert your own code here and remove the following error message 
        
#         S = np.zeros(tmax, dtype=int)
#         end_state_value = self.nStates # should not be nStates+1 since index in python starts from 0
        
#         for t in range(0, tmax):
#             if t == 0:
#                 S[t] = DiscreteD(self.q).rand(1).item()
#             else:
#                 S[t] = DiscreteD(self.A[S[t-1]]).rand(1).item()
            
#             if self.is_finite and S[t] == end_state_value: 
#                 return S[:t]
        
#         return S

    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        #*** Insert your own code here and remove the following error message 
        
        S = np.zeros(tmax, dtype=int)
        
        for t in range(0, tmax):
            if t == 0:
                S[t] = DiscreteD(self.q).rand(1).item()
            else:
                S[t] = DiscreteD(self.A[S[t-1] - 1]).rand(1).item()
            
            if self.is_finite and S[t] == self.endState: 
                return S[:t]
        
        return S
        


    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass
    
    def forward(self, pX):
    # Initialization
        n, t_max = np.shape(pX) # n <=> # of states; t_max <=> sequence length
        c = np.zeros(t_max+1)
        alpha_hat = np.zeros([n,t_max])
        alpha_temp = self.q * pX[:,0] # 5.42
        c[0] = np.sum(alpha_temp) # 5.43
        alpha_hat[:,0] = alpha_temp / c[0] # 5.44
        
    # Forward Step
        for t in range(1, t_max):
            alpha_temp = np.array([])
            for j in range(n):
                alpha_update = pX[j,t] * ( alpha_hat[:,t-1] @ self.A[:,j] ) # 5.50
                alpha_temp = np.append(alpha_temp, alpha_update) # 5.50
            c[t] = np.sum(alpha_temp, axis=0) # 5.51
            alpha_hat[:,t] = alpha_temp / c[t] # 5.52
            
    # Termination
        if self.is_finite:
            c[-1] = alpha_hat[:,-1] @ self.A[:,-1] # 5.53 
        else:
            c = c[0:-1] # length of c is different in infinite and finite case
        
        return alpha_hat, c
                
    def finiteDuration(self):
        pass
    
    def backward(self, pX, c):
    # Initialization
        n, t_max = np.shape(pX)
        beta_hat = np.zeros([n, t_max])
        if self.is_finite:
            beta_end = self.A[:,-1] / (c[-1]*c[-2]) # 5.65
        else:
            beta_end = np.ones(self.A.shape[0]) / c[-1] # 5.64
        beta_hat[:,-1] = beta_end
        
    # Backward Step
        c = np.flip(c) # reverse vector c for simplicity
        c = c[2:] if self.is_finite else c[1:]
        for t in range(c.shape[0]):
            beta_hat_t = np.zeros(n)
            for i in range(n):
                beta_i_t = np.sum(self.A[i,:n] * pX[:,-1-t] * beta_hat[:n,-1-t]) # 5.69
                # print(self.A[i,:self.A.shape[0]])
                # print(pX[:self.A.shape[0],-1-t])
                # print(beta_hat[:self.A.shape[0],-1-t])
                beta_hat_i_t = beta_i_t / c[t] # 5.70
                beta_hat_t[i] = beta_hat_i_t
            beta_hat[:,-2-t] = beta_hat_t # recursive update
            
        return beta_hat

#     def backward(self, p_x, c):
#         beta_hat = []

#         """initialization"""
#         if self.is_finite:
#             beta_hat0 = [beta/(c[-1]*c[-2]) for beta in self.A[:, -1]]  # eq 5.65
#         else:
#             beta_hat0 = [1/c[-1] for i in range(self.A.shape[0])]  # eq 5.64
#         beta_hat.insert(0, beta_hat0)

#         """backward step"""
#         c = c[:-2] if self.is_finite else c[:-1]
#         c = list(c)
#         c.reverse()
#         for t in range(len(c)):
#             beta_hat_t = []
#             for i in range(self.A.shape[0]):
#                 print(self.A.shape[0])
#                 beta_i_t = sum([p_x[j, -t-1]*self.A[i, j]*beta_hat[0][j] for j in range(self.A.shape[0])])
#                 beta_hat_i_t = beta_i_t/c[t]
#                 beta_hat_t.append(beta_hat_i_t)  # eq 5.70
#             beta_hat.insert(0, beta_hat_t)

#         return beta_hat


    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass

        
    

#   def forward(self, pX):
#     # Initialization
#         n, t_max = np.shape(pX) # n <=> # of states; t_max <=> sequence length
#         c = np.zeros(t_max+1)
#         alpha_hat = np.zeros([t_max,n])
#         alpha_temp = self.q * pX[:,0] # 5.42
#         c[0] = np.sum(alpha_temp) # 5.43
#         alpha_hat[0,:] = alpha_temp / c[0] # 5.44
        
#     # Forward Step
#         for t in range(1, t_max):
#             alpha_temp = np.array([])
#             for j in range(n):
#                 alpha_update = pX[j,t] * ( alpha_hat[t-1,:] @ self.A[:,j] ) # 5.50
#                 alpha_temp = np.append(alpha_temp, alpha_update) # 5.50
#             c[t] = np.sum(alpha_temp, axis=0) # 5.51
#             alpha_hat[t,:] = alpha_temp / c[t] # 5.52
            
#     # Termination
#         if self.is_finite:
#             c[-1] = alpha_hat[-1,:] @ self.A[:,-1] # 5.53 
        
#         return alpha_hat, c

