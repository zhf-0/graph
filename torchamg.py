import torch

class TwoGrid:
    def __init__(self,A,p):
        self.A = A
        self.P = p
        self.R = p.T

    def Setup(self):
        A_c = self.R @ self.A @ self.P 
        self.dense_A_c = A_c.to_dense()

    def CoarseSolve(self,b):
        x = torch.linalg.solve(self.dense_A_c, b)
        return x

    def Solve(self,b):
        if len(b.shape) == 1:
            b = b.unsqueeze(1)

        Rb = self.R @ b
        x = self.CoarseSolve(Rb)
        out = self.P @ x

        return out

            

