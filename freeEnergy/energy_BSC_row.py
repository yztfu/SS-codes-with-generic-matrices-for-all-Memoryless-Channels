import vegas
import math
import numpy as np
from scipy import integrate
import functools
import random
import matplotlib.pyplot as plt
import numpy as np

file = open("2rITBSC.txt","w")

for j in range(11):
    B = 2
    R = 0.52 + 0.01 * j
    alpha = math.log(B) / math.log(2) / R / B
    epsilon = 0.01

    def integrated_function(x1,x2,qx_hat):
        temp1=math.exp(0.5*qx_hat+math.sqrt(qx_hat)*x1)
        temp2=math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*x2)
        temp3=1+x1/math.sqrt(qx_hat)
        temp4=-1+x2/math.sqrt(qx_hat)
        temp5=1/(2*math.pi)*math.exp((-x1**2-x2**2)/2)
        return temp5*(temp1*temp3+temp2*temp4)/(temp1+temp2)

    def integral(qx_hat):
        return integrate.dblquad(integrated_function,-5,5,-5,5,args=(qx_hat,))[0]

    
    def integrated_function_out(z,qz_hat):
        gamma=math.sqrt(qz_hat/(1-qz_hat))
        temp1=(0.5+0.5*math.erf(-math.sqrt(2)/2*gamma*z))*(1-2*epsilon)+epsilon
        if(temp1==0):
            return 0
        else:
            return temp1*math.log(temp1)*math.exp(-z*z/2)/math.sqrt(2*math.pi)

    def I_out(qz_hat):
        return 2*integrate.quad(integrated_function_out,-5,5,args=(qz_hat,))[0]

    qx=0.33
    qz=0.2
    qx_hat=1
    qz_hat=1
    for i in range(200):
        #qz_hat=(1-2*(1-B*qx)/(1+math.sqrt(1-4*alpha*(1-B*qx)*qz))+qz_hat)/2
        #qx_hat=(2*alpha*B*qz/(1+math.sqrt(1-4*alpha*(1-B*qx)*qz))+qx_hat)/2
        #qz_hat=1-(2*(1-B*qx)/(1+math.sqrt(1-4*alpha*(1-B*qx)*qz)))
        #qx_hat=2*alpha*B*qz/(1+math.sqrt(1-4*alpha*(1-B*qx)*qz))
        qz_hat=0.9*(1-2*(1-B*qx)/(1+math.sqrt(1-4*alpha*(1-B*qx)*qz)))+0.1*qz_hat
        qx_hat=0.6*(2*alpha*B*qz/(1+math.sqrt(1-4*alpha*(1-B*qx)*qz)))+0.4*qx_hat
        qx=(1/B*(integral(qx_hat))+qx)/2
        qz=(2*(I_out(qz_hat+0.00001)-I_out(qz_hat))/0.00001+qz)/2
        #print(1-B*qx)
    #print(1-B*qx)
    #print(qz_hat)
    #print(qx_hat)
    #print(qx)
    #print(qz)

    def integrated_fuction2(xi1,xi2,qx_hat):
        temp1=math.exp(0.5*qx_hat+math.sqrt(qx_hat)*xi1)
        temp2=math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*xi2)
        temp=math.log(temp1+temp2)
        Pr=1/(2*math.pi)*math.exp(-(xi1**2+xi2**2)/2)
        return Pr*temp

    def I_0(qx,qx_hat):
        return -B/2*qx*qx_hat+integrate.dblquad(integrated_fuction2,-5,5,-5,5,args=(qx_hat,))[0]

    def Iout(qz,qz_hat):
        return -B/2*qz*qz_hat+B*I_out(qz_hat)

    def I_int(qx,qz):
        temp1=math.sqrt(1-4*alpha*(1-B*qx)*qz)
        return -0.5*B*math.log((1+temp1)/2)+0.5*B*temp1+0.5*alpha*B*qz-0.5*B
           
    def phi(qx,qx_hat,qz,qz_hat):
        return I_0(qx,qx_hat)+alpha*Iout(qz,qz_hat)+I_int(qx,qz)

    entropy = phi(qx,qx_hat,qz,qz_hat)

    file.write(str(R))
    file.write(" ")
    file.write(str(entropy))
    file.write(" ")
    result = alpha*B*(epsilon*math.log(epsilon) + (1-epsilon)*math.log(1-epsilon))
    file.write(str(result))
    file.write("\n")
    print(j)

file.close()