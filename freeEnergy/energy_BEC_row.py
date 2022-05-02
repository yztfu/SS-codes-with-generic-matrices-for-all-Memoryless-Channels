import vegas
import math
import numpy as np
from scipy import integrate
import functools
import random
import matplotlib.pyplot as plt
import numpy as np

'''
B=4
R=0.603
alpha=0.5/R
epsilon = 0.1
N=200000 

def f(x):
    temp1=math.exp(0.5*qx_hat+math.sqrt(qx_hat)*x[0])
    temp2=0
    for j in range(1,B):
        temp2=temp2+math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*x[j])
    temp3=math.exp(0.5*qx_hat+math.sqrt(qx_hat)*x[0])*(1+x[0]/(math.sqrt(qx_hat)))
    temp4=(B-1)*math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*x[1])*(-1+x[1]/math.sqrt(qx_hat))
    temp5=0
    for i in range(B):
        temp5=temp5-x[i]**2
    temp6=(1/(2*math.pi))**(B/2)*math.exp(temp5/2)
    return temp6*((temp3+temp4)/(temp1+temp2))

def saddle():
    integ=vegas.Integrator(B*[[-5,5]])
    return integ(f,nitn=10,neval=200000).mean

def integrated_function_out(z,qz_hat):
    gamma=math.sqrt(qz_hat/(1-qz_hat))
    temp1=(0.5+0.5*math.erf(-math.sqrt(2)/2*gamma*z))*(1-epsilon)
    if(temp1==0):
        return 0
    else:
        return temp1*math.log(temp1)*math.exp(-z*z/2)/math.sqrt(2*math.pi)

def I_out(qz_hat):
    return 2*integrate.quad(integrated_function_out,-5,5,args=(qz_hat,))[0]

qx=0.17
qz=0.8
qx_hat=1.2
qz_hat=0.5
#qx=0.9151156908631846/B
#qz=2.219571322031546
#qx_hat=8.015088985231719
#qz_hat=0.8968895832655542
for i in range(50):
    qz_hat=0.9*(1-2*(1-B*qx)/(1+math.sqrt(1-4*alpha*(1-B*qx)*qz)))+0.1*qz_hat
    qx_hat=0.6*(2*alpha*B*qz/(1+math.sqrt(1-4*alpha*(1-B*qx)*qz)))+0.4*qx_hat
    qx=(saddle()/B+qx)/2
    qz=(2*(I_out(qz_hat+0.0001)-I_out(qz_hat))/0.0001+qz)/2
    #print(1-B*qx)
    #print("\n")
#print("result:")
#print("\n")
#print(1-B*qx)
#print("\n")
#print(qz_hat)
#print("\n")
#print(qx_hat)
#print("\n")
#print(B*qx)
#print("\n")
#print(qz)

def f_2(x):
    temp1=math.exp(0.5*qx_hat+math.sqrt(qx_hat)*x[0])
    temp2=0
    for j in range(1,B):
        temp2=temp2+math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*x[j])
    temp3=0
    for i in range(B):
        temp3=temp3-x[i]**2
    return math.log(temp1+temp2)*(1/(2*math.pi))**(B/2)*math.exp(temp3/2)

def I_0(qx,qx_hat):
    temp1=-B/2*qx*qx_hat
    integ=vegas.Integrator(B*[[-5,5]])
    temp2=integ(f_2,nitn=10,neval=200000).mean
    return temp1+temp2

def Iout(qz,qz_hat):
    return -B/2*qz*qz_hat+B*I_out(qz_hat)

def I_int(qx,qz):
    temp1=math.sqrt(1-4*alpha*(1-B*qx)*qz)
    return -0.5*B*math.log((1+temp1)/2)+0.5*B*temp1+0.5*alpha*B*qz-0.5*B

def phi(qx,qx_hat,qz,qz_hat):
    return I_0(qx,qx_hat)+alpha*Iout(qz,qz_hat)+I_int(qx,qz)

entropy = phi(qx,qx_hat,qz,qz_hat)
'''
file = open("4rITBEC.txt","w")

for j in range(11):
    B=4
    R=0.65 + j * 0.01
    alpha=0.5/R
    epsilon = 0.1
    N=200000 

    def f(x):
        temp1=math.exp(0.5*qx_hat+math.sqrt(qx_hat)*x[0])
        temp2=0
        for j in range(1,B):
            temp2=temp2+math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*x[j])
        temp3=math.exp(0.5*qx_hat+math.sqrt(qx_hat)*x[0])*(1+x[0]/(math.sqrt(qx_hat)))
        temp4=(B-1)*math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*x[1])*(-1+x[1]/math.sqrt(qx_hat))
        temp5=0
        for i in range(B):
            temp5=temp5-x[i]**2
        temp6=(1/(2*math.pi))**(B/2)*math.exp(temp5/2)
        return temp6*((temp3+temp4)/(temp1+temp2))

    def saddle():
        integ=vegas.Integrator(B*[[-5,5]])
        return integ(f,nitn=10,neval=200000).mean

    def integrated_function_out(z,qz_hat):
        gamma=math.sqrt(qz_hat/(1-qz_hat))
        temp1=(0.5+0.5*math.erf(-math.sqrt(2)/2*gamma*z))*(1-epsilon)
        if(temp1==0):
            return 0
        else:
            return temp1*math.log(temp1)*math.exp(-z*z/2)/math.sqrt(2*math.pi)

    def I_out(qz_hat):
        return 2*integrate.quad(integrated_function_out,-5,5,args=(qz_hat,))[0]

    qx=0.17
    qz=0.8
    qx_hat=1.2
    qz_hat=0.5
    #qx=0.9151156908631846/B
    #qz=2.219571322031546
    #qx_hat=8.015088985231719
    #qz_hat=0.8968895832655542
    for i in range(50):
        qz_hat=0.9*(1-2*(1-B*qx)/(1+math.sqrt(1-4*alpha*(1-B*qx)*qz)))+0.1*qz_hat
        qx_hat=0.6*(2*alpha*B*qz/(1+math.sqrt(1-4*alpha*(1-B*qx)*qz)))+0.4*qx_hat
        qx=(saddle()/B+qx)/2
        qz=(2*(I_out(qz_hat+0.0001)-I_out(qz_hat))/0.0001+qz)/2
        #print(1-B*qx)
        #print("\n")
    #print("result:")
    #print("\n")
    #print(1-B*qx)
    #print("\n")
    #print(qz_hat)
    #print("\n")
    #print(qx_hat)
    #print("\n")
    #print(B*qx)
    #print("\n")
    #print(qz)

    def f_2(x):
        temp1=math.exp(0.5*qx_hat+math.sqrt(qx_hat)*x[0])
        temp2=0
        for j in range(1,B):
            temp2=temp2+math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*x[j])
        temp3=0
        for i in range(B):
            temp3=temp3-x[i]**2
        return math.log(temp1+temp2)*(1/(2*math.pi))**(B/2)*math.exp(temp3/2)

    def I_0(qx,qx_hat):
        temp1=-B/2*qx*qx_hat
        integ=vegas.Integrator(B*[[-5,5]])
        temp2=integ(f_2,nitn=10,neval=200000).mean
        return temp1+temp2

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
    file.write("\n")
    print(j)

file.close()

