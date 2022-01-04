import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# unpack tuple (N,)
def unpackTuple(X,Y):
    X1 = []
    X2 = []
    X3 = []
    out = []
    for index,item in enumerate(X):
        X1.append(item[0])
        X2.append(item[1])
        X3.append(item[2])
        out.append(Y[index][0])
    return X1,X2,X3,out

# pack tuple (N,3)
def packTuple(x,y,z):
    result = []
    for index, item in enumerate(x):
        result.append([item,y[index],z[index]])
    return result

# renders points based on same frames as above renderer
def pointRender(inputList, outputList):
    inputList = np.array(inputList)
    outputList = np.array(outputList)
    overall = np.concatenate((inputList,outputList), axis=1)
    overall = overall[np.argsort(overall[:,2])]
    z = np.linspace(-10, 10, 400)

    index = 0
    zi = 0
    store = []

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-10,10])

    for index, p in enumerate(overall):
        X,Y,Z,out = p

        if Z>z[zi]:

            plt.savefig(str(Z)+'.png')
            store.append(str(Z)+'.png')
            plt.clf()
            plt.close()
            zi+=1

            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])
            ax.set_zlim([-10,10])

        ax.scatter3D(X, Y, out);

    images = []
    while(store):
        temp = store.pop(0)
        images.append(imageio.imread(temp))
        os.remove(temp)
    imageio.mimsave('pts.gif',images)
