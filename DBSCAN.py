# -*- coding: utf-8 -*-
"""
This is an implementation of DBSCAN following the pseudo code presented in the original paper "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
As a simple program to verify the algorithm, I didn't use R* tree as mentioned in the paper for speed acceleration.

Created on 15.02.2016 by YangZhao
"""
import math
import matplotlib.pyplot as plt
import numpy as np

class Points():
    def __init__(self, _x, _y, _idx):
        self.x = _x
        self.y = _y
        self.idx = _idx
        self.ClId = -1      # -1: UNCLASSIFIED
                            # 0: NOISE
                            # >0: Cluster Id
    def __str__(self):
        result = "Idx: " + str(self.idx) + "\t" + \
                 "x: " + str(self.x) + "\t" + \
                 "y: " + str(self.y) + "\t" + \
                 "ClId: " + str(self.ClId)
        return result

def euclid(p1, p2):
    return math.sqrt( (p1.x - p2.x)**2 + (p1.y - p2.y)**2 )

class DBSCAN():
    def __init__(self, fname, _eps, _minPts):
        self.eps = _eps
        self.minPts = _minPts
        self.clusterId = 0

        self.setOfPoints = []
        with open(fname) as f:  # read data
            idx = 0
            for line in f:
                coord = line.split()
                x = float(coord[0])
                y = float(coord[1])
                self.setOfPoints.append(Points(x, y, idx))
                idx += 1

        n = len(self.setOfPoints)
        self.dist = [ [0 for x in range(n)] for x in range(n)]

        for i in range(n):  # precompute distance matrix
            for j in range(n):
                self.dist[i][j] = (euclid(self.setOfPoints[i], self.setOfPoints[j]), j)
                # print("({0:.2f},{1:d})".format(self.dist[i][j][0], self.dist[i][j][1]), end = ' ')
            # print()
            self.dist[i].sort()
            # print(self.dist[i])


    def fit(self):
        self.clusterId = 1
        for p in self.setOfPoints:
            if p.ClId == -1:
                if self.expandCluster(p):
                    self.clusterId += 1

    def expandCluster(self, point):
        seeds = self.regionQuery(point)
        if len(seeds) < self.minPts:
            point.ClId = 0
            return False

        for ele in seeds:
            ele.ClId = self.clusterId

        del seeds[0]    # point itself is contained in its neighbors

        while(len(seeds) != 0):
            currentP = seeds[0]
            result = self.regionQuery(currentP)
            if len(result) >= self.minPts:
                for resultP in result:
                    if resultP.ClId == -1 or resultP.ClId == 0:
                        if resultP.ClId == -1:
                            seeds.append(resultP)
                        resultP.ClId = self.clusterId
            del seeds[0]
        return True

    def regionQuery(self, point):
        seeds = []
        idx = point.idx
        for ele in self.dist[idx]:
            if ele[0] <= self.eps:
                seeds.append(self.setOfPoints[ele[1]])
        return seeds

    def plot(self):
        colors = plt.cm.Spectral(np.linspace(0, 1, self.clusterId + 1))
        for point in self.setOfPoints:
            plt.plot(point.x, point.y, 'o', markerfacecolor = colors[point.ClId], markeredgecolor = 'k', markersize = 14)
        plt.show()

    def showLabels(self):
        for point in self.setOfPoints:
            print(point)




if __name__ == "__main__":
    fname = "data4DBSCAN"
    eps = 0.3
    minPts = 10
    model = DBSCAN(fname, eps, minPts)
    model.fit()
    model.plot()
    model.showLabels()
