import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import math
import numpy as np
from numpy.random import rand
from numpy import sqrt
from time import time


def find_distances(current_point, data_points, eps):

    neighbors = []

    # find the euclidean distance between each point
    # this is less time efficient, but works with data with more than just two indices
    for pt in data_points:
        sum_pts = 0.0
        for dim in range(0, len(current_point)):

            diff_sq = (current_point[dim] - pt[dim])**2
            sum_pts += diff_sq

        eucl_dist = sqrt(sum_pts)

        if eucl_dist < eps:
            neighbors.append(pt)

    return neighbors


def expand_cluster(current_point, neighbors, clusters, cluster_num, eps, min_pts, data_points, visited_pts):

    # make a new cluster with a new point
    clusters[cluster_num].append(current_point)

    # get density-reachable points from the selected core point's neighbors
    for pt in neighbors:
        if pt not in visited_pts:
            visited_pts.append(pt)
            new_neighbor_pts = find_distances(pt, data_points, eps)

            # if there are a significant amount of neighbors, append to the already existing neighbors
            # if they are not already present
            if len(new_neighbor_pts) >= min_pts:
                for n in new_neighbor_pts:
                    if n not in neighbors:
                        neighbors.append(n)

        # add points to specific cluster
        for c in clusters:
            if pt not in c:
                if pt not in clusters[cluster_num]:
                    clusters[cluster_num].append(pt)

        # starting the first cluster
        if len(clusters) == 0:
            if pt not in clusters[cluster_num]:
                clusters[cluster_num].append(pt)


def dbscan(data_points, eps, min_pts):

    # start timer and initialize variables
    start = time()
    noise_pts = []
    visited_pts = []
    clusters = []

    # start with -1 in case there are no clusters
    cluster_num = -1

    # append pt to visited, find immediate neighbors
    for pt in data_points:
        if pt not in visited_pts:
            visited_pts.append(pt)
            neighbors = find_distances(pt, data_points, eps)
            # print(neighbors)

            # if there are a significant amount of immediate neighbors, start a new cluster
            if len(neighbors) < min_pts:
                noise_pts.append(pt)
            else:
                clusters.append([])
                cluster_num += 1
                expand_cluster(pt, neighbors, clusters, cluster_num, eps, min_pts, data_points, visited_pts)

    # end the timer when algorithm is done with classifications
    end = time()

    # plot the clusters found by the algorithm
    for c in clusters:
        col = [rand(1), rand(1), rand(1)]
        plt.scatter([i[0] for i in c], [i[1] for i in c], color=col)

    plt.show()

    # print results
    print("Number of clusters found: ", len(clusters))
    print("Time taken: ", end-start, "s")


def main():

    condition = True
    while (condition):
        mode = input("Enter sm, md, or lg: ")
        if mode == "sm" or mode == "md" or mode == "lg":
            condition = False
        else:
            print("Please enter only 'sm', 'md', or 'lg'.")

    # small = iris dataset
    if mode == "sm":

        file = open("iris.txt")

        data = []

        for line in file:
            individual_pt = []
            line = line.split(",")
            for idx in line:
                try:
                    individual_pt.append(float(idx))
                except ValueError:
                    continue

            data.append(individual_pt)

        file.close()

        dbscan(data, 1.5, 10)

    # medium dataset has 500 points
    elif mode == "md":

        steps = 500
        h = 0
        k = 0
        r = 1
        labels = np.zeros((steps))
        Y = np.zeros((steps, 2))
        X = np.zeros((steps, 2))
        counter = 0
        theta = np.arange(0, 2 * 3.14, 2 * 3.14 / 250)

        for i in range(2):
            for t in theta:
                X_ = h + r * math.cos(t) + np.random.randn()
                Y_ = k - r * math.sin(t) + np.random.randn()
                X[counter] = np.array([X_, Y_])
                labels[counter] = i
                Y[counter, i] = 1
                counter += 1
            r += 10

        # plot the original clusters
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap=plt.cm.Spectral)
        plt.show()

        data_mat = []
        for i in range(0,len(X)-1):
            new_pt = []
            new_pt.append(X[i][0])
            new_pt.append(X[i][1])
            data_mat.append(new_pt)

        dbscan(data_mat, 5, 20)

    # large has 1000 data points, runs slow
    elif mode == "lg":

        centers = [[3, 3], [9, 9], [1, 12]]
        X,y = make_blobs(n_samples=1000, n_features=2, centers=centers, cluster_std=.5, center_box=(1, 10.0), shuffle=True, random_state=0)

        # plot the original clusters
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()

        data_mat = []
        for i in range(0, len(X) - 1):
            new_pt = []
            new_pt.append(X[i][0])
            new_pt.append(X[i][1])
            data_mat.append(new_pt)

        dbscan(data_mat, 5, 50)

# run main method
main()
