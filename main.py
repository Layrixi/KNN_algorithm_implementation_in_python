import math
import operator

import pandas as pd
import numpy as np
import scipy
import sklearn
from scipy.spatial import KDTree
from sklearn import datasets
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection

import matplotlib.pyplot as plt

def euclidean_distance(point1,point2):
    return np.sqrt(np.sum((point1-point2)**2))

class KNN_regression:
    n_neighbors=1
    use_KDTree=False
    data=None
    targets=None
    kd_tree=None


    def __init__(self, n_neighbors=1, use_KDTree=False):
        self.n_neighbors = n_neighbors
        if n_neighbors%2 == 0:
            self.n_neighbors-=1 #in case n_neighbors is not odd
        self.use_KDTree = use_KDTree


    def fit(self,X,Y):
        #X - values, Y - results
        self.data = np.array(X)
        self.targets = np.array(Y)
        if self.use_KDTree:
            self.kd_tree = KDTree(self.data)

    def predict(self,X):
        #calculate euclidean distance between each point
        #sort points by distances ascending
        #nearest neighbours N from sorted arrays
        targets_array = np.array([])

        if self.use_KDTree:
            for point in X:
                _, closest_neighbours_indices = self.kd_tree.query([point], self.n_neighbors)
                mean_value = np.mean(self.targets[closest_neighbours_indices])  # calculate mean of the target values of the nearest neighbors
                targets_array = np.append(targets_array, mean_value)
        else:
            for point in X:
                euclidean_array = np.array([euclidean_distance(point, point2) for point2 in self.data])
                sorted_indices = euclidean_array.argsort()
                closest_neighbours_indices = sorted_indices[:self.n_neighbors]
                mean_value = np.mean(self.targets[closest_neighbours_indices]) # calculate mean of the target values of the nearest neighbors
                targets_array = np.append(targets_array,mean_value)
        return targets_array

    def score(self,X,Y):
        return metrics.mean_squared_error(Y,X)     #return accuracy

class KNN:
    n_neighbors=1
    use_KDTree=False
    data=None
    targets=None
    kd_tree = None

    def __init__(self, n_neighbors=1, use_KDTree=False):
        self.n_neighbors = n_neighbors
        if n_neighbors%2 == 0:
            self.n_neighbors-=1 #in case n_neighbors is not odd
        self.use_KDTree = use_KDTree

    def fit(self,X,Y):
        #X - values, Y - results
        self.data = np.array(X)
        self.targets = np.array(Y)
        if self.use_KDTree:
            self.kd_tree = KDTree(self.data)

    def predict(self,X):
        #calculate euclidean distance between each point
        #sort points by distances ascending
        #nearest neighbours N from sorted arrays
        targets_array = np.array([])
        if self.use_KDTree:
            for point in X:
                _, closest_neighbours_indices = self.kd_tree.query([point], self.n_neighbors)
                most_common = scipy.stats.mode(self.targets[closest_neighbours_indices])[0]
                targets_array = np.append(targets_array, most_common)
        else:
            for point in X:
                euclidean_array = np.array([euclidean_distance(point, point2) for point2 in self.data])
                sorted_indices = euclidean_array.argsort()
                closest_neighbours_indices = sorted_indices[:self.n_neighbors]
                most_common = scipy.stats.mode(self.targets[closest_neighbours_indices])[0] #0 is the value that is most common, mode is used to get the most common value

                targets_array = np.append(targets_array,most_common)
        return targets_array

    def score(self,X,Y):
        return metrics.accuracy_score(Y,X) *100     #return accuracy

if __name__ == '__main__':
    
    #knn regression with own knn
    X, Y = datasets.make_regression(n_samples=100, n_features=2, noise=0.1)
    #print("Features (X): \n", X)
    #print("Targets (y): \n", y)

    # split the data for testing and training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # standardize the test and training data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # KNN
    k = int(math.sqrt(len(Y_test)))
    if k % 2 == 0:
        k -= 1

    #regression
    knn_regression = KNN_regression(n_neighbors=k)
    knn_regression.fit(X_train,Y_train)
    Y_pred = knn_regression.predict(X_test)
    score = knn_regression.score(Y_test,Y_pred)
    print(score)
    # Plot the training data points
    plt.scatter(X_train[:, 0], Y_train, label="Training Data", color="blue")

    # Plot the test data points and their predictions
    plt.scatter(X_test[:, 0], Y_pred, label="Test Predictions", color="red")

    #plot real data points
    plt.scatter(X_test[:, 0], Y_test, label="Test True", color="green")

    #plot regression line
    pca = PCA(n_components=1)

    X1d_test = pca.fit(X_test).transform(X_test)
    X1d_test=X1d_test.flatten()
    z = np.polyfit(X1d_test, Y_pred, 1)
    p = np.poly1d(z)
    plt.plot(X1d_test, p(X1d_test), "b--")

    X1d_real = pca.fit(X_test).transform(X_test)
    z = np.polyfit(X1d_test, Y_test, 1)
    p = np.poly1d(z)
    plt.plot(X1d_real, p(X1d_real), "r--")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.legend()
    plt.show()


    #KNN for random points with sklearn KNN
    X,Y = datasets.make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=3)
    #print(X)       #values
    #print(Y)       #results/target

    # split the data for testing and training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # standardize the test and training data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # KNN
    k = int(math.sqrt(len(Y_test)))
    if k % 2 == 0:
        k -= 1
    # classify
    skKNN = KNeighborsClassifier(n_neighbors=k, metric='euclidean', p=2)
    skKNN.fit(X_train, Y_train)

    # results
    Y_pred = skKNN.predict(X_test)
    conf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
    print(metrics.accuracy_score(Y_test, Y_pred))

    #plot
    plt.subplot(1,2,1)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = skKNN.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k')
    plt.title('Contour Plot with sklearn KNN')
    plt.xlabel('x')
    plt.ylabel('y')




    #own KNN for random points

    X, Y = datasets.make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=3)
    # print(X)       #values
    # print(Y)       #results/target

    # split the data for testing and training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # standarize the test and training data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    # KNN

    k = int(math.sqrt(len(Y_test)))
    if k % 2 == 0:
        k -= 1
    knn_randompoints = KNN(n_neighbors=k)
    knn_randompoints.fit(X_train,Y_train)
    Y_pred = knn_randompoints.predict(X_test)
    score = knn_randompoints.score(Y_test,Y_pred)

    #plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = knn_randompoints.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(1,2,2)
    plt.contourf(xx, yy, Z, alpha=0.4)

    #plot points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k')

    plt.title('Contour Plot with own KNN')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()



    #own KNN for iris

    iris = datasets.load_iris()
    irisdf = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
    print(irisdf.columns)
    X = irisdf.iloc[:,0:irisdf.columns.size - 1]  # values w/o the last column, since it's the results, 0:size in reality takes size-1
    Y = irisdf.iloc[:, irisdf.columns.size - 1]  # results

    # split the data for testing and training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

    # standarize the test and training data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # KNN

    k = int(math.sqrt(len(Y_test)))
    if k % 2 == 0:
        k -= 1

    #classify
    knn_iris = KNN(n_neighbors=k)
    knn_iris.fit(X_train, Y_train)

    #predict and score for comparing purpose
    Y_pred = knn_iris.predict(X_test)
    score = knn_iris.score(Y_test, Y_pred)
    print(score)



    # own KNN on PCA reduced iris with plot

    pca = PCA(n_components=2)
    iris2d = pca.fit(X).transform(X)

    # KNN for 2d iris
    X = iris2d[:, :]  # values w/o the last column, since it's the results, 0:size in reality takes size-1
    Y = irisdf.iloc[:, irisdf.columns.size - 1]  # results

    # split the data for testing and training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # standarize the test and training data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # KNN
    k = int(math.sqrt(len(Y_test)))
    if k % 2 == 0:
        k -= 1

    # classify
    KNN_iris = KNN(n_neighbors=k)
    KNN_iris.fit(X_train, Y_train)

    # results
    Y_pred = KNN_iris.predict(X_test)
    print(metrics.accuracy_score(Y_test, Y_pred))

    # meshgrid

    KNN_iris.fit(iris2d, Y)

    x_min, x_max = iris2d[:, 0].min() - 1, iris2d[:, 0].max() + 1
    y_min, y_max = iris2d[:, 1].min() - 1, iris2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, len(iris2d[:, 0])),
                         np.linspace(y_min, y_max, len(iris2d[:, 1])))
    Z = KNN_iris.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(iris2d[:, 0], iris2d[:, 1], c=Y, edgecolors='k')
    plt.title('PCA Reduced Iris with own KNN')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    #pca
    pca = PCA(n_components=2)
    iris2d = pca.fit(X).transform(X)


    # sklearn KNN for iris

    iris = datasets.load_iris()
    irisdf = pd.DataFrame(data=np.c_[iris['data'], iris['target']],columns=iris['feature_names'] + ['target'])
    print(irisdf.columns)
    X = irisdf.iloc[:,0:irisdf.columns.size-1]  #values w/o the last column, since it's the results, 0:size in reality takes size-1
    Y = irisdf.iloc[:,irisdf.columns.size-1]    #results

    #split the data for testing and training
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5,random_state=0)

    #standarize the test and training data
    sc_X = StandardScaler()
    X_train  = sc_X.fit_transform(X_train)
    X_test  = sc_X.transform(X_test)

    #KNN
    k = int(math.sqrt(len(Y_test)))
    if k%2 ==0:
        k-=1

    #classify
    skKNN = KNeighborsClassifier(n_neighbors=k,metric='euclidean',p=2)
    skKNN.fit(X_train,Y_train)

    #results
    Y_pred = skKNN.predict(X_test)
    print(metrics.accuracy_score(Y_test,Y_pred))




    # sklearn KNN on PCA reduced iris with plot
    # pca
    pca = PCA(n_components=2)
    iris2d = pca.fit(X).transform(X)

    # KNN for 2d iris
    X = iris2d[:,:]  # values w/o the last column, since it's the results, 0:size in reality takes size-1
    Y = irisdf.iloc[:,irisdf.columns.size-1]    #results

    # split the data for testing and training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # standarize the test and training data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # KNN
    k = int(math.sqrt(len(Y_test)))
    if k % 2 == 0:
        k -= 1

    # classify
    skKNN = KNeighborsClassifier(n_neighbors=k, metric='euclidean', p=2)
    skKNN.fit(X_train, Y_train)

    # results
    Y_pred = skKNN.predict(X_test)
    print(metrics.accuracy_score(Y_test, Y_pred))

    skKNN.fit(iris2d,Y)

    # meshgrid
    x_min, x_max = iris2d[:, 0].min() - 1, iris2d[:, 0].max() + 1
    y_min, y_max = iris2d[:, 1].min() - 1, iris2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, len(iris2d[:,0])),
                         np.linspace(y_min, y_max, len(iris2d[:,1])))
    Z = skKNN.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(iris2d[:, 0], iris2d[:, 1], c=Y, edgecolors='k')
    plt.title('PCA Reduced Iris with sklearn KNN')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    """
    #own KNN for iris with LOO cross-validation
    """
    iris = datasets.load_iris()
    irisdf = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
    print(irisdf.columns)
    X = irisdf.iloc[:,0:irisdf.columns.size - 1]  # values w/o the last column, since it's the results, 0:size in reality takes size-1
    Y = irisdf.iloc[:, irisdf.columns.size - 1]  # results

    # split the data for testing and training
    loo = model_selection.LeaveOneOut()
    X = X.to_numpy()
    Y = Y.to_numpy()
    scores = {k: [] for k in range(1, len(Y))}

    for k in range(1, len(Y)):
        knn_iris = KNN(n_neighbors=k)
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            knn_iris.fit(X_train, Y_train)
            Y_pred = knn_iris.predict(X_test)
            score = knn_iris.score(Y_test, Y_pred)
            scores[k].append(score)  # Append the score to the list of scores for the current k

    for k, score_list in scores.items():
        mean_score = np.mean(score_list)
        print(f"k={k}, mean score={mean_score}")


    # knn regression with own knn for boston
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    bostondf = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([bostondf.values[::2, :], bostondf.values[1::2, :2]])
    target = bostondf.values[1::2, 2]
    bostondf = bostondf.fillna(bostondf.mean())
    #print(bostondf)
    X = bostondf.iloc[:,0:bostondf.columns.size - 1]  # values w/o the last column, since it's the results, 0:size in reality takes size-1
    Y = bostondf.iloc[:, bostondf.columns.size - 1]

    # split the data for testing and training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # standardize the test and training data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    cv10 = model_selection.KFold(n_splits=10)

    # KNN
    k = int(math.sqrt(len(Y_test)))
    if k % 2 == 0:
        k -= 1

    X = X.to_numpy()
    Y = Y.to_numpy()
    scores = {k: [] for k in range(1, len(Y))}

    for k in range(1, 500):
        knn_iris = KNN_regression(n_neighbors=k,use_KDTree=True)
        for train_index, test_index in cv10.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            knn_iris.fit(X_train, Y_train)
            Y_pred = knn_iris.predict(X_test)
            score = knn_iris.score(Y_test, Y_pred)
            scores[k].append(score)  # Append the score to the list of scores for the current k
        print(np.mean(scores[k]))
    for k, score_list in scores.items():
        mean_score = np.mean(score_list)
        print(f"k={k}, mean score={mean_score}")

