import numpy

def calculate_distance(a, b):
    '''
    calculate the Euclidean distance between a and b
    :param a: ndarray shape of (1, n)
    :param b: ndarrya shape of (1, n)
    :return: dist, float
    '''


def find_k_cluster(data, k):
    '''
    Find k clusters in data
    :param data: ndarray shape of (m, n), m: number of samples, n: number of data dimension
    :param k: int, number of cluster to find
    :return:
    centroids: ndarray shape of (k, n)
    error: float, sum of square errors
    '''
    m, n = data.shape
    # initialize k centroids within the sample space
    centroids = np.zeros((k, n))
    for j in range(n):
        centroids[:,j] = np.random.randint(data[:,j].min(), data[:,j].max(), size = k)

    # training on the data until the centroids are not moving
    prev = np.ones((k, n)) * float('inf')
    while centroids != prev:
        new_centroids = np.zeros((k, n))
        cnt = np.zeros(k)
        errors = np.zeros(k)
        for i in range(m):
            dist = float('inf')
            centroid = 0
            for j in range(k):
                temp = np.linalg.norm(centroids[j,:], data[i,:])
                if temp < dist:
                    dist = temp
                    centroid = j
            errors[centroid] += dist
            new_centroids[centroid,:] += data[i,:]
            cnt[centroid] += 1

        new_centroids = np.divide(new_centroids, cnt, out=np.zeros_like(new_centroids), where=cnt!=0)
        prev, centroids = centroids, new_centroids

    return centroids, np.sum(errors)