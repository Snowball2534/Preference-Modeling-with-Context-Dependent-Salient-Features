import numpy as np
from scipy.special import comb
from sklearn.linear_model import LogisticRegression


class data_generation(object):
    '''
    This class generates the synthetic data and judgement vector.
    num_items: how many items to generate
    dim: the number of features
    '''

    def __init__(self, num_items, dim):
        self.num_items = num_items
        self.dim = dim

        self.U = np.random.normal(0, scale=1 / np.sqrt(self.dim), size=(self.dim, self.num_items))
        self.w = 2*np.random.normal(0, scale=1 / np.sqrt(self.dim), size=(self.dim, 1))


'''
This class tests the Theorem 1.
data_generation: generated data
'''


class synthetic_pairs(object):

    def __init__(self, data_generation):
        self.U = data_generation.U
        self.w = data_generation.w
        self.num_items = data_generation.num_items
        self.dim = data_generation.dim

    def selection_function(self, x, y, t):
        '''
        This function selects the most salient features with top-k variance
        by returning the index not used to compare.
        '''
        mu = (x + y) / 2
        variance = (np.square(x - mu) + np.square(y - mu)) / 2
        not_t = len(x) - t
        not_selected_idx = np.argsort(variance)[:not_t]

        return not_selected_idx

    def salient_feature_preference_model(self, x, y, t):
        '''
        This function computes the probability of x beats y with selection function
        '''
        diff = x - y
        diff[self.selection_function(x, y, t)] = 0

        return 1 / (1 + np.exp(-np.dot(diff, self.w)))

    def get_parameters(self, t, m_true):
        '''
        This function computes all the relevant terms from Theorem 1 in the sample complexity result
        b_star, lamb, eta, zeta, beta, m_1, m_2, and the upper error bound for the estimate
        '''
        b_star = 0
        Z = np.zeros((self.dim, self.dim))
        beta = 0
        variance = np.zeros((self.dim, self.dim))
        zeta = 0

        for i in range(self.num_items):
            for j in range(i + 1, self.num_items):
                x = np.copy(self.U[:, i])
                y = np.copy(self.U[:, j])
                diff = x - y
                diff[self.selection_function(x, y, t)] = 0

                # compute b_star
                temp = abs(np.dot(diff, self.w))
                if temp > b_star:
                    b_star = temp

                # compute beta
                temp = np.linalg.norm(diff, ord=np.inf)
                if temp > beta:
                    beta = temp

                # compute the outer product matrix Z
                diff = diff.reshape(self.dim, 1)
                temp = diff @ diff.T
                Z += temp

        # compute the smallest eigenvalue lamb
        EZ = Z / comb(self.num_items, 2)
        lamb = np.linalg.eigvalsh(EZ)[0]

        for i in range(self.num_items):
            for j in range(i + 1, self.num_items):
                x = np.copy(self.U[:, i])
                y = np.copy(self.U[:, j])
                diff = x - y
                diff[self.selection_function(x, y, t)] = 0
                diff = diff.reshape(self.dim, 1)
                Z = diff @ diff.T
                # compute the variance of Z / step 1: summation
                variance += (Z - EZ)@(Z - EZ)
                # compute the largest eigenvalue eta
                temp = np.linalg.eigvalsh(EZ - Z)[-1]
                if temp > zeta:
                    zeta = temp

        # compute the variance of Z / step 2: average
        variance = variance / comb(self.num_items, 2)
        _, svd_value, _ = np.linalg.svd(variance)
        eta = svd_value[-1]
        delta = 1 / self.dim
        # compute the lower bound of the sample
        m1 = (3 * beta ** 2 * np.log(4 * self.dim / delta) * self.dim + 4 * beta * np.log(
            4 * self.dim / delta) * np.sqrt(self.dim)) / 6
        m2 = 8 * np.log(2 * self.dim / delta) * (6 * eta + lamb * zeta) / (3 * lamb ** 2)
        m_ = max(m1, m2)
        # compute the upper bound of the error
        probability = 1 - delta
        temp = np.sqrt((3 * beta ** 2 * np.log(4 * self.dim / delta) * self.dim + 4 * beta * np.sqrt(self.dim) *
                        np.log(4 * self.dim / delta)) / (6 * m_true))
        upper_bound = (4 * (1 + np.exp(b_star)) ** 2 / (np.exp(b_star) * lamb)) * temp

        print('b* is', b_star[0])
        print('lambda is', lamb)
        print('eta is', eta)
        print('zeta is', zeta)
        print('beta is', beta)
        print('m at least', m1)
        print('m at least', m2)
        print('with probability', probability)
        print('upper bound is', upper_bound[0])

        return np.array([b_star[0], lamb, eta, zeta, beta, m1, m2, probability, upper_bound[0]])

    def inconsistency_analysis(self, t):
        '''
        This function analyzes the systematic inconsistent pairwise comparisons by computing the number of violations of
        strong, moderate, and weak stochastic transitivity and the pairwise inconsistency.
        '''
        # compute the preference matrix at first
        P = np.zeros((self.num_items, self.num_items))
        for i in range(self.num_items):
            for j in range(self.num_items):
                if i == j:
                    continue
                P[i][j] = self.salient_feature_preference_model(self.U[:, i], self.U[:, j], t)

        cardinality_T = 0
        weak_st_violations = 0
        moderate_st_violations = 0
        strong_st_violations = 0
        pairwise_inconsistencies = 0
        for i in range(self.num_items):
            for j in range(self.num_items):
                if i == j:
                    continue
                i_utility = np.dot(self.U[:, i], self.w)
                j_utility = np.dot(self.U[:, j], self.w)
                if i > j and (P[i, j] - .5) * (i_utility - j_utility) < 0:
                    pairwise_inconsistencies += 1
                for k in range(self.num_items):
                    if i == k or j == k:
                        continue
                    if P[i, j] > .5 and P[j, k] > .5:
                        cardinality_T += 1
                        if P[i, k] < .5:
                            weak_st_violations += 1
                        if P[i, k] < min(P[i, j], P[j, k]):
                            moderate_st_violations += 1
                        if P[i, k] < max(P[i, j], P[j, k]):
                            strong_st_violations += 1

        return np.array([weak_st_violations / cardinality_T, moderate_st_violations / cardinality_T,
                         strong_st_violations / cardinality_T, pairwise_inconsistencies / comb(self.num_items, 2)])

    def training_data_generation(self, m, t):
        '''
         This function samples the data as training_data for the synthetic experiments with 100*2^(m-1) samples.
        '''

        num_samples = 100*2**(m-1)
        index = np.random.randint(0, self.num_items, size=(num_samples, 2))
        label = np.zeros((num_samples, 1))
        for i in range(num_samples):
            if index[i,0] == index[i,1]:  #remove the case which picks the same item
                if index[i,0] == self.num_items-1:
                    index[i,0]-=1
                else:
                    index[i,0]+=1
            if self.salient_feature_preference_model(self.U[:, index[i, 0]], self.U[:, index[i, 1]], t) > 0.5:
                label[i] = 1
        return index, label

    def train_model(self, training_data, t):
        '''
         This function trains the model using training data generated by sampling.
        '''
        index, label = training_data
        n = len(label)
        feature = np.zeros((n, self.dim))
        for i in range(n):
            not_selected_idx = self.selection_function(self.U[:, index[i, 0]], self.U[:, index[i, 1]], t)
            diff = self.U[:, index[i, 0]]-self.U[:, index[i, 1]]
            diff[not_selected_idx] = 0
            feature[i] = diff
        clf = LogisticRegression(penalty='none', random_state=0).fit(feature, np.ravel(label))

        return clf

    def estimation_error(self, m, t):
        # compute the empirical error
        data = self.training_data_generation(m, t)
        clf = self.train_model(data, t)
        empirical_error = np.log10(np.linalg.norm((clf.coef_.reshape((self.dim, 1)) -self.w), ord=2))

        # compute the theoretical error
        m_true = 100*2**(m-1)
        theoretical_error = np.log10(self.get_parameters(t, m_true)[-1])
        return np.array([empirical_error, theoretical_error])






