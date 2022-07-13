import csv
import math
import random
import sys
import numpy as np
from scipy.optimize import minimize

class GLAD_generate:
    def __init__(self, crowd_file, truth_file, **kwargs):
        self.crowd_file = crowd_file
        self.truth_file = truth_file
        e2wl, w2el, label_set = self.gete2wlandw2el()
        self.e2wl = e2wl
        self.w2el = w2el
        self.workers = self.w2el.keys()
        self.examples = self.e2wl.keys()
        self.label_set = label_set
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    def sigmoid(self, x):
        if (-x)>math.log(sys.float_info.max):
            return 0
        if (-x)<math.log(sys.float_info.min):
            return 1

        return 1/(1+math.exp(-x))

    def logsigmoid(self, x):
        # For large negative x, -log(1 + exp(-x)) = x
        if (-x)>math.log(sys.float_info.max):
            return x
        # For large positive x, -log(1 + exp(-x)) = 0
        if (-x)<math.log(sys.float_info.min):
            return 0

        value = -math.log(1+math.exp(-x))
        #if (math.isinf(value)):
        #    return x

        return value

    def logoneminussigmoid(self, x):
        # For large positive x, -log(1 + exp(x)) = -x
        if (x)>math.log(sys.float_info.max):
            return -x
        # For large negative x, -log(1 + exp(x)) = 0
        if (x)<math.log(sys.float_info.min):
            return 0

        value = -math.log(1+math.exp(x))
        #if (math.isinf(value)):
        #    return -x

        return value

    def kronecker_delta(self, answer, label):
        if answer == label:
            return 1
        else:
            return 0

    def expbeta(self, beta):
        if beta >= math.log(sys.float_info.max):
            return sys.float_info.max
        else:
            return math.exp(beta)

#E step
    def Update_e2lpd(self):  # {example:label}
        self.e2lpd = {}
        for example, worker_label_set in self.e2wl.items():
            lpd = {}
            total_weight = 0

            for tlabel, prob in self.prior.items():
                weight = math.log(prob)
                for (worker, label) in worker_label_set:
                    logsigma = self.logsigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    logoneminussigma = self.logoneminussigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label, tlabel)
                    weight = weight + delta*logsigma + (1-delta)*(logoneminussigma-math.log(len(self.label_set)-1))

                if weight < math.log(sys.float_info.min):
                     lpd[tlabel] = 0
                else:
                    lpd[tlabel] = math.exp(weight)
                total_weight = total_weight + lpd[tlabel]

            for tlabel in lpd:
                if total_weight == 0:
                    lpd[tlabel] = 1.0/len(self.label_set)
                else:
                    lpd[tlabel] = lpd[tlabel]*1.0/total_weight

            self.e2lpd[example] = lpd
#M_step


    def gradientQ(self):

        self.dQalpha={}
        self.dQbeta={}

        for example, worker_label_set in self.e2wl.items():
            dQb = 0
            for (worker, label) in worker_label_set:
                for tlabel in self.prior.keys():
                    sigma = self.sigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label,tlabel)
                    dQb = dQb + self.e2lpd[example][tlabel]*(delta-sigma)*self.alpha[worker]*self.expbeta(self.beta[example])
            self.dQbeta[example] = dQb - (self.beta[example] - self.priorbeta[example])

        for worker, example_label_set in self.w2el.items():
            dQa = 0
            for (example, label) in example_label_set:
                for tlabel in self.prior.keys():
                    sigma = self.sigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label,tlabel)
                    dQa = dQa + self.e2lpd[example][tlabel]*(delta-sigma)*self.expbeta(self.beta[example])
            self.dQalpha[worker] = dQa - (self.alpha[worker] - self.prioralpha[worker])

    def computeQ(self):

        Q = 0
        # the expectation of examples given priors, alpha and beta
        for worker, example_label_set in self.w2el.items():
            for (example, label) in example_label_set:
                logsigma = self.logsigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                logoneminussigma = self.logoneminussigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                for tlabel in self.prior.keys():
                    delta = self.kronecker_delta(label,tlabel)
                    Q = Q + self.e2lpd[example][tlabel]*(delta*logsigma+(1-delta)*(logoneminussigma-math.log(len(self.label_set)-1)))

        # the expectation of the sum of priors over all examples
        for example in self.e2wl.keys():
            for tlabel, prob in self.prior.items():
                Q = Q + self.e2lpd[example][tlabel] * math.log(prob)
        # Gaussian (standard normal) prior for alpha
        for worker in self.w2el.keys():
             Q = Q + math.log((pow(2*math.pi,-0.5)) * math.exp(-pow((self.alpha[worker]-self.prioralpha[worker]),2)/2))
        # Gaussian (standard normal) prior for beta
        for example in self.e2wl.keys():
            Q = Q + math.log((pow(2*math.pi,-0.5)) * math.exp(-pow((self.beta[example]-self.priorbeta[example]),2)/2))
        return Q

    def optimize_f(self, x):
        # unpack x
        i = 0
        for worker in self.workers:
            self.alpha[worker] = x[i]
            i = i + 1
        for example in self.examples:
            self.beta[example] = x[i]
            i = i + 1

        return -self.computeQ() #Flip the sign since we want to minimize

    def optimize_df(self,x):
        # unpack x
        i = 0
        for worker in self.workers:
            self.alpha[worker] = x[i]
            i = i + 1
        for example in self.examples:
            self.beta[example] = x[i]
            i = i + 1

        self.gradientQ()

        # pack x
        der = np.zeros_like(x)
        i = 0
        for worker in self.workers:
            der[i] = -self.dQalpha[worker] #Flip the sign since we want to minimize
            i = i + 1
        for example in self.examples:
            der[i] = -self.dQbeta[example] #Flip the sign since we want to minimize
            i = i + 1

        return der

    def Update_alpha_beta(self):

        x0 = []
        for worker in self.workers:
            x0.append(self.alpha[worker])
        for example in self.examples:
            x0.append(self.beta[example])

        res = minimize(self.optimize_f, x0, method='L-BFGS-B', jac=self.optimize_df,tol=0.01,
              options={'disp': False,'maxiter':25})

        # res = minimize(self.optimize_f, x0, method='CG', jac=self.optimize_df, tol=0.01,
        #        options={'disp': False, 'maxiter': 25})

        self.optimize_f(res.x)

#likelihood
    def computelikelihood(self):
        L = 0

        for example, worker_label_set in self.e2wl.items():
            L_example= 0;
            for tlabel, prob in self.prior.items():
                L_label = prob
                for (worker, label) in worker_label_set:
                    sigma = self.sigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label, tlabel)
                    L_label = L_label * pow(sigma, delta)*pow((1-sigma)/(len(self.label_set)-1),1-delta)
                L_example = L_example +L_label
            L = L + math.log(L_example)

        for worker in self.w2el.keys():
             L = L + math.log((1/pow(2*math.pi,1/2)) * math.exp(-pow((self.alpha[worker]-self.prioralpha[worker]),2)/2))


        for example in self.e2wl.keys():
            L = L + math.log((1/pow(2*math.pi,1/2)) * math.exp(-pow((self.beta[example]-self.priorbeta[example]),2)/2))


#initialization
    def Init_prior(self):
        #uniform probability distribution
        prior = {}
        for label in self.label_set:
            prior[label] = 1.0/len(self.label_set)
        # print(prior)  # {'2': 0.25, '4': 0.25, '1': 0.25, '3': 0.25}
        return prior

    def Init_alpha_beta(self):
        prioralpha={}
        priorbeta={}
        for worker in self.w2el.keys():
            prioralpha[worker] = 1
        for example in self.e2wl.keys():
            priorbeta[example] = 1
        # print(prioralpha)   # {w0:a0, w1:a1, w2:a2, ...}
        # print(priorbeta)    # {e0:b0, e1:b1, e2:b2, ...}
        return prioralpha, priorbeta


    def get_workerquality(self):
        sum_worker = sum(self.alpha.values())
        norm_worker_weight = dict()
        for worker in self.alpha.keys():
            norm_worker_weight[worker] = self.alpha[worker] / sum_worker
        return norm_worker_weight


    def run(self, threshold = 1e-4):

        self.prior = self.Init_prior()
        self.prioralpha, self.priorbeta = self.Init_alpha_beta()
        self.alpha=self.prioralpha
        self.beta=self.priorbeta

        # Q = 0
        self.Update_e2lpd()         # 根据初始化的 alpha, beta, prior 来推断
        Q = self.computeQ()   # 根据推断计算Q
        while True:
            lastQ = Q

            # E-step
            self.Update_e2lpd()
            Q = self.computeQ()
            #print(Q)

            # M-step
            self.Update_alpha_beta()    # 更新 alpha, beta,
            Q = self.computeQ()
            # print(Q)

            # compute the likelihood
            #print self.computelikelihood()

            if (math.fabs((Q-lastQ)/lastQ)) < threshold:
                break

        return self.e2lpd, self.alpha

###################################
# The above is the EM method (a class)
# The following are several external functions
###################################

    def get_accuracy(self):
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth

        tcount = 0
        count = 0
        e2lpd = self.e2lpd
        for e in e2lpd:

            if e not in e2truth:
                continue

            temp = 0
            for label in e2lpd[e]:
                if temp < e2lpd[e][label]:
                    temp = e2lpd[e][label]

            candidate = []

            for label in e2lpd[e]:
                if temp == e2lpd[e][label]:
                    candidate.append(label)

            truth = random.choice(candidate)

            count += 1

            if truth == e2truth[e]:
                tcount += 1

        return tcount*1.0/count

    def generate(self, sample_file, generate_file):

        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth
        f.close()

        f_open = open(sample_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        f_save = open(generate_file, 'w')
        for line in reader:
            example, worker, label = line.split('\t')
            l2p = {}
            for label in self.label_set:
                if label == e2truth[example]:
                    l2p[label] = self.sigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                else:
                    l2p[label] = (1-self.sigmoid(self.alpha[worker]*self.expbeta(self.beta[example])))/(len(self.label_set) - 1)

            p = np.array(list(l2p.values()))
            label = np.random.choice(list(l2p.keys()), p=p.ravel())
            f_save.write(example + '\t' + worker + '\t' + label + '\n')
        f_open.close()
        f_save.close()

    def generate_fixed_annotator(self, exist_annotator, generate_file):
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth
        f.close()


        f_open = open(self.crowd_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        f_save = open(generate_file, 'w')
        for line in reader:
            example, worker, label = line.split('\t')
            if worker in exist_annotator:
                f_save.write(example + '\t' + worker + '\t' + label + '\n')
            else:
                l2p = {}
                for label in self.label_set:
                    if label == e2truth[example]:
                        l2p[label] = self.sigmoid(self.alpha[worker] * self.expbeta(self.beta[example]))
                    else:
                        l2p[label] = (1 - self.sigmoid(self.alpha[worker] * self.expbeta(self.beta[example]))) / (
                                    len(self.label_set) - 1)

                p = np.array(list(l2p.values()))
                label = np.random.choice(list(l2p.keys()), p=p.ravel())
                f_save.write(example + '\t' + worker + '\t' + label + '\n')
        f_open.close()
        f_save.close()
    def generate_fixed_task(self, exist_task, generate_file):
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth
        f.close()


        f_open = open(self.crowd_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        f_save = open(generate_file, 'w')
        for line in reader:
            example, worker, label = line.split('\t')
            if example in exist_task:
                f_save.write(example + '\t' + worker + '\t' + label + '\n')
            else:
                l2p = {}
                for label in self.label_set:
                    if label == e2truth[example]:
                        l2p[label] = self.sigmoid(self.alpha[worker] * self.expbeta(self.beta[example]))
                    else:
                        l2p[label] = (1 - self.sigmoid(self.alpha[worker] * self.expbeta(self.beta[example]))) / (
                                    len(self.label_set) - 1)

                p = np.array(list(l2p.values()))
                label = np.random.choice(list(l2p.keys()), p=p.ravel())
                f_save.write(example + '\t' + worker + '\t' + label + '\n')
        f_open.close()
        f_save.close()
    def gete2wlandw2el(self):
        e2wl = {}
        w2el = {}
        label_set = []

        f = open(self.crowd_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, worker, label = line.split('\t')
            if example not in e2wl:
                e2wl[example] = []
            e2wl[example].append([worker,label])

            if worker not in w2el:
                w2el[worker] = []
            w2el[worker].append([example,label])

            if label not in label_set:
                label_set.append(label)

        return e2wl, w2el, label_set




