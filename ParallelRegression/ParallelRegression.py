import sys
import argparse
import numpy as np
from operator import add
from time import time
from pyspark import SparkContext


def readData(input_file,spark_context):
    """  Read data from an input file and return rdd containing pairs of the form:
                         (x,y)
         where x is a numpy array and y is a real value. The input file should be a 
         'comma separated values' (csv) file: each line of the file should contain x
         followed by y. For example, line:

         1.0,2.1,3.1,4.5

         should be converted to tuple:
        
         (array(1.0,2.1,3.1),4.5)
         

    """
    return spark_context.textFile(input_file)\
                .map(lambda line:line.split(','))\
                .map(lambda words:(words[:-1],words[-1]))\
                .map(lambda (features,target): (np.array([ float(x) for x in features]),float(target)))



def readBeta(input):
    """ Read a vector �� from CSV file input
    """
    with open(input,'r') as fh:
        str_list = fh.read()\
                   .strip()\
                   .split(',')
        return np.array( [float(val) for val in str_list] )

def writeBeta(output,beta):
    """ Write a vector �� to a CSV file ouptut
    """
    with open(output,'w') as fh:
        fh.write(','.join(map(str, beta.tolist()))+'\n')
def estimateGrad(fun,x,delta):
     """ Given a real-valued function fun, estimate its gradient numerically.
     """
     d = len(x)
     grad = np.zeros(d)
     for i in range(d):
         e = np.zeros(d)
         e[i] = 1.0
         grad[i] = (fun(x+delta*e) - fun(x))/delta
     return grad


def lineSearch(fun,x,grad,a=0.2,b=0.6):
    """ Given function fun, a current argument x, and gradient grad, 
        perform backtracking line search to find the next point to move to.
        (see Boyd and Vandenberghe, page 464).

        Parameters a,b  are the parameters of the line search.

        Given function fun, and current argument x, and gradient  ?fun(x), the function finds a t such that
        fun(x - t * grad) <= fun(x) - a t <?fun(x),?fun(x)>

        The return value is the resulting value of t.
    """
    t = 1.0
    while fun(x-t*grad) > fun(x)- a * t *np.dot(grad,grad):
        t = b * t
    return t

def predict(x,beta):
    """ Given vector x containing features and parameter vector ��, 
        return the predicted value: 

                        y = <x,��>   

    """
    return np.dot(x,beta)
    pass

def f(x,y,beta):
    """ Given vector x containing features, true label y, 
        and parameter vector ��, return the square error:

                 f(��;x,y) =  (y - <x,��>)^2      

    """
    return np.power(y-np.dot(x,beta), 2)
    pass






def localGradient(x,y,beta):
    """ Given vector x containing features, true label y, 
        and parameter vector ��, return the gradient ?f of f:

                ?f(��;x,y) =  -2 * (y - <x,��>) * x       

        with respect to parameter vector ��.

        The return value is  ?f.
    """
    return -2*(y-np.dot(x,beta))*x
    pass


def F(data,beta,lam = 0):
    """  Compute the regularized mean square error:

             F(��) = 1/n ��_{(x,y) in data}    f(��;x,y)  + �� ||�� ||_2^2   
                  = 1/n ��_{(x,y) in data} (y- <x,��>)^2 + �� ||�� ||_2^2 

         where n is the number of (x,y) pairs in RDD data. 

         Inputs are:
            - data: an RDD containing pairs of the form (x,y)
            - beta: vector ��
            - lam:  the regularization parameter ��
           

         The return value is F(��).

    """
    n = data.count()
    return data.map(lambda (x,y): f(x,y,beta)).reduce(lambda x,y: x+y)/n + lam*np.sum(beta**2)
    pass
def gradient(data,beta,lam = 0):
    """ Compute the gradient  ?F of the regularized mean square error 
                F(��) = 1/n ��_{(x,y) in data} f(��;x,y) + �� ||�� ||_2^2   
                     = 1/n ��_{(x,y) in data} (y- <x,��>)^2 + �� ||�� ||_2^2   
 
        where n is the number of (x,y) pairs in data. 

        Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector ��
             - lam:  the regularization parameter ��
             

        The return value is an array containing ?F.

    """
    n = data.count()
    return data.map(lambda (x, y): localGradient(x,y,beta)).reduce(lambda x, y: x+y)/n + 2*lam*beta
    pass

def test(data,beta):
    """ Compute the mean square error  

                 MSE(��) =  1/n ��_{(x,y) in data} (y- <x,��>)^2

        of parameter vector �� over the dataset contained in RDD data, where n is the size of RDD data.
        
        Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector ��

        The return value is MSE(��).  
       
    """
    return F(data, beta)
    pass
def train(data,beta_0, lam,max_iter,eps):
    """ Perform gradient descent:


        to  minimize F given by
  
             F(��) = 1/n ��_{(x,y) in data} f(��;x,y) + �� ||�� ||_2^2   

        where
             - data: an rdd containing pairs of the form (x,y)
             - beta_0: the starting vector ��
             - lam:  is the regularization parameter ��
             - max_iter: maximum number of iterations of gradient descent
             - eps: upper bound on the l2 norm of the gradient
             - a,b: parameters used in backtracking line search


        The function performs gradient descent with a gain found through backtracking
        line search. That is it computes

                   
                   ��_k+1 = ��_k - ��_k ?F(��_k) 
                
        where the gain ��_k is given by
        
                  ��_k = lineSearch(F,��_��,?F(��_k))

        and terminates after max_iter iterations or when ||?F(��_k)||_2<��.   

        The function returns:
             -beta: the trained ��, 
             -gradNorm: the norm of the gradient at the trained ��, and
             -k: the number of iterations performed
    """
    beta = beta_0
    for k in range(1, max_iter):
        grad = gradient(data, beta, lam)
        grad_norm=np.linalg.norm(grad)
        print 'Iteration', k
        print 'time elapsed', time()
        print 'F(��_',k,')', F(data, beta, lam)
        print '||?F(��_',k,')||\n', grad_norm
        if grad_norm < eps: break
        gama = lineSearch(lambda x: F(data, x, lam), beta, grad)
        beta = beta - gama * grad
    return beta, grad_norm, k
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Ridge Regression.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--traindata',default=None, help='Input file containing (x,y) pairs, used to train a linear model')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a linear model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter ��')
    parser.add_argument('--max_iter', type=int,default=100, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.01, help='��-tolerance. If the l2_norm gradient is smaller than ��, gradient descent terminates.')
    parser.add_argument('--N',type=int,default=2,help='Level of parallelism')


    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    sc = SparkContext(appName='Parallel Ridge Regression')

    if not args.verbose :
        sc.setLogLevel("ERROR")

    beta = None

    if args.traindata is not None:
        # Train a linear model �� from data with regularization parameter ��, and store it in beta
        print 'Reading training data from',args.traindata
        data = readData(args.traindata,sc)
        data = data.repartition(args.N).cache()

        x,y = data.take(1)[0]
        beta0 = np.zeros(len(x))

        print 'Training on data from',args.traindata,'with �� =',args.lam,', �� =',args.eps,', max iter = ',args.max_iter
        beta, gradNorm, k = train(data,beta_0=beta0,lam=args.lam,max_iter=args.max_iter,eps=args.eps)
        print 'Algorithm ran for',k,'iterations. Converged:',gradNorm<args.eps
        print 'Saving trained �� in',args.beta
        writeBeta(args.beta,beta)


    if args.testdata is not None:
        # Read beta from args.beta, and evaluate its MSE over data
        print 'Reading test data from',args.testdata
        data = readData(args.testdata,sc)
        data = data.repartition(args.N).cache()
         x,y = data.take(1)[0]
        beta0 = np.zeros(len(x))

        print 'Training on data from',args.traindata,'with �� =',args.lam,', �� =',args.eps,', max iter = ',args.max_iter
        beta, gradNorm, k = train(data,beta_0=beta0,lam=args.lam,max_iter=args.max_iter,eps=args.eps)
        print 'Algorithm ran for',k,'iterations. Converged:',gradNorm<args.eps
        print 'Saving trained �� in',args.beta
        writeBeta(args.beta,beta)


    if args.testdata is not None:
        # Read beta from args.beta, and evaluate its MSE over data
        print 'Reading test data from',args.testdata
        data = readData(args.testdata,sc)
        data = data.repartition(args.N).cache()

        print 'Reading beta from',args.beta
        beta = readBeta(args.beta)

        print 'Computing MSE on data',args.testdata
        MSE = test(data,beta)
        print 'MSE is:', MSE

                                                                               277,8         96%





