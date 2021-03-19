import numpy as np
import matplotlib.pyplot as pyplot

class LogisticRegression():

    def intialize_weights_and_bias(self, dimension):
        w = np.full((dimension,1),0.01)
        b = 0.0
        return w, b

    def sigmoid(self, z):
        y_1 = 1/(1+np.exp(-z))
        return y_1
    
    def forward_backward_propagation(self,w,b,x_train,y_train):
        epsilon = 1e-5  
        z = np.dot(w.T,x_train) + b
        y_1 = self.sigmoid(z)
        loss = -y_train*np.log(y_1 + epsilon)-(1-y_train)*np.log(1-y_1 + epsilon)
        cost = (np.sum(loss))/x_train.shape[1]
        derivative_weight = (np.dot(x_train,((y_1-y_train).T)))/x_train.shape[1]
        derivative_bias = np.sum(y_1-y_train)/x_train.shape[1]
        return cost, derivative_weight, derivative_bias

    def update(self, w, b, x_train, y_train, learning_rate, no_iteration):
        cost_list = []
        index = []

        for i in range(no_iteration):

            cost, weight_gradient, bias_gradient = self.forward_backward_propagation(w, b, x_train, y_train)
            w = w - learning_rate * weight_gradient
            b = b - learning_rate * bias_gradient

            if i%20 == 0:
                cost_list.append(cost)
                index.append(i)
                print("cost after iteration %i: %f" %(i, cost))

        plt.plot(index, cost_list)
        plt.show()
        return w, b, weight_gradient, bias_gradient, cost_list

    def predict(self, w, b, x_test):

        z = sigmoid(np.dot(w.T, x_test) + b)
        Y_prediction = np.zeros((1,x_test.shape[1]))

        for i in range(z.shape[1]):
            if(z[0,i]<=0.5):
                Y_prediction[0,i] = 0
            else:
                Y_prediction[0,i] = 1
        
        return Y_prediction

    def logistic_regression(self, x_train, y_train, x_test, y_test, learning_rate, no_iteration):

        dimension = x_train.shape[0]
        w, b = self.intialize_weights_and_bias(dimension)
        w, b, weight_gradient, bias_gradient, cost_list = self.update(w, b, x_train, y_train, learning_rate,no_iteration)
        Y_prediction_test = self.predict(w,b,x_test)
        Y_prediction_train = self.predict(x,b,x_train)

        print("Test Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100,2)))
        print("Train Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100,2)))
