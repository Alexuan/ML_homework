function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values





m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X*theta;
theta_1 = theta;
theta_1(1) = 0;
jval = 0;
jval = -y'*log(sigmoid(z))-(1-y)'*(log(1-sigmoid(z)));
theta_qs = theta_1'*theta_1;
J = 1/m*jval + lambda/(2*m)*theta_qs;
grad = 1/m*(sigmoid(z)-y)'*X+lambda/m*theta_1';

% =============================================================

end
