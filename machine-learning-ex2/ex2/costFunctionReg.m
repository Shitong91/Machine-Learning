function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
J1=0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J1=(-y'*log(sigmoid(X*theta))-(ones(m,1)-y)'*log(ones(m,1)-sigmoid(X*theta)))./m;
J=lambda*sum(theta(2:end).^2)/(2*m)+J1;
grad(1,1)=((sigmoid(X*theta)-y)'*X(:,1))'/m;
grad(2:end,1)=((sigmoid(X*theta)-y)'*X(:,2:end))'./m+lambda*theta(2:end,:)/m;




% =============================================================

end
