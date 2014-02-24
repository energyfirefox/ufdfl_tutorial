function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                      
         
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;

W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
% feedforwarding:
%X = patches(:,1:15);
X = data;
m = size(X)(2);

a1 = X;
z2 = W1 * a1 + repmat(b1,1,m);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2,1,m);
H  = z3; 
a3 = H;
% backpropagation part


delta3 = H - X;	

rho_hat = sum(a2,2)/ m;
sparsity_grad = beta.*(- sparsityParam ./ rho_hat + (1 - sparsityParam) ./ (1 - rho_hat));

KL1 = sparsityParam*log(sparsityParam./rho_hat);
KL2 = (1-sparsityParam)*log((1-sparsityParam)./(1-rho_hat));
KL_divergence = sum(KL1 + KL2);

err2 = W2' * delta3 + sparsity_grad*ones(1, m);

%err2 = W2' * delta3;
delta2 = err2.*(a2.*(1-a2));


W2grad = delta3*a2';
b2grad = delta3*ones(m, 1);

W1grad = delta2*a1';
b1grad = delta2*ones(m, 1);

W1grad = W1grad/m + lambda*W1;
W2grad = W2grad/m + lambda*W2;
b1grad = b1grad/m;
b2grad = b2grad/m;

J = 0.5*sum(sum((H-X).^2));
cost = (1/m)*J + lambda*0.5*(sum(sum(W1.^2))+sum(sum(W2.^2))) + beta*KL_divergence;
%cost = (1/m)*J+lambda*0.5*(sum(sum(W1.^2))+sum(sum(W2.^2)));

%cost = sum(sum(-X.*log(H)-(1-X).*log(1-H)))/m ;
%cost = cost + lambda/2/m*(sum(sum(W1.^2))+sum(sum(W2.^2)));

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.


err = sum(sum((H - X).^2));
disp(err);
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
		 


