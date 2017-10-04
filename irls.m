

function Beta = irls(X,Y, MinTol, RepLimit)



if(length(Y(1,:)) ~= 1)
    Y = Y';
    if(length(Y(1,:)) ~= 1)
    	print('Y Must be a single column vector.');
        Beta = 0;
        return;
    end
end

if(length(X(:,1)) ~= length(Y))
    X = X';
    if(length(X(:,1)) ~= length(Y))
        print('X and Y Must be the same length.');
        Beta = 0;
        return;
    end
end


% initialize XX
XX = [ones(length(X),1), X];

% initialize Beta
Beta = ones(length(XX(1,:)),1);

% initialize Mu
Mu = 1./(1 + exp(-XX*Beta) );

% initialize SqWeights
SqWeights = eye(length(Y));

for i=1:length(Y)
    SqWeights(i,i) = Mu(i)*(1-Mu(i));
end

for j = 1:RepLimit
    
    % Get NewBeta
    NewBeta = (XX' * SqWeights * XX)^-1 * XX' * (SqWeights * XX * Beta + Y - Mu);
    
    if(max(abs(NewBeta-Beta)) < MinTol)
        Beta = NewBeta;
        sprintf('Converged in %i itterations.',j);
        break;
    end
    
    % Update Beta
    Beta = NewBeta;
    
    % Get new Mu
    Mu = 1./(1 + exp(-XX*Beta) );
    
    % Get new SqWeights
    for i=1:length(Y)
        SqWeights(i,i) = Mu(i)*(1-Mu(i));
    end
end


end
%EOF