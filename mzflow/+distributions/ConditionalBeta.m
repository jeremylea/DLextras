classdef ConditionalBeta < distributions.LatentDist
    % ConditionalBeta A learanable multi-variate beta based on conditions
    %   This is based on the same ideas as the splines, but simpler.
    properties
        input_dim     % data dimension
        condition_dim % condition dimension
        info          % simple description string
    end

    properties (Learnable)
        nnet % for this we use a full network to predict A & B.
    end

    methods
        function this = ConditionalBeta(input_dim,condition_dim)
            % Constructor which just needs the data dimensions
            this.input_dim = input_dim;
            this.condition_dim = condition_dim;
            this.info = {"Conditional Beta" input_dim};
            this.nnet = this.DenseReluNetwork(condition_dim,2*input_dim);
            this.InputNames = ["in","cond"];
        end

        function log_prob = log_prob(this, inputs, conditions)
            % this needs two inputs, and follows the same logic as the
            % splines.
            mask = any(inputs <= 0 | inputs >= 1,finddim(inputs,"C"));
            outputs = forward(this.nnet,conditions);
            outputs = reshape(outputs,2,this.input_dim,[]);
            outputs = permute(outputs,[2 3 1]);
            % use softplus to avoid negatives
            a = softplus(outputs(:,:,1));
            b = softplus(outputs(:,:,2));
            loga = (a-1).*log(inputs);
            logb = (b-1).*log(1-inputs); % log1p doesn't support dlarray :-(
            log_prob = loga + logb - betaln(a,b);
            log_prob = sum(log_prob,finddim(inputs,"C"));
            log_prob(mask) = -inf;
            log_prob = stripdims(log_prob);
        end

        function [samples, u] = sample(this, conditions)
            outputs = forward(this.nnet,conditions);
            outputs = reshape(outputs,2,this.input_dim,[]);
            outputs = permute(outputs,[3 2 1]);
            a = extractdata(softplus(outputs(:,:,1)));
            b = extractdata(softplus(outputs(:,:,2)));
            samples = betarnd(a,b);
            u = betacdf(samples,a,b);
        end

        function x = inverse(this, u, conditions)
            outputs = forward(this.nnet,conditions);
            outputs = reshape(outputs,2,this.input_dim,[]);
            outputs = permute(outputs,[3 2 1]);
            a = extractdata(softplus(outputs(:,:,1)));
            b = extractdata(softplus(outputs(:,:,2)));
            x = betainv(u,a,b);
        end
    end
    methods (Static, Access = private)
        function net = DenseReluNetwork(in_dim, out_dim)
            % This creates a very simple predictor network - we're only
            % really looking for a basic sketch of the conditionalized
            % marginal distribution, not a final result.
            layers = [
                featureInputLayer(in_dim,"Normalization","none")
            ];
            for i = 1:1
                layers = [ layers
                    fullyConnectedLayer(out_dim)
                    leakyReluLayer(0.001)
                ]; %#ok<AGROW>
            end
            % initialize bias to 2, since zero is bad...
            layers(end+1) = fullyConnectedLayer(out_dim,"Bias",repmat(2,[out_dim 1]));
            net = dlnetwork(layerGraph(layers));
        end
    end
end

function x = softplus(x)
    x = max(0,x)+log(1+exp(-abs(x)));
end

function y = betaln(z,w)
    y = gammaln(z)+gammaln(w)-gammaln(z+w);
end

function y = gammaln(x)
    y = 0.5*log(2*pi)+(x-0.5).*log(x)-x+0.5*x.*log(x.*sinh(1./x)+1./(810*x.^6));
end
