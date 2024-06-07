classdef Beta < distributions.LatentDist
    % Beta A learnable mutli-variate latent beta distribution
    properties
        input_dim % data dimension
        info      % simple description of layer
    end

    properties (Learnable)
        A % learned A values
        B % learned B values
    end

    methods
        function this = Beta(input_dim)
            % Constructor, which just depends only on dimensions
            this.input_dim = input_dim;
            this.info = {"Learnable Beta" input_dim};
            % 2 seems like a good initial value
            this.A = repmat(2,[1 input_dim]);
            this.B = repmat(2,[1 input_dim]);
        end

        function log_prob = log_prob(this, inputs)
            mask = any(inputs <= 0 | inputs >= 1,finddim(inputs,"C"));
            len = size(inputs,finddim(inputs,"B"));
            a = dlarray(repmat(softplus(this.A),[len 1]),"BC");
            b = dlarray(repmat(softplus(this.B),[len 1]),"BC");
            % we cannout use betapdf here
            loga = (a-1).*log(inputs);
            % even log1p doesn't support dlarray :-(
            logb = (b-1).*log(1-inputs);
            log_prob = loga + logb - betaln(a,b);
            log_prob = sum(log_prob,finddim(inputs,"C"));
            log_prob(mask) = -inf;
            log_prob = stripdims(log_prob);
        end

        function [samples, u] = sample(this, nsamples)
            a = repmat(softplus(this.A),[nsamples 1]);
            b = repmat(softplus(this.B),[nsamples 1]);
            samples = betarnd(a,b);
            % for learnable latent dists, also return the uniform sample
            u = betacdf(samples,a,b);
        end

        function x = inverse(this, u)
            a = repmat(softplus(this.A),[nsamples 1]);
            b = repmat(softplus(this.B),[nsamples 1]);
            x = betainv(u,a,b);
        end
    end
end

% XXX these should probably be moved to helper functions
function x = softplus(x)
    x = max(0,x)+log(1+exp(-abs(x)));
end

function y = betaln(z,w)
    y = gammaln(z)+gammaln(w)-gammaln(z+w);
end

function y = gammaln(x)
    % This approximation is close enough for this type of work
    y = 0.5*log(2*pi)+(x-0.5).*log(x)-x+0.5*x.*log(x.*sinh(1./x)+1./(810*x.^6));
end
