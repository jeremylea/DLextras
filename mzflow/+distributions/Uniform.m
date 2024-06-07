classdef Uniform < distributions.LatentDist
    % Uniform A multi-variate uniform latent distribution
    %   This is the default latent distribution.

    properties
        input_dim % Number of dimensions
        info      % Simple description string
    end

    methods
        function this = Uniform(input_dim)
            % Constructior, with number of dimensions
            this.input_dim = input_dim;
            this.info = {"Uniform" input_dim};
        end

        function log_prob = log_prob(this, inputs)
            % Return constant log probabilities
            mask = any(inputs < 0 | inputs > 1,finddim(inputs,"C"));
            log_prob = repmat(this.input_dim*log(1.0),size(mask));
            log_prob(mask) = -inf;
            log_prob = stripdims(dlarray(log_prob));
        end

        function samples = sample(this, nsamples)
            % Sample the distribution, avoiding a small region right on the
            % edges, which is probably not wanted.
            border = (1e-4)*10.^min(2,this.input_dim);
            samples = 0.5*border+(1-border)*rand([nsamples this.input_dim]);
        end
    end
end

