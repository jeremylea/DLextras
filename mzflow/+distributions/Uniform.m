classdef Uniform < distributions.LatentDist
    properties
        input_dim
        info
    end

    methods
        function this = Uniform(input_dim)
            this.input_dim = input_dim;
            this.info = {"Uniform" input_dim};
        end

        function log_prob = log_prob(this, inputs)
            mask = any(inputs < 0 | inputs > 1,finddim(inputs,"C"));
            log_prob = repmat(this.input_dim*log(1.0),size(mask));
            log_prob(mask) = -inf;
            log_prob = stripdims(dlarray(log_prob));
        end

        function samples = sample(this, nsamples)
            samples = rand([nsamples, this.input_dim]);
        end
    end
end

