classdef Uniform < distributions.LatentDist
    properties
        input_dim
        info
    end

    methods
        function this = Uniform(input_dim)
            this.input_dim = input_dim;
            this.info = {"Uniform", input_dim };
        end

        function log_prob = log_prob(this, inputs)
            mask = all((inputs >= 0) & (inputs <= 1),finddim(inputs,"C"));
            log_prob = repmat(-inf,size(mask));
            log_prob(mask) = -this.input_dim*log(1.0);
            log_prob = dlarray(log_prob,"CB");
        end

        function samples = sample(this, nsamples)
            samples = rand([nsamples, this.input_dim]);
        end
    end
end

