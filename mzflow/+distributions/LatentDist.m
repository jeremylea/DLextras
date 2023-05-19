classdef LatentDist < handle
    properties (Abstract)
        info
    end
    methods (Abstract)
        log_prob(this, inputs);
        sample(this, nsamples);
    end
end
