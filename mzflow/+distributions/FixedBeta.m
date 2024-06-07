classdef FixedBeta < distributions.LatentDist
    % FixedBeta A fixed parameter multi-variate latent beta distribution
    %   For cases where the marginal distributions are bell curves, this
    %   is probably a better starting point than a uniform distribtution,
    %   although it depends on centered and symmetric data.

    properties
        input_dim % dimention number
        info      % basic description
        a         % Beta A parameter
        b         % Beta B parameter
    end

    methods
        function this = FixedBeta(input_dim,param)
            % Constructor, with data dimension and optional beta parameter,
            % which defaults to 13 (close to normal).
            arguments
                input_dim (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse}
                param (1,1) {mustBeReal, mustBePositive, mustBeNonsparse} = 13
            end
            this.input_dim = input_dim;
            this.info = {"FixedBeta" input_dim param};
            this.a = param;
            this.b = param;
        end

        function log_prob = log_prob(this, inputs)
            mask = any(inputs <= 0 | inputs >= 1,finddim(inputs,"C"));
            loga = (this.a-1).*log(inputs);
            logb = (this.b-1).*log(1-inputs); % log1p doesn't support dlarray :-(
            log_prob = loga + logb - betaln(this.a,this.b);
            log_prob = sum(log_prob,finddim(inputs,"C"));
            log_prob(mask) = -inf;
            log_prob = stripdims(log_prob);
        end

        function samples = sample(this, nsamples)
            samples = betarnd(this.a,this.b,[nsamples this.input_dim]);
        end
    end
end
