classdef LatentDist < nnet.layer.Layer ...
        & nnet.layer.Formattable
    % LatentDist A underlying distribution for a normalizing flow
    %   This class captures the underlying distribution for the non-data
    %   side of a normalizing flow.  This is where samples are generated
    %   for the inverse path and log_probabilites are generated for the
    %   forward path and the loss function.
    %
    %   This uses a layer as the base class, even though that's not ideal
    %   because that enables learnable parameters.  We fake the predict
    %   function, for use in the training loop.
    properties (Abstract)
        info                    % Simple description string
    end
    methods (Abstract)
        log_prob(this, inputs); % return log-probability of inputs
        sample(this, nsamples); % sample the distribution
    end
    methods
        function outputs = predict(this, varargin)
            % Abuse predict to directly return log_prob not prob, which
            % would then just get logged in the training loop.
            outputs = this.log_prob(varargin{:});
        end
    end
end
