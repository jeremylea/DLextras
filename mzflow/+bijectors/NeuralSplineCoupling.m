classdef NeuralSplineCoupling < bijectors.Bijector
    properties
        params
        K
        hidden_layers
        hidden_dim
        transformed_dim
        periodic
        upper_dim
        lower_dim
    end
    properties (Learnable)
        nnet
    end
    methods
        function this = NeuralSplineCoupling(input_dim, condition_dim, K, hidden_layers, hidden_dim, transformed_dim, periodic, options)
            arguments
                input_dim (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse}
                condition_dim (1,1) {mustBeInteger, mustBeNonnegative, mustBeNonsparse} = 0
                K (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse} = 16
                hidden_layers (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse} = 2
                hidden_dim (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse} = 128
                transformed_dim (1,1) {mustBeInteger, mustBeNonnegative, mustBeNonsparse} = 2
                periodic (1,1) {mustBeNumericOrLogical, mustBeNonsparse} = false
                options.Name {mustBeText} = ''
                options.Description {mustBeText} = "Neural Spline Coupling layer"
            end
            this.Type = "NeuralSplineCoupling";
            this.Name = char(options.Name);
            this.Description = options.Description;
            this.K = K;
            this.hidden_layers = hidden_layers;
            this.hidden_dim = hidden_dim;
            this.transformed_dim = transformed_dim;
            this.periodic = periodic;
            if condition_dim > 0
                this.InputNames = ["in","cond"];
            else
                this.NumInputs = 1;
            end
            this.OutputNames = ["out","log_det"];
            if this.transformed_dim == 0
                this.upper_dim = floor(input_dim / 2);
                this.lower_dim = input_dim - this.upper_dim;
            else
                this.upper_dim = input_dim - this.transformed_dim;
                this.lower_dim = this.transformed_dim;
            end
            this.nnet = DenseReluNetwork(this.upper_dim+condition_dim,...
                (3*this.K-1+this.periodic)*this.lower_dim, this.hidden_layers, this.hidden_dim);
        end
        function [outputs, log_det] = predict(this, varargin)
            inputs = getinput(this,varargin,"in");
            if this.NumInputs == 1
                conditions = [];
            else
                conditions = getinput(this,varargin,"cond");
            end
            upper = inputs(1:this.upper_dim,:);
            lower = inputs(this.upper_dim+1:end,:);
            [lower, log_det] = spline_params(this,upper,lower,conditions,false);
            outputs = cast([upper; lower],'like',inputs);
        end
        function outputs = inverse(this, inputs, conditions)
            upper = inputs(1:this.upper_dim,:);
            lower = inputs(this.upper_dim+1:end,:);
            lower = spline_params(this,upper,lower,conditions,true);
            outputs = [upper; lower];
        end
        function [inf] = info(this)
            inf = { 'NeuralSplineCoupling', this.n_conditions, this.K, this.hidden_layers, this.hidden_dim, this.transformed_dim, this.periodic};
        end
    end
    methods (Access = private)
        function x = getinput(this,args,name)
            x = args{strcmp(this.InputNames,name)};
        end
        function [outputs, log_det] = spline_params(this, upper, lower, conditions, inverse)
            % Try with a hand rolled network.
            inputs = [upper; conditions];
            outputs = predict(this.nnet,inputs);
            outputs = reshape(outputs,3*this.K-1+this.periodic,this.lower_dim,[]);
            outputs = permute(outputs,[2 3 1]);
            W = softmax(outputs(:,:,1:this.K),3);
            H = softmax(outputs(:,:,this.K+1:2*this.K),3);
            D = softplus(outputs(:,:,2*this.K+1:end));
            xk = W;
            for i = 2:size(W,3)
                xk(:,:,i) = xk(:,:,i-1)+xk(:,:,i);
            end
            xk(:,:,end) = 1;
            xk(:,:,end+1) = 0;
            xk = circshift(xk,1,3);
            yk = H;
            for i = 2:size(H,3)
                yk(:,:,i) =  yk(:,:,i-1)+yk(:,:,i);
            end
            yk(:,:,end) = 1;
            yk(:,:,end+1) = 0;
            yk = circshift(yk,1,3);
            if this.periodic
                dk = [D(:,:,end); D];
            else
                dk = D;
                dk(:,:,end+1:end+2) = 0; % one is illogical, changed to zero. XXX
                dk = circshift(dk,1,3);
            end
            out_of_bounds = lower < 0 | lower > 1;
            masked = lower;
            masked(out_of_bounds) = mod(extractdata(lower(out_of_bounds)),1.0);
            if inverse
                idx = sum(yk <= masked,3);
            else
                idx = sum(xk <= masked,3);
            end
            idx(idx == size(xk,3)) = size(xk,3)-1;
            I1 = repelem((1:size(idx,1))',1,size(idx,2));
            I2 = repmat(1:size(idx,2),size(idx,1),1);
            idxp1 = sub2ind(size(xk),I1,I2,idx+1);
            idx = sub2ind(size(xk),I1,I2,idx);
            xk = xk(idx);
            yk = yk(idx);
            dkp1 = dk(idxp1);
            dk = dk(idx);
            wk = W(idx);
            hk = H(idx);
            sk = wk ./ hk;
            if inverse
                a = hk .* (sk - dk) + (masked - yk) .* (dkp1 + dk - 2 * sk);
                b = hk .* dk - (masked - yk) .* (dkp1 + dk - 2 * sk);
                c = -sk .* (masked - yk);
                relx = 2 * c ./ (-b - sqrt(b.^2 - 4 * a .* c));
                outputs = relx .* wk + xk;
                if ~this.periodic
                    outputs(out_of_bounds) = lower(out_of_bounds);
                end
                log_det = [];
            else
                relx = min(max(0,(masked - xk) ./ wk),1);
                num = hk .* (sk .* relx.^2 + dk .* relx .* (1 - relx));
                den = sk + (dkp1 + dk - 2 .* sk) .* relx .* (1 - relx);
                outputs = yk + num ./ den;
                if ~this.periodic
                    outputs(out_of_bounds) = lower(out_of_bounds);
                end
                dnum = dkp1 .* relx.^2 + 2 .* sk .* relx .* (1 - relx) + dk .* (1 - relx).^2;
                dden = sk + (dkp1 + dk - 2 .* sk) .* relx .* (1 - relx);
                log_det = 2 * log(sk) + log(dnum) - 2*log(dden);
                if ~this.periodic
                    log_det(out_of_bounds) = 0;
                end
            end
            outputs = dlarray(outputs,"CB");
            log_det = dlarray(log_det,"CB");
            outputs = cast(outputs,'like',lower);
        end
    end
end

function net = DenseReluNetwork(in_dim, out_dim, hidden_layers, hidden_dim)
    layers = featureInputLayer(in_dim,"Normalization","none");
    for i = 1:hidden_layers
        layers = [ layers; ...
            fullyConnectedLayer(hidden_dim);
            leakyReluLayer(0.01);
        ]; %#ok<AGROW>
    end
    layers(end+1) = fullyConnectedLayer(out_dim);
    net = dlnetwork(layerGraph(layers));
end

function x = softmax(x,dim)
    x = exp(x);
    x = x./sum(x,dim);
end

function x = softplus(x)
    x = max(0,x)+log(1+exp(-abs(x)));
end

