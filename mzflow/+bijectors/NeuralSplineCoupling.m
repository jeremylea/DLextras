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
                K (1,1) {mustBeInteger, mustBeNonnegative, mustBeNonsparse} = 15
                hidden_layers (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse} = 1
                hidden_dim (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse} = 32*(input_dim+condition_dim);
                transformed_dim (1,1) {mustBeInteger, mustBeNonnegative, mustBeNonsparse} = 0
                periodic (1,1) {mustBeNumericOrLogical, mustBeNonsparse} = false
                options.Name {mustBeText} = ''
                options.Description {mustBeText} = strjoin(["Neural Spline Coupling" num2str(input_dim) num2str(condition_dim) num2str(K) num2str(hidden_layers) num2str(hidden_dim) num2str(transformed_dim) num2str(periodic)])
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
                this.InputNames = "in";
            end
            if this.transformed_dim == 0
                this.upper_dim = floor(input_dim / 2);
                this.lower_dim = input_dim - this.upper_dim;
            else
                this.upper_dim = input_dim - this.transformed_dim;
                this.lower_dim = this.transformed_dim;
            end
            this.nnet = DenseReluNetwork(max(1,this.upper_dim+condition_dim),...
                (3*this.K+2-this.periodic)*this.lower_dim,this.hidden_layers,this.hidden_dim,this.K);
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
            [lower,log_det] = spline_params(this,upper,lower,conditions,false);
            outputs = cast([upper; lower],"like",inputs);
        end
        function outputs = inverse(this, inputs, conditions)
            upper = inputs(1:this.upper_dim,:);
            lower = inputs(this.upper_dim+1:end,:);
            lower = spline_params(this,upper,lower,conditions,true);
            outputs = [upper; lower];
        end
    end
    methods (Access = private)
        function x = getinput(this,args,name)
            x = args{strcmp(this.InputNames,name)};
        end
        function [outputs, log_det] = spline_params(this, upper, lower, conditions, inverse)
            inputs = [upper; conditions];
            if isempty(inputs)
                inputs = dlarray(zeros([1 size(inputs,finddim(inputs,"B"))]),"CB");
            end
            outputs = predict(this.nnet,inputs);
            outputs = reshape(outputs,3*this.K+2-this.periodic,this.lower_dim,[]);
            outputs = permute(outputs,[2 3 1]);
            W = outputs(:,:,1:this.K);
            H = outputs(:,:,this.K+1:2*this.K);
            D = outputs(:,:,2*this.K+1:end);
            W(:,:,end+1) = 0; % pad the last interval to a fixed value
            H(:,:,end+1) = 0;
            W = softmax(W,3);
            H = softmax(H,3);
            D = softplus(D);
            xk = W;
            for i = 2:(size(W,3)-1)
                xk(:,:,i) = xk(:,:,i-1)+xk(:,:,i);
            end
            xk(:,:,end) = 1;
            xk(:,:,end+1) = 0;
            xk = circshift(xk,1,3);
            yk = H;
            for i = 2:(size(H,3)-1)
                yk(:,:,i) =  yk(:,:,i-1)+yk(:,:,i);
            end
            yk(:,:,end) = 1;
            yk(:,:,end+1) = 0;
            yk = circshift(yk,1,3);
            if this.periodic
                D = cat(3,D,D(:,:,1));
            else
                %D(:,:,end+1:end+2) = 0; % XX why one?
                %D = circshift(D,1,3);
            end
            out_of_bounds = lower < 0 | lower > 1;
            masked = lower;
            masked(out_of_bounds) = mod(extractdata(lower(out_of_bounds)),1.0);
            if inverse
                idx = extractdata(sum(yk <= masked,3));
            else
                idx = extractdata(sum(xk <= masked,3));
            end
            idx(idx == size(xk,3)) = size(xk,3)-1;
            I1 = repelem((1:size(idx,1))',1,size(idx,2));
            I2 = repmat(1:size(idx,2),size(idx,1),1);
            idxp1 = sub2ind(size(xk),I1,I2,idx+1);
            idx = sub2ind(size(xk),I1,I2,idx);
            xk = xk(idx);
            yk = yk(idx);
            wk = W(idx);
            hk = H(idx);
            sk = hk./wk;
            dkp1 = D(idxp1);
            dk = D(idx);
            if inverse
                a = hk.*(sk-dk) + (masked-yk).*(dkp1+dk-2*sk);
                b = hk.*dk - (masked-yk).*(dkp1+dk-2*sk);
                c = -sk.*(masked-yk);
                relx = 2*c./(-b-sqrt(b.^2-4*a.*c));
                outputs = min(max(0,relx.* wk + xk),1);
                if ~this.periodic
                    outputs(out_of_bounds) = lower(out_of_bounds);
                end
                log_det = [];
            else
                relx = min(max(0,(masked-xk)./wk),1);
                num = hk.*(sk.*relx.^2 + dk.*relx.*(1-relx));
                den = sk + (dkp1+dk-2*sk).*relx.*(1-relx);
                outputs = min(max(0,yk + num./den),1);
                if ~this.periodic
                    outputs(out_of_bounds) = lower(out_of_bounds);
                end
                dnum = dkp1.*relx.^2 + 2*sk.*relx.*(1-relx) + dk.*(1-relx).^2;
                dden = sk + (dkp1+dk-2*sk).*relx.*(1-relx);
                log_det = 2*log(sk) + log(max(0,dnum)) - 2*log(max(0,dden));
                if ~this.periodic
                    log_det(out_of_bounds) = 0;
                end
                log_det = sum(log_det,1);
            end
            outputs = dlarray(outputs,"CB");
            log_det = stripdims(dlarray(log_det));
            outputs = cast(outputs,"like",lower);
        end
    end
end

function net = DenseReluNetwork(in_dim, out_dim, hidden_layers, hidden_dim, K)
    layers = [
        featureInputLayer(in_dim,"Normalization","none")
        fullyConnectedLayer(in_dim,"BiasInitializer","narrow-normal")
        swishLayer
        %leakyReluLayer(1e-6)
        %tanhLayer
    ];
    for i = 1:hidden_layers
        layers = [ layers
            fullyConnectedLayer(hidden_dim,"BiasInitializer","narrow-normal")
            swishLayer
            %leakyReluLayer(1e-6)
            %tanhLayer
        ]; %#ok<AGROW>
    end
    bi = ones(out_dim,1);
    bi(1:2*K) = 0;
    layers(end+1) = fullyConnectedLayer(out_dim,"Bias",bi);
    net = dlnetwork(layerGraph(layers));
end

function x = softmax(x,dim)
    x = exp(x);
    x = x./sum(x,dim);
end

function x = softplus(x)
    x = max(0,x)+log(1+exp(-abs(x)));
end

