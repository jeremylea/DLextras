classdef NeuralSplineCoupling < bijectors.Bijector
    properties
        params
        K
        hidden_layers
        hidden_dim
        transformed_dim
        end_conditions
        end_dims
        upper_dim
        lower_dim
        spline_dim
    end
    properties (Learnable)
        nnet
    end
    properties (Constant)
        smooth = 0
        inflated = 1
        closed = 2
        unit = 3
        periodic = 4
    end
    methods
        function this = NeuralSplineCoupling(input_dim, condition_dim, K, options)
            arguments
                input_dim (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse}
                condition_dim (1,1) {mustBeInteger, mustBeNonnegative, mustBeNonsparse} = 0
                K (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse} = 32
                options.transformed_dim (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse} = 1
                options.hidden_layers (1,1) {mustBeInteger, mustBeNonnegative, mustBeNonsparse} = 2
                options.hidden_dim (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse} = 8*K*max(1,input_dim+condition_dim)
                options.end_conditions (1,2) {mustBeText,mustBeMember(options.end_conditions,["smooth","inflated","closed","unit","periodic"])} = [ "smooth" "smooth" ];
                options.Name {mustBeText} = ''
                options.Description {mustBeText} = strjoin(["Neural Spline Coupling" num2str(input_dim) num2str(condition_dim) num2str(K) ]) %  num2str(options.hidden_layers) num2str(hidden_dim) num2str(transformed_dim) end_conditions
            end
            this.Type = "NeuralSplineCoupling";
            this.Name = char(options.Name);
            this.Description = options.Description;
            this.K = K-1;
            this.hidden_layers = options.hidden_layers;
            this.transformed_dim = options.transformed_dim;
            this.end_conditions = [this.smooth this.smooth];
            this.end_dims = [0 0];
            if any(strcmp(options.end_conditions,"periodic"))
                assert(all(strcmp(options.end_conditions,"periodic")));
                this.end_conditions = [this.periodic this.periodic];
                this.end_dims = [1 0];
            else
                for i = 1:2
                    switch options.end_conditions(i)
                        case "smooth"
                            % default
                        case "inflated"
                            this.end_conditions(i) = this.inflated;
                            this.end_dims(i) = 2;
                        case "closed"
                            this.end_conditions(i) = this.closed;
                            this.end_dims(i) = 1;
                        case "unit"
                            this.end_conditions(i) = this.unit;
                        otherwise
                            throw("Invalid end conditions");
                    end
                end
            end
            if condition_dim > 0
                this.InputNames = ["in","cond"];
            else
                this.InputNames = "in";
            end
            this.upper_dim = input_dim - this.transformed_dim;
            this.lower_dim = this.transformed_dim;
            this.hidden_dim = options.hidden_dim;
            this.spline_dim = 3*this.K+sum(this.end_dims);
            this.nnet = this.DenseReluNetwork(condition_dim);
        end
        function [outputs, log_det, penalty] = predict(this, varargin)
            inputs = getinput(this,varargin,"in");
            if this.NumInputs == 1
                conditions = [];
            else
                conditions = getinput(this,varargin,"cond");
            end
            upper = inputs(1:this.upper_dim,:);
            lower = inputs(this.upper_dim+1:end,:);
            [lower,log_det,penalty] = spline_params(this,upper,lower,conditions,false);
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
        function [X, log_det, penalty] = spline_params(this, upper, lower, conditions, inverse)
            inputs = [upper; conditions];
            if isempty(inputs)
                inputs = dlarray(zeros([1 size(inputs,finddim(inputs,"B"))]),"CB");
            end
            outputs = cast(forward(this.nnet,inputs),"double");
            outputs = reshape(outputs,this.spline_dim,this.lower_dim,[]);
            outputs = permute(outputs,[2 3 1]);
            W = outputs(:,:,1:this.K);
            H = outputs(:,:,this.K+1:2*this.K);
            D = outputs(:,:,(2*this.K+1):(end-sum(this.end_dims==2)));
            W(:,:,end+1) = 0; % pad the last interval to a fixed value
            H(:,:,end+1) = 0;
            W = softmax(circshift(W,ceil(this.K/2),3),3);
            H = softmax(circshift(H,ceil(this.K/2),3),3);
            D = softplus(D);
            if ~inverse
                penalty = sqrt(W.^2+H.^2)./(sqrt(2)/this.K);
                penalty = 0.1*sum(max(penalty,[],3)-min(penalty,[],3),1);
            else
                penalty = dlarray([]);
            end
            if this.end_conditions(1) == this.inflated % XXX
                ZI = softplus(outputs(:,:,end-(this.end_dims(2)==2)));
                penalty = penalty + 0.1*sum(1-ZI,1);
                zm = ZI >= 0;
                WZ = zeros(size(ZI));
                WH = zeros(size(ZI));
                WZ(zm) = ZI(zm);
                WH(~zm) = -ZI(~zm);
                W = W.*(1-WZ);
                H = H.*(1-WH);
                W = cat(3,WZ,W);
                H = cat(3,WH,H);
            end
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
            if this.end_conditions(1) == this.periodic
                D = cat(3,D,D(:,:,1));
            else
                if this.end_conditions(1) == this.inflated % XXX
                    D(:,:,end+1) = ~zm*100;
                    D = circshift(D,1,3);
                elseif this.end_conditions(1) == this.smooth || this.end_conditions(1) == this.unit
                    D(:,:,end+1) = this.end_conditions(1) == this.unit; % zero or one
                    D = circshift(D,1,3);
                end
                if this.end_conditions(2) == this.inflated % XXX
                    D(:,:,end+1) = ~zm*100;
                elseif this.end_conditions(2) == this.smooth || this.end_conditions(2) == this.unit
                    D(:,:,end+1) = this.end_conditions(2) == this.unit;
                end
            end
            S = H./W;
            S(:,:,end+1) = S(:,:,end);
            D = D.*S;
            if ~inverse
                dxxk = -2*(D(:,:,1:end-1).*(D(:,:,1:end-1)+D(:,:,2:end)-2*S(:,:,1:end-1))+S(:,:,1:end-1).*(D(:,:,1:end-1)-S(:,:,1:end-1)))./H;
                dxxk1 = -2*(S(:,:,1:end-1)+D(:,:,2:end))./W + 2*D(:,:,2:end).*(D(:,:,1:end-1)+D(:,:,2:end))./H;
                penalty = penalty+0.01*sum(max(abs(dxxk1(:,:,1:end-1)-dxxk(:,:,2:end)),[],3),1);
            end
            out_of_bounds = lower < 0 | lower > 1;
            masked = lower;
            masked(out_of_bounds) = mod(extractdata(lower(out_of_bounds)),1.0);
            if inverse
                idx = extractdata(sum(yk <= masked,3));
                idx(idx == size(yk,3)) = size(yk,3)-1;
                if this.end_conditions(1) == this.inflated % XXX
                    izh = ~zm & masked < yk(:,:,2);
                    izw = zm & masked == 0;
                    idx(izw) = 1;
                end
            else
                idx = extractdata(sum(xk <= masked,3));
                idx(idx == size(xk,3)) = size(xk,3)-1;
                if this.end_conditions(1) == this.inflated % XXX
                    izh = ~zm & masked == 0;
                    izw = zm & masked < xk(:,:,2);
                    idx(izh) = 1;
                end
            end
            I1 = repelem((1:size(idx,1))',1,size(idx,2));
            I2 = repmat(1:size(idx,2),size(idx,1),1);
            idxp1 = sub2ind(size(xk),I1,I2,idx+1);
            idx = sub2ind(size(xk),I1,I2,idx);
            xk = xk(idx);
            yk = yk(idx);
            wk = W(idx);
            hk = H(idx);
            sk = hk./max(1e-40,wk); % we'll deal later with the zeros
            dkp1 = D(idxp1);
            dk = D(idx);
            if inverse
                a = hk.*(sk-dk) + (masked-yk).*(dkp1+dk-2*sk);
                b = hk.*dk - (masked-yk).*(dkp1+dk-2*sk);
                c = -sk.*(masked-yk);
                relx = 2*c./(-b-sqrt(b.^2-4*a.*c));
                X = min(max(0,relx.* wk + xk),1);
                if this.end_conditions(1) ~= this.periodic
                    X(out_of_bounds) = lower(out_of_bounds);
                end
                log_det = dlarray([]);
            else
                relx = min(max(0,(masked-xk)./max(1e-40,wk)),1);
                num = hk.*(sk.*relx.^2 + dk.*relx.*(1-relx));
                den = sk + (dkp1+dk-2*sk).*relx.*(1-relx);
                den = max(1e-40,den);
                X = min(max(0,yk + num./den),1);
                if this.end_conditions(1) == this.inflated % XXX
                    X(izw | izh) = 0;
                end
                if this.end_conditions(1) ~= this.periodic
                    X(out_of_bounds) = lower(out_of_bounds);
                end
                dnum = dkp1.*relx.^2 + 2*sk.*relx.*(1-relx) + dk.*(1-relx).^2;
                log_det = 2*log(sk) + log(max(1e-40,dnum)) - 2*log(den);
                if this.end_conditions(1) == this.inflated % XXX
                    log_det(izw) = -10; % punish this badly
                    penalty(izw) = penalty(izw) + 100*sum(ZI(izw)-masked(izw),1);
                    log_det(izh) = log(ZI(izh)); % prob of observing zero
                end
                if this.end_conditions(1) ~= this.periodic
                    log_det(out_of_bounds) = 0;
                end
                log_det = sum(log_det,1);
            end
            log_det = stripdims(log_det);
            penalty = stripdims(penalty);
            X = cast(dlarray(X,"CB"),"like",lower);
        end
        function net = DenseReluNetwork(this, condition_dim)
            in_dim = max(1,this.upper_dim+condition_dim);
            out_dim = this.spline_dim*this.lower_dim;
            layers = featureInputLayer(in_dim,"Normalization","none");
            for i = 1:this.hidden_layers
                n = ceil(in_dim*(1-i/this.hidden_layers)+this.hidden_dim*(i/this.hidden_layers));
                layers = [ layers
                    fullyConnectedLayer(n*this.lower_dim,"BiasInitializer","narrow-normal","BiasLearnRateFactor",10)
                    leakyReluLayer(0.001)
                ]; %#ok<AGROW>
            end
            bi = 0.1*randn(out_dim,1);
            idx = 2*this.K+1:out_dim;
            if this.end_dims(2) == 2
                idx(end) = [];
            end
            if this.end_dims(1) == 2
                idx(end) = [];
            end
            if all(this.end_conditions == this.unit)
                bi(idx) = bi(idx)+0.5413248596027; % softplus 1
            else
                bi(idx) = bi(idx)+1.24751754107991; % softplus 1.5
            end
            if this.end_conditions(1) == this.periodic
                bi(idx(1)) = bi(idx(1))-6.0;
            else
                if this.end_dims(1) > 0
                    bi(idx(1)) = bi(idx(1))-6.0;
                end
                if this.end_dims(2) > 0
                    bi(idx(end)) = bi(idx(end))-6.0;
                end
            end
            layers(end+1) = fullyConnectedLayer(out_dim,"Bias",bi,"BiasLearnRateFactor",0.1);
            net = dlnetwork(layerGraph(layers));
            for i = 1:height(net.Learnables)
                if contains(net.Learnables{i,1},"fc") && contains(net.Learnables{i,2},"Weights")
                    net.Learnables{i,3}{1} = eye(size(net.Learnables{i,3}{1})) + 0.1*net.Learnables{i,3}{1};
                end
            end
        end
    end
end

function x = softmax(x,dim)
    x = exp(x);
    x = x./sum(x,dim);
end

function x = softplus(x)
    x = max(0,x)+log(1+exp(-abs(x)));
end

