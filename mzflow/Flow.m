classdef Flow
    properties (SetAccess = immutable)
        data_columns
        input_dim
        conditional_columns
        autoscale_conditions
        weight_column
    end
    properties (SetAccess = private)
        condition_means
        condition_stds
        latent
        bijector
    end
    properties (Access = private)
        losses = [];
        val_losses = [];
        trailingAvgB = [];
        trailingAvgSqB = [];
        trailingAvgL = [];
        trailingAvgSqL = [];
    end
    methods
        function this = Flow(data_columns, conditional_columns, weight_column, bijector, latent, autoscale_conditions)
            arguments
                data_columns {mustBeText}
                conditional_columns {mustBeText} = ""
                weight_column (1,1) {mustBeText} = ""
                bijector = []
                latent = []
                autoscale_conditions (1,1) {mustBeNumericOrLogical} = true
            end
            this.data_columns = data_columns;
            this.input_dim = length(this.data_columns);
            this.autoscale_conditions = ~~autoscale_conditions;
            if ~isempty(bijector)
                this = set_bijector(this,bijector);
            end
            if ~isempty(latent)
                this = set_lantent(this,latent);
            end
            if isempty(conditional_columns) || all(strcmp(conditional_columns,""))
                this.conditional_columns = [];
                this.condition_means = [];
                this.condition_stds = [];
            else
                this.conditional_columns = conditional_columns;
                this.condition_means = zeros(1,length(this.conditional_columns));
                this.condition_stds = ones(1,length(this.conditional_columns));
            end
            if isempty(weight_column) || strcmp(weight_column,"")
                this.weight_column = [];
            else
                this.weight_column = weight_column;
            end
        end

        function this = set_latent(this, latent)
            if latent.input_dim ~= length(this.data_columns)
                error("The latent distribution has %d dimensions, but data_columns has %d dimensions. They must match!",latent.input_dim,length(this.data_columns));
            end
            if latent.NumInputs == 2
                if latent.condition_dim ~= length(this.conditional_columns)
                    error("The latent distribution has %d condition dimensions, but conditional_columns has %d dimensions. They must match!",latent.conditional_dim,length(this.conditional_columns));
                end
                this.latent = dlnetwork(latent,dlarray(zeros(1,this.input_dim),"BC"),dlarray(zeros(1,length(this.conditional_columns)),"BC"));
            else
                assert(latent.NumInputs == 1);
                this.latent = dlnetwork(latent,dlarray(zeros(1,this.input_dim),"BC"));
            end
        end

        function this = set_uniform_latent(this)
            this = set_latent(this,distributions.Uniform(this.input_dim));
        end

        function this = set_fixedbeta_latent(this,options)
            arguments
                this
                options.param (1,1) {mustBeReal, mustBePositive, mustBeNonsparse}
            end
            if ~isfield(options,"param")
                this = set_latent(this,distributions.FixedBeta(this.input_dim));
            else
                this = set_latent(this,distributions.FixedBeta(this.input_dim,options.param));
            end
        end

        function this = set_default_latent(this)
            this = set_uniform_latent(this);
        end

        function this = set_bijector(this, bijector)
            this.bijector = bijector;
        end

        function this = set_default_bijector(this, inputs, K, input_lim, options)
            arguments
                this
                inputs
                K (1,1) {mustBeInteger, mustBeNonnegative, mustBeNonsparse} = 32
                input_lim {mustBeNonsparse} = [];
                options.shift_dim (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse} = 1
                options.end_conditions (1,2) {mustBeText,mustBeMember(options.end_conditions,["smooth","inflated","closed","unit","periodic"])} = [ "smooth" "smooth" ];
                options.hidden_layers (1,1) {mustBeInteger, mustBeNonnegative, mustBeNonsparse} = 2
                options.hidden_dim (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse} = 8*K*max(1,numel(this.data_columns)+numel(this.conditional_columns))
                options.rolls (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse} = 1
            end
            if isempty(input_lim)
                input_lim = this.default_limits(inputs);
            end
            has_conditions = numel(this.conditional_columns) > 0;
            layers = layerGraph(featureInputLayer(this.input_dim,name='inputs'));
            if has_conditions
                layers = addLayers(layers,[
                    featureInputLayer(numel(this.conditional_columns),name='conditions')
                ]);
            end
            layers = addLayers(layers,[
                bijectors.ShiftBounds(input_lim(1,:),input_lim(2,:),name='shift')
            ]);
            layers = connectLayers(layers,'inputs','shift');
            assert(options.shift_dim > 0 && options.shift_dim <= this.input_dim && mod(this.input_dim,options.shift_dim) == 0);
            for i = 1:options.shift_dim:(options.rolls*this.input_dim)
                sn = ['nsp' num2str(i)];
                rn = ['roll' num2str(i)];
                layers = addLayers(layers,[
                    bijectors.NeuralSplineCoupling(this.input_dim,numel(this.conditional_columns),K,end_conditions=options.end_conditions, ...
                        transformed_dim=options.shift_dim,hidden_layers=options.hidden_layers,hidden_dim=options.hidden_dim,name=sn)
                    bijectors.Roll(options.shift_dim,name=rn)
                ]);
                if ~has_conditions
                    if i == 1
                        layers = connectLayers(layers,'shift/out','nsp1');
                    else
                        layers = connectLayers(layers,['roll' num2str(i-options.shift_dim)],sn);
                    end
                else
                    if i == 1
                        layers = connectLayers(layers,'shift/out','nsp1/in');
                    else
                        layers = connectLayers(layers,['roll' num2str(i-options.shift_dim) '/out'],[sn '/in']);
                    end
                    layers = connectLayers(layers,'conditions',[sn '/cond']);
                end
            end
            this = set_bijector(this,dlnetwork(layers));
        end

        function check_bijector(this)
            if isempty(this.latent)
                error("The latent distribution has not been set up yet! You can do this by calling flow.set_latent(latent), or by calling train, in which case the default distribution will be used.");
            end
            if isempty(this.bijector)
                error("The bijector has not been set up yet! You can do this by calling flow.set_bijector(bijector), or by calling train, in which case the default bijector will be used.");
            end
        end

        function conditions = get_conditions(this, inputs)
            if isempty(this.conditional_columns)
                conditions = zeros(size(inputs,1),0);
            else
                conditions = inputs{:,this.conditional_columns};
                conditions = (conditions - this.condition_means) ./ this.condition_stds;
            end
        end

        function this = scale_conditions(this,inputs)
            if ~isempty(this.conditional_columns) && this.autoscale_conditions
                C = get_conditions(this,inputs);
                [s,m] = std(C,0,1);
                this.condition_means = m;
                this.condition_stds = (s ~= 0).*s+(s == 0);
            end
        end

        function input_lim = default_limits(this,inputs)
            data = inputs{:,this.data_columns};
            mins = floor(min(data,[],1));
            maxs = ceil(max(data,[],1));
            input_lim = [mins; maxs];
        end

        function input_lim = format_limits(this,mins,maxs)
            if isscalar(mins)
                mins = repmat(mins,1,length(this.data_columns));
            else
                mins = reshape(mins,1,length(this.data_columns));
            end
            if isscalar(maxs)
                maxs = repmat(maxs,1,length(this.data_columns));
            else
                maxs = reshape(maxs,1,length(this.data_columns));
            end
            input_lim = [mins; maxs];
        end

        function weights = get_weights(this, inputs)
            if isempty(this.weight_column)
                weights = ones(size(inputs,1),1);
            else
                weights = inputs{:,this.weight_column};
            end
        end

        function [neg_log_prob,log_probs] = neg_log_prob(this, inputs)
            check_bijector(this);
            X = dlarray(inputs{:,this.data_columns},"BC");
            conditions = dlarray(get_conditions(this,inputs),"BC");
            weights = dlarray(get_weights(this,inputs),"BC");
            [~,log_probs] = log_probs_internal(this.bijector,this.latent,X,conditions);
            [~,neg_log_prob] = std(-log_probs,weights,2);
            log_probs = extractdata(gather(log_probs))';
        end

        function pdfs = posterior(this, inputs, column, grid, normalize, batch_size)
            check_bijector(this)
            idx = find(strcmp(this.data_columns,column));
            cols = this.data_columns;
            cols(idx) = [];
            if isempty(this.data_columns) && isempty(this.conditional_columns)
                nrows = 1;
            else
                nrows = size(inputs,1);
            end
            grid = grid(:);
            if nargin < 6 || isempty(batch_size)
                batch_size = nrows;
            end
            pdfs = zeros(nrows,numel(grid));
            for i = 1:batch_size:nrows
                batch_idx = i:min(i+batch_size-1,nrows);
                batch = inputs(batch_idx,:);
                conditions = get_conditions(this,batch);
                %weights = get_weights(this,batch); % XXX Hmmm...
                batch = batch{:,cols};
                batch = [repmat(batch(:,1:idx-1),numel(grid),1);...
                    repelem(grid,size(batch,1),1);...
                    repmat(batch(:,idx:end),numel(grid),1)];
                conditions = repmat(conditions,numel(grid),1);
                batch = dlarray(batch,"BC");
                conditions = dlarray(conditions,"BC");
                [~,log_probs] = log_probs_internal(this.bijector,this.latent,batch,conditions);
                log_probs = reshape(log_probs,[],numel(grid));
                pdfs(batch_idx,:) = exp(log_probs);
            end
            if nargin < 5 || isempty(normalize) || normalize
                pdfs = pdfs./sum(pdfs,2);
            end
            pdfs(isnan(pdfs)) = 0;
        end

        function [x, u] = sample(this, nsamples, conditions, save_conditions, u)
            check_bijector(this);
            if nargin < 5
                u = [];
            elseif ~isempty(u)
                nsamples = size(u,1)/max(1,size(conditions,1));
            end
            if nargin < 4 || isempty(save_conditions)
                save_conditions = true;
            end
            if ~isempty(this.conditional_columns) && isempty(conditions)
                error("Must provide the following conditions\n%s",this.conditional_columns);
            end
            if isempty(this.conditional_columns)
                conditions = zeros(nsamples,0);
            else
                if isa(conditions,"table")
                    conditions = get_conditions(this,conditions);
                end
                conditions = repelem(conditions,nsamples,1);
            end
            [x,u] = this.sample_internal(size(conditions,1),this.bijector,this.latent,conditions,u);
            x = extractdata(x)';
            if isempty(this.conditional_columns) || ~save_conditions
                x = array2table(x,"VariableNames",this.data_columns);
            else
                conditions = conditions .* this.condition_stds + this.condition_means;
                x = array2table([x conditions],"VariableNames",[this.data_columns this.conditional_columns]);
            end
        end

        function [this, losses, val_losses] = train(this, inputs, validation, epochs, batch_size, init_rate, decay, penalty, debug_cb, varargin)
            if nargin < 9, debug_cb = []; end
            if nargin < 8 || isempty(penalty) || penalty < 0, penalty = 0; end
            if nargin < 7 || isempty(decay) || decay <= 0, decay = 0.5; end
            if nargin < 6 || isempty(init_rate) || init_rate < 0, init_rate = 0; end
            if nargin < 5 || isempty(batch_size) || batch_size < 1, batch_size = 1000; end
            if nargin < 4 || isempty(epochs) || epochs < 1, epochs = 100; end
            if nargin < 3, validation = []; end
            if isempty(this.latent)
                this = set_default_latent(this);
            end
            if isempty(this.bijector)
                this = set_default_bijector(this,inputs);
            end
            this = scale_conditions(this,inputs);
            ds = arrayDatastore(inputs(:,[this.data_columns this.conditional_columns this.weight_column]),IterationDimension=1);
            mbq = minibatchqueue(ds,3,...
                MiniBatchSize=batch_size,...
                MiniBatchFcn=@(X) preprocessData(this,X),...
                MiniBatchFormat=["BC" "BC" "BC"]);
            iteration = 0;
            batch_size = min(batch_size,height(inputs));
            epochs = epochs+max(0,length(this.losses)-1);
            if init_rate == 0
                initialLearnRate = (10^(-sqrt(2*this.input_dim)))*(100/epochs)*min(1,batch_size/1000)/(log(height(inputs)/batch_size)+1);
            else
                initialLearnRate = init_rate;
            end
            learnRate = initialLearnRate;
            best = { inf [] [] };
            if ~isempty(validation)
                wd = sum(get_weights(this,inputs));
                wv = sum(get_weights(this,validation));
                [wd,wv] = deal(wd/(wd+wv),wv/(wd+wv));
            end
            for epoch = max(0,length(this.losses)-1):epochs
                if epoch > 0
                    learnRate = initialLearnRate*decay.^log10(epoch);
                    %learnRate = initialLearnRate*exp(-decay*((epoch-1)/(epochs/3))^2);
                    shuffle(mbq);
                    while hasdata(mbq)
                        iteration = iteration+1;
                        [X,C,W] = next(mbq);
                        [~,gradientsB,gradientsL] = dlfeval(@loss_fun,this.bijector,this.latent,X,C,W,penalty);
                        [this.bijector,this.trailingAvgB,this.trailingAvgSqB] = adamupdate(this.bijector,gradientsB,...
                            this.trailingAvgB,this.trailingAvgSqB,iteration,learnRate);
                        if ~isempty(gradientsL)
                            [this.latent,this.trailingAvgL,this.trailingAvgSqL] = adamupdate(this.latent,gradientsL,...
                                this.trailingAvgL,this.trailingAvgSqL,iteration,1e-2); % fix this for now
                        end
                        if ~isempty(debug_cb)
                            debug_cb(this,varargin{:});
                        end
                    end
                end
                if epoch == 0 || epoch >= length(this.losses)
                    this.losses(end+1) = neg_log_prob(this,inputs);
                    if ~isfinite(this.losses(end))
                        disp(strjoin(["Training stopping after epoch ",num2str(epoch)," because training loss diverged."]));
                        break;
                    end
                    if ~isempty(validation)
                        this.val_losses(end+1) = neg_log_prob(this,validation);
                        fprintf("Epoch: %i, LearnRate: %f, Loss: %f, Validation Loss: %f\n",epoch,learnRate,this.losses(end),this.val_losses(end));
                        l = wd*this.losses(end)+wv*this.val_losses(end);
                        if l <= best{1}
                            best = { l this.bijector this.latent };
                        end
                    else
                        fprintf("Epoch: %i, LearnRate: %f, Loss: %f\n",epoch,learnRate,this.losses(end));
                        if this.losses(end) <= best{1}
                            best = { this.losses(end) this.bijector this.latent };
                        end
                    end
                    if ~isempty(debug_cb)
                        debug_cb(this,varargin{:});
                    end
                end
            end
            if ~isempty(best{2})
                this.bijector = best{2};
                this.latent = best{3};
            end
            losses = this.losses;
            val_losses = this.val_losses;
        end
    end
    methods (Access = private)
        function [X,C,W] = preprocessData(this, batch)
            batch = cat(1,batch{:});
            X = batch{:,this.data_columns};
            C = get_conditions(this,batch);
            W = get_weights(this,batch);
        end
    end
    methods (Static)
        function [x, u] = sample_internal(nsample, bijector, latent, conditions, u)
            conditions = dlarray(conditions,"BC");
            if isempty(u)
                if ~isempty(latent.Learnables)
                    if numel(latent.InputNames) == 2
                        [x, u] = sample(latent.Layers(1),conditions);
                    else
                        [x, u] = sample(latent.Layers(1),nsample);
                    end
                else
                    u = sample(latent.Layers(1),nsample);
                    x = u;
                end
            else
                if ~isempty(latent.Learnables)
                    if numel(latent.InputNames) == 2
                        x = inverse(latent.Layers(1),u,conditions);
                    else
                        x = inverse(latent.Layers(1),u);
                    end
                else
                    x = u;
                end
            end
            x = dlarray(x,"BC");
            for i = numel(bijector.Layers):-1:1
                l = bijector.Layers(i);
                if ~isa(l,"bijectors.Bijector")
                    continue;
                end
                x = bijector.Layers(i).inverse(x,conditions);
            end
        end
    end
end

function [loss,gradientsB,gradientsL] = loss_fun(bijector, latent, x, c, w, penalty_scale)
    [bijector,log_probs,penalty,r] = log_probs_internal(bijector,latent,x,c);
    [~,loss] = std(-log_probs,w,2); % fake weighted mean
    r = (r-eye(size(r))).^2./size(r,1).^2; % we want zero correlation
    [gradientsB,gradientsL] = dlgradient(loss+penalty_scale*mean(penalty,"all")+penalty_scale*sum(r,"all"),bijector.Learnables,latent.Learnables);
end

function [bijector, log_probs, penalty, r] = log_probs_internal(bijector, latent, x, conditions)
    if isempty(conditions)
        [x,state] = forward(bijector,x);
    else
        [x,state] = forward(bijector,x,conditions);
    end
    bijector.State = state;
    if numel(latent.InputNames) == 2
        log_probs = forward(latent,x,conditions);
    else
        log_probs = forward(latent,x); % actually latent.Layer(1).log_prob(x)
    end
    penalty = zeros(size(log_probs));
    r = correl(x);
    for i = 1:height(state)
        if strcmp(state{i,"Parameter"},"log_det")
            log_probs = log_probs+state{i,"Value"}{1};
        end
        if strcmp(state{i,"Parameter"},"penalty")
            penalty = penalty+state{i,"Value"}{1};
        end
    end
    log_probs(isnan(log_probs)) = -inf;
end

function r = correl(x)
    r = cov(stripdims(x)');
    d = sqrt(r(logical(eye(size(r)))));
    r = r./d./d';
    r = (r+r')/2;
end
