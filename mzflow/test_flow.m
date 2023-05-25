rng("default") % For reproducibility

close all;

X = betarnd(2,2,10000,1);
data = array2table(X,"VariableNames","x");

fig = figure;
hold on;
histogram(X,20,"Normalization","pdf");
flow = Flow("x");
flow = flow.set_default_bijector(data,2); % set just two bins
[samples,presampled] = flow.sample(1000);
h = histogram(samples{:,1},20,"Normalization","pdf");
hold off;
[flow,losses] = flow.train(data,[],10,[],@debug_cb1,h,presampled);
fig2 = figure;
ax2 = axes(fig2);
plot(losses,'Parent',ax2);
samples = flow.sample(10000);
h.Data = samples{:,1};
grid = linspace(0,1,101);
pdf = flow.posterior(data,"x",grid);
fig3 = figure;
ax3 = axes(fig3);
plot(grid,pdf,'Parent',ax3);
pause(10);
close([fig fig2 fig3]);

[X,labels] = twomoons(10000);
data = array2table([X labels],"VariableNames",["x","y","label"]);
test = array2table(unique(labels),"VariableNames","label");

fig = figure;
hold on;
scatter(X(:,1),X(:,2),[],"blue","filled")
flow = Flow(["x", "y"]);
flow = flow.set_default_bijector(data);
[samples,presampled] = flow.sample(1000);
s = scatter(samples{:,1},samples{:,2},[],"red","filled");
hold off;
[flow,losses] = flow.train(data,[],100,[],@debug_cb2,s,presampled);
fig2 = figure;
ax2 = axes(fig2);
plot(losses,'Parent',ax2);
samples = flow.sample(10000);
s.XData = samples{:,1};
s.YData = samples{:,2};
%grid = jnp.linspace(-2, 2, 100);
%pdfs = flow.posterior(data, column="x", grid=grid);
pause(10);
close([fig fig2]);

fig = figure;
hold on;
cm = colormap(parula(2*height(test)));
scatter(X(:,1),X(:,2),[],cell2mat(arrayfun(@(x) cm(x,:),2*labels-1,'UniformOutput',false)),"filled")
flow = Flow(["x", "y"],"label");
flow = flow.set_default_bijector(data);
[samples,presampled] = flow.sample(1000,test);
s = scatter(samples{:,1},samples{:,2},[],cell2mat(arrayfun(@(x) cm(x,:),2*samples{:,3},'UniformOutput',false)),"filled");
hold off;
[flow,losses] = flow.train(data,[],100,[],@debug_cb3,s,presampled,test);
fig2 = figure;
plot(losses);
samples = flow.sample(10000,test);
s.XData = samples{:,1};
s.YData = samples{:,2};
s.CData = cell2mat(arrayfun(@(x) cm(x,:),2*samples{:,3},'UniformOutput',false));
%grid = jnp.linspace(-2, 2, 100);
%pdfs = flow.posterior(data, column="x", grid=grid);
pause(10);
close([fig fig2]);

function debug_cb1(f,h,presampled)
    samples = f.sample([],[],[],presampled);
    h.Data = samples{:,1};
    drawnow;
end

function debug_cb2(f,s,presampled)
    samples = f.sample([],[],[],presampled);
    s.XData = samples{:,1};
    s.YData = samples{:,2};
    drawnow;
end

function debug_cb3(f,s,presampled,test)
    samples = f.sample([],test,[],presampled);
    s.XData = samples{:,1};
    s.YData = samples{:,2};
    drawnow;
end
