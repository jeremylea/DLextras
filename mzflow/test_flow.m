rng("default") % For reproducibility

close all;

X = betarnd(2,2,10000,1);
data = array2table(X,"VariableNames",{"x"});

fig = figure;
hold on;
histogram(X,20,"Normalization","pdf");
flow = Flow("x");
flow = flow.set_default_bijector(data);
flow.sample(1);
[flow,losses] = flow.train(data,[],10);
samples = flow.sample(10000);
histogram(samples{:,1},20,"Normalization","pdf");
%grid = jnp.linspace(-2, 2, 100);
%pdfs = flow.posterior(data, column="x", grid=grid);
hold off;
pause
close(fig);

[X,label] = twomoons(10000);
data = array2table([X label],"VariableNames",{"x","y","label"});
test = array2table(unique(label),"VariableNames",{"label"});

scatter(X(:,1),X(:,2),[],"red","filled")
%title("Labeled Data")
drawnow;
pause(1);
flow = Flow(["x", "y"]);
flow = flow.set_default_bijector(data);
flow.sample(1);
[flow,losses] = flow.train(data);
samples = flow.sample(10000,test);
scatter(samples{:,1},samples{:,2},[],"red","filled");
%grid = jnp.linspace(-2, 2, 100);
%pdfs = flow.posterior(data, column="x", grid=grid);
pause

scatter(X(:,1),X(:,2),[],label,"filled")
flow = Flow(["x", "y"],"label");
flow = flow.set_default_bijector(data);
flow.sample(1,test);
[flow,losses] = flow.train(data);
samples = flow.sample(10000,test);
scatter(samples{:,1},samples{:,2},[],samples{:,3},"filled");
%grid = jnp.linspace(-2, 2, 100);
%pdfs = flow.posterior(data, column="x", grid=grid);
pause
