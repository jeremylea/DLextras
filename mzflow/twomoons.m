function [X,label,Z] = twomoons(n) % Generate two moons, with n points in each moon.

% Specify the radius and relevant angles for the two moons.
noise = 0.05;
radius = 1;
angle1 = pi;  %+ pi/10;
angle2 = 0; %pi/10;

% Create the bottom moon with a center at (1,0).
bottomTheta = linspace(-angle1,angle2,n)';
bottomX1 = radius.*cos(bottomTheta) + 1.0;
bottomX2 = radius.*sin(bottomTheta) + 0.5;

% Create the top moon with a center at (0,0).
topTheta = linspace(angle1,-angle2,n)';
topX1 = radius.*cos(topTheta);
topX2 = radius.*sin(topTheta);

% Return the moon points and their labels.
X = [bottomX1 bottomX2; topX1 topX2];
label = [ones(n,1); 2*ones(n,1)];

idx = randperm(numel(label));
X = X(idx, :);
label = label(idx);

e = randn(size(X));
X = X + noise * e;
Z = mvnpdf(e);

end
