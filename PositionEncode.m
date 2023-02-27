function P = PositionEncode(n_t, n_c, N)
    if nargin < 3
        N = 10000;
    end
    if isscalar(n_t)
        P = (1:n_t)'-1;
    else
        P = n_t(:);
        n_t = length(P);
    end
    while 10*N < max(P)-min(P)
        N = 10*N;
    end
    A = P ./ N.^((0:2:n_c-1)./n_c);
    P = zeros(n_t,n_c);
    P(:,1:2:n_c) = sin(A);
    if 2*floor(n_c/2) == n_c
        P(:,2:2:n_c) = cos(A);
    else
        P(:,2:2:n_c) = cos(A(:,1:end-1));
    end
return
