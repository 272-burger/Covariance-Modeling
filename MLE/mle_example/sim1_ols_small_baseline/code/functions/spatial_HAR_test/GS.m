%% Gram-Schmidt procedure

function [q, r] = GS(A)
        [m, n] = size(A);

        q = zeros(m, n);
        r = zeros(n, n);

        for k = 1:n
                r(k,k) = norm(A(1:m, k));

                if r(k,k) == 0
                        break;
                end

                q(1:m, k) = A(1:m, k) / r(k,k);

                for j = k+1:n
                        r(k, j) = dot(q(1:m, k), A(1:m, j));
                        A(1:m, j) = A(1:m, j) - r(k, j) * q(1:m, k);
                end
        end