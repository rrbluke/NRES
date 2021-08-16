
function w = create_hanning_window(N)

    w = zeros(N,1);

    for i=1:N
        w(i) = 0.5*(1-cos(2*pi*i/(N+1)));
    end

    w = sqrt(w);
