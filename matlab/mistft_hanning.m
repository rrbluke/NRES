
function x = mistft_hanning(Fx)


[nbin, nfram, nchan] = size(Fx);
wlen = (nbin-1)*2;
shift = wlen/2;

w = create_hanning_window(wlen);

samples = nfram*shift+wlen-shift;


x = zeros(samples, nchan);

for t=1:nfram
    
    idx = (t-1)*shift+1:(t-1)*shift+wlen;
    
    for c=1:nchan
        Fx0 = Fx(:,t,c);
        Fx0(wlen/2+2:wlen) = flipud(conj(Fx0(2:wlen/2)));
        x0 = ifft(Fx0).*w;
        x(idx,c) = x(idx,c) + x0;
    end
    
end



