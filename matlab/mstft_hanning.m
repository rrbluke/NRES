
function Fx = mstft_hanning(x, wlen)

nbin = wlen/2+1;
shift = wlen/2;

samples = length(x);

w = create_hanning_window(wlen);

nfram = ceil( (samples-wlen+shift)/shift );
samples = nfram*shift+wlen-shift;


%zero-pad input
x(end+1:samples) = 0;

Fx = zeros(nbin, nfram);


for t=1:nfram
    
    idx = (t-1)*shift+1:(t-1)*shift+wlen;
    
    Fx0 = fft(x(idx).*w.');
    Fx(:,t) = Fx0(1:nbin);
    
end



