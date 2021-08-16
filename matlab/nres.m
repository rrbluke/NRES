clear all
close all
clc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


load('../predictions/nres.mat');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fs = 16e3;
wlen = 1024;
shift = 512;
[nbin, nfram] = size(Fs);
tvect = (0:nfram-1)*shift/fs;
fvect = (0:nbin-1)*0.5*fs/(nbin-1);

gain = 10;
Fd = Fd*gain;
Fy = Fy*gain;
Fs = Fs*gain;

Fe1 = Fd-Fy;
Fz1 = Fe1.*p_erle;

Fd2 = Fd+Fs;
Fe2 = Fd2-Fy;
Fz2 = Fe2.*p_sdr;

e1 = mistft_hanning(Fe1);
z1 = mistft_hanning(Fz1);
e2 = mistft_hanning(Fe2);
z2 = mistft_hanning(Fz2);
s = mistft_hanning(Fs);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


multiplot(Fe1, Fz1, p_erle, Fe2, Fz2, Fs, p_sdr)
wavplot(e1, z1, e2, z2, s)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
break


figure
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'Position', [1 1 1500 500]);
set(gcf, 'PaperPosition', [0 0 30 10]);
set(gcf, 'renderer', 'zbuffer');
pcolor(tvect, fvect, 20*log10(abs(double(Fd2))));
% pcolor(tvect, fvect, 20*log10(abs(double(Fe2))));
% pcolor(tvect, fvect, 20*log10(abs(double(Fz2))));
% pcolor(tvect, fvect, 20*log10(abs(double(Fs))));
colorbar
set(gca, 'CLim', [-40 40])
shading flat
xlabel('Time  [s]', 'fontsize', 16)
ylabel('Frequency  [Hz]', 'fontsize', 16)
set(gca,'FontSize', 16)

saveas(gcf, 'NRES_Fd2.png')
% saveas(gcf, 'NRES_Fe2.png')
% saveas(gcf, 'NRES_Fz2.png')
% saveas(gcf, 'NRES_Fs.png')



