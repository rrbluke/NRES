function multiplot(varargin)

    AC.vars = varargin;
    AC.names = {};
    
    for i=1:nargin
        AC.names{i} = inputname(i);
        AC.vars{i} = double(squeeze(AC.vars{i}));
    end


    [nbin, nfram] = size(AC.vars{1});

    fs = 16e3;
    wlen = 1024;
    shift = 512;
    framerate = fs/shift;
    
    fvect = (0:nbin-1)*fs*0.5/(nbin-1);
    tvect = (0:nfram-1)/framerate;


    %init figure
    close all
    figsize = [1024 768];
    scnsize = get(0,'ScreenSize');
    center = scnsize(3:4)/2-figsize/2;
    f = figure;
    set(f,'Position',[center figsize])

    AC.pop = uicontrol('Units','normalized','style','popupmenu','position',[0.91 0.87 0.07 0.1],...
              'string',AC.names,'callback',@multiplot_callback);
          
    AC.axes = axes('units','normalized','position',[0.06 0.07 0.915 0.85]);
    

    axes(AC.axes)
    zoom on
    AC.plot(1) = pcolor(tvect,fvect,zeros(nbin,nfram));
    shading flat
    xlabel('time [s]')
    ylabel('frequency [Hz]')
    colorbar
    colormap jet

    
    set(gcf,'Userdata',AC);
    multiplot_callback

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function multiplot_callback(h,eventdata)

    AC = get(gcf,'userdata');
    
    axes(AC.axes)

    
    i = get(AC.pop,'value');

    data = AC.vars{i};
    if ~isreal(data)
        data = 20*log10(abs(data) + 1e-9);
    end
    
    set(AC.plot(1),'CData', data );
    set(AC.plot(1),'ZData', data );
    set(get(AC.axes(1),'Title'), 'String', strrep(AC.names{i}, '_', '\_'))
    
    
    x = data(:);
    x = sort(x);
    xmax = x(round(length(x)*0.99));
    xmin = x(round(length(x)*0.10));
    
    xmin = floor(xmin);
    xmax = ceil(xmax);
    
    if length(AC.names{i}) > 1 && AC.names{i}(1) == 'F'
        xmin = -50;
        xmax = 30;
    end

    set(AC.axes(1),'CLim', [xmin xmax]);


end


