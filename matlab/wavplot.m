
function wavplot(varargin)

    % default samplerate = 16kHz
    fs = 16e3;
    
    % read wavs
    x = [];
    for n=1:nargin
        z = varargin{n};
        
        dims = size(z);
        if numel(dims) > 2
            disp('wav data must be 1D or 2D')
            return
        end
        if dims(2) > dims(1)
            z = z.';
        end
        if numel(z) == 1
            fs = z;
            sprintf('Updated fs to: %d', fs)
        else
            if size(x,1)>0 && size(x,1) ~= size(z,1)
                sprintf('wav data has different length: %d and %d', size(x,1), size(z,1))
                return
            else
                x = [x z];
            end
        end
        
    end
        
  

    %draw figure
    figure
    set(gcf, 'color', [0.5 0.5 1]);
    set(gcf, 'windowbuttondownfcn', @mouse_down);


    [samples, n_wavs] = size(x);
    for n=1:n_wavs
        draw_subplot(n_wavs, n, x(:,n), fs);
    end

end



function draw_subplot(n_wavs, n, x, fs)

    samples = numel(x);
    tvect = linspace(0, samples/fs, samples);
    
    % draw subplot for x
    subplot(n_wavs,1,n)
    plot(tvect, x, 'color', [0 0 1]);
    set(gca, 'color', [0.8 0.8 1]);
    xlim([0, tvect(end)]);
    ylim([-1,1]);
    grid on
    pos = get(gca, 'Position');
    pos(1) = 0.055;
    pos(3) = 0.9;
    set(gca, 'Position', pos);    
    
    % draw vertical line
    h_line = line([0,0], [-1,1], 'color', [0 0.3 0.3]);
    set(h_line, 'Visible', 'off')

    % instantiate player
    h_player = audioplayer(x, fs);
    set(h_player, 'StartFcn', @player_start);
    set(h_player, 'TimerFcn', @player_active);
    set(h_player, 'StopFcn', @player_stop);
    set(h_player, 'UserData', h_line)
    
    data.h_line = h_line;
    data.h_player = h_player;
    set(gca, 'UserData', data);
    
end




function mouse_down(hObject, eventdata)

    data = get(gca, 'UserData');
    h_player = data.h_player;
    
    % get left mouse button press
    if strcmp(get(gcf, 'selectiontype'), 'normal')
        if strcmp(get(h_player, 'Running'), 'off')
            play(h_player)
        else
            stop(h_player)
        end
    end
    
end



function player_start(hObject, eventdata)

    % make line visible
    h_line = get(hObject, 'UserData');
    set(h_line, 'Visible', 'on')

end



function player_active(hObject, eventdata)

    % update line with current player time
    h_line = get(hObject, 'UserData');
    samples = get(hObject, 'CurrentSample');
    fs = get(hObject, 'SampleRate');
    t = samples/fs;
    set(h_line, 'XData', [t,t]);
   
end



function player_stop(hObject, eventdata)

    % make line invisible
    h_line = get(hObject, 'UserData');
    set(h_line, 'Visible', 'off')
    set(h_line, 'XData', [0,0]);

end



