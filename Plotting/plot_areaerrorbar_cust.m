% ----------------------------------------------------------------------- %
% Function plot_areaerrorbar plots data filling the space between the     %
% positive and negative errors using a semi-transparent background        %
%                                                                         %
%   Input parameters:                                                     %
%       - xdata:    x-axis data matrix, with rows corresponding to observations  %
%                   and columns to samples.                               %
%       - ydata:    y-axis data matrix, with rows corresponding to observations  %
%                   and columns to samples.                               %
%       - error:    error data matrix, with rows corresponding to observations  %
%                   and columns to samples.                               %
%       - options:  (Optional) Struct that contains the customized params.%
%           * options.handle:       Figure handle to plot the result.     %
%           * options.color:        RGB color of the filled area/line.    %
%           * options.alpha:        Alpha value for transparency.         %
%           * options.line_width:   Mean line width.                      %
%           * options.hide_line:    Hide/Show the line.                   %
% ----------------------------------------------------------------------- %
%   Example of use:                                                       %
%       data = repmat(sin(1:0.01:2*pi),100,1);                            %
%       data = data + randn(size(data));                                  %
%       plot_areaerrorbar(data);                                          %
% ----------------------------------------------------------------------- %
%   Author:  Victor Martinez-Cagigal                                      %
%   Date:    30/04/2018                                                   %
%   E-mail:  vicmarcag (at) gmail (dot) com                               %
% ----------------------------------------------------------------------- %
function plot_areaerrorbar_cust(xdata, ydata, error, options)

    % Default options
    if(nargin<4)
        options.handle     = figure(1);
        options.color = [128 193 219]./255;    % Blue theme       
        %options.color = [243 169 114]./255;    % Orange theme
        options.alpha      = 0.5;
        options.line_width = 2;
        options.hide_line = 0;
    end
    
    % Plotting the result
    figure(options.handle);
    x_vector = [xdata, fliplr(xdata)];
    patch = fill(x_vector, [ydata+error, fliplr(ydata-error)], options.color);
    % patch = fill(xdata, [ydata+error, ydata-error], options.color);
    set(patch, 'edgecolor', 'none');
    set(patch, 'FaceAlpha', options.alpha);
    hold on;
    if ~options.hide_line
        plot(xdata, ydata, 'color', options.color, ...
        'LineWidth', options.line_width);
    else
        plot(xdata, ydata, 'color', options.color, ...
        'LineStyle','none');
    end
    % hold off;
    
end