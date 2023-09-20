function plot_tsne(ydata, L)

Y_toplot = L; ydata_toplot = ydata;
num_types_toplot = unique(Y_toplot); num_types_toplot = num_types_toplot(~isnan(num_types_toplot));
% markers_toplot = {'o', '+', 'd', '*', '.', 'x'};
markers_toplot = {'s', 'o', 'd', 'p', 'o'}; % Last one used to be 'h'
figure('units','normalized','outerposition',[0 0 1 1]); 
for i = 1:length(num_types_toplot)
    curr_points = Y_toplot == num_types_toplot(i);
    if num_types_toplot(i) ~= 99
        scatter3(ydata_toplot(curr_points,1),ydata_toplot(curr_points,2),ydata_toplot(curr_points,3),15,'filled',markers_toplot{i}); hold on;
    else
        scatter3(ydata_toplot(curr_points,1),ydata_toplot(curr_points,2),ydata_toplot(curr_points,3),20,'filled',markers_toplot{i}); hold on;
        ColOrd = get(gca,'ColorOrder');
        plot3(ydata_toplot(curr_points,1),ydata_toplot(curr_points,2),ydata_toplot(curr_points,3),'Color',ColOrd(i,:),'LineWidth',0.7); hold on;
    end    
end
if length(num_types_toplot) == 4
    legend({'A', 'B', 'C', 'D'});
elseif length(num_types_toplot) == 5
    legend({'A', 'B', 'C', 'D','t'});
end

% ,'MarkerEdgeColor','k'
%     ydata = tsne(X, [], 3, 45, 30); scatter3(ydata(curr_points,1),ydata(curr_points,2),ydata(curr_points,3),15,Y_toplot(curr_points),'filled',markers_toplot{i}); hold on;