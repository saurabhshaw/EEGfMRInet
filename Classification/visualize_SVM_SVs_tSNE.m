function varargout = visualize_SVM_SVs_tSNE(data, data_labels, support_vectors, dim)

marker_size = 20;

% Standardize the data:
data = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));

% t-SNE Plotting
Y_toplot = data_labels;
num_types_toplot = unique(Y_toplot);
if issparse(support_vectors)
    X = cat(1,data,full(support_vectors));
else
    X = cat(1,data,support_vectors);
end

% markers_toplot = {'o', '+', 'd', '*', '.', 'x'};
% markers_toplot = {'s', 'o', 'd', 'p', 'h'};
ydata = tsne(X,[],dim,30,30); % was 45, 30
ydata_onlyData = ydata(1:size(data,1),:);
ydata_SV = ydata(size(data,1)+1 : end,:);
figure; 
for i = 1:length(num_types_toplot)
    curr_points = Y_toplot == num_types_toplot(i);
    if dim == 3
        scatter3(ydata_onlyData(curr_points,1),ydata_onlyData(curr_points,2),ydata_onlyData(curr_points,3),marker_size,'filled','MarkerEdgeColor','k','DisplayName',num2str(num_types_toplot(i))); hold on;
    elseif dim == 2
        scatter(ydata_onlyData(curr_points,1),ydata_onlyData(curr_points,2),marker_size,'filled','MarkerEdgeColor','k','DisplayName',num2str(num_types_toplot(i))); hold on;
    end
end

% Plot Support Vectors
if dim == 3
    scatter3(ydata_SV(:,1),ydata_SV(:,2),ydata_SV(:,3),marker_size*2,'MarkerEdgeColor','k','DisplayName','SV'); hold on;
elseif dim == 2
    scatter(ydata_SV(:,1),ydata_SV(:,2),marker_size*2,'MarkerEdgeColor','k','DisplayName','SV'); hold on;
end

legend_cell = mat2cell(num_types_toplot,repmat([1],[size(num_types_toplot,1) 1]),repmat([1],[1 size(num_types_toplot,2)]));
legend_cell = cellfun(@(x)num2str(x),legend_cell,'un',0);
% legend_cell = {legend_cell,'SV'};
legend(legend_cell);
% legend

varargout{1} = ydata;

    %ydata = tsne(X, [], 3, 45, 30); scatter3(ydata(curr_points,1),ydata(curr_points,2),ydata(curr_points,3),15,Y_toplot(curr_points),'filled',markers_toplot{i}); hold on;

% print('-djpeg','-r500',['tSNE_Sub' num2str(subject_number) '_Ses' num2str(session_number)]);
