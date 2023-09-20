function h = plot_network_timecourse(x_data,net_to_analyze,title_text,net_idx,trial_avg_data,trial_std_data,colorord,plot_errdisp)

% Plot the figure:
h = figure; hold on; % plot(nan(1,25)); % Normalized
% plot(x_data,trial_tc*0.02); legend_text = cell(1,2+length(net_to_analyze)); legend_text = {[conditions{condition_idx} ' Condition']};
legend_text = cell(1); % plot_colorord = get(gca,'ColorOrder');
for i = 1:length(net_to_analyze)
    plot_ydata = smooth(sum(trial_avg_data(:,net_idx{i}),2));
    plot(x_data,plot_ydata,'Color',colorord(i,:));
    legend_text = [legend_text, [net_to_analyze{i}]];
end
% ylim([-0.008 0.01]);
legend(legend_text(~cellfun('isempty',legend_text)),'AutoUpdate', 'off');
switch plot_errdisp
    case 'bar'
        errbar_subsample = 3;
        for i = 1:length(net_to_analyze)
            plot_ydata = smooth(sum(trial_avg_data(:,net_idx{i}),2));
            % errorbar(x_data(1:errbar_subsample:end),smooth(sum(trial_avg_data(1:errbar_subsample:end,net_idx{i}),2)),smooth(mean(trial_std_data(1:errbar_subsample:end,net_idx{i}),2)),'LineStyle','none','Color',plot_colorord(i,:));
            errorbar(x_data(1:errbar_subsample:end),plot_ydata(1:errbar_subsample:end),smooth(mean(trial_std_data(1:errbar_subsample:end,net_idx{i}),2)),'LineStyle','none','Color',colorord(i,:));
        end
        
    case 'area'
        options = []; options.handle = h; options.alpha = 0.2; options.line_width = 2; options.hide_line = 1;
        for i = 1:length(net_to_analyze)
            plot_ydata = smooth(sum(trial_avg_data(:,net_idx{i}),2)); options.color = colorord(i,:);
            plot_areaerrorbar_cust(x_data, plot_ydata', smooth(mean(trial_std_data(:,net_idx{i}),2))', options)
        end        
end
title(title_text);  
xlabel('Time(s)'); xline(0,'--'); yline(0);
