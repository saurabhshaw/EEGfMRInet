function [EEG] = interp_null_values(EEG)

%% interp null values -- DEPRECATED, DO NOT USE
for y = 1:EEG.nbchan
    subset = EEG.data(y,:);
    % subset = EEG.data(y,1:500);
    if sum(isnan(subset)) > 0.7*length(subset)
        [EEG] = pop_select(EEG,'nochannel',y); EEG = eeg_checkset(EEG);
    else
        subset_nan_idx = find(isnan(subset));
        if ~isempty(subset_nan_idx)
            for k = 1:length(subset_nan_idx)
                interp_range = 5;
                if isequal(subset_nan_idx(k),EEG.pnts) || isequal(subset_nan_idx(k),1)
                    EEG.data(y,subset_nan_idx(k)) = 0.0;
                    subset = EEG.data(y,:);
                    continue;
                elseif subset_nan_idx(k) + interp_range > EEG.pnts || subset_nan_idx(k) - interp_range < 1
                    interp_range = 3;
                    if subset_nan_idx(k) + interp_range > EEG.pnts || subset_nan_idx(k) - interp_range < 1
                        EEG.data(y,subset_nan_idx(k)) = 0.0;
                        subset = EEG.data(y,:);
                        continue;
                    end
                end
                interp_x = [subset_nan_idx(k)-interp_range:subset_nan_idx(k)-1,subset_nan_idx(k)+1:subset_nan_idx(k)+interp_range];
                interp_y = subset([subset_nan_idx(k)-interp_range:subset_nan_idx(k)-1,subset_nan_idx(k)+1:subset_nan_idx(k)+interp_range]);


                % nan_locs_to_interp = subset_nan_idx(k);
                if sum(isnan(interp_y)) > 0
                    replace_interp = find(isnan(interp_y));
                    % nan_locs_to_interp = [nan_locs_to_interp,interp_x(replace_interp)];
                    for p = 1:sum(isnan(interp_y))
                        u = replace_interp(1);
                        interp_x(u) = [];
                        interp_y(u) = [];
                        replace_interp = find(isnan(interp_y));
                    end

                end


                EEG.data(y,subset_nan_idx(k)) = interp1(interp_x,interp_y,subset_nan_idx(k),'linear'); EEG = eeg_checkset(EEG); % linear or spline??
                subset = EEG.data(y,:);
                % view run status
                print = sum(isnan(subset));
                disp(print);
                disp(y);
                disp(subset_nan_idx(k));
                % subset = EEG.data(y,1:500);
            end
        end
    end
end
