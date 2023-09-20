
% Takes CONN_x as input and modificed conditions and durations:
num_subjects = length(CONN_x.Setup.conditions.values);
num_conditions = length(CONN_x.Setup.conditions.allnames);

% THIS IS OUR STUDY SPECIFIC -
for i = 1:num_subjects
    for j = 1:num_conditions
        k = 5;
        if (j == 3) || (j == 7)
            sub_trial_data_relevant{i}.block_condition_onset_vect{k}{j} = 0;
            sub_trial_data_relevant{i}.block_condition_duration_vect{k}{j} = Inf;
        end
    end
end

for i = 1:num_subjects
   for j = 1:num_conditions
       num_sessions = length(CONN_x.Setup.conditions.values{i}{j});       
       for k = 1:num_sessions
           CONN_x.Setup.conditions.values{i}{j}{k}{1} = sub_trial_data_relevant{i}.block_condition_onset_vect{k}{j};
           CONN_x.Setup.conditions.values{i}{j}{k}{2} = sub_trial_data_relevant{i}.block_condition_duration_vect{k}{j}; 
       end      
   end   
end