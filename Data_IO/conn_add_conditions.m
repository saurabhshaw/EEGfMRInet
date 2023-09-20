
% Input:
% sub_onset
% sub_duration
% study_conditions
% study_CONNfilename
% study_name
% base_path_data

% Takes CONN_x as input and modificed conditions and durations:
% num_subjects = length(CONN_x.Setup.conditions.values);
% num_conditions = length(CONN_x.Setup.conditions.allnames);
num_subjects = length(sub_onset);
num_conditions = length(study_conditions);
table_struct = [];
table_struct.condition_name= []; table_struct.subject_number= []; table_struct.session_number= [];
table_struct.onsets= []; table_struct.durations= [];

% load(study_CONNfilename)
% CONN_x.Setup.conditions.model{idx}=[];
% CONN_x.Setup.conditions.param(idx)=0;
% CONN_x.Setup.conditions.filter{idx}=[];
% CONN_x.Setup.conditions.names{idx}=name;



for i = 1:num_subjects
   for j = 1:num_conditions
       num_sessions = length(sub_onset{i});       
       for k = 1:num_sessions
           for m = 1:length(sub_onset{i}{k}{j})
               table_struct.condition_name = cat(1,table_struct.condition_name,study_conditions(j));
               table_struct.subject_number = cat(1,table_struct.subject_number,i);
               table_struct.session_number = cat(1,table_struct.session_number,k);
               table_struct.onsets = cat(1,table_struct.onsets,sub_onset{i}{k}{j}(m));
               table_struct.durations = cat(1,table_struct.durations,sub_duration{i}{k}{j}(m));
               % CONN_x_Setup.onsets{i}{j}{k} = sub_onset{i}{k}{j};
               % CONN_x_Setup.durations{i}{j}{k} = sub_duration{i}{k}{j};
           end
       end      
   end   
end
table_data = struct2table(table_struct);
excel_name = ['CONN_Info_' study_name];
writetable(table_data,[base_path_data filesep excel_name '.csv']);

%% Import the conditions from the written .CSV file:
load(study_CONNfilename)

out = conn_importcondition_mod([base_path_data filesep excel_name '.csv'],study_conditions,'deleteall',true);
CONN_x = out;
save(study_CONNfilename,'CONN_x');