function parsave_struct(fname, analyzedData, toappend)
dumvar = 0;
if toappend
    save(fname,'-struct','analyzedData','-v7.3','-append'); % Removed 'dumvar', '-v7.3'
else
    save(fname,'-struct','analyzedData','-v7.3'); % Removed 'dumvar', '-v7.3'
end
end