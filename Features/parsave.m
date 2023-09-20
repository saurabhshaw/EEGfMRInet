function parsave(fname, analyzedData, varargin)
    dumvar = 0; 
    if isempty(varargin)
        save(fname,'dumvar','analyzedData','-v7.3');
    else
        varargin1 = varargin{1};
        save(fname,'dumvar','analyzedData','varargin1','-v7.3');
    end
    
end