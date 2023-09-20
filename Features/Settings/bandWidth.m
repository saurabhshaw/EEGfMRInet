function range = bandWidth(selection)

switch selection
    case 'full'
        range = [1 50];
    case 'delta'
        range = [1 4];
    case 'theta'
        range = [4 8];
    case 'alpha'
        range = [8 13];
    case 'beta'
        range = [13 30];        
    case 'gamma'
        range = [30 50];
    case 'high_gamma'
        range = [50 70];
    otherwise
        range = [1 50];
        
end