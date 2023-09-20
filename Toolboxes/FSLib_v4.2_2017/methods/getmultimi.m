function c = getmultimi(da, dt, weights)
c = zeros(size(da,2),1);
for i = 1:size(da,2)
    c(i) = WeightedMIToolboxMex(4, weights, da(:,i), dt);
    % c(i) = mutualinfo(da(:,i), dt);
end