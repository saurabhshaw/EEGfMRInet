function [testIdx, trainIdx] = testTrainIdx_overlap(numEpochs,window_len,testTrainPartition)

testIdxLen = floor(numEpochs*testTrainPartition);
testIdxRange = numEpochs-testIdxLen;
startTestIdx = floor(rand*(testIdxRange+1))+1; 
testIdx = [startTestIdx:startTestIdx+testIdxLen-1];
exclTestIdx = [max(1,testIdx(1)-(window_len - 1)):min(numEpochs,testIdx(end)+(window_len - 1))]; 
trainIdx = [1:numEpochs];
if ~isinteger(exclTestIdx) exclTestIdx = int64(exclTestIdx); end
trainIdx(exclTestIdx)=[];
end