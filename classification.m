function net=classification()
outputFolder = fullfile('Hittite');
testFolder = fullfile('test');
categories = {'SU', 'EKMEK'};

imdsTrain = imageDatastore(fullfile(outputFolder,categories),'LabelSource','foldernames');
imdsTest = imageDatastore(fullfile(testFolder,categories), 'LabelSource', 'foldernames');

tbl = countEachLabel(imdsTrain);

SU = find(imdsTrain.Labels == 'SU',1);
EKMEK = find(imdsTrain.Labels == 'EKMEK',1);

layers = [
    imageInputLayer([64 64 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'Momentum',0.4,...
    'LearnRateSchedule','piecewise', ...
    'InitialLearnRate',0.00001, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsTest, ...
    'MiniBatchSize',256,...
    'ValidationFrequency',20, ...
    'Verbose',false, ...
    'Plots','training-progress');

gpuDeviceCount;

d = gpuDevice;

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest)

end











































