numUsers = 10;              % Total number of users
numFeatures = 43;          % Number of features per sample
overallAccuracy = [];       % Store accuracy for all users
overallTrainAccuracy = [];
overallFAR = [];
overallFRR = [];
overallEER = [];

% Loop through each user for training and testing
for userID = 1:numUsers
    
    % Load training template for the current user
    trainDataFile = sprintf('User%02d_FD_train_Template.mat', userID);
    trainDataStruct = load(trainDataFile);
    Input_Features = trainDataStruct.userTemplate(:, 1:numFeatures);
    trainLabels = trainDataStruct.userTemplate(:, end); % Labels (1 or 0)

    hiddenLayerSize = [30,20,15]; 
    % Train the neural network
    net = feedforwardnet(hiddenLayerSize); % Single hidden layer NN
    
    % Set deterministic behavior
    net.divideParam.trainRatio = 1.0; % No data division; use all data for training
    net = train(net, Input_Features', trainLabels');
    
    % Load testing template for the current user
    testDataFile = sprintf('User%02d_FD_test_Template.mat', userID);
    testDataStruct = load(testDataFile);
    testData = testDataStruct.userTemplate(:, 1:numFeatures);
    testLabels = testDataStruct.userTemplate(:, end); % True labels for evaluation
    
    % Test the neural network
    output = net(testData');
    predictedLabels = round(output); % Convert scores to binary (0 or 1)


% Calculate FAR and FRR at multiple thresholds
    genuineScores = output(testLabels == 1); % Scores for genuine samples
    impostorScores = output(testLabels == 0); % Scores for impostor samples

    thresholds = linspace(0, 1, 100); % Generate thresholds from 0 to 1
    FAR = zeros(1, length(thresholds));
    FRR = zeros(1, length(thresholds));
    
    for i = 1:length(thresholds)
        threshold = thresholds(i);
        FAR(i) = sum(impostorScores > threshold) / length(impostorScores); % False acceptance rate
        FRR(i) = sum(genuineScores <= threshold) / length(genuineScores); % False rejection rate
    end
    
    % Find the threshold where FAR and FRR are closest
    [~, minIndex] = min(abs(FAR - FRR)); % Index of closest FAR and FRR
    closestFAR = FAR(minIndex);
    closestFRR = FRR(minIndex);
    EER = (closestFAR + closestFRR) / 2; % Average FAR and FRR at the threshold
    
    % Save the results
    overallFAR = [overallFAR; closestFAR];
    overallFRR = [overallFRR; closestFRR];
    overallEER = [overallEER; EER];
    % Accuracy for the current user
    numCorrect = sum(predictedLabels == testLabels');
    accuracy = (numCorrect / length(testLabels)) * 100;
    overallAccuracy = [overallAccuracy; accuracy];
    
    % Display user-specific metrics
    fprintf('User %02d: Accuracy = %.2f%%, FAR = %.2f%%, FRR = %.2f%%, EER = %.2f%%\n', ...
        userID, accuracy, closestFAR * 100, closestFRR * 100, EER * 100);
end

% Display overall metrics
disp('Training and testing completed for all users.');
fprintf('Overall Test Accuracy: %.2f%%\n', mean(overallAccuracy));
fprintf('Overall FAR: %.2f%%\n', mean(overallFAR) * 100);
fprintf('Overall FRR: %.2f%%\n', mean(overallFRR) * 100);
fprintf('Overall EER: %.2f%%\n', mean(overallEER) * 100);
