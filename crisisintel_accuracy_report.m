%% CrisisIntel - automated model accuracy report and validation check (MATLAB)
% Reads the saved accuracy results of the ML models, builds a comparison
% chart, and runs a PASS/FAIL validation against a minimum accuracy threshold.
% Run from the CrisisIntel project root:  crisisintel_accuracy_report

clear; clc; close all;

%% 1. Load results
cycFile = fullfile('Backend','CYCLONE_BACKEND','outputs','model_accuracy_comparison.csv');
eqFile  = fullfile('Backend','EARTHQAUKE_BACKEND','earthquake_outputs','algorithm_accuracy_comparison.csv');

cyc = readtable(cycFile);   % columns: Algorithm, Accuracy, Accuracy_Percentage
eq  = readtable(eqFile);    % columns: Algorithm, Accuracy

%% 2. Build a combined summary table (accuracy in %)
cycPct = cyc.Accuracy * 100;
eqPct  = eq.Accuracy  * 100;

summary = table( ...
    [repmat("Cyclone",  height(cyc),1); repmat("Earthquake", height(eq),1)], ...
    [string(cyc.Algorithm); string(eq.Algorithm)], ...
    [cycPct; eqPct], ...
    'VariableNames', {'Model','Algorithm','AccuracyPercent'});

disp(summary);

%% 3. Grouped bar chart: algorithm vs. accuracy for each model
algos = ["Random Forest","SVM","Logistic Regression"];
data  = nan(numel(algos), 2);
for i = 1:numel(algos)
    data(i,1) = pickAcc(cyc, algos(i)) * 100;
    data(i,2) = pickAcc(eq,  algos(i)) * 100;
end

figure('Name','CrisisIntel accuracy comparison');
b = bar(categorical(algos, algos), data);   % keep RF, SVM, LR order (plain categorical() sorts A-Z and would mislabel bars)
ylabel('Test accuracy (%)');
title('CrisisIntel - model accuracy comparison');
legend({'Cyclone','Earthquake'}, 'Location','southoutside','Orientation','horizontal');
ylim([0 105]); grid on;
saveas(gcf, 'matlab_accuracy_comparison.png');

%% 4. Validation check: every model must reach a minimum accuracy
threshold = 60;   % percent, adjust as required
results = validateAccuracy(summary, threshold);
disp(results);

if all(results.Status == "PASS")
    fprintf('\nVALIDATION PASSED: all models >= %d%%\n', threshold);
else
    fprintf('\nVALIDATION FAILED: see table above\n');
end

writetable(results, 'matlab_validation_report.csv');

%% ---------- local functions ----------
function acc = pickAcc(T, algoName)
    idx = strcmp(string(T.Algorithm), algoName);
    if any(idx)
        acc = T.Accuracy(find(idx,1));
    else
        acc = NaN;
    end
end

function out = validateAccuracy(summary, threshold)
    status = repmat("PASS", height(summary), 1);
    status(summary.AccuracyPercent < threshold) = "FAIL";
    out = [summary, table(status, 'VariableNames', {'Status'})];
end
