%% Fetching data
if ~exist('gt/AIC19/train.mat','file')
    fprintf('Downloading ground truth...\n');
    url = 'https://drive.google.com/uc?export=download&id=1_fWMD3kMawp9hOaHnkpCKIYgpt9vufsQ';
    if ~exist('gt','dir'), mkdir('gt'); end
    if ~exist('gt/AIC19','dir'), mkdir('gt/AIC19'); end
    filename = 'gt/AIC19/train.mat';
    if exist('websave','builtin')
      outfilename = websave(filename,url); % exists from MATLAB 2014b
    else
      outfilename = urlwrite(url, filename);
    end
end
if ~exist('res/AIC19/baseline','dir') || ~exist('res/AIC19/baseline/train.txt','file') %|| ~exist('res/AIC19/baseline/test.txt','file')
    fprintf('Downloading baseline tracker output...\n');
    url = 'https://drive.google.com/uc?export=download&id=1qQ1PJXyrdKb8kU0NWlZtAlWfB4oFv__O';
    if ~exist('res','dir'), mkdir('res'); end
    if ~exist('res/AIC19','dir'), mkdir('res/AIC19'); end
    if ~exist('res/AIC19/baseline','dir'), mkdir('res/AIC19/baseline'); end
    filename = 'res/AIC19/baseline/baseline.zip';
    if exist('websave','builtin')
      outfilename = websave(filename,url); % exists from MATLAB 2014b
    else
      outfilename = urlwrite(url, filename);
    end
    unzip(outfilename,'res/AIC19/baseline/');
    delete(filename);
    % Convert to motchallenge format: Frame, ID, left, top, right, bottom,
    % worldX, worldY
    output_train = dlmread('res/AIC19/baseline/train.txt');
    %output_test = dlmread('res/AIC19/baseline/test.txt');
    
    for cam = 1:40
        filter_train = output_train(:,1) == cam;
        data_train = output_train(filter_train,:);
        data_train = data_train(:,2:end);
        data_train(:,[1 2]) = data_train(:,[2 1]);
        if ~isempty(data_train)
            dlmwrite(sprintf('res/AIC19/baseline/c%03d_train.txt', cam), data_train, 'delimiter', ',', 'precision', 6);
        end

%         filter_test = output_test(:,1) == cam;
%         data_test = output_test(filter_test,:);
%         data_test = data_test(:,2:end);
%         data_test(:,[1 2]) = data_test(:,[2 1]);
%         if ~isempty(data_test)
%             dlmwrite(sprintf('res/AIC19/baseline/c%03d_test.txt', cam), data_test, 'delimiter', ',', 'precision', 6);
%         end
    end
end

%% Evaluation
[allMets, metsBenchmark, metsMultiCam] = evaluateTracking('AIC19-train.txt', 'res/AIC19/baseline/', 'gt/AIC19', 'AIC19');
