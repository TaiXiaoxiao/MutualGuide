clc;clear;close all
addpath('common')
addpath('datasets')
addpath('elm')
addpath('parameters')
addpath('ERS')

%% load HSI datasets and set parameters
database  = 'Salinas';

if strcmp(database,'Indian')
    load('Indian_pines_corrected.mat');load('Indian_pines_gt.mat');
    load('RIP.mat');load('BIP_A.mat');load('BIP_B.mat');load('IWIP_A.mat');load('IWIP_B.mat');
    AF = 'ReLU';      num = 32;      pixel = 80;
    data3D = indian_pines_corrected;      label_gt = indian_pines_gt;
    h_score = 2.5;      l_score = 0.5;
    
elseif strcmp(database,'PaviaU')
    load PaviaU;load PaviaU_gt;
    load('RP.mat');load('BP_A.mat');load('BP_B.mat');load('IWP_A.mat');load('IWP_B.mat');
    AF = 'tanh';       num = 300;       pixel = 23;
    data3D = paviaU;       label_gt = paviaU_gt;
    h_score = 3;      l_score = 0.5;
    
elseif strcmp(database,'Salinas')
    load Salinas_corrected;load Salinas_gt;
    load('RS.mat');load('BS_A.mat');load('BS_B.mat');load('IWS_A.mat');load('IWS_B.mat');
    AF = 'sig';       num = 300;       pixel = 100;
    data3D = salinas_corrected;      label_gt = salinas_gt;
    h_score = 4;      l_score = 0.5;
end

%% SuperPCA as The Feature Extractor
num_PC            =  50;  %Optimal PCA dimension
num_Pixels        =   pixel*sqrt(2).^(-1:1); % Number of superpixels
for inum_Pixel = 1:size(num_Pixels,2)
    num_Pixel = num_Pixels(inum_Pixel);
    data3D = data3D./max(data3D(:));
    % super-pixels segmentation
    labels = cubseg(data3D,num_Pixel);
    % SupePCA DR
    [dataDR] = SuperPCA(data3D,num_PC,labels);
end
[M,N,D] = size(dataDR);

%% Remove the data with label=0
data1 = reshape(dataDR,M*N,D);
label1 = reshape(label_gt,M*N,1);
data = []; label = [];
pos = find((label1 ~= 0));
data = [data; data1(pos, :)];
label = [label; label1(pos)];
label=double(label);

%% Separate training samples and test samples
UniqueLabel = unique(label);
nUniqueLabel = size(UniqueLabel, 1); 
[ndata, ~] = size(data); 
num_train = round(0.05*ndata);
% R = randperm(ndata);
DataTrn  = data(R(1:num_train),:);
LabTrn  = label(R(1:num_train),:);
DataTst = data(R(num_train+1:ndata),:);
LabTst = label(R(num_train+1:ndata),:);

%% Classifier A first training and testing
% [IW1,B1,LW1,AF1,TYPE1] = elmtrain(DataTrn',LabTrn',4000,AF,1); 
LW1 = relmtrain(DataTrn',LabTrn',IW1,B1,AF,1);
[Score_A,PredictTst_A] = elmpredict(DataTst',IW1,B1,LW1,AF,1);
PredictTst_A = PredictTst_A';
% Calculate the initial accuracy
[OA,Kappa,producerA_A] = CalAccuracy(PredictTst_A,LabTst);
AA = mean(producerA_A(:));
fprintf('   ... ... The initial OA, AA, and Kappa of classifier A are %f, %f, %f ... ...\n', OA, AA, Kappa);

%% Classifier B first training and testing
% [IW2,B2,LW2,AF2,TYPE2] = elmtrain(DataTrn',LabTrn',4000,AF,1);
LW2 = relmtrain(DataTrn',LabTrn',IW2,B2,AF,1);
[Score_B,PredictTst_B] = elmpredict(DataTst',IW2,B2,LW2,AF,1);
PredictTst_B = PredictTst_B';
%Calculate the initial accuracy
[OA,Kappa,producerA_B] = CalAccuracy(PredictTst_B,LabTst);
AA = mean(producerA_B(:));
fprintf('   ... ... The initial OA, AA, and Kappa of classifier B are %f, %f, %f ... ...\n', OA, AA, Kappa);

%% Score processing
ScoreTst_A = sort(max(Score_A),2,'descend');
ScoreTst_B = sort(max(Score_B),2,'descend');
B = [];C = [];D = [];
DataTrn_A = DataTrn;
DataTrn_B = DataTrn;
LabTrn_A = LabTrn;
LabTrn_B = LabTrn;
DataRTst_A = DataTst;
DataRTst_B = DataTst;
LabRTst_A = LabTst;
LabRTst_B = LabTst;

%% Iteratively update data
iter = 9;
for x = 1:iter
    ScoreTst_A = sort(ScoreTst_A,2,'descend');
    ScoreTst_B = sort(ScoreTst_B,2,'descend');
    [~,Len_A] = size(ScoreTst_A);
    [~,Len_B] = size(ScoreTst_B);
    t1 = 0; t2 = 0;
    L1 = zeros(Len_A,1);
    L2 = zeros(Len_B,1);
    
    for i = 1:Len_A
        if ((ScoreTst_A(i) <= h_score)&&(ScoreTst_A(i) >= l_score))
            L1(i) = 1;
            t1 = t1+1;
            GuideScore_A(t1) = ScoreTst_A(i);
        end
    end
    for i = 1:Len_B
        if ((ScoreTst_B(i) <= h_score)&&(ScoreTst_B(i) >= l_score))
            L2(i) = 1;
            t2 = t2+1;
            GuideScore_B(t2) = ScoreTst_B(i);
        end
    end
    
    %Add the guide labels to the training set
    [~,Len_GA] = size(GuideScore_A);
    [~,Len_GB] = size(GuideScore_B);
    L3 = zeros(Len_GA,1);
    L4 = zeros(Len_GB,1);
    t3 = 0; t4 = 0;
    for i = 1:Len_GA
        if(GuideScore_A(i) >= GuideScore_A(num))
            L3(i) = 1;
            t3 = t3+1;
        end
    end
    for i = 1:Len_GB
        if(GuideScore_B(i) >= GuideScore_B(num))
            L4(i) = 1;
            t4 = t4+1;
        end
    end
    indexCor1 = find(L1==1);
    indexCor2 = find(L2==1);
    indexCor3 = indexCor1(find(L3==1));
    indexCor4 = indexCor2(find(L4==1));
    GuideLab_A = PredictTst_A(indexCor3);
    GuideLab_B = PredictTst_B(indexCor4);
    
    %The accuracy of the guide labels
    c = 0; d = 0;
    GT_A = LabRTst_A(indexCor1);
    GT_B = LabRTst_B(indexCor2);
    for q = 1:t3
        if(GuideLab_A(q) ~= GT_A(q))
            c = c+1;
        end
    end
    for q = 1:t4
        if(GuideLab_B(q) ~= GT_B(q))
            d = d+1;
        end
    end
    C = [C;c];
    D = [D;d];

    %% Construct training set and test set
    %comparative results
    result_A = [PredictTst_A(indexCor3),LabRTst_A(indexCor3)];
    result_B = [PredictTst_B(indexCor4),LabRTst_B(indexCor4)];
    
    %training set
    DataRTrn_A = [DataTrn_A;DataRTst_B(indexCor4,:)];
    LabRTrn_A = [LabTrn_A;PredictTst_B(indexCor4)];
    DataRTrn_B = [DataTrn_B;DataRTst_A(indexCor3,:)];
    LabRTrn_B = [LabTrn_B;PredictTst_A(indexCor3)];
    
    DataTrn_A = DataRTrn_A;
    LabTrn_A = LabRTrn_A;
    DataTrn_B = DataRTrn_B;
    LabTrn_B = LabRTrn_B;
    
    %test set
    DataRTst_A(indexCor3,:) = [];
    LabRTst_A(indexCor3,:) = [];
    DataRTst_B(indexCor4,:) = [];
    LabRTst_B(indexCor4,:) = [];
    [mtrain,~] = size(DataRTrn_A);
    [mtest,~] = size(DataRTst_A);
    
    %% Classifier A training and testing
    %training
    reLW1 = relmtrain(DataRTrn_A',LabRTrn_A',IW1,B1,AF,1);
    %testing
    [~,Predict_A] = elmpredict(DataTst',IW1,B1,reLW1,AF,1);
    [reScore_A,PredictRTst_A] = elmpredict(DataRTst_A',IW1,B1,reLW1,AF,1);
    PredictRTst_A = PredictRTst_A';
    
    %% Classifier B training and testing
    %training
    reLW2 = relmtrain(DataRTrn_B',LabRTrn_B',IW2,B2,AF,1);
    %testing
    [~,Predict_B] = elmpredict(DataTst',IW2,B2,reLW2,AF,1);
    [reScore_B,PredictRTst_B] = elmpredict(DataRTst_B',IW2,B2,reLW2,AF,1);
    PredictRTst_B = PredictRTst_B';
    
    %% Update prediction results and scores
    PredictTst_A = PredictRTst_A;
    PredictTst_B = PredictRTst_B;
    ScoreTst_A = max(reScore_A);
    ScoreTst_B = max(reScore_B);
end

%% Calculate the final accuracy
[OA_A,Kappa_A,producerA_A] = CalAccuracy(Predict_A,LabTst);
AA_A = mean(producerA_A(:));
fprintf('   ... ... The final OA, AA, and Kappa of classifier A are %f, %f, %f ... ...\n', OA_A, AA_A, Kappa_A);
[OA_B,Kappa_B,producerA_B] = CalAccuracy(Predict_B,LabTst);
AA_B = mean(producerA_B(:));
fprintf('   ... ... The final OA, AA, and Kappa of classifier B are %f, %f, %f ... ...\n', OA_B, AA_B, Kappa_B);

%% Display classification map
if(OA_A >= OA_B)
    Result_OA = OA_A;Result_AA = AA_A;Result_Kappa = Kappa_A;PredictTst = Predict_A';
else
    Result_OA = OA_B;Result_AA = AA_B;Result_Kappa = Kappa_B;PredictTst = Predict_B';
end
Result = [LabTrn;PredictTst];
for i = 1:ndata
    p = R(i);
    Prediction(p) = Result(i);
end
predictL = zeros (M,N);
for i = 1:ndata
    q = pos(i);
    predictL(q) = Prediction(i);
end
figure();
imshow(predictL);