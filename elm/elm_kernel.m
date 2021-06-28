% function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,TY] = elm_kernel(P, T, Elm_Type, Regularization_coefficient, Kernel_type, Kernel_para)

% Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File           - Filename of training data set
% TestingData_File            - Filename of testing data set
% Elm_Type                    - 0 for regression; 1 for (both binary and multi-classes) classification
% Regularization_coefficient  - Regularization coefficient C
% Kernel_type                 - Type of Kernels:
%                                   'RBF_kernel' for RBF Kernel
%                                   'lin_kernel' for Linear Kernel
%                                   'poly_kernel' for Polynomial Kernel
%                                   'wav_kernel' for Wavelet Kernel
%Kernel_para                  - A number or vector of Kernel Parameters. eg. 1, [0.1,10]...
% Output: 
% TrainingTime                - Time (seconds) spent on training ELM
% TestingTime                 - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy            - Training accuracy: 
%                               RMSE for regression or correct classification rate for classification
% TestingAccuracy             - Testing accuracy: 
%                               RMSE for regression or correct classification rate for classification
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm_kernel('sinc_train', 'sinc_test', 0, 1, ''RBF_kernel',100)
% Sample2 classification: elm_kernel('diabetes_train', 'diabetes_test', 1, 1, 'RBF_kernel',100)
%
    %%%%    Authors:    MR HONG-MING ZHOU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       MARCH 2012

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

%%%%%%%%%%% Load training dataset
P = DataTrn;
T = LabTrn;                              %   Release raw training data array

%%%%%%%%%%% Load testing dataset
TV.P = DataTst; 
TV.T = LabTst;

                               %   Release raw testing data array
Regularization_coefficient = 1;
Elm_Type = 1;                           
Kernel_type = 'RBF_kernel';
Kernel_para = 100;
                             
                               
                               
                               
                                                     
                               
C = Regularization_coefficient;
NumberofTrainingData=size(P,1);
NumberofTestingData=size(TV.P,1);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(1,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(i,1) ~= label(j,1)
            j=j+1;
            label(j,1) = sorted_target(i,1);
        end
    end
    number_class=j;
    NumberofOutputNeurons=300;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(j,1) == T(i,1)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(j,1) == TV.T(i,1)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
                                              %   end if of Elm_Type
end

%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
n = size(T,2);
Omega_train = kernel_matrix(P',Kernel_type, Kernel_para);
OutputWeight=((Omega_train+speye(n)/C)\(T')); 
TrainingTime=toc

%%%%%%%%%%% Calculate the training output
Y=(Omega_train * OutputWeight)';                             %   Y: the actual output of the training data

%%%%%%%%%%% Calculate the output of testing input
tic;
Omega_test = kernel_matrix(P',Kernel_type, Kernel_para,TV.P');
TY=(Omega_test' * OutputWeight)';                            %   TY: the actual output of the testing data
TestingTime=toc

%%%%%%%%%% Calculate training & testing classification accuracy

if Elm_Type == REGRESSION
%%%%%%%%%% Calculate training & testing accuracy (RMSE) for regression case
    TrainingAccuracy=sqrt(mse(T - Y))
    TestingAccuracy=sqrt(mse(TV.T - TY))           
end

if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;

    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2)  
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)  
end
    
    
