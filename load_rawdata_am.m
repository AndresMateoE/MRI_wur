%% MRI_RawDataToImage.m
% Script designed to directly read raw data of the Bruker 14T/600MHz and
% 7T/300MHz MRI systems at MAGNEFY. Raw data is converted to a structured
% k-space and subsequently converted to the image space. Data is saved as a
% cell array in the desired location. 


function load_rawdata_am(read_path, save_path)
    addpath('C:\Users\mateo006\Documents\Python codes\MRI_wur\additional', 'C:\Users\mateo006\Documents\Python codes\MRI_wur\datatypes');
    addpath 'C:\Users\mateo006\Documents\Python codes\Matlab';
    addBrukerPaths; %maybe this folder should be in the python codes folder also.

    %read_path = 'C:\Users\mateo006\Documents\MRI\AM7T_250121_SPC_extrudate_1_1_20250121_093706\2\';
    %save_path = 'C:\Users\mateo006\Documents\Processed_data\250203_Test\2\';

    read_path = [read_path, filesep];
    save_path = [save_path, filesep];
    
    %% Initialise 
    % We will use this from python so the function will eat a sigle path
    %read_path = 'C:\Users\mateo006\Documents\MRI\AM7T_250121_SPC_extrudate_1_1_20250121_093706\1\';
    %save_path = 'C:\Users\mateo006\Documents\Processed_data\250131_Test\1';
    dir_experiment = read_path;
    
    
    %Initialise matrix
    MRIdata = cell(size(dir_experiment,2),7); %initialise empty cell matrix to store data
    %disp(MRIdata);
    MRIdata{1,1} = "image data"; MRIdata{1,2} = "k-space"; MRIdata{1,3} = "x-axis"; MRIdata{1,4} = "y-axis"; 
    %disp(MRIdata);
    MRIdata{1,5} = "expnum"; MRIdata{1,6} = "TE, TR, AVG, REP, SLI, RAR, THK"; MRIdata{1,7} = "SEQ"; MRIdata{1,8} = "MISC";
    %disp(MRIdata);
    
    for expr = 1:size(dir_experiment,2)  %I dont get this, but maybe if you have multiples exp at once
        % set paths and load in data
        dir_data = dir_experiment;
    
        Acqp = readBrukerParamFile([dir_data, filesep, 'acqp']); %read acquisition parameters
        Method = readBrukerParamFile([dir_data, filesep, 'method']); %read method file
        rawObj = RawDataObject(dir_data); % Create a RawDataObject, importing the test data
    
    
        %check if all required fields are present and set default values if not
        if ~isfield(Method,'PVM_RareFactor');   Method.PVM_RareFactor = 1;      end
        if ~isfield(Acqp,'NI');                 Acqp.NI = 1;                    end
        if ~isfield(Method,'PVM_NRepetitions'); Method.PVM_NRepetitions = 1;    end
        if ~isfield(Method,'PVM_NAverages');    Method.PVM_NAverages = 1;       end
        if ~isfield(Method,'PVM_NEchoImages');    Method.PVM_NEchoImages = 1;   end
        if ~isfield(Method,'PVM_EncSteps1');    warning('Slice Encoding steps not found in Method file');        end
        if ~isfield(Method,'PVM_ObjOrderList'); warning('Object order not defined in Method file, linear distribution chosen');
            Method.PVM_ObjOrderList = linspace(0,Acqp.NI,Acqp.NI-1); end
        if ~isfield(Acqp,'ACQ_fov');            warning('FOV steps not found in Acqp file'); Acqp.ACQ_fov = 1;   end
    
    
        %set parameters
        rawdata = squeeze(rawObj.data{1}); %take all the raw data
        numSequence = Method.Method;
        numRare = Method.PVM_RareFactor;
        numSlices = Method.PVM_SPackArrNSlices;
        numRepetitions = Method.PVM_NRepetitions;
        numEchoes = Method.PVM_NEchoImages;
        numAverages = Method.PVM_NAverages;
        slice_order = Method.PVM_EncSteps1; %Order of line acquisition by ParaVision
        image_order = Method.PVM_ObjOrderList+1; %images are saved interlaced in the raw data [0 2 1 3]
        
    end
    %%
    writematrix(rawdata, [save_path, filesep, 'rawdata.csv']);
    
    disp('Process finished and csv file saven in:');
    disp(save_path);
end
