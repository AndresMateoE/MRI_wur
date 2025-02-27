function [ frame ] = bruker_getTranspositionFrame( data, Visu, numberFrame )
% [ frame ] = bruker_getTranspositionFrame( data, Visu, numberFrame )
% generates one frame of the dataset with transposition
% 
% IN:
%   data: the image matrix stored in the ImageDataObject or generated with
%         readBruker2dseq
%   Visu: a parameterstruct of visu-parameters
%   numberFrame: the number of the Frame, which should be returned has to
%                be one single integer value 
%    
% OUT:
%   frame: 4-dimensional Image Matrix of one frame

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (c) 2012
% Bruker BioSpin MRI GmbH
% D-76275 Ettlingen, Germany
%
% All Rights Reserved
%
% $Id$
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Check input
if ~isfield(Visu, 'VisuCoreTransposition')
    frame=data(:,:,:,:,numberFrame);
else
    cellstruct{1}=Visu;
    all_here = bruker_requires(cellstruct, {'Visu','VisuCoreTransposition', 'VisuCoreSize', 'VisuCoreFrameCount', 'VisuCoreDim'});
    clear cellstruct;
    if ~all_here
        error('Some parameters are missing');
    end

    if numberFrame < 1 && numberFrame >  Visu.VisuCoreFrameCount
        error('numberFrame is not correct')
    end

    %% localize Variables
    VisuCoreSize=Visu.VisuCoreSize;
    VisuCoreTransposition=Visu.VisuCoreTransposition;
    VisuCoreDim=Visu.VisuCoreDim;

    %% Reshape
    dims=size(data);
    if length(dims)>5
        data=reshape(data,[size(data,1), size(data,2), size(data,3), size(data,4), prod(dims(5:end))]);
    end

    %% start Transposition
    if VisuCoreTransposition(numberFrame) >0
        new_Size=VisuCoreSize;
        ch_dim1=mod(VisuCoreTransposition(numberFrame), VisuCoreDim)+1;
        ch_dim2=VisuCoreTransposition(numberFrame)-1+1;
        tmp=new_Size(ch_dim1);
        new_Size(ch_dim1)=VisuCoreSize(ch_dim2);
        new_Size(ch_dim2)=tmp;
        frame=reshape(data(:,:,:,:,numberFrame), new_Size);

    else
       frame=data(:,:,:,:,numberFrame); 
    end
end
end

