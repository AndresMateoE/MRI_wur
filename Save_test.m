%% Save test

function Save_test() 
    rawdata = [1,2,3,4];
    save_path = 'C:\Users\mateo006\Desktop';
    writematrix(rawdata, [save_path, filesep, 'data_test.csv']);

end