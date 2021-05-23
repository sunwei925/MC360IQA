clear
clc
% Which image are converted to the cubic images
startIndex = 1;
endIndex = 544;
indexRange = startIndex:endIndex;
indexYaw = 0:2:358;

readDatabasePath = 'E:\data\CVIQ';
wirteDatabasePath = 'E:\CVIQcubic';

names = {'Front Face', 'Right Face', 'Back Face', 'Left Face', ...
    'Top Face', 'Bottom Face'};
namesToSave = {'F','R','BA','L','T','BO'};

disp(['*****************************************************************'])
disp(['**********************Start Converting **************************'])
% wait bar
h=waitbar(0,'please wait');
for i = indexRange
    % read the image
    img = imread([readDatabasePath,'\',num2str(i,'%03d'),'.png']);    
    mkdir([wirteDatabasePath,'\',num2str(i,'%03d')])
    for j = indexYaw
        disp(['The current image is ',num2str(i,'%03d'),', current yaw is ',num2str(j,'%03d')])
        tic
        mkdir([wirteDatabasePath,'\',num2str(i,'%03d'),'\',num2str(j,'%03d')])
        out = equi2cubic(img,j);
        for idx = 1:numel(names)
            imwrite(out{idx},[wirteDatabasePath,'\',num2str(i,'%03d'),'\',num2str(j,'%03d'),'\',num2str(i,'%03d'),num2str(j,'%03d'),namesToSave{idx},'.jpg']);
        end
        consumeTime = toc;
        disp(['It consumes ',num2str(consumeTime,'%.3f'),' seconds'])
        
        str=['running...',num2str(((i-startIndex)*180+j/2)/((endIndex-startIndex)*180)*100),'%'];
        waitbar(((i-startIndex)*180+j/2)/((endIndex-startIndex)*180),h,str)
    end 
    

end
delete(h);