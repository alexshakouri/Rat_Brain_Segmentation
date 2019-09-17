
%% Read in the image and the mask

day = ['D03'; 'D07'; 'D28'];

name = ['Veh01 R14-192'; 'Veh02 R14-200']; 
%Training 
%'Veh02 R14-200'; 'Veh03 R14-211'; 'Veh04 R14-350'; 'Veh05 R14-360'; 'Veh06 R14-238'; 'Veh07 R14-256'; 'Veh08 R14-266'; 'Veh09 R14-276'; 'Veh10 R14-286'];
%'MDZ02 R14-195'; 'MDZ03 R14-209'; 'MDZ04 R14-346'; 'MDZ05 R14-355'; 'MDZ06 R14-236'; 'MDZ07 R14-251'; 'MDZ08 R14-265'; 'MDZ09 R14-272'; 'MDZ10 R14-279'];
%'DZP02 R14-194'; 'DZP03 R14-208'; 'DZP04 R14-348'; 'DZP05 R14-358'; 'DZP06 R14-231'; 'DZP07 R14-249'; 'DZP08 R14-260'; 'DZP09 R14-270'; 'DZP10 R14-285']; 
%'DFP02 R14-196'; 'DFP03 R14-206'; 'DFP04 R14-410'; 'DFP05 R14-412'; 'DFP06 R14-235'; 'DFP07 R14-250'; 'DFP08 R14-259'; 'DFP09 R14-273'; 'DFP10 R14-280']; %'DFP02 R14-196'; 'Veh01 R14-192'; 'Veh02 R14-200'];

%Validation
%'DFP01 R14-189'; 'DZP01 R14-187'; 'MDZ01 R14-190'; 'Veh01 R14-192';

%skipped files
%skipping: E:\Research\DIBS 3-28 Imaging Data\Image Data\DZP\DZP02 R14-194\outputPMOD\R14-194-D07-voi\R14-194-D07-voi.nii
%skipping: E:\Research\DIBS 3-28 Imaging Data\Image Data\MDZ\MDZ04 R14-346\outputPMOD\R14-346-D28-voi\R14-346-D28-voi.nii
%skipping: E:\Research\DIBS 3-28 Imaging Data\Image Data\VEH\Veh06 R14-238\outputPMOD\R14-238-D03-voi\R14-238-D03-voi.nii
%skipping: E:\Research\DIBS 3-28 Imaging Data\Image Data\VEH\Veh06 R14-238\R14-238 D07 T2w.img
%skipping: E:\Research\DIBS 3-28 Imaging Data\Image Data\VEH\Veh06 R14-238\R14-238 D28 T2w.img

for indexName = 1:size(name,1)
for indexDay = 1:size(day,1)
%Read in the nifty image as img
tempFileName = ['E:\Research\DIBS 3-28 Imaging Data\Image Data\VEH\' name(indexName,:) '\' name(indexName, 7:end) ' ' day(indexDay,:) ' T2w.img'];
if exist(tempFileName, 'file') == 0
    disp(['skipping: ', tempFileName])
    continue
end
imageRaw = fopen(tempFileName, 'r');
Iimg = fread(imageRaw, 'single', 'ieee-be');
%skip the first 348 bytes (header file)
Zimg = reshape(Iimg, 280,200,44);
%Have to roate it 180 degrees about z
Zimg = imrotate(Zimg,180);
%Zimg = imresize(Zimg, 0.25);

%Read in the nifty mask as raw (VALUES CORRECT/Spaceing CORRECT) 
tempFileName2 = ['E:\Research\DIBS 3-28 Imaging Data\Image Data\VEH\' name(indexName,:) '\outputPMOD\' name(indexName, 7:end) '-' day(indexDay,:) '-voi\' name(indexName, 7:end) '-' day(indexDay,:) '-voi.nii'];
if exist(tempFileName2, 'file') == 0
    disp(['skipping: ', tempFileName2])
    continue
end
imageRaw = fopen(tempFileName2, 'r');
I = fread(imageRaw, 'bit24');
%skip the first 348 bytes (header file)
Inew = I(119:end);
Zmask = reshape(Inew, 280,200,44);
%Zmask = imresize(Zmask, 0.25);

 %only keep certain label 
 %Zmask(Zmask ~= 1) = 0;
 %normalize
 %Zmask(Zmask == 1) = 1;

% %Plot the segmented image
% figure;
% imagesc(Zmask(:,:,26)' .* Zimg(:,:,26)')
% colorbar

%% Preprocess the image/mask
case_id = [ name(indexName,:) '-' day(indexDay,:)];



[slices, mask] = preprocessing3D(Zimg, Zmask);

figure;
imhist(slices(:))

% %Plot the segmented image
% figure;
% imagesc(slices(:,:,26)')
% colorbar
% 
%  figure;
%  imagesc(mask(:,:,26)')
%  colorbar
options.color = true;

for s = size(mask, 3):-1:1
    %One modality
    image = slices(:,:,s);
    
    imwrite(image,  ['E:\Research\Code\brain-segmentation-master\data\dataAllVal_128_testIMG\' case_id '_' num2str(s) '.tif']);
    %%saveastiff(image, ['dataAll_128/' case_id '_' num2str(s) '.tif']);
    imwrite(mask(:, :, s),['E:\Research\Code\brain-segmentation-master\data\dataAllVal_128_testIMG\' case_id '_' num2str(s) '_mask.tif']);
    %%saveastiff(mask(:, :, s), ['dataAll_128/' case_id '_' num2str(s) '_mask.tif'], options);   
end

%Multi-modality
%           preModality = pre(:, :, s);
%           flairModality = flair(:, :, s:);
%           postModality = post(:, :, s);
% 
%           image = cat(3, preModality, flairModality, postModality);
% 
%           saveastiff(image, [rootPath case_id '_' num2str(s) '.tif'], options);
%           saveastiff(mask(:, :, s), [rootPath case_id '_' num2str(s) '_mask.tif']);
%       end


%           saveastiff(mask(:, :, s), [rootPath case_id '_' num2str(s) '_mask.tif']);
end
end
