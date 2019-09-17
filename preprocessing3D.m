function [ slices, mask ] = preprocessing3D( slices, mask )
%PREPROCESSING3D Implements preprocessing of a 3D volume containing slices 
%of a FLAIR, pre-contrast or post-contrast modality together with its 
%segmentation mask.
%
%Examples:
%
%   Basic usecase:
%
%       [slices, mask] = preprocessing3D(slices, mask);
%   
%   If you don't have a segmentation mask and want to preprocess test 
%   images for inference, pass a zeros matrix instead:
%
%       [slices, mask] = preprocessing3D(slices, zeros(size(slices)));
%
%   Saving combined preprocessed slices from 3 modalities:
%
%       rootPath = '/media/username/data/train/';
%       case_id = 'patient_001';
%       [pre, mask] = preprocessing3D(pre, mask);
%       [flair, ~] = preprocessing3D(pre, mask);
%       [post, ~] = preprocessing3D(pre, mask);
%       options.color = true;    
%       for s = size(mask, 3):-1:1
%           preModality = pre(:, :, s);
%           flairModality = flair(:, :, s:);
%           postModality = post(:, :, s);
% 
%           image = cat(3, preModality, flairModality, postModality);
% 
%           saveastiff(image, [rootPath case_id '_' num2str(s) '.tif'], options);
%           saveastiff(mask(:, :, s), [rootPath case_id '_' num2str(s) '_mask.tif']);
%       end

    %make sure that the max in mask is 1 and then rest is below it (divide
    %by 255 to keep it 1-14 in the uint8 and by max(max(max(mask))) to
    %equally divide them out
    mask = mask/14.0;

    % resize to have smaller dimension equal 256 pixels
    if min(size(slices(:, :, 1))) ~= 128

        scale = 128 / min(size(slices(:,:,1)));
        % resize images to 256 with bicubic interpolation
   
        slices = imresize(slices, scale);
  
        % and mask with NN interpolation
        mask = imresize(mask, scale, 'method', 'nearest');

    end

    % center crop to 256x256 square
    slices = center_crop(slices, [128 128]);
    mask = center_crop(mask, [128 128]);

    % fix the rage of pixel values after bicubic interpolation
    slices(slices < 0) = 0;

    % get histogram of an image volume
    [N, edges] = histcounts(slices(:), 'BinWidth', 2);

    % rescale the intensity peak to be at value 100
    minimum = 0; %prctile(slices(slices ~= 0), 3);
    
    start = find(edges >= minimum, 1);
    [~, ind] = max(N(start:end));
    peak_val = edges(ind + start - 1);
    maximum = minimum + max(slices(:)); %((peak_val - minimum)* 20); %2.55);

    slices(slices < minimum) = minimum;
    slices(slices > maximum) = maximum;
    slices = (slices - minimum) ./ (maximum - minimum);
    disp(peak_val)
    disp(max(slices(:)))
    % preprocessed images
    slices = im2uint8(slices);
    mask = im2uint8(mask);

end


function [ image ] = center_crop( image, cropSize )
%CENTER_CROP Center crop of given size

    [p3, p4, ~] = size(image);

    i3_start = max(1, floor((p3 - cropSize(1)) / 2));
    i3_stop = i3_start + cropSize(1) - 1;

    i4_start = max(1, floor((p4 - cropSize(2)) / 2));
    i4_stop = i4_start + cropSize(2) - 1;

    image = image(i3_start:i3_stop, i4_start:i4_stop, :);

end