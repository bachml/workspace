clear;clc;close all;

%% caffe setttings
matCaffe = fullfile(pwd, '../../../matlab')
addpath(genpath(matCaffe));
gpu = 1;
if gpu
   gpu_id = 1;
   caffe.set_mode_gpu();
   caffe.set_device(gpu_id);
else
   caffe.set_mode_cpu();
end
caffe.reset_all();
model = '../evaluation/deploy.prototxt';
weights = '../snapshots/fer2013_experiment_iter_10000.caffemodel';
% caffe.set_mode_gpu();
net = caffe.Net(model, weights, 'test'); % create net and load weights

fileList = importdata('/data/jason/fer2013_dataset/list/fer2013_test_full.txt');
IMAGE_DIM = 256;
CROPPED_DIM = 227;
mean_data = 127.5;
count = 0;
for k = 1:size(fileList.data,1)
    k
    input_imagePath = char(fileList.textdata(k));
    img = imread(input_imagePath);
    if size(img, 3)==1
       img = repmat(img, [1,1,3]);
    end
    im_data = img(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single
    im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data bicubic bilinear
    im_data = (im_data - mean_data) / 128;  % subtract mean_data (already in W x H x C, BGR)
    
    crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
    indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
    n = 1;
    for i = indices
      for j = indices
        crops_data(:, :, :, n) = im_data(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :);
        crops_data(:, :, :, n+5) = crops_data(end:-1:1, :, :, n);
        n = n + 1;
      end
    end
    center = floor(indices(2) / 2) + 1;
    crops_data(:,:,:,5) = ...
      im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
    crops_data(:,:,:,10) = crops_data(end:-1:1, :, :, 5);
    
    res = net.forward({crops_data});
    prob = res{1};
    [a,b] = max(sum(prob,2));
    if fileList.data(k) == b-1
        count = count + 1;
    end

%     caffe_ft = net.blobs('fc6_new').get_data();
end

count/size(fileList.data,1)
    
img = imread('/data/jason/fer2013_dataset/data/FER_32313_1_3.jpg');
%     img     = single(img);
%     img     = (img - 127.5)/128;
%     img     = permute(img, [2,1,3]);
if size(img, 3)==1
   img = repmat(img, [1,1,3]);
end
mean_data = 127.5;

IMAGE_DIM = 256;
CROPPED_DIM = 224;


% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
im_data = img(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single


im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bicubic');  % resize im_data
im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)

% oversample (4 corners, center, and their x-axis flips)
crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
n = 1;
for i = indices
  for j = indices
    crops_data(:, :, :, n) = im_data(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :);
    crops_data(:, :, :, n+5) = crops_data(end:-1:1, :, :, n);
    n = n + 1;
  end
end
center = floor(indices(2) / 2) + 1;
crops_data(:,:,:,5) = ...
  im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
crops_data(:,:,:,10) = crops_data(end:-1:1, :, :, 5);