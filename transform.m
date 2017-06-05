% points_input = load('/data/disk2/caffe/data/LFW/lfw-deepfunneled-points.txt');
% fileList = importdata('/data/disk2/caffe/data/LFW/lfw-deepfunneled.txt');
function [ output_arg ] = init( fileList, points_input)
% base_left_eye_x = 29;
% base_left_eye_y = 49;
% base_right_eye_x = 65;
% base_right_eye_y = 49;
% base_nose_x = 47;
% base_nose_y = 73;
% base_left_mouth_x  = 32;
% base_left_mouth_y  = 90;
% base_right_mouth_x  = 60;
% base_right_mouth_y  = 90;
base_left_eye_x = 89.3095;
base_left_eye_y = 72.9025;
base_right_eye_x = 169.3095;
base_right_eye_y = 72.9025;
base_nose_x = 127.8949;
base_nose_y = 127.0441;
base_left_mouth_x  = 96.8796;
base_left_mouth_y  = 184.8907;
base_right_mouth_x  = 159.1065;
base_right_mouth_y  = 184.7601;


base_points = [base_left_eye_x base_left_eye_y; base_right_eye_x base_right_eye_y; base_nose_x base_nose_y;...
    base_left_mouth_x base_left_mouth_y; base_right_mouth_x base_right_mouth_y];
base_points = base_points;
% arguments
% read_folder = './';
% write_folder = 'cropped_morph/';
read_folder = '/data/disk2/caffe/data/LFW/';
write_folder = '/data/disk2/caffe/data/LFW/align/';
facial_points = points_input ;
% face_size = [1 40];


id_cropping = -1;
aligned_facial_mat = [];
id_flag = '';
for image_k = 1:13233
    if mod(image_k,200) == 0
        image_k
    end
    input_imagePath = [read_folder, char(fileList(image_k))];
    image = imread(input_imagePath);
    id_name_index = strfind(char(fileList(image_k)),'/');
    tempPath = char(fileList(image_k));
    id_folder_name = tempPath(1:id_name_index(2) - 1);
    id_name_index
    id_folder_name
    save_folder = [write_folder, id_folder_name];
    if strcmp(id_flag, id_folder_name) == 0 % not equal, it's a new identity, then make a new folder
        mkdir(save_folder);
        id_flag = id_folder_name;
    end
    
    facial_temp = points_input(image_k, :);
    facial_X = facial_temp([1 3 5 7 9]);
    facial_Y = facial_temp([2 4 6 8 10]);
    facial_points = [facial_X; facial_Y]';
    
    tform = cp2tform(facial_points, base_points, 'nonreflective similarity');
%     crop_image = imtransform(image, tform, 'XData', [1 96], 'YData', [1 112]);
    crop_image = imtransform(image, tform, 'XData', [1 256], 'YData', [1 256]);
    %if size(crop_image,3) == 3
      %  crop_image = rgb2gray(crop_image);
   % end
    
    output_imagePath = [write_folder,char(fileList(image_k))];
    imwrite(crop_image,output_imagePath);
    
    U = facial_points(:,1);
    V = facial_points(:,2);
    [X, Y] = tformfwd(tform, U, V);
    X = round(X);
    Y = round(Y);
    aligned_facial_points = [X(1) Y(1) X(2) Y(2) X(3) Y(3) X(4) Y(4) X(5) Y(5)];
    aligned_facial_mat = [aligned_facial_mat; aligned_facial_points];
end
save aligned_points.mat aligned_facial_mat
end