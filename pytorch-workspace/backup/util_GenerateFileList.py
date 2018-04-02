import glob
import sys
import shutil
import os
import numpy as np
import json
import scipy.io as sio


def caseMsCeleb_splitTrainlistMakeTestlist(filelist):
    fin = file(filelist)
    fout_train = file(filelist+'_train', 'w+')
    fout_test = file(filelist+'_test', 'w+')

    last_label = -1
    while 1:
	line = fin.readline().strip('\n')
	if not line:
	    break
	name = line.split(' ')[0]
	label = int(line.split(' ')[1])
	if last_label != label:
	    fout_test.writelines(line + '\n')
	else:
	    fout_train.writelines(line + '\n')
	last_label = label
	
	

def caseMsCeleb_refineLabel(filelist):
    fin = file(filelist)
    fout = file(filelist+'_refine', 'w+')

    crnt_label = -1
    last_label = -1
    while 1:
	line = fin.readline().strip('\n')
	if not line:
	    break
	name = line.split(' ')[0]
	label = int(line.split(' ')[1])
	if last_label != label:
	    crnt_label = crnt_label + 1
	last_label = label
	fout.writelines(name + ' ' + str(crnt_label) + '\n')
	

def caseMsCeleb_removeFileWithLabel(trainlist, label_txt, threshold):
    #target_label = caseMsCeleb_removeSpecificLabel(label_txt, threshold)
    target_label = np.load('target_label.npy')
    fin = file(trainlist)
    fout = file(trainlist+'_removed', 'w+')
    
    while 1:
	line = fin.readline().strip('\n')
	if not line:
	    break
	name = line.split(' ')[0]
	label = line.split(' ')[1]
	if int(label) in target_label:
	    continue
	fout.writelines(line + '\n')
    
    


def caseMsCeleb_removeSpecificLabel(label_txt, threshold):
    label_list = np.loadtxt(label_txt)
    label_list = label_list.astype('int64') #z
    label_bincount = np.bincount(label_list) #a
    label_unique = np.unique(label_list) #w

    is_specificLabel = label_bincount==int(threshold) # q = a==1
    is_target = is_specificLabel & label_unique # zz = w&q  q
    is_target = is_target & label_unique # zz = w&q  q
    target_label = np.where(is_target==1) #qq
    #print(target_label)
    np.save('target_label', target_label)
    return target_label





def caseMsCeleb_makeListLbaled(fileA, fileB):
    finA = file(fileA)
    finB = file(fileB)
    fout = file(fileA+'_labeled', 'w+')

    lineB = finB.readline().strip('\n')
    crnt_nameB = lineB
    this_label = -1
    lastlabelA = -1
    while 1:
        lineA = finA.readline().strip('\n')
        if not lineA :
            break
        nameA = lineA.split(' ')[0]
        labelA = lineA.split(' ')[1]
        if crnt_nameB == nameA :
            if lastlabelA != labelA :
                this_label = this_label + 1
            fout.writelines(nameA + ' ' + str(this_label) + '\n')
            lineB = finB.readline().strip('\n')
            crnt_nameB = lineB
            lastlabelA = labelA
    fout.close()


def caseMegaFace_writeLandmarks(path):
    f = file(path)
    data = json.load(f)
    data = data['landmarks']
    data = data.encode('UTF-8')
    data = data.replace('(', '').replace('[','').replace(')','').replace(']','').replace(' ','')
    landmarks_list = data.split(',')
    landmarks68 = np.zeros((68,2))
    for i in range(68):
        landmarks68[i, 0] = landmarks_list[2*i]
        landmarks68[i, 1] = landmarks_list[2*i+1]
    eyesL = np.zeros((1,2))
    eyesR = np.zeros((1,2))
    for i in range(36,42,1):
        eyesL[0,0] = eyesL[0,0] + landmarks68[i,0]
        eyesL[0,1] = eyesL[0,1] + landmarks68[i,1]
        eyesR[0,0] = eyesR[0,0] + landmarks68[i+6,0]
        eyesR[0,1] = eyesR[0,1] + landmarks68[i+6,1]
    eyesL = eyesL/6
    eyesR = eyesR/6
    nose = landmarks68[30,:] + landmarks68[33,:]
    nose = nose/2
    mouthL = landmarks68[48,:]
    mouthR = landmarks68[54,:]
    result = np.zeros((5,2))
    result[0,:] = eyesL
    result[1,:] = eyesR
    result[2,:] = nose
    result[3,:] = mouthL
    result[4,:] = mouthR
    sio.savemat(path.split('.')[0] + '.mat', {'landmarks':result})


def util_splitFileIntoTwo(inputfile, targetStr=' '):
    fout_L = file(inputfile.split('.')[0] + '_splitedL.txt', 'w+')
    fout_R = file(inputfile.split('.')[0] + '_splitedR.txt', 'w+')
    reader = file(inputfile)
    while 1:
        line = reader.readline()
        if not line:
            break
	splitindex = line.find(' ')
	strL = line[0:splitindex]
	strR = line[splitindex+1:len(line)-1]

        fout_L.writelines(strL + '\n')
        fout_R.writelines(strR + '\n')





def util_splitFileColumn(inputfile, targetStr=' '):
    fout_L = file(inputfile.split('.')[0] + '_splitedL.txt', 'w+')
    fout_R = file(inputfile.split('.')[0] + '_splitedR.txt', 'w+')
    reader = file(inputfile)
    while 1:
        line = reader.readline()
        if not line:
            break
        parsedLine = line.split(' ')
        strL = parsedLine[0]
        strR = parsedLine[1]
        fout_L.writelines(strL + '\n')
        fout_R.writelines(strR + '\n')


def util_GenFileList(folder):
    '''
    for webface format
    '''
    label = -1
    fout_list = file(folder+'_list.txt', 'w+')
    fout_label = file(folder+'_label.txt', 'w+')
    fout_fileWithLabel = file(folder+'.txt', 'w+')

    current_parsename = ''
    for filename in glob.glob(folder + '/*/*.jpg'):
        c_filename = str(filename)
        foldername = c_filename.split('/')[1]
        if foldername != current_parsename :
            current_parsename = foldername
            label = label + 1
        fout_fileWithLabel.writelines(c_filename + ' ' + str(label) + '\n')
        fout_list.writelines(c_filename + '\n')
        fout_label.writelines(str(label) + '\n')

    fout_fileWithLabel.close()

def util_IterGenFileList(folder):
    f = file('folder_list.txt', 'w+')
    #for filename in glob.glob(folder + '/*'):
    util_IterWritePath(folder, f)
    f.close()

def util_IterWritePath(rootDir, fwrite):
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        if os.path.isdir(path):
            util_IterWritePath(path, fwrite)
        else:
            #do anything
            fwrite.writelines(str(path) + '\n' )
            caseMegaFace_writeLandmarks(path)

def caseID_createTrainList(filelist):
    caseID_removeReplicatedIDphoto(filelist)



def caseID_removeReplicatedIDphoto(filelist):
    output_filename = filelist.split('.')[0] + '_removedRepID.txt'
    fout_classified = file(output_filename, 'w+')
    reader = open(filelist)
    hasWriteIDPhoto = False
    crnt_label = ''

    while True:
        line = reader.readline()
        if not line :
            break
        parsedLine = line.split('/')
        label = parsedLine[len(parsedLine)-3]
        #print(label)
        img_type = parsedLine[len(parsedLine)-1]
        img_type = img_type[0: len(img_type)-1]# remove \n
        if label != crnt_label :
            crnt_label = label
            hasWriteIDPhoto = False
            if img_type == 'card.bmp' :
                fout_classified.writelines(line)
                hasWriteIDPhoto = True
            else :
                fout_classified.writelines(line)
        else :
            if img_type != 'card.bmp' :
                fout_classified.writelines(line)
            if img_type == 'card.bmp' :
                if hasWriteIDPhoto==False :
                    fout_classified.writelines(line)
                    hasWriteIDPhoto = True

    fout_classified.close()
    return output_filename

def util_classifyImageList(filelist):
    fout_classified = file(filelist.split('.')[0] + '_classified.txt', 'w+')
    fout_label = file(filelist.split('.')[0] + '_label.txt', 'w+')
    reader = open(filelist)
    crnt_label = ''
    index_label = -1
    while True:
        line = reader.readline()
        line = line[0:len(line)-1] # remove \n
        if not line :
            break
        parsedLine = line.split('/')
        label = parsedLine[len(parsedLine)-3]
        if label != crnt_label :
            crnt_label = label
            index_label = index_label + 1
            fout_classified.writelines(line + ' ' + str(index_label) + '\n')
            fout_label.writelines(str(index_label) + '\n')
        else :
            fout_classified.writelines(line + ' ' + str(index_label) + '\n')
            fout_label.writelines(str(index_label) + '\n')




def forgenerate_augment():
    f = file("output.txt", "w+")
    for filename in glob.glob(r'webface/*/*.jpg'):
        c_filename = str(filename)
        f.writelines(c_filename[8:len(c_filename)].split('.')[0] + ',' + filename + ',None\n')
    f.close()

def for_augment_webface() :
    count = 0
    label = 0
    fout_train = file('aug_train.txt', 'w+')
    fout_val = file('aug_val.txt', 'w+')
    basename = '/mnt/augment_webface/'
    #for filename in glob.glob(r'/mnt/augment_webface/'):
    while 1:
        if label > 10574 :
            break
        name = str(count).zfill(6)
        foldername = basename + name
        if os.path.exists(foldername):
            val_flag = True
            for img_file in glob.glob(foldername + '/*.jpg'):
                c_imgfile = str(img_file)
                if val_flag == True :
                    fout_val.writelines(c_imgfile + ' ' + str(label) + '\n')
                    val_flag = False
                else :
                    fout_train.writelines(c_imgfile + ' ' + str(label) + '\n')
            for img_file in glob.glob(foldername + '/*/*.jpg'):
                c_imgfile = str(img_file)
                fout_train.writelines(c_imgfile + ' ' + str(label) + '\n')
            label = label + 1
        count = count + 1







def write_filelist_for_id_data():
    f = file("wenface_id_data_list.txt", "w+")
    for filename in glob.glob(r'CAISAdataset/CAISAdataset/*/*.jpg'):
        f.writelines(filename + '\n')
    f.close()

def write_filelist(path):
    '''
    :param path: folder path
    :return: recursively return all path of specific files in the folder
    '''
    numLine = 0
    for filename in glob.glob(r'unique_cacd/*/*.jpg'):
        numLine = numLine + 1
    f = file("filelist.txt", "w+")
    f.writelines(str(numLine) + '\n')
    for filename in glob.glob(r'unique_cacd/*/*.jpg'):
        f.writelines(filename + '\n')
    f.close()

def remove_line_with_specific_char(path):
    reader = open(path)
    writer = file("target.txt", "w+")
    while 1:
        line = reader.readline()
        if not line:
            break
        if line.count(' ') != 4 :
            continue
        writer.writelines(line)
    writer.close()

def countLine(path):
    numLine = 0
    reader = open(path)
    while 1:
        line = reader.readline()
        if not line:
            break
        numLine = numLine + 1
    return numLine

def strseg(path, target_char):
    reader = open(path)
    writer = file("seg_result.txt", "w+")
    while 1:
        line = reader.readline()
        if not line:
            break
        seg_temp = line[0:line.index(target_char)]
        writer.writelines(seg_temp + '\n')
    writer.close()

def delete_folder_which_in_list(path, folderlist):
    reader = open(folderlist)
    while 1:
        line = reader.readline()
        if not line:
            break
        folder_name = line[0:len(line)-2]
        if os.path.exists(path + "/" + folder_name)==0:
            continue
        shutil.rmtree(path + "/" + folder_name)


if __name__ == '__main__':
    #remove_line_with_specific_char('bbox.txt')
    #write_filelist_for_id_data()
    #for_augment_webface()
    #forgenerate()
    folder = sys.argv[1]
    #util_GenFileList(folder)
    #delete_folder_which_in_list('cacd_temp', 'unique_cacd_in_lfw.txt')

    util_IterGenFileList(folder)
    #caseID_createTrainList(sys.argv[1])

    #util_classifyImageList(sys.argv[1])
    #util_splitFileColumn(sys.argv[1], sys.argv[2])
    #util_splitFileIntoTwo(sys.argv[1], sys.argv[2])
    #caseMegaFace_writeLandmarks(sys.argv[1])
    #caseMsCeleb_makeListLbaled(sys.argv[1], sys.argv[2])
    #caseMsCeleb_removeSpecificLabel(sys.argv[1], sys.argv[2])
    #caseMSCeleb_removeFileWithLabel(sys.argv[1], sys.argv[2], sys.argv[3])
    #caseMsCeleb_refineLabel(sys.argv[1])
    #caseMsCeleb_splitTrainlistMakeTestlist(sys.argv[1])


